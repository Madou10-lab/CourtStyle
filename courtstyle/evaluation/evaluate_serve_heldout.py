#!/usr/bin/env python3
"""Held-out evaluation of serve style assignment.

Assigns every record of a held-out tournament to a discovered serve style by
weighted k-nearest-neighbour voting (``courtstyle.evaluation.serve_inference``)
and compares the assignment with the expert-provided annotations of that
tournament. Reports accuracy, balanced accuracy and macro-F1 with bootstrap
confidence intervals, per-style precision, recall and F1, the confusion matrix,
and the association with each annotated style attribute.

The paper evaluates on the Australian Open 2017 (42 server-match records).

Usage:
    python -m courtstyle.evaluation.evaluate_serve_heldout \\
        --community-run-dir outputs/serve/community_run_<timestamp> \\
        --ckpt-path outputs/serve/serve_encoder_best.pt \\
        --preprocessor-path data/serve/processed/preprocessor.pkl \\
        --test-csv data/serve/test_set/australian_open_2017.csv \\
        --annotations-csv data/serve/test_set/australian_open_2017_annotations.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.metrics import precision_recall_fscore_support

# For figures
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================


# Style fields to evaluate
EVAL_FIELDS = [
    "speed_tier",
    "placement_bias",
    "variability",
    "risk",
    "payoff",
    "second_serve_resilience",
]

# Label normalization maps (annotation -> internal scoring vocab)
# Note: We normalize annotation labels to a consistent internal vocab for scoring
LABEL_NORMALIZATION = {
    "speed_tier": {
        "neutral": "mid",  # annotations use "neutral", scoring expects "mid"
        "low": "low",
        "high": "high",
        "unclear": "unclear",
    },
    "placement_bias": {
        # Normalize to lowercase
        "T": "t",
        "t": "t",
        "Wide": "wide",
        "wide": "wide",
        "Body": "body",
        "body": "body",
        "Mixed": "mixed",
        "mixed": "mixed",
        "unclear": "unclear",
    },
    "variability": {
        "neutral": "mid",
        "low": "low",
        "high": "high",
        "unclear": "unclear",
    },
    "risk": {
        "neutral": "mid",
        "low": "low",
        "high": "high",
        "unclear": "unclear",
    },
    "payoff": {
        "neutral": "mid",
        "low": "low",
        "high": "high",
        "unclear": "unclear",
    },
    "second_serve_resilience": {
        "weak": "weak",
        "neutral": "neutral",
        "strong": "strong",
        "unclear": "unclear",
    },
}

# Seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Utility Functions
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically."""
    import os
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
    os.rename(tmp_path, path)


def atomic_write_text(path: Path, content: str) -> None:
    """Write text file atomically."""
    import os
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        f.write(content)
    os.rename(tmp_path, path)


# =============================================================================
# Step 1: Run Inference (or Reuse)
# =============================================================================

def run_inference(
    test_csv: Path,
    community_run_dir: Path,
    out_dir: Path,
    device: str,
    k: Optional[int],
    ckpt_path_override: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run inference pipeline on test data.

    Returns assignment data dict.
    """
    # Import inference pipeline functions
    from courtstyle.evaluation.serve_inference import (
        load_preprocessor,
        preprocess_input,
        safe_torch_load,
        build_encoder,
        df_to_tensors,
        extract_embeddings,
        load_reference_data,
        compute_sigma,
        assign_knn_vote,
        assign_nearest_centroid,
        DEFAULT_PREPROCESSOR_PATH,
        DEFAULT_UNCERTAINTY_CONF_THRESHOLD,
        DEFAULT_UNCERTAINTY_DIST_THRESHOLD,
    )
    import torch
    import hashlib

    # Load run config
    run_config_path = community_run_dir / "run_config.json"
    with open(run_config_path) as f:
        run_config = json.load(f)

    # Resolve k
    k_val = k if k is not None else run_config.get("knn_k", 15)

    # Resolve checkpoint
    ckpt_path = ckpt_path_override or Path(run_config.get("ckpt_path", ""))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}. Pass --ckpt-path.")
    preprocessor_path = preprocessor_path or DEFAULT_PREPROCESSOR_PATH

    # Device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"  Loading preprocessor from {preprocessor_path}...")
    preprocessor = load_preprocessor(preprocessor_path)

    print(f"  Loading test data from {test_csv}...")
    raw_df = pd.read_csv(test_csv)
    print(f"    Loaded {len(raw_df)} rows")

    print("  Preprocessing...")
    processed_df = preprocess_input(raw_df, preprocessor)

    print(f"  Loading encoder from {ckpt_path}...")
    ckpt = safe_torch_load(ckpt_path, dev)
    encoder = build_encoder(ckpt, dev)

    cont_cols = ckpt["cont_cols"]
    cat_names = ckpt["cat_names"]

    x_cont, x_cat = df_to_tensors(processed_df, cont_cols, cat_names)

    print("  Extracting embeddings...")
    embeddings = extract_embeddings(
        encoder, x_cont, x_cat,
        cont_cols, cat_names,
        batch_size=256, device=dev,
    )

    print(f"  Loading reference data from {community_run_dir}...")
    ref_embeddings, ref_labels, centroids = load_reference_data(community_run_dir)
    community_ids = sorted(np.unique(ref_labels))

    sigma = compute_sigma(ref_embeddings, k_val)
    print(f"    sigma={sigma:.4f}, k={k_val}")

    print("  Assigning to communities (kNN vote)...")
    knn_results = assign_knn_vote(
        query_embeddings=embeddings,
        ref_embeddings=ref_embeddings,
        ref_labels=ref_labels,
        k=k_val,
        sigma=sigma,
        uncertainty_conf_threshold=DEFAULT_UNCERTAINTY_CONF_THRESHOLD,
        uncertainty_dist_threshold=DEFAULT_UNCERTAINTY_DIST_THRESHOLD,
    )

    centroid_preds = assign_nearest_centroid(embeddings, centroids, community_ids)

    # Build assignment results
    assignments = []
    for i in range(len(embeddings)):
        row_data = raw_df.iloc[i].to_dict()
        row_str = json.dumps(row_data, sort_keys=True, default=str)
        row_hash = hashlib.md5(row_str.encode()).hexdigest()[:12]

        # Get max confidence
        pred = knn_results["predictions"][i]
        confs = knn_results["confidences"][i]
        max_conf = confs.get(str(pred), confs.get(pred, 0))

        assignment = {
            "instance_id": f"ao2017_{i:06d}",
            "row_index": i,
            "row_hash": row_hash,
            "community_id": int(pred),
            "centroid_prediction": int(centroid_preds[i]),
            "max_confidence": float(max_conf),
            "top2_margin": float(knn_results["top2_margins"][i]),
            "uncertain": bool(knn_results["uncertain"][i]),
            "all_confidences": {str(k): float(v) for k, v in confs.items()},
        }
        assignments.append(assignment)

    # Save artifacts
    inference_dir = out_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_df = pd.DataFrame([{
        "instance_id": a["instance_id"],
        "row_index": a["row_index"],
        "community_id": a["community_id"],
        "centroid_prediction": a["centroid_prediction"],
        "max_confidence": a["max_confidence"],
        "top2_margin": a["top2_margin"],
        "uncertain": a["uncertain"],
    } for a in assignments])
    pred_df.to_csv(inference_dir / "predictions.csv", index=False)

    # assignment.json
    assignment_data = {
        "k": k_val,
        "sigma": float(sigma),
        "n_inputs": len(embeddings),
        "community_ids": [int(c) for c in community_ids],
        "assignments": assignments,
    }
    atomic_write_json(inference_dir / "assignment.json", assignment_data)

    # run_meta.json
    run_meta = {
        "test_csv": str(test_csv),
        "community_run_dir": str(community_run_dir),
        "ckpt_path": str(ckpt_path),
        "k": k_val,
        "sigma": float(sigma),
        "device": str(dev),
        "timestamp": datetime.now().isoformat(),
    }
    atomic_write_json(inference_dir / "run_meta.json", run_meta)

    print(f"    Saved inference artifacts to {inference_dir}")

    return assignment_data


def load_existing_assignment(assignment_json: Path) -> Dict[str, Any]:
    """Load existing assignment JSON."""
    with open(assignment_json) as f:
        return json.load(f)


# =============================================================================
# Step 2: Load and Validate Annotations
# =============================================================================

def load_and_validate_annotations(
    annotations_csv: Path,
    n_expected: int,
) -> pd.DataFrame:
    """
    Load annotations CSV and validate.

    Returns DataFrame with normalized labels for scoring.
    """
    df = pd.read_csv(annotations_csv)

    # Validate row count
    if len(df) != n_expected:
        raise ValueError(
            f"Annotation row count mismatch: expected {n_expected}, got {len(df)}"
        )

    # Validate instance_id uniqueness
    if df["instance_id"].nunique() != len(df):
        raise ValueError("instance_id is not unique in annotations")

    # Check for community_id_gt column (required for community-ID evaluation)
    if "community_id_gt" not in df.columns:
        print("\nERROR: 'community_id_gt' column not found in annotations.")
        print("See docs/data_format.md for the expected annotation format.")
        sys.exit(1)

    # Normalize labels for scoring
    for field in EVAL_FIELDS:
        if field not in df.columns:
            raise ValueError(f"Missing required field: {field}")

        norm_map = LABEL_NORMALIZATION.get(field, {})
        if norm_map:
            df[f"{field}_norm"] = df[field].map(
                lambda x: norm_map.get(str(x), str(x).lower())
            )
        else:
            df[f"{field}_norm"] = df[field].astype(str).str.lower()

    return df


# =============================================================================
# Step 3: Evaluation Metrics
# =============================================================================

def cramers_v(contingency_table: np.ndarray) -> Tuple[float, float]:
    """
    Compute Cramér's V and chi-square p-value from contingency table.

    Returns (cramers_v, p_value).
    """
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1

    if min_dim == 0 or n == 0:
        return 0.0, p

    v = np.sqrt(chi2 / (n * min_dim))
    return float(v), float(p)


def compute_majority_mapping(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, str]:
    """
    Compute majority label mapping for each predicted cluster.

    Returns dict mapping cluster_id -> majority_label.
    """
    mapping = {}
    for cluster_id in np.unique(predictions):
        mask = predictions == cluster_id
        cluster_labels = labels[mask]
        if len(cluster_labels) > 0:
            unique, counts = np.unique(cluster_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            mapping[int(cluster_id)] = str(majority_label)
    return mapping


def apply_majority_mapping(
    predictions: np.ndarray,
    mapping: Dict[int, str],
) -> np.ndarray:
    """Apply majority mapping to get predicted labels."""
    return np.array([mapping.get(int(p), "unknown") for p in predictions])


def compute_field_metrics(
    community_ids: np.ndarray,
    labels: np.ndarray,
    field_name: str,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a single field.

    Returns dict with NMI, ARI, Cramér's V, chi2_p, accuracy, macro_f1, balanced_accuracy.
    """
    # Filter out unclear labels
    mask = labels != "unclear"
    comm_filtered = community_ids[mask]
    labels_filtered = labels[mask]

    n_valid = len(labels_filtered)
    if n_valid == 0:
        return {
            "field": field_name,
            "n": 0,
            "nmi": np.nan,
            "ari": np.nan,
            "cramers_v": np.nan,
            "chi2_p": np.nan,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
        }

    # Convert labels to numeric for NMI/ARI
    label_to_int = {lbl: i for i, lbl in enumerate(np.unique(labels_filtered))}
    labels_int = np.array([label_to_int[lbl] for lbl in labels_filtered])

    # 3A: Association metrics
    nmi = normalized_mutual_info_score(labels_int, comm_filtered)
    ari = adjusted_rand_score(labels_int, comm_filtered)

    # Contingency table for Cramér's V
    unique_labels = sorted(np.unique(labels_filtered))
    unique_comms = sorted(np.unique(comm_filtered))

    cont_table = np.zeros((len(unique_comms), len(unique_labels)), dtype=int)
    comm_to_idx = {c: i for i, c in enumerate(unique_comms)}
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    for c, lbl in zip(comm_filtered, labels_filtered):
        cont_table[comm_to_idx[c], label_to_idx[lbl]] += 1

    v, chi2_p = cramers_v(cont_table)

    # 3B: Majority mapping scores
    majority_map = compute_majority_mapping(comm_filtered, labels_filtered)
    predicted_labels = apply_majority_mapping(comm_filtered, majority_map)

    accuracy = np.mean(predicted_labels == labels_filtered)

    # Balanced accuracy and macro-F1
    unique_true = np.unique(labels_filtered)
    bal_acc = balanced_accuracy_score(labels_filtered, predicted_labels)
    macro_f1 = f1_score(labels_filtered, predicted_labels, average="macro", zero_division=0)

    return {
        "field": field_name,
        "n": n_valid,
        "nmi": float(nmi),
        "ari": float(ari),
        "cramers_v": float(v),
        "chi2_p": float(chi2_p),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "majority_mapping": majority_map,
        "contingency_table": cont_table.tolist(),
        "contingency_labels": unique_labels,
        "contingency_communities": unique_comms,
    }


def compute_uncertainty_analysis(
    predictions_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute uncertainty analysis metrics.

    Returns dict with overall uncertain rate, per-community rates,
    and confidence-correctness relationship.
    """
    n_total = len(predictions_df)
    n_uncertain = predictions_df["uncertain"].sum()

    # Overall uncertain rate
    uncertain_rate = n_uncertain / n_total if n_total > 0 else 0

    # Per-community uncertain rate
    per_community = predictions_df.groupby("community_id").agg({
        "uncertain": ["sum", "count"],
    })
    per_community.columns = ["uncertain_count", "total"]
    per_community["uncertain_rate"] = per_community["uncertain_count"] / per_community["total"]

    # Confidence quartile analysis (using speed_tier as example field)
    # Compute correctness under majority mapping for each instance
    confidence_analysis = {}

    for field in EVAL_FIELDS:
        norm_col = f"{field}_norm"
        if norm_col not in merged_df.columns:
            continue

        # Get majority mapping for this field
        mask = merged_df[norm_col] != "unclear"
        if mask.sum() == 0:
            continue

        comm_ids = merged_df.loc[mask, "community_id"].values
        labels = merged_df.loc[mask, norm_col].values
        confs = merged_df.loc[mask, "max_confidence"].values

        majority_map = compute_majority_mapping(comm_ids, labels)
        predicted = apply_majority_mapping(comm_ids, majority_map)
        correct = (predicted == labels).astype(int)

        # Bin by confidence quartiles (handle duplicate edges gracefully)
        try:
            # First try with labels
            quartiles = pd.qcut(confs, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            # If duplicates cause fewer bins, don't use labels
            quartiles = pd.qcut(confs, q=4, duplicates="drop")

        quartile_df = pd.DataFrame({
            "quartile": quartiles,
            "correct": correct,
            "confidence": confs,
        })

        quartile_stats = quartile_df.groupby("quartile", observed=True).agg({
            "correct": ["mean", "count"],
            "confidence": ["min", "max", "mean"],
        })

        confidence_analysis[field] = {
            "quartile_accuracy": {
                str(q): float(quartile_stats.loc[q, ("correct", "mean")])
                for q in quartile_stats.index
            },
            "quartile_counts": {
                str(q): int(quartile_stats.loc[q, ("correct", "count")])
                for q in quartile_stats.index
            },
        }

    return {
        "overall_uncertain_rate": float(uncertain_rate),
        "n_uncertain": int(n_uncertain),
        "n_total": n_total,
        "per_community": per_community.to_dict("index"),
        "confidence_analysis": confidence_analysis,
    }


# =============================================================================
# Community-ID Evaluation (Pred vs GT)
# =============================================================================

def compute_community_id_metrics(
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute community-ID classification metrics (predicted vs ground truth).

    Args:
        pred_ids: Predicted community IDs
        gt_ids: Ground truth community IDs

    Returns:
        Dict with accuracy, balanced_accuracy, macro_f1, nmi, ari, confusion_matrix
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Filter out NaN ground truth
    mask = ~pd.isna(gt_ids)
    pred_filtered = pred_ids[mask].astype(int)
    gt_filtered = gt_ids[mask].astype(int)

    n_valid = len(gt_filtered)
    if n_valid == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "nmi": np.nan,
            "ari": np.nan,
            "confusion_matrix": [],
            "confusion_labels": [],
        }

    # Classification metrics
    accuracy = accuracy_score(gt_filtered, pred_filtered)
    bal_acc = balanced_accuracy_score(gt_filtered, pred_filtered)
    macro_f1 = f1_score(gt_filtered, pred_filtered, average="macro", zero_division=0)

    # Clustering metrics
    nmi = normalized_mutual_info_score(gt_filtered, pred_filtered)
    ari = adjusted_rand_score(gt_filtered, pred_filtered)

    # Confusion matrix (rows=GT, cols=Pred)
    labels = sorted(set(gt_filtered) | set(pred_filtered))
    conf_mat = confusion_matrix(gt_filtered, pred_filtered, labels=labels)

    return {
        "n": n_valid,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "nmi": float(nmi),
        "ari": float(ari),
        "confusion_matrix": conf_mat.tolist(),
        "confusion_labels": [int(l) for l in labels],
    }


def save_community_metrics_csv(metrics: Dict, out_path: Path) -> None:
    """Save community-ID metrics as CSV."""
    df = pd.DataFrame([{
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "nmi": metrics["nmi"],
        "ari": metrics["ari"],
    }])
    df.to_csv(out_path, index=False, float_format="%.4f")


def save_community_metrics_latex(metrics: Dict, out_path: Path) -> None:
    """Save community-ID metrics as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Community-ID Classification: Predicted vs Annotated}",
        r"\label{tab:test_eval_community}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"N & Accuracy & Bal. Acc. & Macro-F1 & NMI & ARI \\",
        r"\midrule",
    ]

    n = metrics["n"]
    acc = f"{metrics['accuracy']:.3f}" if not np.isnan(metrics['accuracy']) else "--"
    bal = f"{metrics['balanced_accuracy']:.3f}" if not np.isnan(metrics['balanced_accuracy']) else "--"
    f1 = f"{metrics['macro_f1']:.3f}" if not np.isnan(metrics['macro_f1']) else "--"
    nmi = f"{metrics['nmi']:.3f}" if not np.isnan(metrics['nmi']) else "--"
    ari = f"{metrics['ari']:.3f}" if not np.isnan(metrics['ari']) else "--"

    lines.append(f"{n} & {acc} & {bal} & {f1} & {nmi} & {ari} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    atomic_write_text(out_path, "\n".join(lines))


def save_community_confusion_csv(metrics: Dict, out_path: Path) -> None:
    """Save confusion matrix as CSV (rows=GT, cols=Pred)."""
    labels = metrics["confusion_labels"]
    conf_mat = np.array(metrics["confusion_matrix"])

    df = pd.DataFrame(conf_mat, index=labels, columns=labels)
    df.index.name = "gt_community"
    df.columns.name = "pred_community"
    df.to_csv(out_path)


def plot_community_confusion(metrics: Dict, out_dir: Path) -> None:
    """Plot confusion matrix for community-ID classification."""
    set_publication_style()

    labels = metrics["confusion_labels"]
    conf_mat = np.array(metrics["confusion_matrix"])

    # Compute proportions (normalize by row = GT)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_mat_norm = conf_mat / row_sums

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.heatmap(
        conf_mat_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[f"C{l}" for l in labels],
        yticklabels=[f"C{l}" for l in labels],
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Proportion'},
    )
    ax.set_xlabel("Predicted Community")
    ax.set_ylabel("Annotated Community")
    #ax.set_title("Confusion Matrix (Row-Normalized)")

    plt.tight_layout()
    fig.savefig(out_dir / "test_eval_community_confusion.png")
    fig.savefig(out_dir / "test_eval_community_confusion.pdf")
    plt.close(fig)


# =============================================================================
# Output Generation
# =============================================================================

def save_metrics_csv(metrics: List[Dict], out_path: Path) -> None:
    """Save metrics as CSV."""
    df = pd.DataFrame(metrics)
    cols = ["field", "NMI", "ARI", "cramers_v", "chi2_p", "accuracy", "macro_f1", "balanced_accuracy", "n"]

    # Rename for output
    df = df.rename(columns={"nmi": "NMI", "ari": "ARI"})

    # Reorder columns
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_path, index=False, float_format="%.4f")


def save_metrics_latex(metrics: List[Dict], out_path: Path) -> None:
    """Save metrics as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Test Set Evaluation: Community-Attribute Association}",
        r"\label{tab:test_eval_metrics}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Field & NMI & ARI & Cramér's V & Accuracy & Macro-F1 & Bal. Acc. \\",
        r"\midrule",
    ]

    for m in metrics:
        field = m["field"].replace("_", r"\_")
        nmi = f"{m['nmi']:.3f}" if not np.isnan(m['nmi']) else "--"
        ari = f"{m['ari']:.3f}" if not np.isnan(m['ari']) else "--"
        v = f"{m['cramers_v']:.3f}" if not np.isnan(m['cramers_v']) else "--"
        acc = f"{m['accuracy']:.3f}" if not np.isnan(m['accuracy']) else "--"
        f1 = f"{m['macro_f1']:.3f}" if not np.isnan(m['macro_f1']) else "--"
        bal = f"{m['balanced_accuracy']:.3f}" if not np.isnan(m['balanced_accuracy']) else "--"

        lines.append(f"{field} & {nmi} & {ari} & {v} & {acc} & {f1} & {bal} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    atomic_write_text(out_path, "\n".join(lines))


def save_contingency_csv(
    metrics: Dict,
    field: str,
    out_path: Path,
) -> None:
    """Save contingency table as CSV with proportions."""
    table = np.array(metrics["contingency_table"])
    labels = metrics["contingency_labels"]
    communities = metrics["contingency_communities"]

    # Create DataFrame with counts
    df = pd.DataFrame(table, index=communities, columns=labels)
    df.index.name = "community"

    # Add row totals
    df["total"] = df.sum(axis=1)

    # Add proportions
    for lbl in labels:
        df[f"{lbl}_prop"] = df[lbl] / df["total"]

    df.to_csv(out_path, float_format="%.3f")


# =============================================================================
# Figure Generation
# =============================================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def plot_uncertainty(
    predictions_df: pd.DataFrame,
    uncertainty_analysis: Dict,
    out_dir: Path,
) -> None:
    """Plot uncertainty analysis figures."""
    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Uncertain rate by community
    ax1 = axes[0]
    per_comm = uncertainty_analysis["per_community"]
    comms = sorted(per_comm.keys())
    rates = [per_comm[c]["uncertain_rate"] for c in comms]

    colors = sns.color_palette("coolwarm", len(comms))
    ax1.bar([f"C{c}" for c in comms], rates, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel("Uncertain Rate")
    ax1.set_xlabel("Community")
    ax1.set_title("Uncertainty by Community")
    ax1.set_ylim(0, max(0.5, max(rates) * 1.2) if rates else 0.5)

    # Add overall line
    overall = uncertainty_analysis["overall_uncertain_rate"]
    ax1.axhline(y=overall, color='red', linestyle='--', linewidth=1.5,
                label=f'Overall: {overall:.1%}')
    ax1.legend()

    # Right: Confidence histogram
    ax2 = axes[1]
    confs = predictions_df["max_confidence"].values

    ax2.hist(confs, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Max Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")
    ax2.axvline(x=np.median(confs), color='red', linestyle='--',
                label=f'Median: {np.median(confs):.2f}')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "test_eval_uncertainty.png")
    fig.savefig(out_dir / "test_eval_uncertainty.pdf")
    plt.close(fig)


def bootstrap_community_metrics(
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for community-ID metrics."""
    rng = np.random.default_rng(seed)
    n = len(gt_ids)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pred_sample = pred_ids[idx]
        gt_sample = gt_ids[idx]
        acc = np.mean(pred_sample == gt_sample)
        bal_acc = balanced_accuracy_score(gt_sample, pred_sample)
        macro_f1 = f1_score(gt_sample, pred_sample, average="macro", zero_division=0)
        rows.append((acc, bal_acc, macro_f1))
    return pd.DataFrame(rows, columns=["accuracy", "balanced_accuracy", "macro_f1"])


def save_community_metrics_ci(
    metrics: Dict[str, Any],
    boot_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Save bootstrap CI summary for community-ID metrics."""
    rows = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        series = boot_df[metric]
        rows.append({
            "metric": metric,
            "mean": float(metrics[metric]),
            "ci_low": float(np.percentile(series, 2.5)),
            "ci_high": float(np.percentile(series, 97.5)),
            "n": int(metrics["n"]),
            "n_boot": int(len(series)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_eval_community_metrics_ci.csv", index=False, float_format="%.4f")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Community-ID Metrics with Bootstrap 95\% CI}",
        r"\label{tab:test_eval_community_ci}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Mean & CI Low & CI High & N \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['metric']} & {row['mean']:.3f} & {row['ci_low']:.3f} & {row['ci_high']:.3f} & {int(row['n'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    atomic_write_text(out_dir / "test_eval_community_metrics_ci.tex", "\n".join(lines))
    return df


def plot_community_metrics_ci(ci_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot community-ID metric CIs."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    labels = ["Accuracy", "Balanced Acc", "Macro-F1"]
    means = [ci_df.loc[ci_df["metric"] == m, "mean"].values[0] for m in metrics]
    low = [ci_df.loc[ci_df["metric"] == m, "ci_low"].values[0] for m in metrics]
    high = [ci_df.loc[ci_df["metric"] == m, "ci_high"].values[0] for m in metrics]
    yerr = np.array([np.array(means) - np.array(low), np.array(high) - np.array(means)])

    ax.bar(labels, means, color=sns.color_palette("Blues", 3), edgecolor="black", linewidth=0.5)
    ax.errorbar(labels, means, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Community-ID Metrics (95% CI)")
    plt.tight_layout()
    fig.savefig(out_dir / "test_eval_community_metrics_ci.png")
    fig.savefig(out_dir / "test_eval_community_metrics_ci.pdf")
    plt.close(fig)


def save_per_class_prf(
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
    out_dir: Path,
    labels: List[int],
) -> pd.DataFrame:
    """Compute and save per-class precision/recall/F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        gt_ids, pred_ids, labels=labels, zero_division=0
    )
    df = pd.DataFrame({
        "community_id": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    df.to_csv(out_dir / "test_eval_per_class_metrics.csv", index=False, float_format="%.4f")
    return df


def plot_per_class_prf(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot per-class precision/recall/F1."""
    set_publication_style()
    plot_df = df.copy()
    plot_df["community_id"] = plot_df["community_id"].astype(str)
    hue_order = [str(c) for c in sorted(df["community_id"].tolist())]
    n_communities = len(hue_order)
    palette = sns.color_palette("viridis", n_colors=n_communities)
    melted = plot_df.melt(
        id_vars=["community_id"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=melted,
        x="metric",
        y="score",
        hue="community_id",
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    #ax.set_title("Per-Community Precision/Recall/F1")
    # Legend with correct color-label mapping
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=[f"C{c}" for c in hue_order],
        title="Community",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "test_eval_per_class_prf.png", bbox_inches="tight")
    fig.savefig(out_dir / "test_eval_per_class_prf.pdf", bbox_inches="tight")
    plt.close(fig)


def compute_train_test_profile_consistency(
    run_dir: Path,
    test_df: pd.DataFrame,
    pred_ids: np.ndarray,
    features: List[str],
) -> pd.DataFrame:
    """Compute train vs test profile means per community."""
    profiles = pd.read_csv(run_dir / "communities" / "community_profiles.csv")
    rows = []

    for _, row in profiles.iterrows():
        cid = int(row["community_id"])
        test_subset = test_df[pred_ids == cid]
        for feat in features:
            train_mean = float(row.get(f"mean_{feat}", np.nan))
            test_mean = float(test_subset[feat].mean()) if feat in test_subset.columns else np.nan
            rows.append({
                "community_id": cid,
                "feature": feat,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "abs_diff": abs(train_mean - test_mean) if not np.isnan(train_mean) and not np.isnan(test_mean) else np.nan,
            })

    return pd.DataFrame(rows)


def plot_train_test_profile_consistency(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot train vs test feature means per community."""
    set_publication_style()
    communities = sorted(df["community_id"].unique())
    fig, axes = plt.subplots(1, len(communities), figsize=(12, 4), sharex=True, sharey=True)
    if len(communities) == 1:
        axes = [axes]

    for ax, cid in zip(axes, communities):
        sub = df[df["community_id"] == cid]
        ax.scatter(sub["train_mean"], sub["test_mean"], s=45, color="#1f77b4")
        min_val = min(sub["train_mean"].min(), sub["test_mean"].min())
        max_val = max(sub["train_mean"].max(), sub["test_mean"].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
        if sub["train_mean"].notna().all() and sub["test_mean"].notna().all():
            r = np.corrcoef(sub["train_mean"], sub["test_mean"])[0, 1]
        else:
            r = np.nan
        ax.set_title(f"C{cid} (r={r:.2f})")
        ax.set_xlabel("Train mean")
        ax.set_ylabel("Test mean")

    plt.tight_layout()
    fig.savefig(out_dir / "test_eval_train_test_profile_consistency.png")
    fig.savefig(out_dir / "test_eval_train_test_profile_consistency.pdf")
    plt.close(fig)


def generate_summary_md(
    metrics: List[Dict],
    uncertainty_analysis: Dict,
    n_instances: int,
    n_communities: int,
    out_path: Path,
    community_metrics: Dict[str, Any],
    ci_df: pd.DataFrame,
) -> None:
    """Generate paper summary snippet."""
    lines = [
        "## Test Set Evaluation Summary",
        "",
        f"- Community-ID accuracy {community_metrics['accuracy']:.3f} with 95% CI "
        f"[{ci_df.loc[ci_df['metric']=='accuracy','ci_low'].values[0]:.3f}, "
        f"{ci_df.loc[ci_df['metric']=='accuracy','ci_high'].values[0]:.3f}] on n={community_metrics['n']}.",
        f"- Balanced accuracy {community_metrics['balanced_accuracy']:.3f} and macro-F1 {community_metrics['macro_f1']:.3f} "
        f"(see `test_eval_community_metrics_ci.png`).",
        "- Per-community precision/recall/F1 shows class-wise performance (see `test_eval_per_class_prf.png`).",
        "- Train vs test profile consistency is summarized by feature-level mean alignment "
        "(see `test_eval_train_test_profile_consistency.png`).",
        f"- Uncertainty flags are {uncertainty_analysis['overall_uncertain_rate']:.1%} "
        f"({uncertainty_analysis['n_uncertain']}/{uncertainty_analysis['n_total']}).",
    ]

    atomic_write_text(out_path, "\n".join(lines))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Held-out evaluation of serve style assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--community-run-dir",
        type=Path,
        required=True,
        help="Community discovery run directory",
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Raw test data CSV",
    )

    parser.add_argument(
        "--annotations-csv",
        type=Path,
        required=True,
        help="Expert-provided style annotations of the held-out records (see docs/data_format.md)",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <community-run-dir>/test_set_eval)",
    )

    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Encoder checkpoint. Defaults to the path recorded in the discovery run.",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=None,
        help="Preprocessor fitted by courtstyle.data.serve_dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (default: auto)",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="kNN neighbors (default: from run_config.json)",
    )

    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, evaluate from existing assignment",
    )

    parser.add_argument(
        "--assignment-json",
        type=Path,
        default=None,
        help="Path to existing assignment.json to reuse",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Validate inputs
    if not args.community_run_dir.exists():
        print(f"ERROR: Community run directory not found: {args.community_run_dir}")
        sys.exit(1)

    if not args.test_csv.exists():
        print(f"ERROR: Test CSV not found: {args.test_csv}")
        sys.exit(1)

    if not args.annotations_csv.exists():
        print(f"ERROR: Annotations CSV not found: {args.annotations_csv}")
        sys.exit(1)

    # Output directory
    out_dir = args.out_dir or args.community_run_dir / "test_set_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    print(f"  Community run: {args.community_run_dir}")
    print(f"  Test CSV: {args.test_csv}")
    print(f"  Annotations: {args.annotations_csv}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Run Inference (or Reuse)
    # =========================================================================
    print("\n[Step 1] Running inference...")

    if args.skip_inference and args.assignment_json:
        print(f"  Loading existing assignment from {args.assignment_json}")
        assignment_data = load_existing_assignment(args.assignment_json)
    elif args.skip_inference:
        # Look for existing assignment in out_dir
        existing_path = out_dir / "inference" / "assignment.json"
        if existing_path.exists():
            print(f"  Loading existing assignment from {existing_path}")
            assignment_data = load_existing_assignment(existing_path)
        else:
            print("ERROR: --skip-inference requires --assignment-json or existing inference output")
            sys.exit(1)
    else:
        assignment_data = run_inference(
            test_csv=args.test_csv,
            community_run_dir=args.community_run_dir,
            out_dir=out_dir,
            device=args.device,
            k=args.k,
            ckpt_path_override=args.ckpt_path,
            preprocessor_path=args.preprocessor_path,
        )

    # Extract predictions DataFrame
    predictions_df = pd.DataFrame([{
        "instance_id": a["instance_id"],
        "row_index": a["row_index"],
        "community_id": a["community_id"],
        "centroid_prediction": a["centroid_prediction"],
        "max_confidence": a["max_confidence"],
        "top2_margin": a["top2_margin"],
        "uncertain": a["uncertain"],
    } for a in assignment_data["assignments"]])

    n_instances = len(predictions_df)
    community_ids = assignment_data["community_ids"]
    n_communities = len(community_ids)

    print(f"  {n_instances} instances assigned to {n_communities} communities")

    # =========================================================================
    # Step 2: Load and Validate Annotations
    # =========================================================================
    print("\n[Step 2] Loading and validating annotations...")

    annotations_df = load_and_validate_annotations(
        args.annotations_csv,
        n_expected=n_instances,
    )
    print(f"  Loaded {len(annotations_df)} annotations")
    print(f"  Fields: {EVAL_FIELDS}")

    # Merge predictions and annotations (include community_id_gt for community-ID evaluation)
    merge_cols = ["instance_id", "row_index", "community_id_gt"] + EVAL_FIELDS + \
                 [f"{f}_norm" for f in EVAL_FIELDS]
    merged_df = predictions_df.merge(
        annotations_df[merge_cols],
        on=["instance_id", "row_index"],
        how="left",
    )

    # =========================================================================
    # Step 3: Compute Evaluation Metrics
    # =========================================================================
    print("\n[Step 3] Computing evaluation metrics...")

    all_metrics = []
    all_metrics_detailed = {}

    for field in EVAL_FIELDS:
        norm_field = f"{field}_norm"

        comm_ids = merged_df["community_id"].values
        labels = merged_df[norm_field].values

        metrics = compute_field_metrics(comm_ids, labels, field)
        all_metrics.append(metrics)
        all_metrics_detailed[field] = metrics

        print(f"  {field}:")
        print(f"    NMI={metrics['nmi']:.3f}, ARI={metrics['ari']:.3f}, "
              f"Cramér's V={metrics['cramers_v']:.3f} (p={metrics['chi2_p']:.4f})")
        print(f"    Accuracy={metrics['accuracy']:.1%}, "
              f"Macro-F1={metrics['macro_f1']:.3f}, "
              f"Balanced Acc={metrics['balanced_accuracy']:.3f}")

    # Uncertainty analysis
    print("\n[Step 3C] Computing uncertainty analysis...")
    uncertainty_analysis = compute_uncertainty_analysis(predictions_df, merged_df)
    print(f"  Overall uncertain rate: {uncertainty_analysis['overall_uncertain_rate']:.1%}")

    # =========================================================================
    # Step 3D: Community-ID Evaluation (Pred vs GT)
    # =========================================================================
    print("\n[Step 3D] Computing community-ID evaluation (pred vs gt)...")

    community_metrics = compute_community_id_metrics(
        merged_df["community_id"].values,
        merged_df["community_id_gt"].values,
    )

    n_valid_gt = community_metrics["n"]
    n_missing_gt = n_instances - n_valid_gt

    print(f"  Valid GT instances: {n_valid_gt}/{n_instances}")
    if n_missing_gt > 0:
        print(f"  (Skipped {n_missing_gt} instances with missing community_id_gt)")
    print(f"  Accuracy={community_metrics['accuracy']:.1%}, "
          f"Balanced Acc={community_metrics['balanced_accuracy']:.3f}")
    print(f"  Macro-F1={community_metrics['macro_f1']:.3f}, "
          f"NMI={community_metrics['nmi']:.3f}, ARI={community_metrics['ari']:.3f}")

    # =========================================================================
    # Save Outputs
    # =========================================================================
    print("\n[Step 4] Saving outputs...")

    # Tables
    save_metrics_csv(all_metrics, out_dir / "test_eval_metrics.csv")
    print(f"  Saved: test_eval_metrics.csv")

    save_metrics_latex(all_metrics, out_dir / "test_eval_metrics.tex")
    print(f"  Saved: test_eval_metrics.tex")

    # Contingency tables
    for field in EVAL_FIELDS:
        save_contingency_csv(
            all_metrics_detailed[field],
            field,
            out_dir / f"test_eval_contingency_{field}.csv",
        )
    print(f"  Saved: test_eval_contingency_*.csv")

    # Merged predictions
    merged_df.to_csv(out_dir / "test_eval_predictions.csv", index=False)
    print(f"  Saved: test_eval_predictions.csv")

    # Community-ID metrics
    save_community_metrics_csv(community_metrics, out_dir / "test_eval_community_metrics.csv")
    print(f"  Saved: test_eval_community_metrics.csv")

    save_community_metrics_latex(community_metrics, out_dir / "test_eval_community_metrics.tex")
    print(f"  Saved: test_eval_community_metrics.tex")

    save_community_confusion_csv(community_metrics, out_dir / "test_eval_community_confusion.csv")
    print(f"  Saved: test_eval_community_confusion.csv")

    # Full results JSON
    full_results = {
        "metrics": all_metrics,
        "community_metrics": community_metrics,
        "community_evaluation": {
            "n": community_metrics.get("n"),
            "accuracy": community_metrics.get("accuracy"),
            "balanced_accuracy": community_metrics.get("balanced_accuracy"),
            "macro_f1": community_metrics.get("macro_f1"),
            "nmi": community_metrics.get("nmi"),
            "ari": community_metrics.get("ari"),
        },
        "uncertainty_analysis": uncertainty_analysis,
        "n_instances": n_instances,
        "n_communities": n_communities,
        "community_ids": community_ids,
        "timestamp": datetime.now().isoformat(),
    }
    atomic_write_json(out_dir / "test_eval_results.json", full_results)
    print(f"  Saved: test_eval_results.json")

    # Figures
    print("\n[Step 5] Generating figures...")
    # Community-ID metrics with bootstrap CI
    gt_ids = merged_df["community_id_gt"].values
    pred_ids = merged_df["community_id"].values
    mask = ~pd.isna(gt_ids)
    pred_filtered = pred_ids[mask].astype(int)
    gt_filtered = gt_ids[mask].astype(int)
    boot_df = bootstrap_community_metrics(pred_filtered, gt_filtered, n_boot=2000, seed=RANDOM_SEED)
    ci_df = save_community_metrics_ci(community_metrics, boot_df, out_dir)
    plot_community_metrics_ci(ci_df, out_dir)
    print(f"  Saved: test_eval_community_metrics_ci.*")

    # Per-class precision/recall/F1
    labels = sorted(set(gt_filtered) | set(pred_filtered))
    per_class_df = save_per_class_prf(pred_filtered, gt_filtered, out_dir, labels)
    plot_per_class_prf(per_class_df, out_dir)
    print(f"  Saved: test_eval_per_class_prf.*")

    # Train vs test profile consistency
    test_raw_df = pd.read_csv(args.test_csv)
    features = [
        "Avg Speed", "1st Avg", "1st StDev", "2nd Avg",
        "%_serves_T", "%_serves_Wide", "%_serves_Body",
    ]
    profile_df = compute_train_test_profile_consistency(
        args.community_run_dir,
        test_raw_df,
        predictions_df["community_id"].values,
        features,
    )
    profile_df.to_csv(out_dir / "test_eval_train_test_profile_consistency.csv", index=False, float_format="%.4f")
    plot_train_test_profile_consistency(profile_df, out_dir)
    print(f"  Saved: test_eval_train_test_profile_consistency.*")

    # Uncertainty and confusion matrix
    plot_uncertainty(predictions_df, uncertainty_analysis, out_dir)
    print(f"  Saved: test_eval_uncertainty.png/pdf")

    plot_community_confusion(community_metrics, out_dir)
    print(f"  Saved: test_eval_community_confusion.png/pdf")

    # Summary
    generate_summary_md(
        all_metrics,
        uncertainty_analysis,
        n_instances,
        n_communities,
        out_dir / "test_eval_summary.md",
        community_metrics,
        ci_df,
    )
    print(f"  Saved: test_eval_summary.md")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Instances evaluated: {n_instances}")
    print(f"  Communities: {n_communities}")
    print(f"  Output directory: {out_dir}")
    print("\n  Key Results:")

    avg_nmi = np.nanmean([m["nmi"] for m in all_metrics])
    avg_acc = np.nanmean([m["accuracy"] for m in all_metrics])
    print(f"    Average NMI: {avg_nmi:.3f}")
    print(f"    Average Accuracy: {avg_acc:.1%}")
    print(f"    Uncertain Rate: {uncertainty_analysis['overall_uncertain_rate']:.1%}")

    print("\n  Per-Field NMI:")
    for m in all_metrics:
        print(f"    {m['field']}: {m['nmi']:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()

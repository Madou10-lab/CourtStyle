"""Rally style discovery on frozen encoder embeddings.

The trained rally encoder embeds every rally of the fit split. Rally length is
removed from the embedding space by subtracting, from each embedding, the mean
embedding of its length bin (bins fitted on the fit split). K-Means with a fixed
``K`` then partitions the fit split, and every other split is assigned to the
nearest centroid by cosine similarity.

The run also writes the evidence used to judge the partition:

* clustering stability over re-seeded K-Means runs (pairwise ARI / NMI),
* confound NMIs against match, handedness pair, rally-length bin and outcome,
* negative controls that shuffle transitions within and across rallies,
* length-leakage diagnostics before and after residualisation,
* per-style state / action / delta profiles (z-scores against the fit split),
  profiles by rally phase, closest-to-centroid exemplar rallies and figures.

Usage:
    python -m courtstyle.discovery.rally_discovery --config configs/rally.yaml \\
        --checkpoint outputs/rally/models/best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset

from courtstyle.data.rally_transitions import (
    EXCLUDED_FROM_TOKENS,
    RallyDatasetBundle,
    _build_transition_raw_feature_table,
    _safe_int,
    build_rally_dataset,
    load_split_info,
)
from courtstyle.models.rally_encoder import RallySSLModel

HAS_PLOTTING = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_PLOTTING = False

HAS_SEABORN = True
try:
    import seaborn as sns  # type: ignore
except Exception:
    HAS_SEABORN = False

HAS_UMAP = True
try:
    import umap  # type: ignore
except Exception:
    HAS_UMAP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    paths = {
        "run": run_dir,
        "embeddings": run_dir / "embeddings",
        "communities": run_dir / "communities",
        "validation": run_dir / "validation",
        "figures": run_dir / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    (paths["validation"] / "negative_controls").mkdir(parents=True, exist_ok=True)
    return paths


# =============================================================================
# Features, bins and figures
# =============================================================================

CURATED_TRANSITION_FEATURES = [
    "speed",
    "spin_rpm",
    "bounce_coordinate_y",
    "net_clearance_z_debug",
    "is_volley",
    "is_return",
    "player_position_at_hit_y",
    "player_opponent_position_at_hit_y",
]



def bin_rally_len(rally_len: Sequence[int]) -> List[str]:
    # Match existing binning logic
    bins = [0, 1, 3, 5, 8, 999]
    labels = ["1", "2-3", "4-5", "6-8", "9+"]
    out: List[str] = []
    for v in rally_len:
        vv = int(v)
        for i in range(len(bins) - 1):
            if bins[i] < vv <= bins[i + 1]:
                out.append(labels[i])
                break
        else:
            out.append(labels[-1])
    return out



def compute_handedness_pair_map(track_shots_df: pd.DataFrame) -> Dict[Tuple[Any, Any], str]:
    if "dominant_hand_R" not in track_shots_df.columns:
        return {}

    pair_map: Dict[Tuple[Any, Any], str] = {}
    for (mid, pidx), group in track_shots_df.groupby(["match_id", "point_index"], sort=False):
        raw_vals = group["dominant_hand_R"].dropna().unique().tolist()
        parsed_vals: set[int] = set()
        for v in raw_vals:
            try:
                iv = int(float(v))
            except (TypeError, ValueError):
                continue
            if iv in (0, 1):
                parsed_vals.add(iv)

        if parsed_vals == {1}:
            pair = "both_right"
        elif parsed_vals == {0}:
            pair = "both_left"
        elif parsed_vals == {0, 1}:
            pair = "mixed"
        else:
            pair = "unknown"
        pair_map[(mid, _safe_int(pidx))] = pair
    return pair_map



def plot_heatmap(out_path: Path, mat: np.ndarray, y_labels: List[str], x_labels: List[str], title: str) -> None:
    if not (HAS_PLOTTING and HAS_SEABORN):
        return
    try:
        plt.figure(figsize=(max(10, 0.5 * len(x_labels)), max(6, 0.35 * len(y_labels))))
        sns.heatmap(mat, cmap="coolwarm", center=0.0, yticklabels=y_labels, xticklabels=x_labels)
        plt.title(title)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception:
        return



def plot_radar_profiles(
    out_path: Path,
    profiles_df: Any,
    radar_features: Sequence[str],
    topk: int,
    title: str,
) -> None:
    if not HAS_PLOTTING:
        return
    try:
        # Use delta z-scores (response change) for interpretability
        z_cols = [f"z_delta_{f}" for f in radar_features if f"z_delta_{f}" in profiles_df.columns]
        if len(z_cols) < 3:
            return

        df = profiles_df.copy()
        df = df.sort_values("count", ascending=False).head(int(topk))
        if len(df) == 0:
            return

        values = df[z_cols].to_numpy(dtype=float)
        labels = df["community_id"].astype(int).tolist()

        angles = np.linspace(0, 2 * np.pi, len(z_cols), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)

        for i in range(len(labels)):
            v = values[i].tolist()
            v += v[:1]
            ax.plot(angles, v, linewidth=2, label=f"comm {labels[i]}")
            ax.fill(angles, v, alpha=0.1)

        ax.set_title(title)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("z_delta_", "") for c in z_cols])
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception:
        return



def plot_umap_communities(out_path: Path, embeddings: np.ndarray, labels: np.ndarray, title: str) -> None:
    if not (HAS_PLOTTING and HAS_UMAP):
        return
    try:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        xy = reducer.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=labels, s=6, cmap="tab20", alpha=0.85)
        plt.title(title)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.colorbar(sc, fraction=0.04, pad=0.02)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception:
        return



# =============================================================================
# Clustering and validation
# =============================================================================

def _contiguous_labels(labels: np.ndarray) -> np.ndarray:
    uniq = sorted(int(v) for v in np.unique(labels).tolist())
    mp = {u: i for i, u in enumerate(uniq)}
    return np.array([mp[int(v)] for v in labels], dtype=int)



def cluster_kmeans(embeddings: np.ndarray, *, k: int, seed: int, n_init: int = 20) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    return km.fit_predict(embeddings).astype(int)


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = _contiguous_labels(labels)
    centroids = []
    for cid in sorted(np.unique(labels).tolist()):
        centroids.append(embeddings[labels == cid].mean(axis=0))
    return np.vstack(centroids).astype(np.float32)



def assign_by_centroid_cosine(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    e = normalize(embeddings, norm="l2")
    c = normalize(centroids, norm="l2")
    sim = e @ c.T
    return np.argmax(sim, axis=1).astype(int)



def compute_cluster_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    labels = _contiguous_labels(labels.astype(int))
    uniq, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    max_frac = float(counts.max() / n) if n > 0 else 1.0
    min_size = int(counts.min()) if len(counts) else 0
    metrics: Dict[str, Any] = {
        "n_clusters": int(len(uniq)),
        "max_frac": max_frac,
        "min_size": min_size,
    }
    if len(uniq) > 1 and len(uniq) < len(labels):
        emb_norm = normalize(embeddings, norm="l2")
        metrics["silhouette"] = float(silhouette_score(emb_norm, labels, metric="cosine"))
        metrics["davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
    else:
        metrics["silhouette"] = -1.0
        metrics["davies_bouldin"] = -1.0
        metrics["calinski_harabasz"] = -1.0
    return metrics



def compute_stability(
    embeddings: np.ndarray,
    *,
    k: int,
    base_seed: int,
    n_runs: int,
    n_init: int = 20,
) -> Dict[str, Any]:
    """Pairwise ARI / NMI between K-Means partitions with seeds ``base_seed + i * 997``."""
    labels_list: List[np.ndarray] = []
    for i in range(int(n_runs)):
        seed = int(base_seed + i * 997)
        labels = cluster_kmeans(embeddings, k=k, seed=seed, n_init=n_init)
        labels_list.append(labels)

    aris: List[float] = []
    nmis: List[float] = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
            nmis.append(float(normalized_mutual_info_score(labels_list[i], labels_list[j])))
    if not aris:
        return {"n_runs": int(n_runs), "ari_mean": 1.0, "ari_std": 0.0, "nmi_mean": 1.0, "nmi_std": 0.0}
    return {
        "n_runs": int(n_runs),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "nmi_mean": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
    }



def _build_handedness_map(shots_df: pd.DataFrame) -> Dict[Tuple[Any, Any], str]:
    try:
        return compute_handedness_pair_map(shots_df)
    except Exception:
        return {}



def _maybe_nmi(labels: Sequence[int], values: Sequence[Any]) -> Optional[float]:
    vv = [str(x) for x in values]
    if len(set(vv)) < 2 or len(set(labels)) < 2:
        return None
    return float(normalized_mutual_info_score(labels, vv))



def compute_confound_metrics(
    labels_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> Dict[str, Any]:
    df = labels_df.copy()
    labels = df["community_id"].astype(int).tolist()

    out: Dict[str, Any] = {}
    out["nmi_match_id"] = _maybe_nmi(labels, df["match_id"].astype(str).tolist())

    handed_map = _build_handedness_map(shots_df)
    handed = [
        handed_map.get((str(m), _safe_int(p)), "unknown")
        for m, p in zip(df["match_id"], df["point_index"])
    ]
    out["nmi_handedness_pair"] = _maybe_nmi(labels, handed)

    rally_len_bins = bin_rally_len(df["n_shots"].astype(int).tolist()) if "n_shots" in df.columns else []
    out["nmi_rally_len_bin"] = _maybe_nmi(labels, rally_len_bins) if rally_len_bins else None

    merged = df.merge(outcomes_df, on=["match_id", "point_index"], how="left")
    for col in ["result", "reason", "conclusion_action", "conclusion_stroke", "conclusion_result"]:
        if col in merged.columns:
            vals = merged[col].fillna("MISSING").astype(str).tolist()
            out[f"nmi_outcome_{col}"] = _maybe_nmi(labels, vals)
        else:
            out[f"nmi_outcome_{col}"] = None
    return out



def _shuffle_within_rallies(seq_arrays: Sequence[np.ndarray], seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for s in seq_arrays:
        if len(s) <= 1:
            out.append(s.copy())
            continue
        perm = rng.permutation(len(s))
        out.append(s[perm].copy())
    return out



def _shuffle_across_rallies(seq_arrays: Sequence[np.ndarray], seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = [len(s) for s in seq_arrays]
    if not lengths:
        return []
    all_tokens = np.concatenate(seq_arrays, axis=0)
    perm = rng.permutation(len(all_tokens))
    shuffled = all_tokens[perm]
    out: List[np.ndarray] = []
    start = 0
    for L in lengths:
        out.append(shuffled[start : start + L].copy())
        start += L
    return out



def build_outcome_heatmap(
    *,
    labels_train_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    fig_path: Path,
) -> pd.DataFrame:
    merged = labels_train_df.merge(outcomes_df, on=["match_id", "point_index"], how="left")
    merged["result"] = merged["result"].fillna("MISSING").astype(str) if "result" in merged.columns else "MISSING"
    tab = (
        merged.groupby(["community_id", "result"], sort=False)
        .size()
        .reset_index(name="count")
    )
    pivot = tab.pivot(index="community_id", columns="result", values="count").fillna(0.0)
    pivot = pivot.div(np.clip(pivot.sum(axis=1).to_numpy()[:, None], 1e-8, None))
    mat = pivot.to_numpy(dtype=float)
    y_labels = [f"community_{int(v)}" for v in pivot.index.tolist()]
    x_labels = [str(c) for c in pivot.columns.tolist()]
    plot_heatmap(fig_path, mat, y_labels, x_labels, "Outcome Distribution by Community")
    return pivot.reset_index()



# =============================================================================
# Length residualisation and leakage diagnostics
# =============================================================================

def _get_length_var(seq_meta: pd.DataFrame, length_var: str) -> np.ndarray:
    """Extract the chosen length variable as a float array aligned to seq_meta."""
    if length_var not in seq_meta.columns:
        raise ValueError(
            f"length_var={length_var!r} not in seq_meta columns: {list(seq_meta.columns)}"
        )
    return seq_meta[length_var].to_numpy(dtype=np.float64)



def fit_residualizer_bin_center(
    embeddings: np.ndarray,
    length_vals: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit bin-center residualization: compute mean embedding per rally_len_bin
    on train_all data.

    Uses the same ``bin_rally_len`` as ``compute_confound_metrics``
    (bins: 1, 2-3, 4-5, 6-8, 9+).
    """
    int_lengths = [int(round(v)) for v in length_vals]
    bins = bin_rally_len(int_lengths)
    unique_bins = sorted(set(bins))

    bin_means: Dict[str, np.ndarray] = {}
    for b in unique_bins:
        mask = np.array([bb == b for bb in bins], dtype=bool)
        bin_means[b] = embeddings[mask].mean(axis=0).astype(np.float64)

    global_mean = embeddings.mean(axis=0).astype(np.float64)

    params: Dict[str, Any] = {
        "method": "bin_center",
        "bin_means": {b: m.tolist() for b, m in bin_means.items()},
        "global_mean": global_mean.tolist(),
        "n_train": int(len(embeddings)),
        "bins_observed": unique_bins,
        "backoff_rule": "unseen bins use global mean",
    }
    return params



def apply_residualizer_bin_center(
    embeddings: np.ndarray,
    length_vals: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Subtract bin mean (fitted on train_all) from each embedding."""
    int_lengths = [int(round(v)) for v in length_vals]
    bins = bin_rally_len(int_lengths)

    bin_means = {b: np.array(m, dtype=np.float64) for b, m in params["bin_means"].items()}
    global_mean = np.array(params["global_mean"], dtype=np.float64)

    out = embeddings.astype(np.float64).copy()
    for i, b in enumerate(bins):
        mean_vec = bin_means.get(b, global_mean)
        out[i] -= mean_vec
    return out.astype(np.float32)



def fit_and_apply_residualization(
    emb_by_split: Dict[str, np.ndarray],
    meta_by_split: Dict[str, pd.DataFrame],
    *,
    method: str,
    length_var: str,
    l2_normalize: bool,
    fit_split: str = "train_all",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Fit on fit_split, apply to ALL splits (including controls later).
    Returns (residualized_emb_by_split, resid_params).
    """
    if method == "none":
        return dict(emb_by_split), {"method": "none"}

    train_emb = emb_by_split[fit_split]
    train_lengths = _get_length_var(meta_by_split[fit_split], length_var)

    if method != "bin_center":
        raise ValueError(f"Unknown residualization method: {method!r}")
    params = fit_residualizer_bin_center(train_emb, train_lengths)

    params["length_var"] = length_var
    params["l2_normalize"] = l2_normalize

    out: Dict[str, np.ndarray] = {}
    for split, emb in emb_by_split.items():
        lengths = _get_length_var(meta_by_split[split], length_var)
        resid = apply_residualizer_bin_center(emb, lengths, params)
        if l2_normalize:
            norms = np.linalg.norm(resid, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            resid = (resid / norms).astype(np.float32)
        out[split] = resid

    logger.info(
        "Residualization applied: method=%s length_var=%s l2_norm=%s",
        method, length_var, l2_normalize,
    )
    return out, params



def apply_residualization_to_array(
    emb: np.ndarray,
    length_vals: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Apply already-fitted residualization to a single embedding array
    (used for negative-control embeddings)."""
    method = params.get("method", "none")
    if method == "none":
        return emb
    if method != "bin_center":
        raise ValueError(f"Unknown method in params: {method!r}")
    resid = apply_residualizer_bin_center(emb, length_vals, params)
    if params.get("l2_normalize", False):
        norms = np.linalg.norm(resid, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        resid = (resid / norms).astype(np.float32)
    return resid



def compute_length_leakage(
    emb_by_split: Dict[str, np.ndarray],
    meta_by_split: Dict[str, pd.DataFrame],
    *,
    ridge_alpha: float = 1.0,
    fit_split: str = "train_all",
) -> Dict[str, Any]:
    """
    Measure how predictable rally length is from the embedding space.

    Fit Ridge on fit_split, evaluate R^2 on all available splits for
    both n_transitions and n_shots.  Also compute length-bin classification
    accuracy via LogisticRegression (if possible).

    Returns a dict suitable for JSON serialisation.
    """
    train_emb = emb_by_split[fit_split]
    results: Dict[str, Any] = {"ridge_alpha": float(ridge_alpha), "fit_split": fit_split}

    for target_var in ["n_transitions", "n_shots"]:
        if target_var not in meta_by_split[fit_split].columns:
            results[target_var] = {"error": f"column {target_var} not in seq_meta"}
            continue

        y_train = meta_by_split[fit_split][target_var].to_numpy(dtype=np.float64)

        # Ridge regression
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(train_emb, y_train)

        var_results: Dict[str, Any] = {}
        for split in sorted(emb_by_split.keys()):
            y_split = meta_by_split[split][target_var].to_numpy(dtype=np.float64)
            y_pred = ridge.predict(emb_by_split[split])
            r2 = float(r2_score(y_split, y_pred))
            var_results[f"r2_{split}"] = r2

        results[target_var] = var_results

    # Length-bin classification accuracy (logistic regression)
    if "n_shots" in meta_by_split[fit_split].columns:
        y_train_shots = meta_by_split[fit_split]["n_shots"].astype(int).tolist()
        y_train_bins = bin_rally_len(y_train_shots)
        unique_bins = sorted(set(y_train_bins))
        if len(unique_bins) >= 2:
            try:
                lr = LogisticRegression(
                    max_iter=500, solver="lbfgs",
                    C=1.0, random_state=42,
                )
                lr.fit(train_emb, y_train_bins)
                bin_results: Dict[str, Any] = {}
                for split in sorted(emb_by_split.keys()):
                    y_split_shots = meta_by_split[split]["n_shots"].astype(int).tolist()
                    y_split_bins = bin_rally_len(y_split_shots)
                    acc = float(lr.score(emb_by_split[split], y_split_bins))
                    bin_results[f"accuracy_{split}"] = acc
                bin_results["n_classes"] = len(unique_bins)
                bin_results["classes"] = unique_bins
                results["length_bin_classification"] = bin_results
            except Exception as e:
                results["length_bin_classification"] = {"error": str(e)}
        else:
            results["length_bin_classification"] = {"error": "fewer than 2 unique bins"}

    return results



def save_length_leakage(
    leakage: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Save length_leakage.json and length_leakage.csv to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "length_leakage.json", leakage)

    # Flatten to CSV rows
    csv_rows: List[Dict[str, Any]] = []
    for target_var in ["n_transitions", "n_shots"]:
        entry = leakage.get(target_var, {})
        if isinstance(entry, dict) and "error" not in entry:
            for key, val in entry.items():
                csv_rows.append({"target": target_var, "metric": key, "value": float(val)})
    bin_cls = leakage.get("length_bin_classification", {})
    if isinstance(bin_cls, dict) and "error" not in bin_cls:
        for key, val in bin_cls.items():
            if key.startswith("accuracy_"):
                csv_rows.append({"target": "length_bin", "metric": key, "value": float(val)})
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_dir / "length_leakage.csv", index=False)



# =============================================================================
# Embeddings and data
# =============================================================================

class _SeqDataset(Dataset):
    def __init__(self, seq_arrays: Sequence[np.ndarray]) -> None:
        self.seq_arrays = list(seq_arrays)

    def __len__(self) -> int:
        return len(self.seq_arrays)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.seq_arrays[idx]



def _pad_batch(seqs: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [int(s.shape[0]) for s in seqs]
    max_len = max(lengths)
    feat_dim = int(seqs[0].shape[1])
    x = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        L = int(s.shape[0])
        x[i, :L, :] = torch.from_numpy(s.astype(np.float32, copy=False))
    return x, torch.tensor(lengths, dtype=torch.long)



@torch.no_grad()
def encode_rally_embeddings(
    model: RallySSLModel,
    seq_arrays: Sequence[np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = _SeqDataset(seq_arrays)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_pad_batch,
    )
    outs: List[np.ndarray] = []
    model.eval()
    for x, lengths in loader:
        x = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        emb, _ = model(x, lengths)
        outs.append(emb.cpu().numpy().astype(np.float32))
    if outs:
        return np.concatenate(outs, axis=0)
    return np.zeros((0, model.encoder.d_model), dtype=np.float32)



def _load_shots_outcomes(
    *,
    split: str,
    train_shots_csv: Path,
    train_outcomes_csv: Path,
    test_shots_csv: Path,
    test_outcomes_csv: Path,
    split_info: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split in {"train_all", "val", "train"}:
        shots = pd.read_csv(train_shots_csv, low_memory=False)
        outcomes = pd.read_csv(train_outcomes_csv, low_memory=False)
        shots["match_id"] = shots["match_id"].astype(str)
        outcomes["match_id"] = outcomes["match_id"].astype(str)
        if split == "val":
            _, val_ids = load_split_info(split_info)
            id_set = set(str(x) for x in val_ids)
            shots = shots[shots["match_id"].isin(id_set)].reset_index(drop=True)
            outcomes = outcomes[outcomes["match_id"].isin(id_set)].reset_index(drop=True)
        elif split == "train":
            train_ids, _ = load_split_info(split_info)
            id_set = set(str(x) for x in train_ids)
            shots = shots[shots["match_id"].isin(id_set)].reset_index(drop=True)
            outcomes = outcomes[outcomes["match_id"].isin(id_set)].reset_index(drop=True)
    elif split == "test":
        shots = pd.read_csv(test_shots_csv, low_memory=False)
        outcomes = pd.read_csv(test_outcomes_csv, low_memory=False)
        shots["match_id"] = shots["match_id"].astype(str)
        outcomes["match_id"] = outcomes["match_id"].astype(str)
    else:
        raise ValueError(f"Unknown split={split}")

    shots["point_index"] = shots["point_index"].apply(_safe_int)
    shots["shot_index"] = shots["shot_index"].apply(_safe_int)
    if "point_index" in outcomes.columns:
        outcomes["point_index"] = outcomes["point_index"].apply(_safe_int)
    return shots, outcomes



# =============================================================================
# Style profiles and exemplars
# =============================================================================

def _profile_table(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    global_mean = df[list(value_cols)].mean()
    global_std = df[list(value_cols)].std(ddof=0).replace(0, np.nan)
    for cid in sorted(df["community_id"].astype(int).unique().tolist()):
        sub = df[df["community_id"].astype(int) == int(cid)]
        row: Dict[str, Any] = {"community_id": int(cid), "count": int(len(sub))}
        for fn, col in zip(feature_names, value_cols):
            m = float(sub[col].mean())
            gstd = global_std[col]
            z = 0.0 if (not np.isfinite(gstd) or pd.isna(gstd)) else float((m - global_mean[col]) / gstd)
            row[f"mean_{fn}"] = m
            row[f"z_{fn}"] = z
        rows.append(row)
    return pd.DataFrame(rows)



def build_profiles_and_figures(
    *,
    transitions_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    raw_cols: List[str],
    dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Build per-cluster state / action / delta profiles from the raw token columns,
    and generate heatmap + radar figures.
    """
    raw_df = _build_transition_raw_feature_table(
        transitions_df, shots_df=shots_df, raw_cols=raw_cols,
    )
    raw_feature_df = pd.DataFrame(
        raw_df[[f"state_raw_{c}" for c in raw_cols]
               + [f"action_raw_{c}" for c in raw_cols]
               + [f"delta_raw_{c}" for c in raw_cols]].to_numpy(dtype=np.float32),
        columns=[f"state_raw_{c}" for c in raw_cols]
                + [f"action_raw_{c}" for c in raw_cols]
                + [f"delta_raw_{c}" for c in raw_cols],
    )
    # NaN handling
    raw_feature_df = raw_feature_df.fillna(0.0)

    # Join transition rows to rally-level cluster labels
    trans = transitions_df[["rally_id"]].copy().reset_index(drop=True)
    lab = labels_df[["rally_id", "community_id"]].copy()
    joined = trans.merge(lab, on="rally_id", how="left", validate="many_to_one")
    if joined["community_id"].isna().any():
        n_miss = int(joined["community_id"].isna().sum())
        raise RuntimeError(
            f"Failed to align {n_miss} transition rows with rally cluster labels."
        )
    raw_feature_df["community_id"] = joined["community_id"].astype(int).values

    state_cols = [f"state_raw_{c}" for c in raw_cols]
    action_cols = [f"action_raw_{c}" for c in raw_cols]
    delta_cols = [f"delta_raw_{c}" for c in raw_cols]

    state_profile = _profile_table(raw_feature_df, raw_cols, state_cols)
    action_profile = _profile_table(raw_feature_df, raw_cols, action_cols)
    delta_profile = _profile_table(raw_feature_df, raw_cols, delta_cols)

    state_profile.to_csv(dirs["communities"] / "community_state_profiles_raw.csv", index=False)
    action_profile.to_csv(dirs["communities"] / "community_action_profiles_raw.csv", index=False)
    delta_profile.to_csv(dirs["communities"] / "community_delta_profiles_raw.csv", index=False)
    _write_json(
        dirs["communities"] / "profile_columns.json",
        {"raw_columns": list(raw_cols), "n_raw_columns": len(raw_cols)},
    )

    # ── heatmaps ─────────────────────────────────────────────────────────
    def _heatmap(profile_df: pd.DataFrame, out_name: str, title: str) -> None:
        z_cols = [f"z_{c}" for c in raw_cols if f"z_{c}" in profile_df.columns]
        mat = profile_df[z_cols].to_numpy(dtype=float)
        y_labels = [f"community_{int(v)}" for v in profile_df["community_id"].astype(int).tolist()]
        x_labels = [c.replace("z_", "") for c in z_cols]
        try:
            plot_heatmap(dirs["figures"] / out_name, mat, y_labels, x_labels, title)
        except Exception as e:
            logger.warning("Heatmap %s skipped: %s", out_name, e)

    _heatmap(state_profile, "profile_heatmap_state.png", "Style state profiles (z vs global)")
    _heatmap(action_profile, "profile_heatmap_action.png", "Style action profiles (z vs global)")
    _heatmap(delta_profile, "profile_heatmap_delta.png", "Style delta profiles (z vs global)")

    # ── radar ────────────────────────────────────────────────────────────
    merged_delta = delta_profile[["community_id", "count"]].copy()
    for c in raw_cols:
        zc = f"z_{c}"
        if zc in delta_profile.columns:
            merged_delta[f"z_delta_{c}"] = delta_profile[zc].values
    radar_features = [f for f in CURATED_TRANSITION_FEATURES if f in raw_cols]
    try:
        plot_radar_profiles(
            dirs["figures"] / "radar_profiles.png",
            merged_delta,
            radar_features=radar_features,
            topk=min(8, len(merged_delta)),
            title="Style delta profiles",
        )
    except Exception as e:
        logger.warning("Radar plot skipped: %s", e)

    return {"raw_columns": list(raw_cols)}



def _assign_phases(n_transitions: int) -> List[str]:
    """Assign early/mid/late phase labels for transitions in one rally.

    1 transition  → ["mid"]
    2 transitions → ["early", "late"]
    3+ transitions → np.array_split-like into 3 groups
    """
    if n_transitions <= 0:
        return []
    if n_transitions == 1:
        return ["mid"]
    if n_transitions == 2:
        return ["early", "late"]
    phases: List[str] = [""] * n_transitions
    splits = np.array_split(np.arange(n_transitions), 3)
    for phase_name, indices in zip(["early", "mid", "late"], splits):
        for i in indices:
            phases[int(i)] = phase_name
    return phases



def _profile_table_by_phase(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Profile table grouped by (community_id, phase).

    Z-scores are computed against the global distribution WITHIN each phase,
    not overall. This ensures temporal patterns are relative to what is
    typical at that rally stage.
    """
    rows: List[Dict[str, Any]] = []
    for phase in ["early", "mid", "late"]:
        phase_df = df[df["phase"] == phase]
        if len(phase_df) == 0:
            continue
        phase_mean = phase_df[list(value_cols)].mean()
        phase_std = phase_df[list(value_cols)].std(ddof=0).replace(0, np.nan)
        for cid in sorted(phase_df["community_id"].astype(int).unique().tolist()):
            sub = phase_df[phase_df["community_id"].astype(int) == int(cid)]
            row: Dict[str, Any] = {
                "community_id": int(cid),
                "phase": phase,
                "count": int(len(sub)),
            }
            for fn, col in zip(feature_names, value_cols):
                m = float(sub[col].mean())
                gstd = phase_std[col]
                z = 0.0 if (not np.isfinite(gstd) or pd.isna(gstd)) else float(
                    (m - phase_mean[col]) / gstd
                )
                row[f"mean_{fn}"] = m
                row[f"z_{fn}"] = z
            rows.append(row)
    return pd.DataFrame(rows)



def build_phase_profiles(
    *,
    transitions_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    raw_cols: List[str],
    dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """Build temporal (early/mid/late) profiles per community.

    Output CSVs (under communities/):
        community_{state,action,delta}_profiles_tpp3_raw.csv
        Each row: community_id, phase, count, mean_*, z_*
        z_* is computed vs global distribution WITHIN that phase.

    Output heatmaps (under figures/):
        profile_heatmap_{state,action,delta}_tpp3.png
        Rows use "C{cid}_{phase}" labels.
    """
    raw_df = _build_transition_raw_feature_table(
        transitions_df, shots_df=shots_df, raw_cols=raw_cols,
    )
    all_feature_cols = (
        [f"state_raw_{c}" for c in raw_cols]
        + [f"action_raw_{c}" for c in raw_cols]
        + [f"delta_raw_{c}" for c in raw_cols]
    )
    raw_feature_df = pd.DataFrame(
        raw_df[all_feature_cols].to_numpy(dtype=np.float32),
        columns=all_feature_cols,
    ).fillna(0.0)

    # Join community labels (rally-level → transition-level)
    trans_meta = transitions_df[["rally_id", "transition_index"]].copy().reset_index(
        drop=True
    )
    lab = labels_df[["rally_id", "community_id"]].copy()
    joined = trans_meta.merge(lab, on="rally_id", how="left", validate="many_to_one")
    if joined["community_id"].isna().any():
        n_miss = int(joined["community_id"].isna().sum())
        raise RuntimeError(
            f"Phase profiles: failed to align {n_miss} transitions with labels."
        )
    raw_feature_df["community_id"] = joined["community_id"].astype(int).values
    raw_feature_df["rally_id"] = joined["rally_id"].values
    raw_feature_df["transition_index"] = trans_meta["transition_index"].values

    # Compute n_transitions per rally
    rally_sizes = trans_meta.groupby("rally_id").size().to_dict()
    raw_feature_df["_n_trans"] = (
        raw_feature_df["rally_id"].map(rally_sizes).astype(int)
    )

    # Build phase lookup: (n_transitions, transition_index) → phase
    max_n = int(raw_feature_df["_n_trans"].max()) + 1
    phase_lookup: Dict[Tuple[int, int], str] = {}
    for n in range(1, max_n):
        for idx, p in enumerate(_assign_phases(n)):
            phase_lookup[(n, idx)] = p

    raw_feature_df["phase"] = [
        phase_lookup.get((int(nt), int(ti)), "mid")
        for nt, ti in zip(
            raw_feature_df["_n_trans"].values,
            raw_feature_df["transition_index"].values,
        )
    ]

    state_cols = [f"state_raw_{c}" for c in raw_cols]
    action_cols = [f"action_raw_{c}" for c in raw_cols]
    delta_cols = [f"delta_raw_{c}" for c in raw_cols]

    state_tpp3 = _profile_table_by_phase(raw_feature_df, raw_cols, state_cols)
    action_tpp3 = _profile_table_by_phase(raw_feature_df, raw_cols, action_cols)
    delta_tpp3 = _profile_table_by_phase(raw_feature_df, raw_cols, delta_cols)

    state_tpp3.to_csv(
        dirs["communities"] / "community_state_profiles_tpp3_raw.csv", index=False
    )
    action_tpp3.to_csv(
        dirs["communities"] / "community_action_profiles_tpp3_raw.csv", index=False
    )
    delta_tpp3.to_csv(
        dirs["communities"] / "community_delta_profiles_tpp3_raw.csv", index=False
    )

    # Heatmap rows are labelled "C{cid}_{phase}".
    def _heatmap_phase(
        profile_df: pd.DataFrame, out_name: str, title: str
    ) -> None:
        z_cols = [f"z_{c}" for c in raw_cols if f"z_{c}" in profile_df.columns]
        if not z_cols or len(profile_df) == 0:
            return
        mat = profile_df[z_cols].to_numpy(dtype=float)
        y_labels = [
            f"C{int(r['community_id'])}_{r['phase']}"
            for _, r in profile_df.iterrows()
        ]
        x_labels = [c.replace("z_", "") for c in z_cols]
        try:
            plot_heatmap(dirs["figures"] / out_name, mat, y_labels, x_labels, title)
        except Exception as e:
            logger.warning("Phase heatmap %s skipped: %s", out_name, e)

    _heatmap_phase(
        state_tpp3, "profile_heatmap_state_tpp3.png",
        "Style state profiles by phase (z within phase)",
    )
    _heatmap_phase(
        action_tpp3, "profile_heatmap_action_tpp3.png",
        "Style action profiles by phase (z within phase)",
    )
    _heatmap_phase(
        delta_tpp3, "profile_heatmap_delta_tpp3.png",
        "Style delta profiles by phase (z within phase)",
    )

    n_phases = int(raw_feature_df["phase"].nunique())
    logger.info(
        "Phase profiles: %d phases, %d state rows, %d action rows, %d delta rows",
        n_phases, len(state_tpp3), len(action_tpp3), len(delta_tpp3),
    )
    return {
        "n_phases": n_phases,
        "state_rows": len(state_tpp3),
        "action_rows": len(action_tpp3),
        "delta_rows": len(delta_tpp3),
    }



def export_exemplar_rallies(
    *,
    emb: np.ndarray,
    seq_meta: pd.DataFrame,
    centroids: np.ndarray,
    labels: np.ndarray,
    shots_df: pd.DataFrame,
    dirs: Dict[str, Path],
    n_exemplars: int = 5,
    output_shots: bool = True,
    split_name: str = "train_all",
) -> pd.DataFrame:
    """Export N closest-to-centroid rallies per community.

    Returns exemplars DataFrame and writes:
      communities/exemplars.csv
      communities/exemplars/<per-rally shot CSVs>
    """
    from sklearn.metrics.pairwise import cosine_distances

    exemplar_dir = dirs["communities"] / "exemplars"
    if output_shots:
        exemplar_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for cid in sorted(np.unique(labels).tolist()):
        mask = labels == cid
        cid_emb = emb[mask]
        cid_meta = seq_meta[mask].reset_index(drop=True)

        centroid = centroids[int(cid)].reshape(1, -1)
        dists = cosine_distances(cid_emb, centroid).flatten()

        n_pick = min(n_exemplars, len(cid_meta))
        top_idx = np.argsort(dists)[:n_pick]

        for rank, idx in enumerate(top_idx):
            row_meta = cid_meta.iloc[int(idx)]
            row = {
                "split": split_name,
                "community_id": int(cid),
                "rally_id": str(row_meta["rally_id"]),
                "match_id": str(row_meta["match_id"]),
                "point_index": int(row_meta["point_index"]),
                "n_shots": int(row_meta.get("n_shots", 0)),
                "n_transitions": int(row_meta.get("n_transitions", 0)),
                "distance_to_centroid": float(dists[int(idx)]),
                "rank_within_community": rank,
            }
            all_rows.append(row)

            if output_shots:
                mid = str(row_meta["match_id"])
                pidx = int(row_meta["point_index"])
                rally_shots = shots_df[
                    (shots_df["match_id"].astype(str) == mid)
                    & (shots_df["point_index"].apply(_safe_int) == pidx)
                ].sort_values("shot_index")
                fname = f"exemplar_{split_name}_C{cid}_rank{rank}_{mid}_{pidx}.csv"
                rally_shots.to_csv(exemplar_dir / fname, index=False)

    exemplars_df = pd.DataFrame(all_rows)
    exemplars_df.to_csv(dirs["communities"] / "exemplars.csv", index=False)
    logger.info(
        "Exported %d exemplars (%d per community) to %s",
        len(exemplars_df), n_exemplars, dirs["communities"] / "exemplars.csv",
    )
    return exemplars_df



# =============================================================================
# Discovery for one K
# =============================================================================

def discover(
    *,
    k: int,
    fit_emb: np.ndarray,
    emb_by_split: Dict[str, np.ndarray],
    split_bundles: Dict[str, RallyDatasetBundle],
    split_outcomes: Dict[str, pd.DataFrame],
    meta_by_split: Dict[str, pd.DataFrame],
    resid_params: Dict[str, Any],
    model: RallySSLModel,
    raw_cols: List[str],
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    fit_split: str,
    device: torch.device,
    active_splits: List[str],
) -> Dict[str, Any]:
    """Run discovery for one K.

    Clustering, stability, confounds, negative controls, profiles, phase profiles,
    exemplars and the decision summary. Returns the decision dict.
    """
    eval_splits = [s for s in active_splits if s != fit_split]

    # ── clustering ────────────────────────────────────────────────────────
    logger.info("Clustering K=%d on fit_split=%s ...", k, fit_split)
    train_labels = cluster_kmeans(fit_emb, k=k, seed=args.seed, n_init=args.n_init)
    train_metrics = compute_cluster_metrics(fit_emb, train_labels)
    logger.info(
        "  K=%d  silhouette=%.4f  max_frac=%.3f  min_size=%d",
        k, train_metrics.get("silhouette", -1.0),
        train_metrics.get("max_frac", 1.0), train_metrics.get("min_size", 0),
    )

    # ── feasibility ───────────────────────────────────────────────────────
    k_feasible = True
    if train_metrics.get("max_frac", 1.0) > args.max_frac:
        logger.warning(
            "  GIANT BUCKET: max_frac=%.3f > %.3f for K=%d",
            train_metrics["max_frac"], args.max_frac, k,
        )
        k_feasible = False
    if train_metrics.get("min_size", 0) < args.min_size:
        logger.warning(
            "  SMALL CLUSTER: min_size=%d < %d for K=%d",
            train_metrics["min_size"], args.min_size, k,
        )
        k_feasible = False

    # ── centroids + label assignment ──────────────────────────────────────
    centroids = compute_centroids(fit_emb, train_labels)
    np.save(dirs["communities"] / "community_centroids.npy", centroids)

    labels_fit_df = split_bundles[fit_split].seq_meta.copy()
    labels_fit_df["community_id"] = train_labels.astype(int)
    labels_fit_df.to_csv(
        dirs["communities"] / f"community_labels_{fit_split}.csv", index=False
    )

    assigned_labels: Dict[str, np.ndarray] = {fit_split: train_labels}
    for split in eval_splits:
        if split not in emb_by_split:
            continue
        lbl = assign_by_centroid_cosine(emb_by_split[split], centroids)
        assigned_labels[split] = lbl
        out_df = split_bundles[split].seq_meta.copy()
        out_df["community_id"] = lbl.astype(int)
        out_df.to_csv(
            dirs["communities"] / f"community_labels_{split}.csv", index=False
        )

    # ── stability ─────────────────────────────────────────────────────────
    logger.info("  Stability (%d runs) ...", args.stability_runs)
    stability = compute_stability(
        fit_emb, k=k, base_seed=args.seed,
        n_runs=args.stability_runs, n_init=args.n_init,
    )
    _write_json(dirs["validation"] / "stability.json", stability)
    logger.info(
        "  Stability: ARI=%.4f (+/-%.4f)  NMI=%.4f (+/-%.4f)",
        stability["ari_mean"], stability["ari_std"],
        stability["nmi_mean"], stability["nmi_std"],
    )

    # ── confounds ─────────────────────────────────────────────────────────
    logger.info("  Confound metrics ...")
    confounds: Dict[str, Dict[str, Any]] = {}
    for split in active_splits:
        if split not in split_bundles:
            continue
        # The fit split is stored under the "train" key, which rally_interpretation reads
        split_key = "train" if split == fit_split else split
        lbl_df = split_bundles[split].seq_meta.copy()
        lbl_df["community_id"] = assigned_labels.get(split, train_labels).astype(int)
        confounds[split_key] = compute_confound_metrics(
            labels_df=lbl_df,
            shots_df=split_bundles[split].shots_df,
            outcomes_df=split_outcomes[split],
        )
    _write_json(dirs["validation"] / "confound_metrics.json", confounds)
    for sk, cm in confounds.items():
        logger.info(
            "    Confounds [%s]: match_id=%.4f  handedness=%.4f  rally_len=%.4f",
            sk, cm.get("nmi_match_id") or 0.0,
            cm.get("nmi_handedness_pair") or 0.0,
            cm.get("nmi_rally_len_bin") or 0.0,
        )

    # ── negative controls ─────────────────────────────────────────────────
    logger.info("  Negative controls at K=%d ...", k)
    neg_results: Dict[str, Dict[str, Any]] = {}
    for mode in ["shuffle_within_rally", "shuffle_across_rallies"]:
        if mode == "shuffle_within_rally":
            seqs_ctrl = _shuffle_within_rallies(
                split_bundles[fit_split].seq_arrays, seed=args.seed + 9001,
            )
        else:
            seqs_ctrl = _shuffle_across_rallies(
                split_bundles[fit_split].seq_arrays, seed=args.seed + 9002,
            )
        emb_ctrl = encode_rally_embeddings(
            model, seqs_ctrl, batch_size=args.batch_size, device=device,
        )
        if resid_params.get("method", "none") != "none":
            length_vals_ctrl = _get_length_var(
                meta_by_split[fit_split], args.length_var,
            )
            emb_ctrl = apply_residualization_to_array(
                emb_ctrl, length_vals_ctrl, resid_params,
            )
        labels_ctrl = cluster_kmeans(emb_ctrl, k=k, seed=args.seed, n_init=args.n_init)
        metrics_ctrl = compute_cluster_metrics(emb_ctrl, labels_ctrl)
        stability_ctrl = compute_stability(
            emb_ctrl, k=k,
            base_seed=args.seed, n_runs=max(2, args.stability_runs), n_init=args.n_init,
        )
        out_dir_ctrl = dirs["validation"] / "negative_controls" / mode
        out_dir_ctrl.mkdir(parents=True, exist_ok=True)
        np.save(out_dir_ctrl / "rally_embeddings.npy", emb_ctrl)
        pd.DataFrame({
            "rally_id": split_bundles[fit_split].seq_meta["rally_id"].tolist(),
            "community_id": labels_ctrl.astype(int).tolist(),
        }).to_csv(out_dir_ctrl / "community_labels_train.csv", index=False)
        _write_json(out_dir_ctrl / "metrics.json", {
            "community_metrics": metrics_ctrl,
            "stability": stability_ctrl,
        })
        neg_results[mode] = {
            "community_metrics": metrics_ctrl,
            "stability": stability_ctrl,
        }
        logger.info(
            "    %s: silhouette=%.4f  stability_ari=%.4f",
            mode,
            metrics_ctrl.get("silhouette", -1.0),
            stability_ctrl.get("ari_mean", 0.0),
        )

    # ── profiles + figures ────────────────────────────────────────────────
    logger.info("  Profiles and figures ...")
    profile_info = build_profiles_and_figures(
        transitions_df=split_bundles[fit_split].transitions_df,
        shots_df=split_bundles[fit_split].shots_df,
        labels_df=labels_fit_df,
        raw_cols=raw_cols,
        dirs={"communities": dirs["communities"], "figures": dirs["figures"]},
    )

    # ── profiles by rally phase ───────────────────────────────────────────
    logger.info("  Building phase profiles ...")
    try:
        tpp3_info = build_phase_profiles(
            transitions_df=split_bundles[fit_split].transitions_df,
            shots_df=split_bundles[fit_split].shots_df,
            labels_df=labels_fit_df,
            raw_cols=raw_cols,
            dirs={"communities": dirs["communities"], "figures": dirs["figures"]},
        )
    except Exception as e:
        logger.warning("  Phase profiles skipped: %s", e)
        tpp3_info = {"error": str(e)}

    # ── exemplar export ───────────────────────────────────────────────────
    if args.exemplars_per_community > 0:
        logger.info(
            "  Exporting exemplars (%d per community) ...",
            args.exemplars_per_community,
        )
        try:
            export_exemplar_rallies(
                emb=fit_emb,
                seq_meta=split_bundles[fit_split].seq_meta,
                centroids=centroids,
                labels=train_labels,
                shots_df=split_bundles[fit_split].shots_df,
                dirs=dirs,
                n_exemplars=args.exemplars_per_community,
                output_shots=args.exemplar_output_shots,
                split_name=fit_split,
            )
        except Exception as e:
            logger.warning("  Exemplar export skipped: %s", e)

    # ── outcome heatmap ───────────────────────────────────────────────────
    try:
        outcome_heat = build_outcome_heatmap(
            labels_train_df=labels_fit_df,
            outcomes_df=split_outcomes[fit_split],
            fig_path=dirs["figures"] / "outcome_heatmap.png",
        )
        outcome_heat.to_csv(
            dirs["communities"] / "community_outcome_profiles_train.csv",
            index=False,
        )
    except Exception as e:
        logger.warning("  Outcome heatmap skipped: %s", e)

    # ── UMAP ──────────────────────────────────────────────────────────────
    try:
        plot_umap_communities(
            dirs["figures"] / "umap_communities.png",
            fit_emb, train_labels,
            "Rally styles (fit split)",
        )
    except Exception as e:
        logger.warning("  UMAP plot skipped: %s", e)

    # ── community metrics ─────────────────────────────────────────────────
    community_metrics = {"k_selected": k, **train_metrics}
    _write_json(dirs["validation"] / "community_metrics.json", community_metrics)

    # ── decision summary ──────────────────────────────────────────────────
    silhouette_main = float(train_metrics.get("silhouette", -1.0))
    ari_main = float(stability.get("ari_mean", 0.0))

    neg_sil_drop = []
    neg_ari_drop = []
    for mode in ["shuffle_within_rally", "shuffle_across_rallies"]:
        mm = neg_results[mode]["community_metrics"]
        ss = neg_results[mode]["stability"]
        neg_sil_drop.append(float(mm.get("silhouette", -1.0)) < silhouette_main)
        neg_ari_drop.append(float(ss.get("ari_mean", 0.0)) < ari_main)
    negative_controls_ok = all(a or b for a, b in zip(neg_sil_drop, neg_ari_drop))

    confound_threshold = 0.3
    confound_checks = {
        "nmi_match_id": confounds.get("train", {}).get("nmi_match_id"),
        "nmi_handedness_pair": confounds.get("train", {}).get("nmi_handedness_pair"),
        "nmi_rally_len_bin": confounds.get("train", {}).get("nmi_rally_len_bin"),
        "nmi_outcome_result": confounds.get("train", {}).get("nmi_outcome_result"),
        "nmi_outcome_reason": confounds.get("train", {}).get("nmi_outcome_reason"),
    }
    confound_violations: Dict[str, float] = {}
    for key, value in confound_checks.items():
        if value is not None and float(value) > confound_threshold:
            confound_violations[key] = float(value)
    confound_ok = len(confound_violations) == 0
    stability_ok = ari_main >= 0.7
    silhouette_ok = silhouette_main > 0.05

    passed = bool(
        k_feasible and silhouette_ok and stability_ok
        and negative_controls_ok and confound_ok
    )
    if passed:
        conclusion = (
            "Rally styles pass the stability, negative-control and confound criteria."
        )
    else:
        reasons = []
        if not k_feasible:
            reasons.append("no feasible K under giant-bucket/min-size constraints")
        if not silhouette_ok:
            reasons.append(f"low silhouette ({silhouette_main:.4f})")
        if not stability_ok:
            reasons.append(f"low stability ARI ({ari_main:.4f})")
        if not negative_controls_ok:
            reasons.append("negative controls did not degrade")
        if not confound_ok:
            reasons.append(f"confound violations: {confound_violations}")
        conclusion = f"No clean schemas: {'; '.join(reasons)}"

    decision: Dict[str, Any] = {
        "pass": passed,
        "conclusion": conclusion,
        "k_feasible": k_feasible,
        "criteria": {
            "k_feasible": bool(k_feasible),
            "silhouette_ok": bool(silhouette_ok),
            "stability_ari_ok": bool(stability_ok),
            "negative_controls_degrade_ok": bool(negative_controls_ok),
            "confound_ok": bool(confound_ok),
        },
        "main_metrics": {
            "k_selected": k,
            "silhouette_train": silhouette_main,
            "stability_ari_mean": ari_main,
            "stability_nmi_mean": float(stability.get("nmi_mean", 0.0)),
            **{key: val for key, val in confound_checks.items()},
        },
        "max_frac_threshold": args.max_frac,
        "min_size_threshold": args.min_size,
        "confound_threshold": confound_threshold,
        "confound_violations": confound_violations,
        "negative_controls": neg_results,
        "length_residualization": {
            "method": resid_params.get("method", "none"),
            "length_var": args.length_var,
            "l2_normalize": args.resid_l2_normalize,
        },
        "tpp3_info": tpp3_info,
        "fit_split": fit_split,
        "active_splits": active_splits,
    }
    _write_json(dirs["validation"] / "decision_summary.json", decision)

    logger.info("=" * 60)
    logger.info("DECISION (K=%d): pass=%s", k, passed)
    logger.info("  %s", conclusion)
    logger.info("=" * 60)

    return decision



# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rally style discovery on frozen encoder embeddings.")
    p.add_argument("--config", type=Path, default=Path("configs/rally.yaml"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/rally/models/best.pt"),
                   help="Rally encoder checkpoint written by courtstyle.training.train_rally.")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/rally/discovery"))
    p.add_argument(
        "--splits", type=str, default="train_all,val,test",
        help=(
            "Comma-separated splits. The first is clustered, the others are assigned to "
            "the nearest centroid. Valid: train_all, train, val, test."
        ),
    )
    p.add_argument("--exemplars-per-community", type=int, default=5,
                   help="Closest-to-centroid rallies exported per style (0 disables).")
    p.add_argument("--no-exemplar-output-shots", action="store_false", dest="exemplar_output_shots",
                   help="Do not write per-exemplar shot CSVs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding rallies.")
    p.add_argument("--limit-rallies", type=int, default=0,
                   help="If >0, keep only the first N rallies per split (smoke test).")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    cfg = yaml.safe_load(cli.config.read_text())
    data_cfg, disc_cfg = cfg["data"], cfg["discovery"]

    args = argparse.Namespace(
        seed=int(disc_cfg["seed"]),
        fixed_k=int(disc_cfg["fixed_k"]),
        n_init=int(disc_cfg["n_init"]),
        stability_runs=int(disc_cfg["stability_runs"]),
        max_frac=float(disc_cfg["max_frac"]),
        min_size=int(disc_cfg["min_size"]),
        length_residualize=str(disc_cfg["length_residualize"]),
        length_var=str(disc_cfg["length_var"]),
        resid_l2_normalize=bool(disc_cfg["resid_l2_normalize"]),
        batch_size=int(cli.batch_size),
        exemplars_per_community=int(cli.exemplars_per_community),
        exemplar_output_shots=bool(cli.exemplar_output_shots),
        train_shots_csv=Path(data_cfg["train_shots"]),
        train_outcomes_csv=Path(data_cfg["train_outcomes"]),
        test_shots_csv=Path(data_cfg["test_shots"]),
        test_outcomes_csv=Path(data_cfg["test_outcomes"]),
        split_info=Path(data_cfg["split_info"]),
        sequence_checkpoint=cli.checkpoint,
    )
    limit_rallies = cli.limit_rallies if cli.limit_rallies > 0 else None
    set_all_seeds(args.seed)

    active_splits = [s.strip() for s in cli.splits.split(",") if s.strip()]
    valid_splits = {"train_all", "train", "val", "test"}
    for s in active_splits:
        if s not in valid_splits:
            raise ValueError(f"Unknown split '{s}'. Valid: {sorted(valid_splits)}")
    if not active_splits:
        raise ValueError("--splits must specify at least one split.")
    fit_split = active_splits[0]
    logger.info("Active splits: %s  (fit_split=%s)", active_splits, fit_split)

    needed = [args.sequence_checkpoint, args.split_info]
    if any(s != "test" for s in active_splits):
        needed += [args.train_shots_csv, args.train_outcomes_csv]
    if "test" in active_splits:
        needed += [args.test_shots_csv, args.test_outcomes_csv]
    if not args.sequence_checkpoint.exists():
        sys.exit(f"Missing checkpoint: {args.sequence_checkpoint}. Train the encoder with courtstyle.training.train_rally first.")
    for path in needed:
        if not path.exists():
            sys.exit(f"Missing input file: {path}. See docs/data_format.md for the expected rally inputs.")

    # ── load checkpoint ──────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", args.sequence_checkpoint)
    seq_ckpt = torch.load(args.sequence_checkpoint, map_location="cpu", weights_only=False)
    model_cfg = seq_ckpt["model_config"]
    raw_cols: List[str] = seq_ckpt["raw_columns"]
    token_columns: List[str] = seq_ckpt["token_columns"]
    leaked = [c for c in raw_cols if c in EXCLUDED_FROM_TOKENS]
    if leaked:
        raise RuntimeError(f"Checkpoint raw_columns contain excluded columns: {leaked}")

    model = RallySSLModel(
        input_dim=int(model_cfg["input_dim"]),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        pooling=str(model_cfg["pooling"]),
        max_len=int(model_cfg["max_len"]),
        proj_dim=int(model_cfg["proj_dim"]),
        encoder_block=str(model_cfg.get("encoder_block", "vanilla")),
        modern_use_gated_mha=bool(model_cfg.get("modern_use_gated_mha", True)),
    )
    model.load_state_dict(seq_ckpt["sequence_model_state"], strict=True)
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cli.work_dir
    dirs = _ensure_dirs(run_dir)
    logger.info("Discovery run dir: %s", run_dir)

    # ── build datasets for each active split ─────────────────────────────
    split_bundles: Dict[str, RallyDatasetBundle] = {}
    split_outcomes: Dict[str, pd.DataFrame] = {}
    for split in active_splits:
        logger.info("Building split=%s ...", split)
        shots_df, outcomes_df = _load_shots_outcomes(
            split=split,
            train_shots_csv=args.train_shots_csv,
            train_outcomes_csv=args.train_outcomes_csv,
            test_shots_csv=args.test_shots_csv,
            test_outcomes_csv=args.test_outcomes_csv,
            split_info=args.split_info,
        )
        bundle = build_rally_dataset(
            shots_df,
            raw_cols=raw_cols,
            min_rally_len=int(data_cfg["min_rally_len"]),
            limit_rallies=limit_rallies,
            track_name=split,
        )
        if bundle.token_matrix.shape[1] != model_cfg["input_dim"]:
            raise RuntimeError(
                f"Token dim {bundle.token_matrix.shape[1]} != "
                f"checkpoint input_dim {model_cfg['input_dim']} for split={split}"
            )
        split_bundles[split] = bundle
        split_outcomes[split] = outcomes_df
        logger.info(
            "  split=%s  rallies=%d  transitions=%d",
            split, len(bundle.seq_arrays), len(bundle.transitions_df),
        )

    # ── encode embeddings ────────────────────────────────────────────────
    raw_emb_by_split: Dict[str, np.ndarray] = {}
    for split, bundle in split_bundles.items():
        logger.info("Encoding embeddings for split=%s ...", split)
        emb = encode_rally_embeddings(
            model, bundle.seq_arrays, batch_size=args.batch_size, device=device,
        )
        raw_emb_by_split[split] = emb
        np.save(dirs["embeddings"] / f"rally_embeddings_{split}_raw.npy", emb)
        bundle.seq_meta.to_csv(dirs["embeddings"] / f"rally_embeddings_{split}_metadata.csv", index=False)
        logger.info("  %s embedding shape: %s", split, emb.shape)

    meta_by_split: Dict[str, pd.DataFrame] = {s: split_bundles[s].seq_meta for s in split_bundles}
    leakage_dir = dirs["run"] / "length_residualization"
    leakage_dir.mkdir(parents=True, exist_ok=True)

    # ── length leakage before residualisation ────────────────────────────
    leakage_raw = compute_length_leakage(raw_emb_by_split, meta_by_split, fit_split=fit_split)
    leakage_raw["stage"] = "before_residualization"
    save_length_leakage(leakage_raw, leakage_dir)

    # ── length residualisation ───────────────────────────────────────────
    logger.info(
        "Length residualization: method=%s  length_var=%s  l2_norm=%s",
        args.length_residualize, args.length_var, args.resid_l2_normalize,
    )
    emb_by_split, resid_params = fit_and_apply_residualization(
        raw_emb_by_split, meta_by_split,
        method=args.length_residualize,
        length_var=args.length_var,
        l2_normalize=args.resid_l2_normalize,
        fit_split=fit_split,
    )
    _write_json(leakage_dir / "resid_params.json", resid_params)
    for split in emb_by_split:
        np.save(dirs["embeddings"] / f"rally_embeddings_{split}.npy", emb_by_split[split])

    if resid_params.get("method", "none") != "none":
        leakage_resid = compute_length_leakage(emb_by_split, meta_by_split, fit_split=fit_split)
        leakage_resid["stage"] = "after_residualization"
        save_length_leakage(leakage_resid, leakage_dir / "after_resid")

    # ── discovery at the configured K ────────────────────────────────────
    discover(
        k=args.fixed_k,
        fit_emb=emb_by_split[fit_split],
        emb_by_split=emb_by_split,
        split_bundles=split_bundles,
        split_outcomes=split_outcomes,
        meta_by_split=meta_by_split,
        resid_params=resid_params,
        model=model,
        raw_cols=raw_cols,
        args=args,
        dirs=dirs,
        fit_split=fit_split,
        device=device,
        active_splits=active_splits,
    )

    run_config: Dict[str, Any] = {
        "timestamp": timestamp,
        "sequence_checkpoint": str(args.sequence_checkpoint),
        "raw_columns": raw_cols,
        "token_columns": token_columns,
        "active_splits": active_splits,
        "fit_split": fit_split,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split_dataset_stats": {k: v.stats for k, v in split_bundles.items()},
        "length_residualization": {
            "method": resid_params.get("method", "none"),
            "params": {k: v for k, v in resid_params.items() if k != "coef"},
            "leakage_dir": str(leakage_dir),
        },
    }
    _write_json(dirs["run"] / "run_config.json", run_config)
    logger.info("Discovery complete. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()

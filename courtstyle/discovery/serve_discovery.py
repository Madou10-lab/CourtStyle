#!/usr/bin/env python3
"""
Serve Style Community Discovery Pipeline

Discovers serve styles as communities of the frozen serve embedding space.

This pipeline:
1. Loads the trained encoder and extracts record embeddings.
2. Builds a cosine k-nearest-neighbour graph, symmetrised over the directed
   edges, with Gaussian affinities whose bandwidth is the median distance.
3. Runs Leiden community detection at resolution 1.0.
4. Merges the fine-grained communities by Ward linkage on their standardised
   feature profiles, down to the target number of styles (3 in the paper).
5. Validates the partition and writes profiles, tables and figures.

Usage:
    python -m courtstyle.discovery.serve_discovery \\
        --ckpt-path outputs/serve/serve_encoder_best.pt \\
        --processed-dir data/serve/processed --work-dir outputs/serve

Output structure:
    community_run_YYYYMMDD_HHMMSS/
    ├── embeddings/
    │   ├── train_embeddings.npy
    │   ├── val_embeddings.npy
    │   └── metadata.csv
    ├── communities/
    │   ├── community_labels_train.csv
    │   ├── community_labels_val.csv
    │   ├── community_centroids.npy
    │   └── community_profiles.csv
    ├── validation/
    │   ├── community_metrics.json
    │   ├── parameter_sensitivity.csv
    │   ├── stability_analysis.csv
    │   ├── statistical_tests.csv
    │   └── baseline_comparison.csv
    ├── figures/
    │   ├── umap_communities.png
    │   ├── umap_surface.png
    │   ├── resolution_sensitivity.png
    │   ├── community_outcome_heatmap.png
    │   ├── community_radar_profiles.png
    │   ├── stability_boxplot.png
    │   └── community_sizes.png
    └── run_config.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from courtstyle.models.serve_encoder import ContrastiveHead, ModernTabularEncoder, PredictionHead

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports
try:
    import leidenalg
    import igraph as ig
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    leidenalg = None
    ig = None

try:
    from networkx.algorithms.community import louvain_communities
except ImportError:
    louvain_communities = None

try:
    from community import community_louvain
except ImportError:
    community_louvain = None

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


WIDTH_EPS = 1e-6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str | None) -> List[float]:
    if not value:
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str | None) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def safe_torch_load(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        try:
            from torch.serialization import add_safe_globals
            scalar_type = getattr(np, "_core", np.core).multiarray.scalar
            add_safe_globals([scalar_type])
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            print("WARNING: weights_only load failed, falling back to weights_only=False.")
            try:
                import numpy.core as np_core
                sys.modules.setdefault("numpy._core", np_core)
                sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
            except Exception:
                pass
            return torch.load(path, map_location=device, weights_only=False)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def infer_surface_cols(df: pd.DataFrame) -> List[str]:
    surface_cols_all = [c for c in df.columns if c.startswith("surface_")]
    required = ["surface_Hard", "surface_Grass"]
    missing = [c for c in required if c not in surface_cols_all]
    if missing:
        raise ValueError(f"Missing required surface columns: {missing}")
    return required + [c for c in surface_cols_all if c not in required]


def assert_surface_binary(df: pd.DataFrame, surface_cols: List[str]) -> None:
    required = ["surface_Hard", "surface_Grass"]
    extra = [c for c in surface_cols if c not in required]
    if extra:
        extra_sum = df[extra].fillna(0).to_numpy().sum(axis=1)
        bad_extra = np.where(extra_sum > WIDTH_EPS)[0]
        if bad_extra.size:
            raise ValueError(f"Surface extra columns not zero. Bad rows: {bad_extra[:10].tolist()}")

    hard = df["surface_Hard"].fillna(0).to_numpy()
    grass = df["surface_Grass"].fillna(0).to_numpy()
    total = hard + grass
    bad = np.where(np.abs(total - 1.0) > WIDTH_EPS)[0]
    if bad.size:
        raise ValueError(f"surface_Hard + surface_Grass must sum to 1. Bad rows: {bad[:10].tolist()}")


def to_tensors(
    df: pd.DataFrame,
    cont_cols: List[str],
    cat_names: List[str],
    surface_cols: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    x_cont = torch.tensor(df[cont_cols].to_numpy(dtype=np.float32))

    cat_indices = []
    for prefix in cat_names:
        cols = [c for c in df.columns if c.startswith(prefix + "_")]
        if not cols:
            raise ValueError(f"No columns found for categorical prefix '{prefix}_'.")
        arr = df[cols].to_numpy(dtype=np.float32)
        cat_indices.append(arr.argmax(axis=1))
    if cat_indices:
        x_cat = torch.tensor(np.stack(cat_indices, axis=1), dtype=torch.long)
    else:
        x_cat = torch.zeros((len(df), 0), dtype=torch.long)

    surface_arr = df[surface_cols].to_numpy(dtype=np.float32)
    surface_labels = surface_arr.argmax(axis=1).astype(int)
    return x_cont, x_cat, surface_labels


# =============================================================================
# Model Building
# =============================================================================

def build_models(ckpt: Dict, device: torch.device) -> Tuple[ModernTabularEncoder, ContrastiveHead, PredictionHead]:
    cont_cols = ckpt["cont_cols"]
    cat_names = ckpt["cat_names"]
    cardinalities = ckpt["cardinalities"]
    config = ckpt.get("config", {})

    if isinstance(cardinalities, dict):
        categorical_cardinalities = cardinalities
    else:
        categorical_cardinalities = dict(zip(cat_names, cardinalities))

    proj_dim = None
    if "head" in ckpt:
        for key, value in ckpt["head"].items():
            if key.endswith("net.3.weight"):
                proj_dim = int(value.shape[0])
                break
    if proj_dim is None:
        proj_dim = int(config.get("proj_dim", 64))

    encoder = ModernTabularEncoder(
        continuous_cols=cont_cols,
        categorical_cardinalities=categorical_cardinalities,
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 8),
        depth=config.get("depth", 4),
        expansion_n=config.get("expansion_n", 4),
        ff_multiplier=config.get("ff_multiplier", 4),
        dropout=config.get("dropout", 0.1),
        encoder_block=config.get("encoder_block", "mhc"),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=True)

    head = ContrastiveHead(
        d_model=config.get("d_model", 128),
        proj_dim=proj_dim,
    ).to(device)
    predictor = PredictionHead(proj_dim=proj_dim).to(device)

    if "head" in ckpt:
        head.load_state_dict(ckpt["head"], strict=True)
    if "predictor" in ckpt:
        predictor.load_state_dict(ckpt["predictor"], strict=True)

    return encoder, head, predictor


@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    predictor: torch.nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    cont_cols: List[str],
    cat_names: List[str],
    batch_size: int,
    device: torch.device,
    space: str,
) -> np.ndarray:
    encoder.eval()
    head.eval()
    predictor.eval()

    outs = []
    n_samples = x_cont.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bc = x_cont[start:end].to(device)
        bcat = x_cat[start:end].to(device)

        x_cont_dict = {name: bc[:, j] for j, name in enumerate(cont_cols)}
        x_cat_dict = {name: bcat[:, j] for j, name in enumerate(cat_names)}

        z = encoder(x_cont_dict, x_cat_dict)
        if space == "encoder":
            out = z
        else:
            proj = head(z)
            if space == "projection":
                out = proj
            elif space == "predictor":
                out = predictor(proj)
            else:
                raise ValueError(f"Unknown space: {space}")

        outs.append(out.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


# =============================================================================
# Graph Construction & Community Detection
# =============================================================================

def build_knn_graph(embeddings: np.ndarray, k: int, metric: str = "cosine") -> Tuple[nx.Graph, float]:
    """Build KNN graph with Gaussian kernel weights. Returns graph and sigma used."""
    knn = kneighbors_graph(embeddings, n_neighbors=k, metric=metric, mode="distance", include_self=False)
    dist_matrix = knn.maximum(knn.T)  # Symmetrize
    coo = dist_matrix.tocoo()

    distances = coo.data
    sigma = float(np.median(distances)) if distances.size > 0 else 1.0

    G = nx.Graph()
    G.add_nodes_from(range(embeddings.shape[0]))
    for i, j, d in zip(coo.row, coo.col, coo.data):
        if i < j:  # Avoid duplicate edges
            sim = np.exp(-(d ** 2) / sigma) if sigma > 0 else 0.0
            G.add_edge(int(i), int(j), weight=float(sim))
    return G, sigma


def run_leiden(G: nx.Graph, n_nodes: int, resolution: float, seed: int) -> np.ndarray:
    """Run Leiden community detection."""
    if HAS_LEIDEN:
        g_ig = ig.Graph()
        g_ig.add_vertices(n_nodes)
        edges = [(int(u), int(v), float(d["weight"])) for u, v, d in G.edges(data=True)]
        g_ig.add_edges([(u, v) for u, v, _ in edges])
        g_ig.es["weight"] = [w for _, _, w in edges]
        partition = leidenalg.find_partition(
            g_ig,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        return np.array(partition.membership, dtype=int)

    if louvain_communities is not None:
        communities = louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
        labels = np.full(n_nodes, -1, dtype=int)
        for cid, comm in enumerate(communities):
            for node in comm:
                labels[int(node)] = int(cid)
        return labels

    if community_louvain is not None:
        partition = community_louvain.best_partition(G, weight="weight", random_state=seed, resolution=resolution)
        return np.array([partition.get(i, -1) for i in range(n_nodes)], dtype=int)

    raise RuntimeError("No community detection backend available.")




def compute_modularity(G: nx.Graph, labels: np.ndarray) -> float:
    """Compute modularity of partition."""
    try:
        from networkx.algorithms.community import modularity
        communities = {}
        for idx, lbl in enumerate(labels):
            communities.setdefault(int(lbl), set()).add(int(idx))
        return float(modularity(G, list(communities.values()), weight="weight"))
    except Exception:
        return float("nan")


# =============================================================================
# Profile-Based Hierarchical Merging
# =============================================================================

def compute_profile_vectors(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
) -> Tuple[np.ndarray, List[int]]:
    """Compute z-score profile vector for each community."""
    global_mean = features[feature_cols].mean()
    global_std = features[feature_cols].std(ddof=0).replace(0, 1e-8)

    community_ids = sorted(np.unique(labels))
    profiles = []

    for cid in community_ids:
        mask = labels == cid
        subset = features.loc[mask, feature_cols]
        mean_vals = subset.mean()
        z_scores = (mean_vals - global_mean) / global_std
        profiles.append(z_scores.values)

    return np.vstack(profiles), community_ids


def hierarchical_merge_communities(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    target_n: int,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray]:
    """
    Merge fine-grained communities based on profile similarity.

    Args:
        features: Feature dataframe
        labels: Original community labels
        feature_cols: Columns to use for profile computation
        target_n: Target number of merged communities
        linkage_method: Hierarchical clustering method ('ward', 'average', 'complete')

    Returns:
        merged_labels: New community labels after merging
        merge_map: Dict mapping new community ID to list of original IDs
        profile_distances: Pairwise profile distance matrix (original communities)
    """
    # Compute profile vectors for each original community
    profile_vectors, original_ids = compute_profile_vectors(features, labels, feature_cols)
    n_original = len(original_ids)

    if n_original <= target_n:
        print(f"  Already have {n_original} communities <= target {target_n}, no merging needed")
        merge_map = {i: [original_ids[i]] for i in range(n_original)}
        return labels.copy(), merge_map, np.zeros((n_original, n_original))

    # Compute pairwise distances between profiles
    profile_distances = squareform(pdist(profile_vectors, metric="euclidean"))

    # Hierarchical clustering on profile vectors
    Z = linkage(profile_vectors, method=linkage_method)

    # Cut to get target_n clusters
    merged_cluster_ids = fcluster(Z, t=target_n, criterion="maxclust")

    # Build merge map: new_id -> [original_ids]
    merge_map = {}
    for i, (orig_id, new_id) in enumerate(zip(original_ids, merged_cluster_ids)):
        new_id = int(new_id) - 1  # fcluster is 1-indexed
        if new_id not in merge_map:
            merge_map[new_id] = []
        merge_map[new_id].append(int(orig_id))

    # Create merged labels
    old_to_new = {}
    for new_id, orig_list in merge_map.items():
        for orig_id in orig_list:
            old_to_new[orig_id] = new_id

    merged_labels = np.array([old_to_new[int(lbl)] for lbl in labels], dtype=int)

    return merged_labels, merge_map, profile_distances


def compute_profile_distinctiveness(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    Compute metrics measuring how distinct the community profiles are.

    Returns:
        - mean_pairwise_distance: Average Euclidean distance between profile vectors
        - min_pairwise_distance: Minimum distance (worst case)
        - profile_spread: Std of distances
    """
    profile_vectors, _ = compute_profile_vectors(features, labels, feature_cols)

    if len(profile_vectors) < 2:
        return {"mean_pairwise_distance": 0, "min_pairwise_distance": 0, "profile_spread": 0}

    distances = pdist(profile_vectors, metric="euclidean")

    return {
        "mean_pairwise_distance": float(np.mean(distances)),
        "min_pairwise_distance": float(np.min(distances)),
        "max_pairwise_distance": float(np.max(distances)),
        "profile_spread": float(np.std(distances)),
    }


def find_optimal_merge_target(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    min_communities: int = 3,
    max_communities: int = None,
    linkage_method: str = "ward",
) -> pd.DataFrame:
    """
    Sweep through different merge targets to find optimal distinctiveness.

    Returns DataFrame with metrics for each target_n.
    """
    n_original = len(np.unique(labels))
    if max_communities is None:
        max_communities = n_original - 1

    results = []

    for target_n in range(min_communities, max_communities + 1):
        merged_labels, merge_map, _ = hierarchical_merge_communities(
            features, labels, feature_cols, target_n, linkage_method
        )

        distinctiveness = compute_profile_distinctiveness(features, merged_labels, feature_cols)

        # Also compute sizes
        sizes = pd.Series(merged_labels).value_counts()

        results.append({
            "target_n": target_n,
            "mean_profile_distance": distinctiveness["mean_pairwise_distance"],
            "min_profile_distance": distinctiveness["min_pairwise_distance"],
            "max_profile_distance": distinctiveness["max_pairwise_distance"],
            "profile_spread": distinctiveness["profile_spread"],
            "size_mean": sizes.mean(),
            "size_std": sizes.std(),
            "size_cv": sizes.std() / sizes.mean() if sizes.mean() > 0 else 0,
        })

    return pd.DataFrame(results)


def select_merge_target(merge_sweep: pd.DataFrame, requested_target: int) -> int:
    """Pick target_n that maximizes distinctiveness within <= requested_target."""
    if merge_sweep.empty:
        return requested_target
    candidates = merge_sweep[merge_sweep["target_n"] <= requested_target]
    if candidates.empty:
        candidates = merge_sweep
    ranked = candidates.sort_values(
        by=["mean_profile_distance", "min_profile_distance", "size_cv"],
        ascending=[False, False, True],
    )
    return int(ranked.iloc[0]["target_n"])


def plot_merge_analysis(
    merge_sweep: pd.DataFrame,
    profile_distances: np.ndarray,
    merge_map: Dict[int, List[int]],
    out_dir: Path,
) -> None:
    """Generate visualization for the merge analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean profile distance vs target_n
    ax = axes[0, 0]
    ax.plot(merge_sweep["target_n"], merge_sweep["mean_profile_distance"], "o-", linewidth=2, markersize=8)
    ax.fill_between(
        merge_sweep["target_n"],
        merge_sweep["min_profile_distance"],
        merge_sweep["max_profile_distance"],
        alpha=0.2,
    )
    ax.set_xlabel("Number of Communities", fontsize=12)
    ax.set_ylabel("Profile Distance", fontsize=12)
    ax.set_title("Profile Distinctiveness vs Community Count", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark the "sweet spot" - where min_distance is still reasonable
    # A good target is where min_distance is above some threshold

    # Plot 2: Original profile distance heatmap
    ax = axes[0, 1]
    if profile_distances.size > 0:
        im = ax.imshow(profile_distances, cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Original Community", fontsize=12)
        ax.set_ylabel("Original Community", fontsize=12)
        ax.set_title("Pairwise Profile Distances (Original)", fontsize=14)
        plt.colorbar(im, ax=ax, label="Euclidean Distance")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
        ax.set_title("Profile Distance Matrix")

    # Plot 3: Size coefficient of variation
    ax = axes[1, 0]
    ax.plot(merge_sweep["target_n"], merge_sweep["size_cv"], "s-", linewidth=2, markersize=8, color="seagreen")
    ax.set_xlabel("Number of Communities", fontsize=12)
    ax.set_ylabel("Size CV (Std/Mean)", fontsize=12)
    ax.set_title("Community Size Balance", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 4: Merge tree visualization (text)
    ax = axes[1, 1]
    ax.axis("off")

    merge_text = "Merge Map:\n\n"
    for new_id in sorted(merge_map.keys()):
        orig_ids = merge_map[new_id]
        merge_text += f"  Community {new_id}: merged from {orig_ids}\n"

    ax.text(0.1, 0.9, merge_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Merge Mapping", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_dir / "merge_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Validation Metrics
# =============================================================================

def compute_community_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    G: nx.Graph,
) -> Dict[str, float]:
    """Compute comprehensive community validation metrics."""
    n_communities = len(np.unique(labels))

    metrics = {
        "n_communities": n_communities,
        "n_samples": len(labels),
    }

    if n_communities > 1 and n_communities < len(labels):
        metrics["silhouette_cosine"] = float(silhouette_score(embeddings, labels, metric="cosine"))
        metrics["davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))

    if G is not None:
        metrics["modularity"] = compute_modularity(G, labels)

    # Community size statistics
    sizes = pd.Series(labels).value_counts()
    metrics["size_mean"] = float(sizes.mean())
    metrics["size_std"] = float(sizes.std())
    metrics["size_min"] = int(sizes.min())
    metrics["size_max"] = int(sizes.max())
    metrics["size_cv"] = float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else 0.0

    return metrics


def compute_stability_metrics(
    embeddings: np.ndarray,
    base_labels: np.ndarray,
    community_fn: Callable[[int], np.ndarray],
    seeds: List[int],
) -> Dict[str, Any]:
    """Compute stability across different random seeds."""
    all_labels = [base_labels]
    for seed in seeds:
        all_labels.append(community_fn(seed))

    ari_scores = []
    nmi_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))
            nmi_scores.append(normalized_mutual_info_score(all_labels[i], all_labels[j]))

    return {
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "ari_min": float(np.min(ari_scores)),
        "ari_max": float(np.max(ari_scores)),
        "nmi_mean": float(np.mean(nmi_scores)),
        "nmi_std": float(np.std(nmi_scores)),
        "nmi_min": float(np.min(nmi_scores)),
        "nmi_max": float(np.max(nmi_scores)),
        "n_runs": len(all_labels),
        "ari_scores": ari_scores,
        "nmi_scores": nmi_scores,
    }


def compute_parameter_sensitivity(
    embeddings: np.ndarray,
    base_labels: np.ndarray,
    k_values: List[int],
    resolution_values: List[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep k and resolution parameters to show robustness."""
    results = []

    for k in k_values:
        G, sigma = build_knn_graph(embeddings, k)
        for res in resolution_values:
            labels = run_leiden(G, len(embeddings), res, seed)
            n_comm = len(np.unique(labels))

            modularity = compute_modularity(G, labels)
            ari_vs_base = adjusted_rand_score(base_labels, labels)
            nmi_vs_base = normalized_mutual_info_score(base_labels, labels)

            sil = float(silhouette_score(embeddings, labels, metric="cosine")) if n_comm > 1 else float("nan")

            results.append({
                "k": k,
                "resolution": res,
                "n_communities": n_comm,
                "modularity": modularity,
                "silhouette": sil,
                "ari_vs_base": ari_vs_base,
                "nmi_vs_base": nmi_vs_base,
            })

    return pd.DataFrame(results)


def compute_statistical_tests(
    outcomes: pd.DataFrame,
    labels: np.ndarray,
    outcome_cols: List[str],
) -> pd.DataFrame:
    """Compute ANOVA/Kruskal-Wallis tests for outcome differentiation."""
    results = []

    for col in outcome_cols:
        values = outcomes[col].values
        groups = [values[labels == c] for c in np.unique(labels)]

        # Remove groups with insufficient data
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            continue

        # Kruskal-Wallis (non-parametric)
        try:
            h_stat, kw_pvalue = stats.kruskal(*groups)
        except Exception:
            h_stat, kw_pvalue = float("nan"), float("nan")

        # One-way ANOVA (parametric)
        try:
            f_stat, anova_pvalue = stats.f_oneway(*groups)
        except Exception:
            f_stat, anova_pvalue = float("nan"), float("nan")

        # Effect size (eta-squared for ANOVA)
        try:
            all_values = np.concatenate(groups)
            grand_mean = all_values.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = ((all_values - grand_mean)**2).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
        except Exception:
            eta_squared = float("nan")

        results.append({
            "outcome": col,
            "kruskal_h": h_stat,
            "kruskal_pvalue": kw_pvalue,
            "anova_f": f_stat,
            "anova_pvalue": anova_pvalue,
            "eta_squared": eta_squared,
            "n_groups": len(groups),
            "significant_001": kw_pvalue < 0.001 if not np.isnan(kw_pvalue) else False,
            "significant_005": kw_pvalue < 0.05 if not np.isnan(kw_pvalue) else False,
        })

    return pd.DataFrame(results)


def compute_baseline_comparison(
    embeddings: np.ndarray,
    leiden_labels: np.ndarray,
    G: nx.Graph,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Leiden to baseline clustering methods."""
    n_communities = len(np.unique(leiden_labels))
    results = []

    # Leiden (reference)
    results.append({
        "method": "Leiden",
        "n_clusters": n_communities,
        "silhouette": float(silhouette_score(embeddings, leiden_labels, metric="cosine")),
        "modularity": compute_modularity(G, leiden_labels),
        "davies_bouldin": float(davies_bouldin_score(embeddings, leiden_labels)),
    })

    # K-Means
    try:
        kmeans = KMeans(n_clusters=n_communities, random_state=seed, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        results.append({
            "method": "K-Means",
            "n_clusters": n_communities,
            "silhouette": float(silhouette_score(embeddings, kmeans_labels, metric="cosine")),
            "modularity": compute_modularity(G, kmeans_labels),
            "davies_bouldin": float(davies_bouldin_score(embeddings, kmeans_labels)),
        })
    except Exception as e:
        print(f"K-Means failed: {e}")

    # Spectral Clustering
    try:
        spectral = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            random_state=seed,
        )
        # Build affinity matrix from graph
        adj_matrix = nx.to_numpy_array(G, weight="weight")
        spectral_labels = spectral.fit_predict(adj_matrix)
        results.append({
            "method": "Spectral",
            "n_clusters": n_communities,
            "silhouette": float(silhouette_score(embeddings, spectral_labels, metric="cosine")),
            "modularity": compute_modularity(G, spectral_labels),
            "davies_bouldin": float(davies_bouldin_score(embeddings, spectral_labels)),
        })
    except Exception as e:
        print(f"Spectral clustering failed: {e}")

    # HDBSCAN (if available)
    if HAS_HDBSCAN:
        try:
            hdbscan = HDBSCAN(min_cluster_size=max(5, len(embeddings) // 20), metric="cosine")
            hdbscan_labels = hdbscan.fit_predict(embeddings)
            n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
            if n_hdbscan > 1:
                # Only score non-noise points
                mask = hdbscan_labels >= 0
                results.append({
                    "method": "HDBSCAN",
                    "n_clusters": n_hdbscan,
                    "silhouette": float(silhouette_score(embeddings[mask], hdbscan_labels[mask], metric="cosine")),
                    "modularity": compute_modularity(G, hdbscan_labels),
                    "davies_bouldin": float(davies_bouldin_score(embeddings[mask], hdbscan_labels[mask])),
                    "noise_fraction": float((hdbscan_labels < 0).mean()),
                })
        except Exception as e:
            print(f"HDBSCAN failed: {e}")

    return pd.DataFrame(results)


# =============================================================================
# Community Profiles
# =============================================================================

def compute_community_profiles(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compute z-score profiles for each community."""
    global_mean = features[feature_cols].mean()
    global_std = features[feature_cols].std(ddof=0).replace(0, np.nan)

    rows = []
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        subset = features.loc[mask, feature_cols]

        entry = {"community_id": int(cid), "count": int(mask.sum())}
        for col in feature_cols:
            mean_val = subset[col].mean()
            entry[f"mean_{col}"] = float(mean_val)
            entry[f"z_{col}"] = float((mean_val - global_mean[col]) / global_std[col]) if pd.notna(global_std[col]) else 0.0

        rows.append(entry)

    return pd.DataFrame(rows)


def compute_community_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[int]]:
    """Compute centroid for each community."""
    community_ids = sorted(np.unique(labels))
    centroids = []
    for cid in community_ids:
        mask = labels == cid
        centroids.append(embeddings[mask].mean(axis=0))
    return np.vstack(centroids), community_ids


# =============================================================================
# Visualization
# =============================================================================

def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    cmap_name: str = "tab20",
) -> None:
    """UMAP scatter plot."""
    plt.figure(figsize=(10, 8))
    uniq = sorted(np.unique(labels))
    cmap = plt.get_cmap(cmap_name, max(len(uniq), 2))

    for i, cid in enumerate(uniq):
        mask = labels == cid
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(i)],
            label=f"{cid}",
            s=20,
            alpha=0.7,
        )

    plt.legend(title="Community", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_community_sizes(sizes: pd.DataFrame, out_path: Path) -> None:
    """Bar plot of community sizes."""
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(sizes)))
    bars = plt.bar(sizes["community_id"].astype(str), sizes["count"], color=colors)

    plt.title("Community Sizes", fontsize=14)
    plt.xlabel("Community ID", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # Add count labels on bars
    for bar, count in zip(bars, sizes["count"]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_surface_heatmap(surface_props: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of community x surface proportions."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(surface_props, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Community × Surface Proportions", fontsize=14)
    plt.xlabel("Surface (0=Hard, 1=Grass)", fontsize=12)
    plt.ylabel("Community", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_resolution_sensitivity(sensitivity_df: pd.DataFrame, out_path: Path) -> None:
    """Plot showing how metrics change with resolution parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Modularity vs resolution
    ax = axes[0]
    for k in sensitivity_df["k"].unique():
        subset = sensitivity_df[sensitivity_df["k"] == k]
        ax.plot(subset["resolution"], subset["modularity"], marker="o", label=f"k={k}")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Modularity")
    ax.set_title("Modularity vs Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ARI vs base
    ax = axes[1]
    for k in sensitivity_df["k"].unique():
        subset = sensitivity_df[sensitivity_df["k"] == k]
        ax.plot(subset["resolution"], subset["ari_vs_base"], marker="o", label=f"k={k}")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("ARI vs Base")
    ax.set_title("Agreement with Base Partition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_outcome_heatmap(
    profiles: pd.DataFrame,
    outcome_cols: List[str],
    out_path: Path,
) -> None:
    """Heatmap of z-scores for each community."""
    z_cols = [f"z_{col}" for col in outcome_cols if f"z_{col}" in profiles.columns]

    if not z_cols:
        print("WARNING: No z-score columns found for outcome heatmap.")
        return

    z_data = profiles[["community_id"] + z_cols].set_index("community_id")
    z_data.columns = [c.replace("z_", "") for c in z_data.columns]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        z_data,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1.5,
        vmax=1.5,
        linewidths=0.5,
    )
    plt.title("Community Outcome Profiles (Z-Scores)", fontsize=14)
    plt.xlabel("Outcome Variable", fontsize=12)
    plt.ylabel("Community", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_radar_profiles(
    profiles: pd.DataFrame,
    feature_cols: List[str],
    out_path: Path,
    max_communities: int = 6,
) -> None:
    """Radar plot of community profiles."""
    z_cols = [f"z_{col}" for col in feature_cols if f"z_{col}" in profiles.columns]

    if len(z_cols) < 3:
        print("WARNING: Not enough features for radar plot.")
        return

    # Select subset of communities if too many
    communities = profiles["community_id"].unique()[:max_communities]
    n_features = len(z_cols)

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))

    for i, cid in enumerate(communities):
        row = profiles[profiles["community_id"] == cid].iloc[0]
        values = [row[col] for col in z_cols]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=f"Community {cid}", color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("z_", "") for c in z_cols], fontsize=9)
    ax.set_ylim(-2, 2)
    ax.set_title("Community Profiles (Z-Scores)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stability_boxplot(stability: Dict[str, Any], out_path: Path) -> None:
    """Boxplot of stability scores."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ARI
    ax = axes[0]
    ax.boxplot([stability["ari_scores"]], labels=["ARI"])
    ax.axhline(y=stability["ari_mean"], color="r", linestyle="--", label=f"Mean={stability['ari_mean']:.3f}")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Stability: ARI Across Seeds")
    ax.legend()
    ax.set_ylim(0, 1)

    # NMI
    ax = axes[1]
    ax.boxplot([stability["nmi_scores"]], labels=["NMI"])
    ax.axhline(y=stability["nmi_mean"], color="r", linestyle="--", label=f"Mean={stability['nmi_mean']:.3f}")
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("Stability: NMI Across Seeds")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_baseline_comparison(comparison_df: pd.DataFrame, out_path: Path) -> None:
    """Bar plot comparing methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    methods = comparison_df["method"].values
    x = np.arange(len(methods))
    width = 0.6

    # Modularity
    ax = axes[0]
    ax.bar(x, comparison_df["modularity"], width, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Modularity")
    ax.set_title("Modularity (higher = better)")
    ax.axhline(y=comparison_df[comparison_df["method"] == "Leiden"]["modularity"].values[0],
               color="r", linestyle="--", alpha=0.5)

    # Silhouette
    ax = axes[1]
    ax.bar(x, comparison_df["silhouette"], width, color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette (higher = better)")

    # Davies-Bouldin
    ax = axes[2]
    ax.bar(x, comparison_df["davies_bouldin"], width, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_title("Davies-Bouldin (lower = better)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve style discovery: k-NN graph, Leiden and Ward profile merging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--work-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--ckpt-path", type=Path,
        default=Path("outputs/serve/serve_encoder_best.pt"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/serve/processed"),
                        help="Directory with train/val data and outcome sidecars.")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--val-path", type=Path, default=None)
    parser.add_argument("--train-outcomes-path", type=Path, default=None)
    parser.add_argument("--val-outcomes-path", type=Path, default=None)
    parser.add_argument("--space", type=str, default="encoder", choices=["encoder", "projection", "predictor"])
    parser.add_argument("--use-exported-embeddings", action="store_true")
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-runs", type=int, default=10)
    parser.add_argument("--bootstrap-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip parameter sensitivity analysis.")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline method comparison.")
    parser.add_argument("--no-profile-merging", dest="use_profile_merging", action="store_false",
                        help="Disable the Ward merge of fine-grained communities (enabled in the paper).")
    parser.add_argument("--knn-k", type=int, default=15, help="Neighbours per node in the k-NN graph.")
    parser.add_argument("--target-communities", type=int, default=3,
                        help="Target number of communities after profile-based merging.")
    parser.add_argument("--merge-linkage", type=str, default="ward",
                        choices=["ward", "average", "complete"],
                        help="Linkage method for hierarchical profile merging.")
    parser.add_argument("--fine-resolution", type=float, default=1.0,
                        help="Resolution for the fine-grained Leiden partition before merging.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist for the default plot.")
    parser.add_argument("--umap-n-neighbors", type=int, default=None,
                        help="UMAP n_neighbors for the default plot (defaults to fixed kNN value).")
    parser.add_argument("--umap-sweep", action="store_true",
                        help="Generate additional UMAP plots over a parameter grid.")
    parser.add_argument("--umap-min-dist-grid", type=str, default=None,
                        help="Comma-separated min_dist values for sweep (e.g., 0.01,0.05,0.1).")
    parser.add_argument("--umap-n-neighbors-grid", type=str, default=None,
                        help="Comma-separated n_neighbors values for sweep (e.g., 5,15,30).")
    args = parser.parse_args()
    args.train_path = args.train_path or args.processed_dir / "train_data.csv"
    args.val_path = args.val_path or args.processed_dir / "val_data.csv"
    args.train_outcomes_path = args.train_outcomes_path or args.processed_dir / "train_outcomes.csv"
    args.val_outcomes_path = args.val_outcomes_path or args.processed_dir / "val_outcomes.csv"
    for path in (args.ckpt_path, args.train_path, args.val_path):
        if not path.exists():
            parser.exit(1, f"Missing input file: {path}\n")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    KNN_K = args.knn_k
    SENSITIVITY_K_VALUES = [10, 15, 20, 25]

    # Create output directories
    run_dir = args.work_dir / f"community_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    emb_dir = run_dir / "embeddings"
    comm_dir = run_dir / "communities"
    val_dir = run_dir / "validation"
    fig_dir = run_dir / "figures"

    for d in [emb_dir, comm_dir, val_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")
    print("=" * 70)

    # Load checkpoint and data
    print("Loading checkpoint and data...")
    ckpt = safe_torch_load(args.ckpt_path, device)
    cont_cols = ckpt["cont_cols"]
    cat_names = ckpt["cat_names"]

    df_train = pd.read_csv(args.train_path)
    df_val = pd.read_csv(args.val_path)

    surface_cols = infer_surface_cols(df_train)
    assert_surface_binary(df_train, surface_cols)
    assert_surface_binary(df_val, surface_cols)

    x_cont_train, x_cat_train, surface_train = to_tensors(df_train, cont_cols, cat_names, surface_cols)
    x_cont_val, x_cat_val, surface_val = to_tensors(df_val, cont_cols, cat_names, surface_cols)

    # Build models and extract embeddings
    print("Extracting embeddings...")
    encoder, head, predictor = build_models(ckpt, device)

    embeddings_loaded = False
    if args.use_exported_embeddings:
        for base in [args.work_dir, args.ckpt_path.parent]:
            train_path = base / "train_embeddings.npy"
            val_path = base / "val_embeddings.npy"
            if train_path.exists() and val_path.exists():
                train_emb = np.load(train_path)
                val_emb = np.load(val_path)
                embeddings_loaded = True
                print(f"  Loaded embeddings from {base}")
                break

    if not embeddings_loaded:
        train_emb = extract_embeddings(
            encoder, head, predictor, x_cont_train, x_cat_train,
            cont_cols, cat_names, args.batch_size, device, args.space
        )
        val_emb = extract_embeddings(
            encoder, head, predictor, x_cont_val, x_cat_val,
            cont_cols, cat_names, args.batch_size, device, args.space
        )

    all_emb = np.vstack([train_emb, val_emb])
    all_surface = np.concatenate([surface_train, surface_val])
    n_train = len(train_emb)

    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Val embeddings: {val_emb.shape}")
    print(f"  Total: {all_emb.shape}")

    # Save embeddings
    np.save(emb_dir / "train_embeddings.npy", train_emb)
    np.save(emb_dir / "val_embeddings.npy", val_emb)
    np.save(emb_dir / "all_embeddings.npy", all_emb)
    np.save(emb_dir / "train_surface_labels.npy", surface_train)
    np.save(emb_dir / "val_surface_labels.npy", surface_val)

    metadata = pd.DataFrame({
        "index": list(range(len(all_emb))),
        "split": ["train"] * n_train + ["val"] * (len(all_emb) - n_train),
        "surface_label": all_surface,
    })
    metadata.to_csv(emb_dir / "metadata.csv", index=False)

    # Build graph and run community detection
    print("\nBuilding KNN graph and detecting communities...")
    G, sigma = build_knn_graph(all_emb, KNN_K)
    print(f"  KNN k={KNN_K}, sigma={sigma:.4f}")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Use fine-grained resolution 1.0 when profile merging is enabled (Leiden only)
    if args.use_profile_merging:
        if args.fine_resolution != 1.0:
            print(f"  Fine-grained partition uses resolution 1.0 (requested {args.fine_resolution})")
        initial_resolution = 1.0
    else:
        initial_resolution = args.leiden_resolution

    def community_fn(seed: int) -> np.ndarray:
        return run_leiden(G, len(all_emb), initial_resolution, seed)

    community_labels = community_fn(args.seed)
    n_communities = len(np.unique(community_labels))
    print(f"  Detected {n_communities} communities (resolution={initial_resolution})")

    # Profile-based hierarchical merging
    merge_map = None
    profile_distances = None
    original_labels = None

    selected_target = None
    if args.use_profile_merging and n_communities > args.target_communities:
        print(f"\nApplying profile-based hierarchical merging...")
        print(f"  Target: {args.target_communities} communities")
        print(f"  Linkage method: {args.merge_linkage}")

        # Need features for profile computation
        all_features = pd.concat([df_train[cont_cols], df_val[cont_cols]], ignore_index=True)
        physics_cols = [c for c in cont_cols if not c.startswith("surface_") and not c.startswith("hand_")]

        # Save original (fine-grained) labels
        original_labels = community_labels.copy()

        # Run sweep to show merge options
        print("  Sweeping merge targets for analysis...")
        max_target = min(n_communities - 1, args.target_communities)
        min_target = max(2, min(max_target, 3))
        merge_sweep = find_optimal_merge_target(
            all_features, community_labels, physics_cols,
            min_communities=min_target,
            max_communities=max_target,
            linkage_method=args.merge_linkage,
        )
        merge_sweep.to_csv(val_dir / "merge_target_sweep.csv", index=False)

        # Select target with maximum distinctiveness (within requested target)
        selected_target = select_merge_target(merge_sweep, args.target_communities)
        print(f"  Selected target_n={selected_target} (requested={args.target_communities})")

        # Perform actual merge
        community_labels, merge_map, profile_distances = hierarchical_merge_communities(
            all_features, community_labels, physics_cols,
            target_n=selected_target,
            linkage_method=args.merge_linkage,
        )

        n_communities_merged = len(np.unique(community_labels))
        print(f"  Merged to {n_communities_merged} communities")

        # Compute distinctiveness metrics
        distinctiveness = compute_profile_distinctiveness(all_features, community_labels, physics_cols)
        print(f"  Mean profile distance: {distinctiveness['mean_pairwise_distance']:.3f}")
        print(f"  Min profile distance: {distinctiveness['min_pairwise_distance']:.3f}")

        # Save merge analysis
        with open(comm_dir / "merge_map.json", "w") as f:
            json.dump({str(k): v for k, v in merge_map.items()}, f, indent=2)

        np.save(comm_dir / "original_community_labels.npy", original_labels)
        np.save(comm_dir / "profile_distances.npy", profile_distances)

        # Plot merge analysis
        plot_merge_analysis(merge_sweep, profile_distances, merge_map, fig_dir)

        n_communities = n_communities_merged

    train_labels = community_labels[:n_train]
    val_labels = community_labels[n_train:]

    # Save community labels
    pd.DataFrame({
        "index": df_train.index.to_numpy(),
        "community_id": train_labels,
        "surface_label": surface_train,
    }).to_csv(comm_dir / "community_labels_train.csv", index=False)

    pd.DataFrame({
        "index": df_val.index.to_numpy(),
        "community_id": val_labels,
        "surface_label": surface_val,
    }).to_csv(comm_dir / "community_labels_val.csv", index=False)

    # Community centroids
    centroids, community_ids = compute_community_centroids(all_emb, community_labels)
    np.save(comm_dir / "community_centroids.npy", centroids)

    # Community sizes
    sizes = pd.Series(community_labels).value_counts().sort_index().reset_index()
    sizes.columns = ["community_id", "count"]
    sizes.to_csv(comm_dir / "community_sizes.csv", index=False)

    # Surface proportions
    contingency = pd.crosstab(community_labels, all_surface)
    contingency.to_csv(val_dir / "community_surface_contingency.csv")
    surface_props = contingency.div(contingency.sum(axis=1), axis=0)
    surface_props.to_csv(val_dir / "community_surface_proportions.csv")

    # ==========================================================================
    # Validation Metrics
    # ==========================================================================
    print("\nComputing validation metrics...")

    # Core metrics
    metrics = compute_community_metrics(all_emb, community_labels, G)
    metrics["surface_nmi"] = float(normalized_mutual_info_score(community_labels, all_surface))
    metrics["knn_k"] = KNN_K
    metrics["resolution"] = args.leiden_resolution
    metrics["sigma"] = sigma

    # Stability across seeds
    print("  Computing seed stability...")
    stability_seeds = [args.seed + i + 1 for i in range(args.stability_runs)]
    stability = compute_stability_metrics(all_emb, community_labels, community_fn, stability_seeds)
    metrics["stability"] = {
        "ari_mean": stability["ari_mean"],
        "ari_std": stability["ari_std"],
        "nmi_mean": stability["nmi_mean"],
        "nmi_std": stability["nmi_std"],
    }

    # Parameter sensitivity
    if not args.skip_sensitivity:
        print("  Computing parameter sensitivity...")
        n_samples = len(all_emb)
        k_values = [
            k for k in SENSITIVITY_K_VALUES
            if 1 < k < n_samples
        ]
        if KNN_K < n_samples:
            k_values.append(KNN_K)
        k_values = sorted(set(k_values))
        if not k_values:
            k_values = [max(2, min(KNN_K, n_samples - 1))]
        resolution_values = [0.5, 0.75, 1.0, 1.25, 1.5]
        sensitivity_df = compute_parameter_sensitivity(
            all_emb, community_labels, k_values, resolution_values, args.seed
        )
        sensitivity_df.to_csv(val_dir / "parameter_sensitivity.csv", index=False)
        plot_resolution_sensitivity(sensitivity_df, fig_dir / "resolution_sensitivity.png")

    # Baseline comparison
    if not args.skip_baselines:
        print("  Computing baseline comparison...")
        baseline_df = compute_baseline_comparison(all_emb, community_labels, G, args.seed)
        baseline_df.to_csv(val_dir / "baseline_comparison.csv", index=False)
        plot_baseline_comparison(baseline_df, fig_dir / "baseline_comparison.png")

    # Statistical tests on outcomes
    print("  Computing statistical tests...")
    outcomes_train = pd.read_csv(args.train_outcomes_path)
    outcomes_val = pd.read_csv(args.val_outcomes_path)
    all_outcomes = pd.concat([outcomes_train, outcomes_val], ignore_index=True)
    outcome_cols = all_outcomes.select_dtypes(include=[np.number]).columns.tolist()

    stat_tests = compute_statistical_tests(all_outcomes, community_labels, outcome_cols)
    stat_tests.to_csv(val_dir / "statistical_tests.csv", index=False)

    n_significant = (stat_tests["significant_005"]).sum()
    metrics["outcome_tests"] = {
        "n_outcomes_tested": len(stat_tests),
        "n_significant_005": int(n_significant),
        "pct_significant": float(n_significant / len(stat_tests)) if len(stat_tests) > 0 else 0,
    }

    # Community profiles
    print("  Computing community profiles...")
    all_features = pd.concat([df_train[cont_cols], df_val[cont_cols]], ignore_index=True)
    profiles = compute_community_profiles(all_features, community_labels, cont_cols)
    profiles.to_csv(comm_dir / "community_profiles.csv", index=False)

    # Outcome profiles
    outcome_profiles = compute_community_profiles(all_outcomes, community_labels, outcome_cols)
    outcome_profiles.to_csv(comm_dir / "community_outcome_profiles.csv", index=False)

    # Save metrics
    with open(val_dir / "community_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # ==========================================================================
    # Figures
    # ==========================================================================
    print("\nGenerating figures...")

    # Community sizes
    plot_community_sizes(sizes, fig_dir / "community_sizes.png")

    # Surface heatmap
    plot_surface_heatmap(surface_props, fig_dir / "community_surface_heatmap.png")

    # Outcome heatmap
    plot_outcome_heatmap(outcome_profiles, outcome_cols[:10], fig_dir / "community_outcome_heatmap.png")

    # Radar profiles
    physics_cols = [c for c in cont_cols if not c.startswith("surface_") and not c.startswith("hand_")]
    plot_radar_profiles(profiles, physics_cols[:8], fig_dir / "community_radar_profiles.png")

    # Stability boxplot
    plot_stability_boxplot(stability, fig_dir / "stability_boxplot.png")

    # UMAP
    if HAS_UMAP:
        print("  Computing UMAP...")
        umap_n_neighbors = args.umap_n_neighbors if args.umap_n_neighbors is not None else KNN_K
        umap_min_dist = args.umap_min_dist
        umap_model = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=args.seed,
        )
        coords = umap_model.fit_transform(all_emb)
        np.save(emb_dir / "umap_coords.npy", coords)

        plot_umap(coords, community_labels, "UMAP: Serve Style Communities", fig_dir / "umap_communities.png")
        plot_umap(coords, all_surface, "UMAP: Surface Type", fig_dir / "umap_surface.png", cmap_name="coolwarm")

        if args.umap_sweep:
            nn_list = parse_int_list(args.umap_n_neighbors_grid) or [5, 10, 15, 30]
            md_list = parse_float_list(args.umap_min_dist_grid) or [0.01, 0.05, 0.1]
            print(f"  UMAP sweep: n_neighbors={nn_list} | min_dist={md_list}")
            for nn in nn_list:
                for md in md_list:
                    if nn == umap_n_neighbors and abs(md - umap_min_dist) < 1e-9:
                        continue
                    umap_model = umap.UMAP(
                        n_neighbors=nn,
                        min_dist=md,
                        metric="cosine",
                        random_state=args.seed,
                    )
                    coords = umap_model.fit_transform(all_emb)
                    md_str = f"{md:.3f}".replace(".", "p")
                    coords_path = emb_dir / f"umap_coords_nn{nn}_md{md_str}.npy"
                    np.save(coords_path, coords)

                    plot_umap(
                        coords,
                        community_labels,
                        f"UMAP: Communities (nn={nn}, md={md})",
                        fig_dir / f"umap_communities_nn{nn}_md{md_str}.png",
                    )
                    plot_umap(
                        coords,
                        all_surface,
                        f"UMAP: Surface (nn={nn}, md={md})",
                        fig_dir / f"umap_surface_nn{nn}_md{md_str}.png",
                        cmap_name="coolwarm",
                    )
    else:
        print("  WARNING: UMAP not available, skipping UMAP plots.")

    # ==========================================================================
    # Save config
    # ==========================================================================
    config = {
        "run_dir": str(run_dir),
        "ckpt_path": str(args.ckpt_path),
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "space": args.space,
        "knn_k": KNN_K,
        "leiden_resolution": args.leiden_resolution,
        "umap": {
            "min_dist": args.umap_min_dist,
            "n_neighbors": args.umap_n_neighbors if args.umap_n_neighbors is not None else KNN_K,
            "sweep": args.umap_sweep,
            "min_dist_grid": args.umap_min_dist_grid,
            "n_neighbors_grid": args.umap_n_neighbors_grid,
        },
        "seed": args.seed,
        "stability_runs": args.stability_runs,
        "timestamp": datetime.now().isoformat(),
        "profile_merging": {
            "enabled": args.use_profile_merging,
            "target_communities": args.target_communities if args.use_profile_merging else None,
            "fine_resolution": args.fine_resolution if args.use_profile_merging else None,
            "linkage_method": args.merge_linkage if args.use_profile_merging else None,
            "selected_target": selected_target if args.use_profile_merging and merge_map else None,
            "merge_map": {str(k): v for k, v in merge_map.items()} if merge_map else None,
        },
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMMUNITY DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  Communities detected: {n_communities}")
    if args.use_profile_merging and merge_map:
        n_original = len(set(oid for ids in merge_map.values() for oid in ids))
        print(f"  (Merged from {n_original} fine-grained communities)")
    print(f"  Modularity: {metrics.get('modularity', 'N/A'):.4f}")
    print(f"  Silhouette: {metrics.get('silhouette_cosine', 'N/A'):.4f}")
    print(f"  Surface NMI: {metrics['surface_nmi']:.4f} (low = surface-independent)")
    print(f"  Stability ARI: {stability['ari_mean']:.3f} ± {stability['ari_std']:.3f}")
    print(f"  Stability NMI: {stability['nmi_mean']:.3f} ± {stability['nmi_std']:.3f}")
    print(f"  Significant outcome tests: {n_significant}/{len(stat_tests)} (p<0.05)")
    print(f"\nOutputs written to: {run_dir}")


if __name__ == "__main__":
    main()

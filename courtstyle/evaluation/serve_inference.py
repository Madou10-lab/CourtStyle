"""Assignment of new serve records to discovered styles.

A new server-match record is preprocessed with the scaler fitted on the training
split, embedded by the frozen serve encoder, and assigned by weighted k-nearest-
neighbour voting over the reference embeddings of the discovery run:

1. find the k nearest reference embeddings under cosine distance;
2. weight each neighbour by ``w_i = exp(-d_i^2 / sigma)``;
3. score every style by the summed weights of its neighbours;
4. predict the highest-scoring style, with confidence ``score / sum(scores)``.

``k`` is taken from the discovery run and ``sigma`` is the median neighbour
distance of the reference set. Style discovery is not re-run for new records.
"""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from courtstyle.data.serve_dataset import (
    DROP_AFTER_REPARAM,
    DROP_COLS,
    OUTCOME_COLS,
    _align_schema,
    _clean_one_hot,
    _coerce_numeric,
    _enforce_one_hot,
    _reparameterize_speed_features,
)
from courtstyle.models.serve_encoder import ModernTabularEncoder


DEFAULT_PREPROCESSOR_PATH = Path("data/serve/processed/preprocessor.pkl")

DEFAULT_UNCERTAINTY_CONF_THRESHOLD = 0.45  # max confidence below this = uncertain


DEFAULT_UNCERTAINTY_DIST_THRESHOLD = 0.5   # min distance above this = uncertain


def load_preprocessor(path: Path) -> Dict[str, Any]:
    """
    Load preprocessor artifacts from joblib pickle.

    Returns dict with:
        - numeric_pipeline: fitted sklearn Pipeline (Imputer + Scaler)
        - cont_cols: list of continuous column names
        - cat_cols: list of categorical column names
        - schema_cols: final feature schema (column order)
        - outcome_cols, drop_cols, drop_after_reparam, derived_features
    """
    try:
        import joblib
        artifacts = joblib.load(path)
        return artifacts
    except Exception as e:
        error_msg = str(e)
        if "numpy" in error_msg.lower() or "sklearn" in error_msg.lower():
            print(
                f"ERROR: Cannot load preprocessor.pkl due to environment mismatch.\n"
                f"  {error_msg}\n\n"
                f"Your current numpy/sklearn environment cannot load preprocessor.pkl.\n"
                f"Regenerate it in the environment used for training:\n"
                f"  python -m courtstyle.data.serve_dataset --real-input data/serve/fingerprint_serve.csv\n"
            )
            sys.exit(1)
        raise


def preprocess_input(
    df: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> pd.DataFrame:
    """
    Preprocess raw input data identically to training.

    Steps:
    1. Clean one-hot groups (surface_, hand_)
    2. Reparameterize speed features (headroom + deltas)
    3. Drop columns: outcome_cols + drop_cols + drop_after_reparam
    4. Enforce one-hot validity
    5. Align to schema (fill missing: 0 for categorical, NaN for continuous)
    6. Coerce continuous to numeric
    7. Apply numeric_pipeline.transform to cont_cols
    8. Cast continuous to float32

    Args:
        df: Raw input DataFrame (like fingerprint_serve.csv format)
        preprocessor: Dict from load_preprocessor()

    Returns:
        Preprocessed DataFrame aligned to training schema
    """
    numeric_pipeline = preprocessor["numeric_pipeline"]
    cont_cols = preprocessor["cont_cols"]
    cat_cols = preprocessor["cat_cols"]
    schema_cols = preprocessor["schema_cols"]
    outcome_cols = preprocessor.get("outcome_cols", OUTCOME_COLS)
    drop_cols = preprocessor.get("drop_cols", DROP_COLS)
    drop_after_reparam = preprocessor.get("drop_after_reparam", DROP_AFTER_REPARAM)

    # Step 1: Clean one-hot groups
    df = _clean_one_hot(df, ["surface_", "hand_"])

    # Step 2: Feature reparameterization (headroom + directional deltas)
    df = _reparameterize_speed_features(df)

    # Step 3: Drop outcome and unnecessary columns
    all_drop = outcome_cols + drop_cols + drop_after_reparam
    df = df.drop(columns=[c for c in all_drop if c in df.columns], errors="ignore")

    # Step 4: Enforce one-hot validity
    df = _enforce_one_hot(df, ["surface_", "hand_"])

    # Step 5: Align to training schema
    df = _align_schema(
        df,
        columns=schema_cols,
        fill_missing=True,
        continuous_cols=cont_cols,
    )

    # Step 6: Coerce continuous to numeric
    df = _coerce_numeric(df, cont_cols)

    # Step 7: Apply fitted numeric pipeline (imputer + scaler)
    df.loc[:, cont_cols] = numeric_pipeline.transform(df[cont_cols])

    # Step 8: Cast to float32
    df[cont_cols] = df[cont_cols].astype("float32")

    return df


def _load_sanitized_checkpoint(path: Path, device: torch.device) -> Dict:
    """
    Load checkpoint from a potentially dirty zip file.

    Some checkpoints may contain extra entries (like PNG files) at the root level
    that can break torch.load. This function creates a sanitized temporary zip.
    """
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a zip file")

    stem = path.stem
    expected_prefix = f"{stem}/"

    with zipfile.ZipFile(path, 'r') as zf:
        names = zf.namelist()

        # Check if there are root-level entries that shouldn't be there
        valid_entries = [n for n in names if n.startswith(expected_prefix) or n.startswith("archive/")]
        invalid_entries = [n for n in names if n not in valid_entries and not n.endswith('/')]

        if not invalid_entries:
            # No cleanup needed, re-raise to try next method
            raise ValueError("No invalid entries found, not a dirty zip issue")

        print(f"  Sanitizing checkpoint: removing {len(invalid_entries)} invalid entries")

        # Create sanitized temporary zip
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(tmp_path, 'w') as zf_clean:
                for name in valid_entries:
                    zf_clean.writestr(name, zf.read(name))

            # Load from sanitized zip
            return torch.load(tmp_path, map_location=device, weights_only=False)
        finally:
            tmp_path.unlink(missing_ok=True)


def safe_torch_load(path: Path, device: torch.device) -> Dict:
    """
    Safely load a PyTorch checkpoint with multiple fallbacks.

    Handles:
    - Standard torch.load with weights_only=True
    - numpy._core compatibility issues
    - Dirty zip files with extra entries (like PNG files)
    """
    path = Path(path)

    # First attempt: standard weights_only load
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e1:
        pass

    # Second attempt: add safe globals for numpy scalar type
    try:
        from torch.serialization import add_safe_globals
        scalar_type = getattr(np, "_core", np.core).multiarray.scalar
        add_safe_globals([scalar_type])
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e2:
        pass

    # Third attempt: alias numpy._core to numpy.core
    try:
        import numpy.core as np_core
        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e3:
        pass

    # Fourth attempt: sanitize dirty zip (remove root-level entries like PNGs)
    try:
        return _load_sanitized_checkpoint(path, device)
    except Exception as e4:
        pass

    # Final fallback: weights_only=False
    print(f"WARNING: All safe load methods failed for {path}, using weights_only=False")
    try:
        import numpy.core as np_core
        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
    except Exception:
        pass
    return torch.load(path, map_location=device, weights_only=False)


def build_encoder(ckpt: Dict, device: torch.device) -> ModernTabularEncoder:
    """Build encoder from checkpoint."""
    cont_cols = ckpt["cont_cols"]
    cat_names = ckpt["cat_names"]
    cardinalities = ckpt["cardinalities"]
    config = ckpt.get("config", {})

    if isinstance(cardinalities, dict):
        categorical_cardinalities = cardinalities
    else:
        categorical_cardinalities = dict(zip(cat_names, cardinalities))

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
    encoder.eval()

    return encoder


def df_to_tensors(
    df: pd.DataFrame,
    cont_cols: List[str],
    cat_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert preprocessed DataFrame to model input tensors.

    Args:
        df: Preprocessed DataFrame
        cont_cols: Continuous column names (from checkpoint)
        cat_names: Categorical prefixes (e.g., ['hand'])

    Returns:
        x_cont: (N, len(cont_cols)) float tensor
        x_cat: (N, len(cat_names)) long tensor of indices
    """
    # Continuous features
    x_cont = torch.tensor(df[cont_cols].to_numpy(dtype=np.float32))

    # Categorical indices (argmax of one-hot columns)
    cat_indices = []
    for prefix in cat_names:
        cols = [c for c in df.columns if c.startswith(prefix + "_")]
        if not cols:
            raise ValueError(f"No columns found for categorical prefix '{prefix}_'")
        arr = df[cols].to_numpy(dtype=np.float32)
        cat_indices.append(arr.argmax(axis=1))

    if cat_indices:
        x_cat = torch.tensor(np.stack(cat_indices, axis=1), dtype=torch.long)
    else:
        x_cat = torch.zeros((len(df), 0), dtype=torch.long)

    return x_cont, x_cat


@torch.no_grad()
def extract_embeddings(
    encoder: ModernTabularEncoder,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    cont_cols: List[str],
    cat_names: List[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract encoder CLS embeddings."""
    encoder.eval()

    outs = []
    n_samples = x_cont.shape[0]

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bc = x_cont[start:end].to(device)
        bcat = x_cat[start:end].to(device)

        # Build input dicts
        x_cont_dict = {name: bc[:, j] for j, name in enumerate(cont_cols)}
        x_cat_dict = {name: bcat[:, j] for j, name in enumerate(cat_names)}

        # Encoder forward -> CLS output (d_model=128)
        z = encoder(x_cont_dict, x_cat_dict)
        outs.append(z.cpu().numpy())

    return np.concatenate(outs, axis=0)


def load_reference_data(community_run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load reference embeddings and labels from community run.

    Returns:
        embeddings: (N, 128) all embeddings
        labels: (N,) community labels (train first, val second)
        centroids: (K, 128) community centroids
    """
    emb_dir = community_run_dir / "embeddings"
    comm_dir = community_run_dir / "communities"

    # Load embeddings
    embeddings = np.load(emb_dir / "all_embeddings.npy")

    # Load labels (train first, val second - matching all_embeddings order)
    train_labels_df = pd.read_csv(comm_dir / "community_labels_train.csv")
    val_labels_df = pd.read_csv(comm_dir / "community_labels_val.csv")

    # Sort by index to ensure correct order
    train_labels_df = train_labels_df.sort_values("index").reset_index(drop=True)
    val_labels_df = val_labels_df.sort_values("index").reset_index(drop=True)

    labels = np.concatenate([
        train_labels_df["community_id"].values,
        val_labels_df["community_id"].values,
    ])

    # Load centroids
    centroids = np.load(comm_dir / "community_centroids.npy")

    return embeddings, labels, centroids


def compute_sigma(embeddings: np.ndarray, k: int) -> float:
    """
    Compute sigma for Gaussian kernel weights exactly like training.

    sigma = median of edge distances in kNN graph
    """
    knn = kneighbors_graph(
        embeddings,
        n_neighbors=k,
        metric="cosine",
        mode="distance",
        include_self=False,
    )
    # Symmetrize
    dist_matrix = knn.maximum(knn.T)
    distances = dist_matrix.data

    if distances.size > 0:
        return float(np.median(distances))
    return 1.0


def assign_knn_vote(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    sigma: float,
    exclude_self_indices: Optional[np.ndarray] = None,
    uncertainty_conf_threshold: float = DEFAULT_UNCERTAINTY_CONF_THRESHOLD,
    uncertainty_dist_threshold: float = DEFAULT_UNCERTAINTY_DIST_THRESHOLD,
) -> Dict[str, Any]:
    """
    Assign queries to communities via weighted kNN vote.

    For each query:
    1. Find k nearest neighbors in ref_embeddings (cosine distance)
    2. For each neighbor with distance d_i, compute weight w_i = exp(-(d_i^2)/sigma)
    3. Score each community: score[c] = sum(w_i for neighbors where label[i]==c)
    4. Predict c_hat = argmax(score)
    5. Confidence p[c] = score[c] / sum(scores)

    Args:
        query_embeddings: (Q, D) query embeddings
        ref_embeddings: (N, D) reference embeddings
        ref_labels: (N,) reference labels
        k: number of neighbors
        sigma: Gaussian kernel bandwidth
        exclude_self_indices: (Q,) if provided, exclude these indices from neighbors
                              (for self-check mode where query is in ref)
        uncertainty_conf_threshold: max confidence below this = uncertain
        uncertainty_dist_threshold: min distance above this = uncertain

    Returns:
        Dict with predictions, scores, confidences, neighbors info
    """
    # Build nearest neighbor index
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")  # +1 in case of self-exclusion
    nn.fit(ref_embeddings)

    # Find neighbors
    distances, indices = nn.kneighbors(query_embeddings)

    community_ids = sorted(np.unique(ref_labels))
    n_communities = len(community_ids)
    cid_to_idx = {c: i for i, c in enumerate(community_ids)}

    results = {
        "predictions": [],
        "scores": [],
        "confidences": [],
        "top2_margins": [],
        "neighbor_indices": [],
        "neighbor_distances": [],
        "uncertain": [],
    }

    for q_idx in range(len(query_embeddings)):
        q_distances = distances[q_idx]
        q_indices = indices[q_idx]

        # Exclude self if needed
        if exclude_self_indices is not None:
            self_idx = exclude_self_indices[q_idx]
            mask = q_indices != self_idx
            q_distances = q_distances[mask][:k]
            q_indices = q_indices[mask][:k]
        else:
            q_distances = q_distances[:k]
            q_indices = q_indices[:k]

        # Compute weights using Gaussian kernel
        weights = np.exp(-(q_distances ** 2) / sigma) if sigma > 0 else np.ones_like(q_distances)

        # Score each community
        scores = np.zeros(n_communities)
        for d, idx, w in zip(q_distances, q_indices, weights):
            lbl = ref_labels[idx]
            scores[cid_to_idx[lbl]] += w

        # Normalize to get confidence
        total_score = scores.sum()
        confidences = scores / total_score if total_score > 0 else np.zeros(n_communities)

        # Prediction
        pred_idx = np.argmax(scores)
        pred = community_ids[pred_idx]

        # Top-2 margin (difference between top-1 and top-2 confidence)
        sorted_conf = np.sort(confidences)[::-1]
        margin = sorted_conf[0] - sorted_conf[1] if len(sorted_conf) > 1 else sorted_conf[0]

        # OOD check: uncertain if max confidence below threshold or nearest distance too large
        max_conf = confidences.max()
        min_dist = q_distances[0] if len(q_distances) > 0 else float('inf')
        uncertain = max_conf < uncertainty_conf_threshold or min_dist > uncertainty_dist_threshold

        results["predictions"].append(int(pred))
        results["scores"].append({int(community_ids[i]): float(s) for i, s in enumerate(scores)})
        results["confidences"].append({int(community_ids[i]): float(c) for i, c in enumerate(confidences)})
        results["top2_margins"].append(float(margin))
        results["neighbor_indices"].append([int(x) for x in q_indices])
        results["neighbor_distances"].append([float(x) for x in q_distances])
        results["uncertain"].append(bool(uncertain))

    return results


def assign_nearest_centroid(
    query_embeddings: np.ndarray,
    centroids: np.ndarray,
    community_ids: List[int],
) -> List[int]:
    """Assign queries to nearest centroid (cosine distance)."""
    from sklearn.metrics.pairwise import cosine_distances

    dists = cosine_distances(query_embeddings, centroids)
    nearest = dists.argmin(axis=1)
    return [community_ids[i] for i in nearest]


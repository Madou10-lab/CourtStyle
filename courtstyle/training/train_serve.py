"""Two-stage contrastive training of the serve encoder.

Stage 1 pre-trains on synthetic serve records. Stage 2 fine-tunes on the real
training records and selects the checkpoint on the validation split.

Both stages optimise the symmetric MoCo v3 objective. For each batch, two views
of every record are drawn by SCARF-style corruption. The online path
(encoder -> projector -> predictor) produces queries, the momentum path
(encoder -> projector, EMA of the online weights) produces keys, and InfoNCE is
applied in both directions with in-batch negatives.

Checkpoint selection penalises dimensional collapse:

    score = val_loss + rank_weight * (1 - effective_rank / d_model)

and the epoch with the lowest score is kept.

Usage:
    python -m courtstyle.training.train_serve --config configs/serve.yaml
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from courtstyle.models.serve_encoder import ContrastiveHead, ModernTabularEncoder, PredictionHead
from courtstyle.training.contrastive import MoCoV3Loss


# =============================================================================
# Data, augmentation and diagnostics
# =============================================================================

class TeeLogger:
    """Write output to both console and a log file."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.log_path = log_path

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def setup_logging(log_dir: Path) -> TeeLogger:
    """Setup logging to both console and file (overwrites each run)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_log.txt"

    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Logging to: {log_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    return tee


def set_seeds(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ForensicLogger:
    """
    Forensic logging for contrastive learning diagnostics.

    Computes intrinsic metrics to detect:
      - Model collapse (uniformity spike)
      - Physics lock (correlation → 1.0)
      - Dimensional collapse (low effective rank)

    Metrics:
      - Alignment Loss: Avg L2 distance between positive pairs (should decrease)
      - Uniformity Loss: log(mean(exp(-2*||x-y||^2))) (should stay low/negative)
      - Effective Rank: How many dimensions are used (higher = better)
      - Physics Correlation: Embedding norm vs avg speed (should be moderate, not 1.0)

    Reference: Wang & Isola (2020) "Understanding Contrastive Representation Learning"
    """

    def __init__(self, d_model: int = 128, speed_col_idx: Optional[int] = None):
        """
        Args:
            d_model: Embedding dimension (for rank reporting)
            speed_col_idx: Index of 'avg_speed' in continuous features (for physics corr)
        """
        self.d_model = d_model
        self.speed_col_idx = speed_col_idx

    @torch.no_grad()
    def compute_alignment(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> float:
        """
        Compute alignment loss: average L2 distance between positive pairs.

        Args:
            z1: Embeddings from view 1, shape (batch, d_model)
            z2: Embeddings from view 2, shape (batch, d_model)

        Returns:
            Average Euclidean distance (should decrease during training)
        """
        # Normalize embeddings
        z1_norm = nn.functional.normalize(z1, dim=1)
        z2_norm = nn.functional.normalize(z2, dim=1)

        # Compute pairwise L2 distance for positive pairs
        distances = torch.norm(z1_norm - z2_norm, dim=1)  # (batch,)
        return distances.mean().item()

    @torch.no_grad()
    def compute_uniformity(
        self,
        z: torch.Tensor,
        t: float = 2.0
    ) -> float:
        """
        Compute uniformity loss: log-expectation of Gaussian potential.

        Formula: log(mean(exp(-t * ||x - y||^2)))

        Args:
            z: Embeddings, shape (batch, d_model)
            t: Temperature parameter (default 2.0 per Wang & Isola)

        Returns:
            Uniformity loss (should stay low/negative; spike indicates collapse)
        """
        z_norm = nn.functional.normalize(z, dim=1)

        # Compute pairwise squared distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        # For normalized vectors: = 2 - 2*x.y = 2*(1 - x.y)
        sq_pdist = 2 - 2 * (z_norm @ z_norm.T)  # (batch, batch)

        # Mask diagonal (self-similarity)
        batch_size = z.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        sq_pdist = sq_pdist[mask]

        # Gaussian potential
        gaussian_potential = torch.exp(-t * sq_pdist)

        # Log-expectation
        uniformity = torch.log(gaussian_potential.mean() + 1e-8)
        return uniformity.item()

    @torch.no_grad()
    def compute_mean_norm_unitvecs(self, z: torch.Tensor) -> float:
        """
        Compute anisotropy metric: ||mean(normalize(z))||.

        For isotropic embeddings in D dims, this should be ~1/sqrt(D).
        Values close to 1.0 indicate a strong shared direction (anisotropy),
        which can make cosine-based neighborhoods nearly trivial.
        """
        if z.numel() == 0:
            return 0.0
        z_norm = nn.functional.normalize(z, dim=1)
        mean_vec = z_norm.mean(dim=0)
        return mean_vec.norm(p=2).item()

    @torch.no_grad()
    def compute_effective_rank(
        self,
        z: torch.Tensor
    ) -> Tuple[int, float, List[float]]:
        """
        Compute effective rank via SVD analysis.

        Returns:
            effective_rank: Number of singular values capturing 99% of variance
            top3_ratio: Ratio of top 3 singular values to total variance
            top_svs: Top 5 singular values (normalized)
        """
        # Center embeddings
        z_centered = z - z.mean(dim=0, keepdim=True)

        # SVD
        try:
            _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        except RuntimeError:
            # Fallback for numerical issues
            return self.d_model, 0.0, [0.0] * 5

        # Variance explained by each component
        var_explained = S ** 2
        total_var = var_explained.sum()

        if total_var < 1e-8:
            return 0, 1.0, [0.0] * 5

        # Normalized singular values
        var_ratio = var_explained / total_var

        # Effective rank: number of components for 99% variance
        cumsum = torch.cumsum(var_ratio, dim=0)
        effective_rank = (cumsum < 0.99).sum().item() + 1

        # Top 3 ratio
        top3_ratio = var_ratio[:3].sum().item() if len(var_ratio) >= 3 else 1.0

        # Top 5 singular values (normalized)
        top_svs = var_ratio[:5].tolist()
        top_svs = top_svs + [0.0] * (5 - len(top_svs))  # Pad if needed

        return effective_rank, top3_ratio, top_svs

    @torch.no_grad()
    def compute_physics_correlation(
        self,
        z: torch.Tensor,
        x_cont: torch.Tensor
    ) -> float:
        """
        Compute Spearman correlation between embedding norm and avg speed.

        Args:
            z: Embeddings, shape (batch, d_model)
            x_cont: Continuous features, shape (batch, n_features)

        Returns:
            Spearman correlation (should be moderate, not locked at 1.0)
        """
        if self.speed_col_idx is None:
            return 0.0

        # Embedding norms
        embedding_norms = torch.norm(z, dim=1).cpu().numpy()

        # Avg speed from input
        avg_speeds = x_cont[:, self.speed_col_idx].cpu().numpy()

        # Handle constant arrays
        if np.std(embedding_norms) < 1e-8 or np.std(avg_speeds) < 1e-8:
            return 0.0

        # Spearman correlation
        corr, _ = spearmanr(embedding_norms, avg_speeds)

        return corr if not np.isnan(corr) else 0.0

    def compute_all_metrics(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        x_cont: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all forensic metrics.

        Args:
            z1: Embeddings from view 1
            z2: Embeddings from view 2
            x_cont: Original continuous features (for physics correlation)

        Returns:
            Dictionary with all metrics
        """
        # Combine embeddings for uniformity and rank
        z_combined = torch.cat([z1, z2], dim=0)

        # Compute metrics
        alignment = self.compute_alignment(z1, z2)
        uniformity = self.compute_uniformity(z_combined)
        mean_norm_unitvecs = self.compute_mean_norm_unitvecs(z1)
        effective_rank, top3_ratio, top_svs = self.compute_effective_rank(z_combined)
        physics_corr = self.compute_physics_correlation(z1, x_cont)

        return {
            "alignment": alignment,
            "uniformity": uniformity,
            "mean_norm_unitvecs": mean_norm_unitvecs,
            "effective_rank": effective_rank,
            "top3_ratio": top3_ratio,
            "physics_corr": physics_corr,
            "sv1": top_svs[0],
            "sv2": top_svs[1],
            "sv3": top_svs[2],
        }

    @staticmethod
    def format_log_line(
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        total_dims: int,
        prefix: str = "",
    ) -> str:
        """Format a single log line with all metrics."""
        mean_norm = metrics.get("mean_norm_unitvecs")
        mean_norm_str = f" | MeanNorm: {mean_norm:.3f}" if mean_norm is not None else ""
        return (
            f"{prefix}Ep {epoch:3d}: L={loss:.4f} | "
            f"Align: {metrics['alignment']:.3f} | "
            f"Unif: {metrics['uniformity']:.2f} | "
            f"Rank: {metrics['effective_rank']:3d}/{total_dims} | "
            f"PhysCorr: {metrics['physics_corr']:.3f}"
            f"{mean_norm_str}"
        )


class TabularAugmenter:
    """
    SCARF-compliant tabular data augmentation for contrastive learning.

    Generates (View1, View2) pairs via:
      - Continuous: Column-wise shuffle (replace with values from other rows)
        This is proper SCARF corruption - masked values come from the marginal
        distribution, not just "0" which means "mean" after standardization.
      - Categorical: Mask to "unknown" token (cardinality index) - NOT random swap
        Random swap breaks positive pair semantics (e.g., hand L↔R represents
        different players semantically).
      - Noise: Add small Gaussian noise to prevent overfitting

    References:
      - SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
        (https://arxiv.org/abs/2106.15147)
    """

    def __init__(
        self,
        mask_prob: float = 0.3,
        noise_std: float = 0.02,
        categorical_mask_prob: float = 0.15,
    ) -> None:
        self.mask_prob = mask_prob
        self.noise_std = noise_std
        self.categorical_mask_prob = categorical_mask_prob

    def augment_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment continuous features using SCARF-style column-wise shuffle.

        Instead of masking to 0 (which means "mean" after StandardScaler),
        we replace masked values with values from other rows in the batch.
        This samples from the empirical marginal distribution of each feature.

        Args:
            x: (batch, n_features) continuous tensor

        Returns:
            Augmented tensor with column-wise shuffled corruption + noise
        """
        batch_size, n_features = x.shape

        # Add small Gaussian noise first (to all values)
        noise = torch.randn_like(x) * self.noise_std
        out = x + noise

        # SCARF-style corruption: replace masked values with values from other rows
        # For each feature column, shuffle the batch dimension independently
        mask = torch.rand_like(x) < self.mask_prob

        if mask.any():
            # Generate random permutation indices for each column
            # This samples from the marginal distribution of each feature
            for col in range(n_features):
                col_mask = mask[:, col]
                if col_mask.any():
                    # Random indices from other rows in the batch
                    perm_indices = torch.randperm(batch_size, device=x.device)
                    # Replace masked values with values from permuted rows
                    out[col_mask, col] = x[perm_indices[col_mask], col]

        return out

    def augment_categorical(
        self, x: torch.Tensor, cardinalities: List[int]
    ) -> torch.Tensor:
        """
        Augment categorical features by masking to "unknown" token.

        IMPORTANT: We do NOT randomly swap categories (e.g., hand L↔R) because
        that breaks positive pair semantics - two augmented views of the "same"
        sample would represent different players semantically.

        Instead, masked categories are set to an "unknown" token index, which is
        the cardinality value (one past the last valid index). The embedding layer
        must be sized to handle this extra token.

        Args:
            x: (batch, n_cats) integer tensor of category indices
            cardinalities: List of cardinality for each categorical column

        Returns:
            Augmented tensor with some categories masked to "unknown"
        """
        if x.numel() == 0:
            return x

        out = x.clone()
        mask = torch.rand_like(x, dtype=torch.float) < self.categorical_mask_prob

        # Set masked categories to "unknown" token (cardinality index)
        # This requires embedding layers to have cardinality+1 embeddings
        for i, card in enumerate(cardinalities):
            unknown_idx = card  # Index beyond the valid categories
            col_mask = mask[:, i]
            out[col_mask, i] = unknown_idx

        return out

    def __call__(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        cardinalities: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one augmented view."""
        aug_cont = self.augment_continuous(x_cont)
        aug_cat = self.augment_categorical(x_cat, cardinalities)
        return aug_cont, aug_cat


def load_processed_data(
    path: Path,
    cont_cols: List[str],
    cat_cols: List[str],
    surface_cols: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Load pre-processed CSV directly into tensors.
    Data are already standardised by courtstyle.data.serve_dataset.

    Args:
        path: Path to processed CSV
        cont_cols: List of continuous column names
        cat_cols: List of categorical column names (one-hot prefixes like 'hand_')
        surface_cols: Optional list of surface one-hot columns for domain labels

    Returns:
        x_cont: (N, n_cont) float tensor
        x_cat: (N, n_cat_groups) long tensor (indices, not one-hot)
        surface_labels: (N,) long tensor of surface indices, or None
    """
    df = pd.read_csv(path)

    # Continuous features - already scaled
    x_cont = torch.from_numpy(df[cont_cols].to_numpy(dtype="float32"))

    # Categorical features - convert one-hot to indices
    cat_tensors = []
    cat_prefixes = list(set(c.rsplit("_", 1)[0] + "_" for c in cat_cols))

    for prefix in sorted(cat_prefixes):
        cols = [c for c in cat_cols if c.startswith(prefix)]
        if cols:
            one_hot = df[cols].to_numpy()
            indices = one_hot.argmax(axis=1).astype("int64")
            cat_tensors.append(torch.from_numpy(indices))

    if cat_tensors:
        x_cat = torch.stack(cat_tensors, dim=1)
    else:
        x_cat = torch.zeros((len(df), 0), dtype=torch.long)

    # Extract surface labels for domain adversarial training
    surface_labels = None
    if surface_cols:
        existing_surface_cols = [c for c in surface_cols if c in df.columns]
        if existing_surface_cols:
            surface_one_hot = df[existing_surface_cols].to_numpy()
            surface_labels = torch.from_numpy(
                surface_one_hot.argmax(axis=1).astype("int64")
            )

    return x_cont, x_cat, surface_labels


def _assert_surface_binary(
    path: Path,
    hard_col: str,
    grass_col: str,
    extra_cols: List[str],
    split_name: str,
) -> None:
    cols = [hard_col, grass_col] + extra_cols
    df = pd.read_csv(path, usecols=cols)
    hard = df[hard_col].to_numpy(dtype="float32")
    grass = df[grass_col].to_numpy(dtype="float32")
    row_sum = hard + grass
    if not np.allclose(row_sum, 1.0):
        bad = int(np.sum(~np.isclose(row_sum, 1.0)))
        raise AssertionError(
            f"{split_name}: surface_Hard + surface_Grass must sum to 1. Bad rows: {bad}"
        )
    for col in extra_cols:
        vals = df[col].to_numpy(dtype="float32")
        if not np.isclose(vals, 0.0).all():
            bad = int(np.sum(~np.isclose(vals, 0.0)))
            raise AssertionError(
                f"{split_name}: {col} must be all zeros. Bad rows: {bad}"
            )


def create_dataloader(
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    surface_labels: Optional[torch.Tensor] = None,
    balanced_sampling: bool = False,
) -> DataLoader:
    """
    Create DataLoader from tensors, optionally including surface labels.

    Args:
        x_cont: Continuous features tensor
        x_cat: Categorical features tensor
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        surface_labels: Optional surface labels for domain adversarial training
        balanced_sampling: If True, use WeightedRandomSampler to balance by surface

    Returns:
        DataLoader instance
    """
    if surface_labels is not None:
        dataset = TensorDataset(x_cont, x_cat, surface_labels)
    else:
        dataset = TensorDataset(x_cont, x_cat)

    sampler = None
    if balanced_sampling and surface_labels is not None:
        # Compute inverse class frequency weights
        unique_labels, counts = torch.unique(surface_labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        # Assign weight to each sample based on its class
        sample_weights = class_weights[surface_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(surface_labels),
            replacement=True,
        )
        # When using sampler, shuffle must be False
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
    )


@torch.no_grad()
def momentum_update(
    query_encoder: nn.Module,
    query_head: nn.Module,
    key_encoder: nn.Module,
    key_head: nn.Module,
    momentum: float = 0.99,
) -> None:
    """MoCo v3 momentum update: EMA of query to key."""
    for param_q, param_k in zip(query_encoder.parameters(), key_encoder.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data, alpha=(1.0 - momentum))
    for param_q, param_k in zip(query_head.parameters(), key_head.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data, alpha=(1.0 - momentum))


def tensors_to_dict(
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    cont_names: List[str],
    cat_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Convert stacked tensors to dict format for encoder."""
    x_cont_dict = {name: x_cont[:, i] for i, name in enumerate(cont_names)}
    x_cat_dict = {name: x_cat[:, i] for i, name in enumerate(cat_names)}
    return x_cont_dict, x_cat_dict



# =============================================================================
# Checkpoint selection
# =============================================================================

def selection_score(val_loss: float, effective_rank: float, d_model: int, rank_weight: float) -> float:
    """Checkpoint selection score, lower is better.

    Adds a penalty for dimensional collapse to the validation loss:
    ``val_loss + rank_weight * (1 - effective_rank / d_model)``.
    """
    return float(val_loss + rank_weight * (1.0 - float(effective_rank) / float(d_model)))


# =============================================================================
# Training and validation epochs
# =============================================================================

def pretrain_epoch(
    query_encoder: ModernTabularEncoder,
    query_head: ContrastiveHead,
    predictor: PredictionHead,
    key_encoder: ModernTabularEncoder,
    key_head: ContrastiveHead,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: MoCoV3Loss,
    augmenter: TabularAugmenter,
    cont_names: List[str],
    cat_names: List[str],
    cardinalities: List[int],
    device: torch.device,
    momentum: float = 0.99,
) -> float:
    """One MoCo v3 epoch on synthetic records (stage 1).

    For each batch: two augmented views, queries from the online path, keys from
    the momentum path, symmetric loss ``ctr(q1, k2) + ctr(q2, k1)``, gradient
    clipping at norm 1.0 per module, then a momentum update of the key path.

    Returns:
        Mean loss over the epoch.
    """
    query_encoder.train()
    query_head.train()
    predictor.train()
    key_encoder.train()
    key_head.train()

    total_loss = 0.0
    for x_cont, x_cat in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)

        optimizer.zero_grad()

        aug_cont1, aug_cat1 = augmenter(x_cont, x_cat, cardinalities)
        aug_cont2, aug_cat2 = augmenter(x_cont, x_cat, cardinalities)
        x_cont_d1, x_cat_d1 = tensors_to_dict(aug_cont1, aug_cat1, cont_names, cat_names)
        x_cont_d2, x_cat_d2 = tensors_to_dict(aug_cont2, aug_cat2, cont_names, cat_names)

        z1 = query_encoder(x_cont_d1, x_cat_d1)
        z2 = query_encoder(x_cont_d2, x_cat_d2)
        q1 = predictor(query_head(z1))
        q2 = predictor(query_head(z2))

        with torch.no_grad():
            k1 = key_head(key_encoder(x_cont_d1, x_cat_d1))
            k2 = key_head(key_encoder(x_cont_d2, x_cat_d2))

        loss = criterion(q1, q2, k1, k2)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(query_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)

        optimizer.step()
        momentum_update(query_encoder, query_head, key_encoder, key_head, momentum)
        total_loss += loss.item()

    return total_loss / len(loader)


def finetune_epoch(
    query_encoder: ModernTabularEncoder,
    query_head: ContrastiveHead,
    predictor: PredictionHead,
    key_encoder: ModernTabularEncoder,
    key_head: ContrastiveHead,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: MoCoV3Loss,
    augmenter: TabularAugmenter,
    cont_names: List[str],
    cat_names: List[str],
    cardinalities: List[int],
    device: torch.device,
    forensic_logger: ForensicLogger,
    momentum: float = 0.99,
) -> Tuple[float, Dict[str, float]]:
    """One MoCo v3 epoch on real records (stage 2), with embedding diagnostics.

    Same objective as ``pretrain_epoch``. Stage 2 applies no gradient clipping.
    Encoder embeddings of both views are accumulated to compute the effective
    rank and related diagnostics used for checkpoint selection.

    Returns:
        Mean loss over the epoch and the diagnostic metrics.
    """
    query_encoder.train()
    query_head.train()
    predictor.train()
    key_encoder.train()
    key_head.train()

    total_loss = 0.0
    all_z1, all_z2, all_x_cont = [], [], []

    for batch in loader:
        # The training loader also yields surface labels for balanced sampling.
        x_cont, x_cat = batch[0], batch[1]
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)

        optimizer.zero_grad()

        aug_cont1, aug_cat1 = augmenter(x_cont, x_cat, cardinalities)
        aug_cont2, aug_cat2 = augmenter(x_cont, x_cat, cardinalities)
        x_cont_d1, x_cat_d1 = tensors_to_dict(aug_cont1, aug_cat1, cont_names, cat_names)
        x_cont_d2, x_cat_d2 = tensors_to_dict(aug_cont2, aug_cat2, cont_names, cat_names)

        z1 = query_encoder(x_cont_d1, x_cat_d1)
        z2 = query_encoder(x_cont_d2, x_cat_d2)
        with torch.no_grad():
            all_z1.append(z1.detach())
            all_z2.append(z2.detach())
            all_x_cont.append(x_cont.detach())

        q1 = predictor(query_head(z1))
        q2 = predictor(query_head(z2))

        with torch.no_grad():
            k1 = key_head(key_encoder(x_cont_d1, x_cat_d1))
            k2 = key_head(key_encoder(x_cont_d2, x_cat_d2))

        loss = criterion(q1, q2, k1, k2)
        loss.backward()
        optimizer.step()
        momentum_update(query_encoder, query_head, key_encoder, key_head, momentum)
        total_loss += loss.item()

    metrics = forensic_logger.compute_all_metrics(
        torch.cat(all_z1, dim=0), torch.cat(all_z2, dim=0), torch.cat(all_x_cont, dim=0)
    )
    return total_loss / len(loader), metrics


@torch.no_grad()
def validate_epoch(
    query_encoder: ModernTabularEncoder,
    query_head: ContrastiveHead,
    predictor: PredictionHead,
    key_encoder: ModernTabularEncoder,
    key_head: ContrastiveHead,
    loader: DataLoader,
    criterion: MoCoV3Loss,
    augmenter: TabularAugmenter,
    cont_names: List[str],
    cat_names: List[str],
    cardinalities: List[int],
    device: torch.device,
    forensic_logger: ForensicLogger,
) -> Tuple[float, Dict[str, float]]:
    """MoCo v3 validation loss on augmented views, with embedding diagnostics.

    Returns:
        Mean validation loss and the diagnostic metrics.
    """
    query_encoder.eval()
    query_head.eval()
    predictor.eval()
    key_encoder.eval()
    key_head.eval()

    total_loss = 0.0
    all_z1, all_z2, all_x_cont = [], [], []

    for batch in loader:
        x_cont, x_cat = batch[0], batch[1]
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)

        aug_cont1, aug_cat1 = augmenter(x_cont, x_cat, cardinalities)
        aug_cont2, aug_cat2 = augmenter(x_cont, x_cat, cardinalities)
        x_cont_d1, x_cat_d1 = tensors_to_dict(aug_cont1, aug_cat1, cont_names, cat_names)
        x_cont_d2, x_cat_d2 = tensors_to_dict(aug_cont2, aug_cat2, cont_names, cat_names)

        all_z1.append(query_encoder(x_cont_d1, x_cat_d1))
        all_z2.append(query_encoder(x_cont_d2, x_cat_d2))
        all_x_cont.append(x_cont)

        q1 = predictor(query_head(query_encoder(x_cont_d1, x_cat_d1)))
        q2 = predictor(query_head(query_encoder(x_cont_d2, x_cat_d2)))
        k1 = key_head(key_encoder(x_cont_d1, x_cat_d1))
        k2 = key_head(key_encoder(x_cont_d2, x_cat_d2))

        total_loss += criterion(q1, q2, k1, k2).item()

    metrics = forensic_logger.compute_all_metrics(
        torch.cat(all_z1, dim=0), torch.cat(all_z2, dim=0), torch.cat(all_x_cont, dim=0)
    )
    return total_loss / len(loader), metrics


@torch.no_grad()
def compute_embeddings(
    encoder: ModernTabularEncoder,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    cont_names: List[str],
    cat_names: List[str],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed records with a frozen encoder.

    Returns:
        Array of shape (n_records, d_model).
    """
    encoder.eval()
    outputs = []
    for start in range(0, x_cont.size(0), batch_size):
        end = min(start + batch_size, x_cont.size(0))
        x_cont_d, x_cat_d = tensors_to_dict(
            x_cont[start:end].to(device), x_cat[start:end].to(device), cont_names, cat_names
        )
        outputs.append(encoder(x_cont_d, x_cat_d).cpu().numpy())
    emb = np.concatenate(outputs, axis=0)
    if not np.isfinite(emb).all():
        raise ValueError("Embeddings contain non-finite values.")
    return emb


# =============================================================================
# Entry point
# =============================================================================

def build_schema(train_csv: Path) -> Tuple[List[str], List[str], List[str], Dict[str, int], List[int]]:
    """Derive the encoder input schema from the processed training file.

    Handedness one-hot columns form one categorical group. Surface columns are
    excluded from the encoder input and used only for balanced sampling.
    """
    all_cols = pd.read_csv(train_csv, nrows=1).columns.tolist()
    surface_cols_all = [c for c in all_cols if c.startswith("surface_")]
    for required in ("surface_Hard", "surface_Grass"):
        if required not in all_cols:
            raise ValueError(f"Missing required surface column: {required}")
    cat_cols = [c for c in all_cols if c.startswith("hand_")]
    cont_cols = [c for c in all_cols if c not in cat_cols and c not in surface_cols_all]
    cat_names = sorted(set(c.rsplit("_", 1)[0] for c in cat_cols))
    categorical_cardinalities: Dict[str, int] = {}
    cardinalities: List[int] = []
    for prefix in cat_names:
        card = len([c for c in cat_cols if c.startswith(prefix + "_")])
        categorical_cardinalities[prefix] = card
        cardinalities.append(card)
    return cont_cols, cat_cols, cat_names, categorical_cardinalities, cardinalities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage MoCo v3 training of the CourtStyle serve encoder.")
    p.add_argument("--config", type=Path, default=Path("configs/serve.yaml"))
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="Directory with train_data.csv, val_data.csv and synthetic_processed.csv. "
                        "Overrides data.processed_dir in the config.")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/serve"),
                   help="Output directory for checkpoints, logs and embeddings.")
    p.add_argument("--skip-pretraining", action="store_true",
                   help="Load the stage-1 checkpoint from --work-dir instead of pre-training.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_cfg, tr_cfg = cfg["encoder"], cfg["training"]

    processed_dir = args.processed_dir or Path(cfg["data"]["processed_dir"])
    train_csv = processed_dir / "train_data.csv"
    val_csv = processed_dir / "val_data.csv"
    syn_csv = processed_dir / "synthetic_processed.csv"
    for path in (train_csv, val_csv, syn_csv):
        if not path.exists():
            sys.exit(f"Missing input file: {path}. Run courtstyle.data.serve_dataset first.")

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(work_dir)

    seed = int(cfg["data"]["seed"])
    set_seeds(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | seed: {seed}")

    d_model = int(enc_cfg["d_model"])
    cont_cols, cat_cols, cat_names, categorical_cardinalities, cardinalities = build_schema(train_csv)
    print(f"Encoder input: {len(cont_cols)} continuous columns, categorical {cat_names} {cardinalities}")

    speed_col_idx = next(
        (i for i, c in enumerate(cont_cols) if "avg_speed" in c.lower().replace(" ", "_")), None
    )
    forensic_logger = ForensicLogger(d_model=d_model, speed_col_idx=speed_col_idx)

    surface_cols = ["surface_Hard", "surface_Grass"]
    all_cols = pd.read_csv(train_csv, nrows=1).columns.tolist()
    surface_extra = [c for c in all_cols if c.startswith("surface_") and c not in surface_cols]
    _assert_surface_binary(train_csv, "surface_Hard", "surface_Grass", surface_extra, split_name="train")
    _assert_surface_binary(val_csv, "surface_Hard", "surface_Grass", surface_extra, split_name="val")

    x_cont_syn, x_cat_syn, _ = load_processed_data(syn_csv, cont_cols, cat_cols, surface_cols=None)
    x_cont_train, x_cat_train, surface_train = load_processed_data(train_csv, cont_cols, cat_cols, surface_cols)
    x_cont_val, x_cat_val, surface_val = load_processed_data(val_csv, cont_cols, cat_cols, surface_cols)

    p1, p2 = tr_cfg["phase1"], tr_cfg["phase2"]
    loader_syn = create_dataloader(x_cont_syn, x_cat_syn, int(p1["batch_size"]), shuffle=True, drop_last=True)
    # Surface-balanced sampling on the real training records.
    loader_train = create_dataloader(
        x_cont_train, x_cat_train, int(p2["batch_size"]), shuffle=True, drop_last=True,
        surface_labels=surface_train, balanced_sampling=True,
    )
    loader_val = create_dataloader(
        x_cont_val, x_cat_val, int(p2["batch_size"]), shuffle=False, drop_last=False,
        surface_labels=surface_val,
    )

    query_encoder = ModernTabularEncoder(
        continuous_cols=cont_cols,
        categorical_cardinalities=categorical_cardinalities,
        d_model=d_model,
        n_heads=int(enc_cfg["n_heads"]),
        depth=int(enc_cfg["depth"]),
        expansion_n=int(enc_cfg["expansion_n"]),
        ff_multiplier=int(enc_cfg["ff_multiplier"]),
        dropout=float(enc_cfg["dropout"]),
        encoder_block=str(enc_cfg["encoder_block"]),
    ).to(device)
    query_head = ContrastiveHead(d_model=d_model, proj_dim=int(enc_cfg["proj_dim"])).to(device)
    predictor = PredictionHead(proj_dim=int(enc_cfg["proj_dim"])).to(device)
    key_encoder = copy.deepcopy(query_encoder)
    key_head = copy.deepcopy(query_head)
    for param in list(key_encoder.parameters()) + list(key_head.parameters()):
        param.requires_grad = False

    criterion = MoCoV3Loss(temperature=float(tr_cfg["temperature"]))
    aug = tr_cfg["augmentation"]
    augmenter_p1 = TabularAugmenter(mask_prob=float(p1["mask_prob"]), noise_std=float(aug["noise_std"]),
                                    categorical_mask_prob=float(aug["categorical_mask_prob"]))
    augmenter_p2 = TabularAugmenter(mask_prob=float(p2["mask_prob"]), noise_std=float(aug["noise_std"]),
                                    categorical_mask_prob=float(aug["categorical_mask_prob"]))
    momentum = float(tr_cfg["momentum"])
    weight_decay = float(tr_cfg["weight_decay"])
    params = list(query_encoder.parameters()) + list(query_head.parameters()) + list(predictor.parameters())

    def checkpoint(extra: Dict) -> Dict:
        state = {
            "encoder": query_encoder.state_dict(),
            "head": query_head.state_dict(),
            "predictor": predictor.state_dict(),
            "key_encoder": key_encoder.state_dict(),
            "key_head": key_head.state_dict(),
            "cont_cols": cont_cols,
            "cat_names": cat_names,
            "cardinalities": cardinalities,
            "config": {k: enc_cfg[k] for k in
                       ("d_model", "n_heads", "depth", "expansion_n", "ff_multiplier", "dropout", "encoder_block")},
        }
        state.update(extra)
        return state

    # Stage 1: pre-training on synthetic records.
    pretrained_ckpt = work_dir / "serve_encoder_pretrained.pt"
    phase1_losses: List[float] = []
    if args.skip_pretraining:
        if not pretrained_ckpt.exists():
            sys.exit(f"--skip-pretraining set but {pretrained_ckpt} not found.")
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
        for module, key in ((query_encoder, "encoder"), (query_head, "head"), (predictor, "predictor"),
                            (key_encoder, "key_encoder"), (key_head, "key_head")):
            module.load_state_dict(ckpt[key], strict=True)
        print(f"Loaded stage-1 checkpoint: {pretrained_ckpt}")
    else:
        optimizer = optim.AdamW(params, lr=float(p1["lr"]), weight_decay=weight_decay)
        for epoch in range(1, int(p1["epochs"]) + 1):
            loss = pretrain_epoch(query_encoder, query_head, predictor, key_encoder, key_head, loader_syn,
                                  optimizer, criterion, augmenter_p1, cont_cols, cat_names, cardinalities,
                                  device, momentum)
            phase1_losses.append(loss)
            if epoch % 10 == 0 or epoch == 1:
                print(f"[Stage 1] epoch {epoch:3d} | loss {loss:.4f}")
        torch.save(checkpoint({}), pretrained_ckpt)

    # Stage 2: fine-tuning on real records with checkpoint selection.
    optimizer = optim.AdamW(params, lr=float(p2["lr"]), weight_decay=weight_decay)
    rank_weight = float(tr_cfg["select_rank_weight"])
    best_score, best_epoch = float("inf"), None
    best_ckpt = work_dir / "serve_encoder_best.pt"
    history = []

    for epoch in range(1, int(p2["epochs"]) + 1):
        train_loss, train_metrics = finetune_epoch(
            query_encoder, query_head, predictor, key_encoder, key_head, loader_train, optimizer, criterion,
            augmenter_p2, cont_cols, cat_names, cardinalities, device, forensic_logger, momentum,
        )
        val_loss, val_metrics = validate_epoch(
            query_encoder, query_head, predictor, key_encoder, key_head, loader_val, criterion,
            augmenter_p2, cont_cols, cat_names, cardinalities, device, forensic_logger,
        )
        score = selection_score(val_loss, val_metrics["effective_rank"], d_model, rank_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_effective_rank": val_metrics["effective_rank"], "selection_score": score})

        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(checkpoint({"epoch": epoch, "val_loss": float(val_loss),
                                   "selection_score": float(score), "forensic_metrics": val_metrics}), best_ckpt)

        marker = " *best*" if epoch == best_epoch else ""
        print(ForensicLogger.format_log_line(epoch, train_loss, train_metrics, total_dims=d_model, prefix="[Train] "))
        print(ForensicLogger.format_log_line(epoch, val_loss, val_metrics, total_dims=d_model, prefix="[Val]   ")
              + f" | score {score:.4f}{marker}")

    pd.DataFrame(history).to_csv(work_dir / "training_history.csv", index=False)
    if phase1_losses:
        pd.DataFrame({"epoch": range(1, len(phase1_losses) + 1), "loss": phase1_losses}).to_csv(
            work_dir / "pretraining_history.csv", index=False)
    print(f"Selected epoch {best_epoch} (score {best_score:.4f}): {best_ckpt}")

    # Export embeddings from the selected checkpoint.
    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    query_encoder.load_state_dict(best["encoder"], strict=True)
    np.save(work_dir / "train_embeddings.npy",
            compute_embeddings(query_encoder, x_cont_train, x_cat_train, cont_cols, cat_names, device))
    np.save(work_dir / "val_embeddings.npy",
            compute_embeddings(query_encoder, x_cont_val, x_cat_val, cont_cols, cat_names, device))
    np.save(work_dir / "train_surface_labels.npy", surface_train.numpy())
    np.save(work_dir / "val_surface_labels.npy", surface_val.numpy())
    print(f"Embeddings written to {work_dir}")


if __name__ == "__main__":
    main()

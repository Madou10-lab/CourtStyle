"""Contrastive training of the rally encoder.

Each training rally yields three views per step. Two are independent
augmentations of the rally, and the third is an augmentation of a copy whose
transition order has been permuted. The objective is

    L = NT-Xent(z1, z2; tau) + lambda * relu(cos(e1, e_shuffled) - margin)

where ``z`` are projection-head outputs and ``e`` are encoder embeddings. The
first term pulls the two views together against the rest of the batch. The
second term pushes the rally away from its shuffled copy, so that the encoder
cannot reduce a rally to order-invariant shot statistics.

Views are drawn with a per-item random generator. Training views change from
one epoch to the next and validation views are fixed, so the validation loss
is comparable across epochs. The checkpoint with the lowest validation loss is
kept.

Usage:
    python -m courtstyle.training.train_rally --config configs/rally.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from courtstyle.data.rally_transitions import (
    _safe_int,
    build_rally_dataset,
    column_groups_to_json,
    load_split_info,
    select_raw_columns,
)
from courtstyle.models.rally_encoder import RallySSLModel
from courtstyle.training.contrastive import nt_xent_loss, order_repulsion_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "run": root,
        "models": root / "models",
        "logs": root / "logs",
        "data": root / "data",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# Views
# =============================================================================

class DomainAwareSequenceAugmenter:
    """
    Sequence augmenter that is column-name aware and applies
    domain-aligned invariances to rally token sequences.
    """

    def __init__(
        self,
        *,
        column_groups: Dict[str, Any],
        crop_min_ratio: float = 0.6,
        token_dropout: float = 0.1,
        feature_mask_prob: float = 0.05,
        noise_std: float = 0.01,
        flip_x_prob: float = 0.5,
        flip_y_prob: float = 0.5,
        player_swap_prob: float = 0.0,
    ) -> None:
        self.crop_min_ratio = float(crop_min_ratio)
        self.token_dropout = float(token_dropout)
        self.feature_mask_prob = float(feature_mask_prob)
        self.noise_std = float(noise_std)
        self.flip_x_prob = float(flip_x_prob)
        self.flip_y_prob = float(flip_y_prob)
        self.player_swap_prob = float(player_swap_prob)

        self.x_token_indices = np.array(column_groups["x_token_indices"], dtype=int)
        self.y_token_indices = np.array(column_groups["y_token_indices"], dtype=int)
        self.swap_pairs = [
            (int(a), int(b))
            for a, b in column_groups["player_opponent_swap_token_pairs"]
        ]

    def __call__(self, seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = seq.astype(np.float32, copy=True)
        n = x.shape[0]
        if n < 1:
            raise ValueError("Sequence must have at least 1 token.")

        if n > 1 and self.crop_min_ratio < 1.0:
            min_len = max(1, int(math.floor(n * self.crop_min_ratio)))
            crop_len = int(rng.integers(min_len, n + 1))
            start = int(rng.integers(0, n - crop_len + 1))
            x = x[start : start + crop_len]

        n2 = x.shape[0]
        if n2 > 1 and self.token_dropout > 0:
            keep = rng.random(n2) >= self.token_dropout
            if not np.any(keep):
                keep[int(rng.integers(0, n2))] = True
            x = x[keep]

        if self.player_swap_prob > 0 and rng.random() < self.player_swap_prob:
            for i_a, i_b in self.swap_pairs:
                x[:, i_a], x[:, i_b] = x[:, i_b].copy(), x[:, i_a].copy()

        if self.flip_x_prob > 0 and len(self.x_token_indices) > 0:
            if rng.random() < self.flip_x_prob:
                x[:, self.x_token_indices] *= -1.0

        if self.flip_y_prob > 0 and len(self.y_token_indices) > 0:
            if rng.random() < self.flip_y_prob:
                x[:, self.y_token_indices] *= -1.0

        if self.feature_mask_prob > 0:
            mask = rng.random(x.shape) < self.feature_mask_prob
            x[mask] = 0.0

        if self.noise_std > 0:
            x = x + rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        return x.astype(np.float32, copy=False)


class RallyTripletDataset(Dataset):
    """Returns (view1, view2, view_shuffled) for one rally."""

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        augmenter: DomainAwareSequenceAugmenter,
        *,
        seed: int,
        deterministic: bool,
    ) -> None:
        self.sequences = list(sequences)
        self.augmenter = augmenter
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self._call_count = 0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq = self.sequences[idx]
        if self.deterministic:
            base = self.seed + idx * 10007
        else:
            self._call_count += 1
            base = self.seed + self._call_count * 7919 + idx * 10007

        rng1 = np.random.default_rng(base)
        rng2 = np.random.default_rng(base ^ 0x9E3779B1)
        rng_perm = np.random.default_rng(base ^ 0x6C078965)
        rng_shuf = np.random.default_rng(base ^ 0x517CC1B7)

        v1 = self.augmenter(seq, rng1)
        v2 = self.augmenter(seq, rng2)

        perm = rng_perm.permutation(len(seq))
        shuffled_seq = seq[perm].copy()
        v_shuf = self.augmenter(shuffled_seq, rng_shuf)

        return v1, v2, v_shuf


def _pad_sequences(seqs: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [int(s.shape[0]) for s in seqs]
    max_len = max(lengths)
    feat_dim = int(seqs[0].shape[1])
    batch = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        L = int(s.shape[0])
        batch[i, :L, :] = torch.from_numpy(s.astype(np.float32, copy=False))
    return batch, torch.tensor(lengths, dtype=torch.long)


def collate_triplet(
    batch: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[torch.Tensor, ...]:
    t1, l1 = _pad_sequences([item[0] for item in batch])
    t2, l2 = _pad_sequences([item[1] for item in batch])
    ts, ls = _pad_sequences([item[2] for item in batch])
    return t1, l1, t2, l2, ts, ls


# =============================================================================
# Training
# =============================================================================

def run_epoch(
    model: RallySSLModel,
    loader: DataLoader,
    *,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    temperature: float,
    grad_clip: float,
    order_loss_lambda: float,
    order_margin: float,
) -> Dict[str, float]:
    """One pass over ``loader``. Trains when ``optimizer`` is given, otherwise evaluates."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_contrastive = 0.0
    total_order = 0.0
    total_combined = 0.0
    n_batches = 0

    for t1, l1, t2, l2, ts, ls in loader:
        t1 = t1.to(device, non_blocking=True)
        l1 = l1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        l2 = l2.to(device, non_blocking=True)
        ts = ts.to(device, non_blocking=True)
        ls = ls.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            e1, z1 = model(t1, l1)
            _, z2 = model(t2, l2)
            e_shuf, _ = model(ts, ls)

            loss_c = nt_xent_loss(z1, z2, temperature=temperature)
            loss_o = order_repulsion_loss(e1, e_shuf, margin=order_margin)
            loss = loss_c + order_loss_lambda * loss_o

        if optimizer is not None:
            loss.backward()
            if grad_clip > 0:
                grad_params = [
                    p
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.requires_grad
                ]
                if grad_params:
                    nn.utils.clip_grad_norm_(grad_params, max_norm=grad_clip)
            optimizer.step()

        total_contrastive += float(loss_c.item())
        total_order += float(loss_o.item())
        total_combined += float(loss.item())
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "contrastive": total_contrastive / denom,
        "order": total_order / denom,
        "combined": total_combined / denom,
    }


def _load_split_shots(
    shots_csv: Path,
    split_info: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    shots = pd.read_csv(shots_csv, low_memory=False)
    shots["match_id"] = shots["match_id"].astype(str)
    shots["point_index"] = shots["point_index"].apply(_safe_int)
    shots["shot_index"] = shots["shot_index"].apply(_safe_int)

    train_ids, val_ids = load_split_info(split_info)
    train_set = set(str(x) for x in train_ids)
    val_set = set(str(x) for x in val_ids)
    train_df = shots[shots["match_id"].isin(train_set)].reset_index(drop=True)
    val_df = shots[shots["match_id"].isin(val_set)].reset_index(drop=True)
    info = {
        "n_shots_all": int(len(shots)),
        "n_matches_all": int(shots["match_id"].nunique()),
        "n_train_match_ids": int(len(train_set)),
        "n_val_match_ids": int(len(val_set)),
        "n_shots_train": int(len(train_df)),
        "n_shots_val": int(len(val_df)),
    }
    return train_df, val_df, info


def _save_checkpoint(
    model: RallySSLModel,
    path: Path,
    *,
    cfg: Dict[str, Any],
    raw_cols: List[str],
    token_columns: List[str],
    column_groups: Dict[str, Any],
    input_dim: int,
    best_epoch: int,
    best_val_loss: float,
    dataset_stats: Dict[str, Any],
) -> None:
    enc, tr = cfg["encoder"], cfg["training"]
    ckpt: Dict[str, Any] = {
        "sequence_model_state": model.state_dict(),
        "model_config": {
            "input_dim": int(input_dim),
            "d_model": int(enc["d_model"]),
            "n_layers": int(enc["n_layers"]),
            "n_heads": int(enc["n_heads"]),
            "dropout": float(enc["dropout"]),
            "pooling": str(enc["pooling"]),
            "max_len": int(cfg["data"]["max_len"]),
            "proj_dim": int(enc["proj_dim"]),
            "encoder_block": str(enc["encoder_block"]),
            "modern_use_gated_mha": bool(enc.get("modern_use_gated_mha", True)),
        },
        "augment_config": {k: float(v) for k, v in tr["augmentation"].items()},
        "order_loss_config": {
            "order_loss_lambda": float(tr["order_loss_lambda"]),
            "order_margin": float(tr["order_margin"]),
        },
        "raw_columns": raw_cols,
        "token_columns": token_columns,
        "column_groups": column_groups_to_json(column_groups),
        "config": cfg,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "dataset_stats": dataset_stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive training of the CourtStyle rally encoder.")
    p.add_argument("--config", type=Path, default=Path("configs/rally.yaml"))
    p.add_argument("--shots-csv", type=Path, default=None,
                   help="Processed training shots. Overrides data.train_shots in the config.")
    p.add_argument("--split-info", type=Path, default=None,
                   help="Train/validation match split. Overrides data.split_info in the config.")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/rally"),
                   help="Output directory for checkpoints, logs and sequence metadata.")
    p.add_argument("--limit-rallies", type=int, default=0,
                   help="If >0, keep only the first N rallies per split (smoke test).")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, enc_cfg, tr_cfg = cfg["data"], cfg["encoder"], cfg["training"]

    shots_csv = args.shots_csv or Path(data_cfg["train_shots"])
    split_info_path = args.split_info or Path(data_cfg["split_info"])
    for path in (shots_csv, split_info_path):
        if not path.exists():
            sys.exit(f"Missing input file: {path}. See docs/data_format.md for the expected rally inputs.")
    limit_rallies = args.limit_rallies if args.limit_rallies > 0 else None

    seed = int(data_cfg["seed"])
    set_all_seeds(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.work_dir
    dirs = _ensure_dirs(run_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s | seed: %d", device, seed)

    logger.info("Loading shots from %s", shots_csv)
    train_shots, val_shots, split_info = _load_split_shots(shots_csv, split_info_path)
    logger.info("Train shots: %d  Val shots: %d", len(train_shots), len(val_shots))

    raw_cols = select_raw_columns(train_shots)
    logger.info("Selected %d raw columns for tokens", len(raw_cols))

    min_rally_len = int(data_cfg["min_rally_len"])
    logger.info("Building TRAIN dataset ...")
    train_bundle = build_rally_dataset(
        train_shots,
        raw_cols=raw_cols,
        min_rally_len=min_rally_len,
        limit_rallies=limit_rallies,
        track_name="train",
    )
    logger.info("Building VAL dataset ...")
    val_bundle = build_rally_dataset(
        val_shots,
        raw_cols=raw_cols,
        min_rally_len=min_rally_len,
        limit_rallies=limit_rallies,
        track_name="val",
    )

    input_dim = train_bundle.token_matrix.shape[1]
    if val_bundle.token_matrix.shape[1] != input_dim:
        raise RuntimeError(
            f"Token dim mismatch: train={input_dim} val={val_bundle.token_matrix.shape[1]}"
        )

    max_len = int(data_cfg["max_len"])
    column_groups = train_bundle.column_groups
    max_seq_train = max(s.shape[0] for s in train_bundle.seq_arrays)
    max_seq_val = max(s.shape[0] for s in val_bundle.seq_arrays)
    max_seq = max(max_seq_train, max_seq_val)
    if max_seq > max_len:
        raise RuntimeError(
            f"Observed max sequence length {max_seq} > data.max_len={max_len}. Increase data.max_len."
        )

    logger.info(
        "input_dim=%d train_rallies=%d val_rallies=%d max_seq=%d",
        input_dim,
        len(train_bundle.seq_arrays),
        len(val_bundle.seq_arrays),
        max_seq,
    )

    aug_kwargs = dict(column_groups=column_groups, **tr_cfg["augmentation"])
    train_ds = RallyTripletDataset(
        train_bundle.seq_arrays,
        DomainAwareSequenceAugmenter(**aug_kwargs),
        seed=seed,
        deterministic=False,
    )
    val_ds = RallyTripletDataset(
        val_bundle.seq_arrays,
        DomainAwareSequenceAugmenter(**aug_kwargs),
        seed=seed + 777,
        deterministic=True,
    )

    batch_size = int(tr_cfg["batch_size"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_triplet,
        num_workers=0,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_triplet,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Train loader has zero batches; increase data or lower training.batch_size.")

    model = RallySSLModel(
        input_dim=input_dim,
        d_model=int(enc_cfg["d_model"]),
        n_layers=int(enc_cfg["n_layers"]),
        n_heads=int(enc_cfg["n_heads"]),
        dropout=float(enc_cfg["dropout"]),
        pooling=str(enc_cfg["pooling"]),
        max_len=max_len,
        proj_dim=int(enc_cfg["proj_dim"]),
        encoder_block=str(enc_cfg["encoder_block"]),
        modern_use_gated_mha=bool(enc_cfg.get("modern_use_gated_mha", True)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=float(tr_cfg["lr"]), weight_decay=float(tr_cfg["weight_decay"])
    )
    epoch_kwargs = dict(
        device=device,
        temperature=float(tr_cfg["temperature"]),
        grad_clip=float(tr_cfg["grad_clip"]),
        order_loss_lambda=float(tr_cfg["order_loss_lambda"]),
        order_margin=float(tr_cfg["order_margin"]),
    )
    ckpt_kwargs = dict(
        cfg=cfg,
        raw_cols=raw_cols,
        token_columns=train_bundle.token_columns,
        column_groups=column_groups,
        input_dim=input_dim,
        dataset_stats={"train": train_bundle.stats, "val": val_bundle.stats},
    )

    best_val = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(tr_cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, **epoch_kwargs)
        val_metrics = run_epoch(model, val_loader, optimizer=None, **epoch_kwargs)

        history.append({
            "epoch": int(epoch),
            "train_contrastive": train_metrics["contrastive"],
            "train_order": train_metrics["order"],
            "train_combined": train_metrics["combined"],
            "val_contrastive": val_metrics["contrastive"],
            "val_order": val_metrics["order"],
            "val_combined": val_metrics["combined"],
        })

        print(
            f"[Epoch {epoch:03d}] "
            f"train: c={train_metrics['contrastive']:.4f} o={train_metrics['order']:.4f} "
            f"tot={train_metrics['combined']:.4f} | "
            f"val: c={val_metrics['contrastive']:.4f} o={val_metrics['order']:.4f} "
            f"tot={val_metrics['combined']:.4f}"
        )

        val_loss = val_metrics["combined"]
        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            _save_checkpoint(
                model, dirs["models"] / "best.pt",
                best_epoch=best_epoch, best_val_loss=best_val, **ckpt_kwargs,
            )

    _save_checkpoint(
        model, dirs["models"] / "last.pt",
        best_epoch=best_epoch, best_val_loss=best_val, **ckpt_kwargs,
    )

    with (dirs["logs"] / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(history[0].keys()) if history else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in history:
            w.writerow(r)

    train_bundle.seq_meta.to_csv(dirs["data"] / "train_sequence_metadata.csv", index=False)
    val_bundle.seq_meta.to_csv(dirs["data"] / "val_sequence_metadata.csv", index=False)
    train_bundle.transitions_df.to_csv(dirs["data"] / "train_transitions.csv", index=False)
    val_bundle.transitions_df.to_csv(dirs["data"] / "val_transitions.csv", index=False)

    _write_json(
        dirs["run"] / "token_columns.json",
        {
            "raw_columns": raw_cols,
            "token_columns": train_bundle.token_columns,
            "n_raw_columns": len(raw_cols),
            "n_token_dims": input_dim,
            "column_groups": column_groups_to_json(column_groups),
        },
    )
    _write_json(
        dirs["run"] / "dataset_stats.json",
        {
            "train": train_bundle.stats,
            "val": val_bundle.stats,
        },
    )
    _write_json(
        dirs["run"] / "run_config.json",
        {
            "timestamp": timestamp,
            "config": cfg,
            "split_info_summary": split_info,
            "n_train_rallies": len(train_bundle.seq_arrays),
            "n_val_rallies": len(val_bundle.seq_arrays),
            "input_dim": input_dim,
            "max_seq_train": max_seq_train,
            "max_seq_val": max_seq_val,
            "n_model_params": n_params,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
    )

    logger.info("Training complete. Best epoch=%d best_val_loss=%.6f", best_epoch, best_val)
    logger.info("Run dir: %s", run_dir)


if __name__ == "__main__":
    main()

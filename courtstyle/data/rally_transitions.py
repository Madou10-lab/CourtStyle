"""Rally shot-transition sequences.

Builds one ordered sequence per rally from the preprocessed shot table. Each
token describes one transition between consecutive shots and concatenates three
blocks over the same 31 curated shot variables:

    [state (current shot), action (next shot), delta (next - current)]

giving 93 dimensions per token. Consecutive shots attributed to the same player
are skipped, and rallies with fewer than two transitions are dropped.

Identity and confound columns (match, player, handedness) are never part of the
token vector. The module also exposes the column groups used by the domain-aware
augmentations: court x and y coordinates for axis flips, and player/opponent
column pairs.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pandas is required for this pipeline. Install it in your environment "
            "(e.g., conda install pandas)."
        ) from e
    return pd


def _extract_raw_player_id(value: Any) -> Optional[int]:
    # Same parsing as courtstyle.data.rally_dataset: "(2, 1)" -> 2
    if value is None:
        return None
    try:
        if value != value:  # NaN check
            return None
    except Exception:
        pass

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, tuple) and parsed:
            return int(parsed[0])
    except Exception:
        pass

    s = str(value).strip().lstrip("(")
    for sep in [",", " "]:
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    try:
        return int(s)
    except Exception:
        return None


def build_base_transitions(
    track_shots_df: Any,
    track_name: str,
    trim_start: int,
    trim_end: int,
    match_player_name_map: Optional[Dict[Tuple[str, int], str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    pd = _require_pandas()

    required = ["match_id", "point_index", "shot_index", "player", "player_opponent"]
    missing = [c for c in required if c not in track_shots_df.columns]
    if missing:
        raise ValueError(f"Missing required shot metadata columns: {missing}")

    rows: List[Dict[str, Any]] = []
    n_rallies_total = track_shots_df.groupby(["match_id", "point_index"]).ngroups
    n_rallies_after_trim_lt2 = 0
    n_skipped_same_player = 0

    for (mid, pidx), group in track_shots_df.groupby(["match_id", "point_index"], sort=False):
        group = group.sort_values("shot_index", kind="mergesort")
        rally_len_raw = len(group)

        if trim_end > 0:
            trimmed = group.iloc[trim_start : max(rally_len_raw - trim_end, trim_start)]
        else:
            trimmed = group.iloc[trim_start:]

        rally_len_after = len(trimmed)
        if rally_len_after < 2:
            n_rallies_after_trim_lt2 += 1
            continue

        # Build consecutive opponent->response transitions
        for i in range(rally_len_after - 1):
            state_row = trimmed.iloc[i]
            action_row = trimmed.iloc[i + 1]

            state_player = state_row["player"]
            action_player = action_row["player"]

            if state_player == action_player:
                n_skipped_same_player += 1
                continue

            row: Dict[str, Any] = {
                "match_id": str(mid),
                "point_index": _safe_int(pidx),
                "state_shot_index": _safe_int(state_row["shot_index"]),
                "action_shot_index": _safe_int(action_row["shot_index"]),
                "state_player": str(state_player),
                "action_player": str(action_player),
                # Source keys for action shot (for negative controls; equals state rally in baseline)
                "action_match_id": str(mid),
                "action_point_index": _safe_int(pidx),
                "rally_len_raw": int(rally_len_raw),
                "rally_len_after_trim": int(rally_len_after),
                "track_name": track_name,
            }

            if match_player_name_map:
                state_pid = _extract_raw_player_id(state_player)
                action_pid = _extract_raw_player_id(action_player)
                row["state_player_name"] = (
                    match_player_name_map.get((str(mid), int(state_pid))) if state_pid is not None else None
                )
                row["action_player_name"] = (
                    match_player_name_map.get((str(mid), int(action_pid))) if action_pid is not None else None
                )

            rows.append(row)

    df = pd.DataFrame(rows)
    counters = {
        "n_rallies_total": int(n_rallies_total),
        "n_rallies_after_trim_lt2": int(n_rallies_after_trim_lt2),
        "n_skipped_same_player": int(n_skipped_same_player),
        "n_transitions": int(len(df)),
    }
    return df, counters



def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        sv = str(v)
        if sv in seen:
            continue
        seen.add(sv)
        out.append(sv)
    return out


def _select_interpretable_shot_columns(shots_df: Any) -> Dict[str, Any]:
    cols = [str(c) for c in shots_df.columns.tolist()]
    col_set = set(cols)

    # Primary curated numeric set requested by design.
    preferred_numeric = [
        "speed",
        "spin_rpm",
        "net_clearance_z_debug",
        "bounce_coordinate_y",
        "bounce_coordinate_x",
        "player_position_at_hit_y",
        "player_position_at_hit_x",
        "player_opponent_position_at_hit_y",
        "player_opponent_position_at_hit_x",
        "player_position_at_bounce_y",
        "player_position_at_bounce_x",
        "player_opponent_position_at_bounce_y",
        "player_opponent_position_at_bounce_x",
        "start_pos_x",
        "start_pos_y",
        "end_pos_x",
        "end_pos_y",
        "duration",
        "distance_travelled_by_player",
    ]
    numeric_cols = [c for c in preferred_numeric if c in col_set]

    # Include additional available placement-oriented columns.
    extra_xy: List[str] = []
    for c in cols:
        lc = c.lower()
        if c in numeric_cols:
            continue
        if lc.endswith("_x") or lc.endswith("_y"):
            if any(t in lc for t in ["position", "coordinate", "pos", "bounce"]):
                extra_xy.append(c)
    numeric_cols.extend(extra_xy)

    binary_flags = [
        c
        for c in ["is_volley", "is_return", "is_groundstroke", "is_rally", "is_overhead_smash"]
        if c in col_set
    ]
    stroke_cols = [c for c in cols if c.startswith("stroke_")]
    spin_cols = [c for c in cols if c.startswith("spin_type_")]

    selected = _ordered_unique(numeric_cols + binary_flags + stroke_cols + spin_cols)

    # Keep only columns that are numeric-convertible in practice.
    pd = _require_pandas()
    numeric_convertible: List[str] = []
    for c in selected:
        s = pd.to_numeric(shots_df[c], errors="coerce")
        if s.notna().any():
            numeric_convertible.append(c)

    selected = _ordered_unique(numeric_convertible)
    return {
        "selected_columns": selected,
        "numeric_like_columns": [c for c in selected if c in set(numeric_cols + extra_xy)],
        "binary_flag_columns": [c for c in selected if c in set(binary_flags)],
        "stroke_columns": [c for c in selected if c.startswith("stroke_")],
        "spin_type_columns": [c for c in selected if c.startswith("spin_type_")],
    }


def _build_transition_raw_feature_table(
    transitions_df: Any,
    shots_df: Any,
    raw_cols: Sequence[str],
) -> Any:
    pd = _require_pandas()
    key_cols = ["match_id", "point_index", "shot_index"]

    shots = shots_df.copy()
    shots["match_id"] = shots["match_id"].astype(str)
    shots["point_index"] = shots["point_index"].apply(_safe_int)
    shots["shot_index"] = shots["shot_index"].apply(_safe_int)

    dups = shots.duplicated(subset=key_cols, keep=False)
    if bool(dups.any()):
        ex = shots.loc[dups, key_cols].head(10).to_dict("records")
        raise RuntimeError(
            f"Shot key uniqueness violated for transition join. duplicate_keys={int(dups.sum())} examples={ex}"
        )

    shots_lookup = shots.set_index(key_cols)[list(raw_cols)]

    tdf = transitions_df.copy().reset_index(drop=True)
    tdf["match_id"] = tdf["match_id"].astype(str)
    tdf["point_index"] = tdf["point_index"].apply(_safe_int)
    tdf["state_shot_index"] = tdf["state_shot_index"].apply(_safe_int)
    tdf["action_match_id"] = tdf["action_match_id"].astype(str)
    tdf["action_point_index"] = tdf["action_point_index"].apply(_safe_int)
    tdf["action_shot_index"] = tdf["action_shot_index"].apply(_safe_int)

    state_idx = pd.MultiIndex.from_frame(
        tdf[["match_id", "point_index", "state_shot_index"]].rename(
            columns={"state_shot_index": "shot_index"}
        )
    )
    action_idx = pd.MultiIndex.from_frame(
        tdf[["action_match_id", "action_point_index", "action_shot_index"]].rename(
            columns={
                "action_match_id": "match_id",
                "action_point_index": "point_index",
                "action_shot_index": "shot_index",
            }
        )
    )

    state_exists = state_idx.isin(shots_lookup.index)
    action_exists = action_idx.isin(shots_lookup.index)

    n_state_missing = int((~state_exists).sum())
    n_action_missing = int((~action_exists).sum())
    if n_state_missing > 0 or n_action_missing > 0:
        bad_state = tdf.loc[~state_exists, ["match_id", "point_index", "state_shot_index"]].head(8).to_dict("records")
        bad_action = tdf.loc[
            ~action_exists,
            ["action_match_id", "action_point_index", "action_shot_index"],
        ].head(8).to_dict("records")
        raise RuntimeError(
            "Transition->shot join failed: "
            f"state_missing={n_state_missing}, action_missing={n_action_missing}, "
            f"state_examples={bad_state}, action_examples={bad_action}"
        )

    state_raw = shots_lookup.reindex(state_idx).reset_index(drop=True)
    action_raw = shots_lookup.reindex(action_idx).reset_index(drop=True)

    state_raw = state_raw.apply(pd.to_numeric, errors="coerce").astype("float32")
    action_raw = action_raw.apply(pd.to_numeric, errors="coerce").astype("float32")
    delta_raw = (action_raw - state_raw).astype("float32")

    out_df = tdf.copy()
    for c in raw_cols:
        out_df[f"state_raw_{c}"] = state_raw[c].to_numpy(dtype=np.float32)
    for c in raw_cols:
        out_df[f"action_raw_{c}"] = action_raw[c].to_numpy(dtype=np.float32)
    for c in raw_cols:
        out_df[f"delta_raw_{c}"] = delta_raw[c].to_numpy(dtype=np.float32)

    print(
        "[raw-profile] join alignment OK: "
        f"n_transitions={len(tdf)}, state_mapped={int(state_exists.sum())}, action_mapped={int(action_exists.sum())}"
    )
    return out_df



def load_split_info(split_info_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load train/val match IDs from split_info.json.

    Args:
        split_info_path: Path to split_info.json

    Returns:
        (train_match_ids, val_match_ids)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is malformed
    """
    if not split_info_path.exists():
        raise FileNotFoundError(f"split_info.json not found at {split_info_path}")

    with split_info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    train_ids = data.get("train_match_ids", [])
    val_ids = data.get("val_match_ids", [])

    if not val_ids:
        raise ValueError(f"No val_match_ids in {split_info_path}")

    return train_ids, val_ids



logger = logging.getLogger(__name__)


EXCLUDED_FROM_TOKENS: frozenset[str] = frozenset(
    {
        "match_id",
        "point_index",
        "shot_index",
        "player",
        "player_opponent",
        "dominant_hand_R",
    }
)


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return int(float(x))


def rally_key(match_id: Any, point_index: Any) -> str:
    return f"{str(match_id)}::{_safe_int(point_index)}"


def normalize_shots_df(shots_df: pd.DataFrame) -> pd.DataFrame:
    out = shots_df.copy()
    out["match_id"] = out["match_id"].astype(str)
    out["point_index"] = out["point_index"].apply(_safe_int)
    out["shot_index"] = out["shot_index"].apply(_safe_int)
    return out


def select_raw_columns(shots_df: pd.DataFrame) -> List[str]:
    """
    Select the curated interpretable raw columns, explicitly filtering out
    identity / confound columns.

    Returns the ordered list of column names that form the *base*
    of each token dimension (before state/action/delta tripling).
    """
    info = _select_interpretable_shot_columns(shots_df)
    raw_cols = [c for c in info["selected_columns"] if c not in EXCLUDED_FROM_TOKENS]

    # Hard-fail if any excluded column leaked through
    leaked = [c for c in raw_cols if c in EXCLUDED_FROM_TOKENS]
    if leaked:
        raise RuntimeError(
            f"Identity or confound columns leaked into token columns: {leaked}"
        )

    if not raw_cols:
        raise RuntimeError("select_raw_columns produced zero columns.")

    logger.info("Token base columns (%d): %s", len(raw_cols), raw_cols)
    return raw_cols


def build_column_groups(raw_cols: List[str]) -> Dict[str, Any]:
    """
    Compute column-group metadata for domain-aware augmentation.

    Returns a dict with:
        n_raw_cols:                      int
        n_token_dims:                    int (= 3 * n_raw_cols)
        x_raw_indices:                   list[int]  (indices into raw_cols)
        y_raw_indices:                   list[int]
        x_token_indices:                 list[int]  (indices into the 3N token vector)
        y_token_indices:                 list[int]
        player_opponent_swap_raw_pairs:  list[tuple[int,int]]
        player_opponent_swap_token_pairs: list[tuple[int,int]]
        raw_columns:                     list[str]
    """
    n = len(raw_cols)

    # ── x / y feature indices ────────────────────────────────────────────
    x_raw = [i for i, c in enumerate(raw_cols) if c.endswith("_x")]
    y_raw = [i for i, c in enumerate(raw_cols) if c.endswith("_y")]

    def _expand_to_token(raw_indices: List[int]) -> List[int]:
        out: List[int] = []
        for i in raw_indices:
            out.extend([i, n + i, 2 * n + i])  # state, action, delta
        return out

    x_tok = _expand_to_token(x_raw)
    y_tok = _expand_to_token(y_raw)

    # ── player / opponent swap pairs ─────────────────────────────────────
    player_prefix = "player_position_"
    opponent_prefix = "player_opponent_position_"

    # Build index lookups (be careful NOT to match opponent_ under player_)
    player_map: Dict[str, int] = {}
    opponent_map: Dict[str, int] = {}
    for i, c in enumerate(raw_cols):
        if c.startswith(opponent_prefix):
            opponent_map[c] = i
        elif c.startswith(player_prefix):
            player_map[c] = i

    swap_raw: List[Tuple[int, int]] = []
    for pcol, pidx in sorted(player_map.items()):
        # player_position_at_hit_x -> player_opponent_position_at_hit_x
        ocol = pcol.replace("player_position_", "player_opponent_position_", 1)
        if ocol in opponent_map:
            swap_raw.append((pidx, opponent_map[ocol]))

    swap_tok: List[Tuple[int, int]] = []
    for pi, oi in swap_raw:
        swap_tok.append((pi, oi))                  # state
        swap_tok.append((n + pi, n + oi))          # action
        swap_tok.append((2 * n + pi, 2 * n + oi))  # delta

    groups: Dict[str, Any] = {
        "n_raw_cols": n,
        "n_token_dims": 3 * n,
        "x_raw_indices": x_raw,
        "y_raw_indices": y_raw,
        "x_token_indices": x_tok,
        "y_token_indices": y_tok,
        "player_opponent_swap_raw_pairs": [list(p) for p in swap_raw],
        "player_opponent_swap_token_pairs": [list(p) for p in swap_tok],
        "raw_columns": list(raw_cols),
    }
    logger.info(
        "Column groups: %d x-features, %d y-features, %d swap pairs (token-level)",
        len(x_tok), len(y_tok), len(swap_tok),
    )
    return groups


@dataclass
class RallyDatasetBundle:
    """All data artefacts for one split."""
    shots_df: pd.DataFrame
    transitions_df: pd.DataFrame
    token_matrix: np.ndarray
    token_columns: List[str]
    raw_columns: List[str]
    column_groups: Dict[str, Any]
    seq_arrays: List[np.ndarray]
    seq_meta: pd.DataFrame
    stats: Dict[str, Any]


def _limit_rallies_df(
    shots_df: pd.DataFrame,
    limit_rallies: Optional[int],
) -> pd.DataFrame:
    if limit_rallies is None or int(limit_rallies) <= 0:
        return shots_df
    keys = (
        shots_df[["match_id", "point_index"]]
        .drop_duplicates()
        .sort_values(["match_id", "point_index"], ascending=[True, True], kind="mergesort")
        .head(int(limit_rallies))
    )
    return shots_df.merge(keys, on=["match_id", "point_index"], how="inner")


def _build_rally_sequences(
    transitions_df: pd.DataFrame,
    token_matrix: np.ndarray,
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    """
    Group per-transition token rows into variable-length per-rally sequences.

    Returns (seq_arrays, seq_meta) where seq_meta has columns:
        rally_id, match_id, point_index, n_transitions, n_shots
    """
    if len(transitions_df) != token_matrix.shape[0]:
        raise ValueError(
            f"Row mismatch: transitions={len(transitions_df)} tokens={token_matrix.shape[0]}"
        )
    for col in ("transition_index", "rally_id"):
        if col not in transitions_df.columns:
            raise ValueError(f"transitions_df missing column: {col}")

    tdf = transitions_df.copy().reset_index(drop=True)
    tdf["_tok_row"] = np.arange(len(tdf), dtype=int)
    tdf = tdf.sort_values(["rally_id", "transition_index"], kind="mergesort").reset_index(drop=True)

    seqs: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    for rid, g in tdf.groupby("rally_id", sort=False):
        idx = g["_tok_row"].to_numpy(dtype=int)
        seq = token_matrix[idx].astype(np.float32, copy=False)
        seqs.append(seq)
        n_shots = int(g["rally_len_raw"].iloc[0]) if "rally_len_raw" in g.columns else int(len(g) + 1)
        rows.append(
            {
                "rally_id": str(rid),
                "match_id": str(g["match_id"].iloc[0]),
                "point_index": _safe_int(g["point_index"].iloc[0]),
                "n_transitions": int(len(g)),
                "n_shots": n_shots,
            }
        )

    meta_df = pd.DataFrame(rows)

    # Sanity: rally_id uniqueness
    if meta_df["rally_id"].duplicated().any():
        raise RuntimeError("Duplicate rally_id detected in sequence metadata.")

    return seqs, meta_df


def build_rally_dataset(
    shots_df: pd.DataFrame,
    *,
    raw_cols: List[str],
    min_rally_len: int = 2,
    limit_rallies: Optional[int] = None,
    track_name: str = "rally",
) -> RallyDatasetBundle:
    """
    End-to-end pipeline: shots → transitions → token matrix → per-rally sequences.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level processed dataframe (must contain meta + feature columns).
    raw_cols : list[str]
        Curated raw column names (from ``select_raw_columns``).
    min_rally_len : int
        Minimum rally length (in shots) to include.
    limit_rallies : int | None
        If >0, keep only the first N rallies (for debugging / dry-run).
    track_name : str
        Name passed to ``build_base_transitions``.

    Returns
    -------
    RallyDatasetBundle
    """
    required = ["match_id", "point_index", "shot_index", "player", "player_opponent"]
    missing = [c for c in required if c not in shots_df.columns]
    if missing:
        raise ValueError(f"shots_df missing required columns: {missing}")

    shots_norm = normalize_shots_df(shots_df)

    # ── filter by rally length ───────────────────────────────────────────
    rally_lens = (
        shots_norm.groupby(["match_id", "point_index"], sort=False)
        .size()
        .reset_index(name="rally_len")
    )
    keep = rally_lens[rally_lens["rally_len"] >= int(min_rally_len)][["match_id", "point_index"]]
    kept = shots_norm.merge(keep, on=["match_id", "point_index"], how="inner")
    kept = _limit_rallies_df(kept, limit_rallies=limit_rallies).reset_index(drop=True)

    n_rallies_kept = kept.groupby(["match_id", "point_index"]).ngroups
    logger.info(
        "Rally dataset [%s]: %d rallies kept (min_rally_len=%d), %d shots",
        track_name, n_rallies_kept, min_rally_len, len(kept),
    )

    # ── transitions ──────────────────────────────────────────────────────
    transitions_df, counters = build_base_transitions(
        track_shots_df=kept,
        track_name=track_name,
        trim_start=0,
        trim_end=0,
        match_player_name_map=None,
    )
    transitions_df = transitions_df.copy().reset_index(drop=True)
    if len(transitions_df) == 0:
        raise RuntimeError(f"Rally dataset [{track_name}]: build_base_transitions produced zero transitions.")

    transitions_df["match_id"] = transitions_df["match_id"].astype(str)
    transitions_df["point_index"] = transitions_df["point_index"].apply(_safe_int)
    transitions_df["state_shot_index"] = transitions_df["state_shot_index"].apply(_safe_int)
    transitions_df["action_shot_index"] = transitions_df["action_shot_index"].apply(_safe_int)
    transitions_df["action_match_id"] = transitions_df["action_match_id"].astype(str)
    transitions_df["action_point_index"] = transitions_df["action_point_index"].apply(_safe_int)
    transitions_df = transitions_df.sort_values(
        ["match_id", "point_index", "state_shot_index", "action_shot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    transitions_df["transition_index"] = (
        transitions_df.groupby(["match_id", "point_index"], sort=False).cumcount()
    )
    transitions_df["rally_id"] = [
        rally_key(m, p)
        for m, p in zip(transitions_df["match_id"], transitions_df["point_index"])
    ]

    logger.info(
        "Rally dataset [%s]: %d transitions from %d rallies",
        track_name, len(transitions_df), transitions_df["rally_id"].nunique(),
    )

    # ── token matrix [state_raw, action_raw, delta_raw] ──────────────────
    # Verify raw_cols don't contain excluded columns
    leaked = [c for c in raw_cols if c in EXCLUDED_FROM_TOKENS]
    if leaked:
        raise RuntimeError(f"raw_cols contain excluded columns: {leaked}")

    raw_df = _build_transition_raw_feature_table(
        transitions_df, shots_df=kept, raw_cols=raw_cols
    )
    state_cols = [f"state_raw_{c}" for c in raw_cols]
    action_cols = [f"action_raw_{c}" for c in raw_cols]
    delta_cols = [f"delta_raw_{c}" for c in raw_cols]
    token_columns = state_cols + action_cols + delta_cols

    token_matrix = raw_df[token_columns].to_numpy(dtype=np.float32)
    # Replace NaN / inf with 0 to avoid downstream issues
    token_matrix = np.nan_to_num(token_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if token_matrix.shape[0] != len(transitions_df):
        raise RuntimeError(
            f"Token matrix rows ({token_matrix.shape[0]}) != transitions ({len(transitions_df)})"
        )

    # ── column groups ────────────────────────────────────────────────────
    column_groups = build_column_groups(raw_cols)

    # ── per-rally sequences ──────────────────────────────────────────────
    seq_arrays, seq_meta = _build_rally_sequences(transitions_df, token_matrix)

    # ── stats ────────────────────────────────────────────────────────────
    trans_per_rally = transitions_df.groupby("rally_id", sort=False).size()
    stats: Dict[str, Any] = {
        "track_name": track_name,
        "n_shots_input": int(len(shots_norm)),
        "n_shots_kept": int(len(kept)),
        "n_rallies_kept": int(n_rallies_kept),
        "n_transitions": int(len(transitions_df)),
        "n_sequences": int(len(seq_arrays)),
        "n_token_dims": int(token_matrix.shape[1]),
        "n_raw_cols": int(len(raw_cols)),
        "min_rally_len": int(min_rally_len),
        "transitions_per_rally_mean": float(trans_per_rally.mean()),
        "transitions_per_rally_max": int(trans_per_rally.max()),
        "build_counters": counters,
    }

    return RallyDatasetBundle(
        shots_df=kept,
        transitions_df=transitions_df,
        token_matrix=token_matrix,
        token_columns=token_columns,
        raw_columns=raw_cols,
        column_groups=column_groups,
        seq_arrays=seq_arrays,
        seq_meta=seq_meta,
        stats=stats,
    )


def column_groups_to_json(column_groups: Dict[str, Any]) -> Dict[str, Any]:
    """Convert column_groups to a JSON-safe dict (lists, not tuples/ndarray)."""
    out: Dict[str, Any] = {}
    for k, v in column_groups.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [
                item.tolist() if isinstance(item, np.ndarray) else
                list(item) if isinstance(item, tuple) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


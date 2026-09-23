"""Rally shot preprocessing.

Input is one row per tracked shot from Hawk-Eye logs, with a directory of
per-match point tables that provide point outcomes and player handedness.

* Matches are split whole into training and test partitions
  (``GroupShuffleSplit`` on ``match_id``).
* Player identifiers are kept as metadata and never used as features.
* Point outcomes (result, reason, break point, ...) are written to sidecar files
  and used only for profiling and validation.
* Missing stroke types are imputed by a random forest fitted on the training
  matches. Numeric features are imputed and standardised with statistics fitted
  on the training matches only.
* Shot outputs keep ``match_id``, ``point_index`` and ``shot_index`` so that
  rally sequences can be reconstructed.

Usage:
    python -m courtstyle.data.rally_dataset \\
        --input data/rally/shots_merged.csv --points-dir data/rally/points \\
        --output-dir data/rally
"""
from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = Path("data/rally/shots_merged.csv")
DEFAULT_POINTS_DIR = Path("data/rally/points")
DEFAULT_OUTPUT_DIR = Path("data/rally")
DEFAULT_PREPROCESSOR_PATH = Path("data/rally/preprocessor.pkl")


# =============================================================================
# FEATURE CONTRACT - Pattern-first clustering
# =============================================================================
# Metadata columns: kept for rally reconstruction and attribution analysis, NOT model features.
# Player IDs are metadata because communities should represent patterns, not "who played".
META_COLS = [
    "match_id",
    "point_index",
    "shot_index",
    "player",           # Metadata only - for attribution analysis
    "player_opponent",  # Metadata only - for attribution analysis
]

# Model feature columns: 39 features (original 41 minus player and player_opponent).
# These are the ONLY columns used for embedding/clustering.
FEATURE_COLS = [
    "speed",
    "spin_rpm",
    "is_return",
    "is_rally",
    "is_volley",
    "is_groundstroke",
    "hit_peak_z",
    "bounce_peak_z",
    "start_pos_x",
    "start_pos_y",
    "start_pos_z",
    "start_speed",
    "end_pos_x",
    "end_pos_y",
    "ball_acceleration",
    "bounce_coordinate_x",
    "bounce_coordinate_y",
    "bounce",
    "rebound_net",
    "player_position_at_hit_x",
    "player_position_at_hit_y",
    "player_opponent_position_at_hit_x",
    "player_opponent_position_at_hit_y",
    "player_position_at_bounce_x",
    "player_position_at_bounce_y",
    "player_opponent_position_at_bounce_x",
    "player_opponent_position_at_bounce_y",
    "duration",
    "net_clearance_z_debug",
    "is_overhead_smash",
    "distance_travelled_by_player",
    "stroke_backhand",
    "stroke_forehand",
    "stroke_serve",
    "spin_type_back",
    "spin_type_none",
    "spin_type_top",
    "spin_type_unknown",
    "dominant_hand_R",
]

# Numeric columns to be (median) imputed + StandardScaled.
NUMERIC_TO_SCALE = [
    "speed",
    "spin_rpm",
    "hit_peak_z",
    "bounce_peak_z",
    "start_pos_x",
    "start_pos_y",
    "start_pos_z",
    "start_speed",
    "end_pos_x",
    "end_pos_y",
    "ball_acceleration",
    "bounce_coordinate_x",
    "bounce_coordinate_y",
    "player_position_at_hit_x",
    "player_position_at_hit_y",
    "player_opponent_position_at_hit_x",
    "player_opponent_position_at_hit_y",
    "player_position_at_bounce_x",
    "player_position_at_bounce_y",
    "player_opponent_position_at_bounce_x",
    "player_opponent_position_at_bounce_y",
    "duration",
    "net_clearance_z_debug",
    "distance_travelled_by_player",
]

# Fixed categorical spaces used in the original project artifacts.
STROKE_CATEGORIES = ["backhand", "forehand", "serve"]
SPIN_TYPE_CATEGORIES = ["back", "none", "top", "unknown"]


# =============================================================================
# RALLY-LEVEL OUTCOME COLUMNS (Sidecar Strategy)
# =============================================================================
# These columns are extracted from points CSVs for validation/interpretation.
# They are saved separately and NEVER included in model features.
OUTCOME_COLS = [
    "match_id",         # Join key
    "point_index",      # Join key
    "point_name",       # Descriptive identifier
    "result",           # Point result (Played, Ace, DoubleFault, etc.)
    "reason",           # Reason for point ending
    "is_match_point",
    "is_set_point",
    "is_break_point",
    "is_deciding_point",
    "tie_break",
    "final_set",
    "final_set_tie_break",
    "conclusion_stroke",
    "conclusion_result",
    "conclusion_event_ID",
    "conclusion_action",
    "num_shots",        # Rally length - useful for tactical interpretation
    "duration",         # Point duration
]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _extract_player_id(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, float) and value.is_integer():
        return float(value)
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, tuple) and parsed:
            return float(parsed[0])
    except Exception:
        pass
    # Fallback: try regex-free parsing for formats like "(2, 1)"
    s = str(value).strip().lstrip("(")
    for sep in [",", " "]:
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    try:
        return float(int(s))
    except Exception:
        return np.nan


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    s = series.astype("string").str.strip().str.lower()
    mapped = s.map({"true": True, "false": False, "1": True, "0": False})
    return mapped.astype("boolean")


def _extract_distances_from_string(val: str) -> Tuple[float, float]:
    try:
        d = ast.literal_eval(val)
    except Exception:
        return (np.nan, np.nan)
    if not isinstance(d, dict):
        return (np.nan, np.nan)
    dist_by_1 = np.nan
    dist_by_2 = np.nan
    for key, value in d.items():
        try:
            player_id = key[0] if isinstance(key, tuple) and len(key) > 0 else None
        except Exception:
            player_id = None
        if player_id == 1:
            dist_by_1 = value
        elif player_id == 2:
            dist_by_2 = value
    try:
        dist_by_1 = float(dist_by_1) if dist_by_1 is not None else np.nan
    except Exception:
        dist_by_1 = np.nan
    try:
        dist_by_2 = float(dist_by_2) if dist_by_2 is not None else np.nan
    except Exception:
        dist_by_2 = np.nan
    return (dist_by_1, dist_by_2)


def parse_distance_travelled_by_player(distance_series: pd.Series, player_id: pd.Series) -> pd.Series:
    """
    Per-shot distance covered by the hitting player:
    - `distance_travelled` is a dict-string like "{(2, 1): 3.124, (1, 1): 4.171}"
    - Extract (player 1 distance, player 2 distance) per row
    - Select by current `player` (1 or 2)
    """
    distance_str = distance_series.astype("string")
    unique_vals = distance_str.dropna().unique().tolist()

    dist1_map: Dict[str, float] = {}
    dist2_map: Dict[str, float] = {}
    for val in unique_vals:
        d1, d2 = _extract_distances_from_string(str(val))
        dist1_map[str(val)] = d1
        dist2_map[str(val)] = d2

    d1 = distance_str.map(dist1_map)
    d2 = distance_str.map(dist2_map)
    out = np.where(player_id == 1, d1, np.where(player_id == 2, d2, np.nan))
    return pd.to_numeric(pd.Series(out, index=distance_series.index), errors="coerce")


def load_dominant_hand_map(points_dir: Path) -> Dict[str, str]:
    """
    Infer each player's dominant hand from the per-match point tables:
      - server/receiver player_id (1/2) + player_1_handeness/player_2_handeness
      - majority vote per player_name gives a stable mapping.
    """
    if not points_dir.exists():
        return {}

    votes: Dict[str, Dict[str, int]] = {}

    def vote(player_name: str, hand: str) -> None:
        if not player_name or hand not in {"R", "L"}:
            return
        votes.setdefault(player_name, {})
        votes[player_name][hand] = votes[player_name].get(hand, 0) + 1

    point_files = sorted(p for p in points_dir.iterdir() if p.suffix == ".csv" and p.name.endswith(".points.csv"))
    for file_path in point_files:
        df = pd.read_csv(file_path, low_memory=False)
        required_cols = {
            "server",
            "receiver",
            "server_name",
            "receiver_name",
            "player_1_handeness",
            "player_2_handeness",
        }
        if not required_cols.issubset(df.columns):
            continue

        server_id = df["server"].apply(_extract_player_id)
        receiver_id = df["receiver"].apply(_extract_player_id)
        hand1 = (
            df["player_1_handeness"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"right": "R", "left": "L"})
        )
        hand2 = (
            df["player_2_handeness"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"right": "R", "left": "L"})
        )

        for name, pid, h1, h2 in zip(df["server_name"], server_id, hand1, hand2):
            hand = h1 if pid == 1 else h2 if pid == 2 else None
            vote(str(name), str(hand) if hand is not None else "")

        for name, pid, h1, h2 in zip(df["receiver_name"], receiver_id, hand1, hand2):
            hand = h1 if pid == 1 else h2 if pid == 2 else None
            vote(str(name), str(hand) if hand is not None else "")

    dominant_hand: Dict[str, str] = {}
    for player_name, counts in votes.items():
        if counts:
            dominant_hand[player_name] = max(counts.items(), key=lambda kv: kv[1])[0]
    return dominant_hand


def load_points_outcomes(points_dir: Path) -> pd.DataFrame:
    """
    Load rally-level outcome data from all points CSV files.

    This function extracts outcome columns (result, reason, is_break_point, etc.) from
    points files for use as sidecars in validation/interpretation. These outcomes are
    NOT model features - they are saved separately to prevent leakage.

    Join key: (match_id, point_index)
    - match_id is parsed from the filename prefix (e.g., "SM004_..." -> "SM004")
    - point_index comes from the 'index' column in points CSV

    Args:
        points_dir: Path to directory containing *.points.csv files

    Returns:
        DataFrame with one row per (match_id, point_index), containing outcome columns
    """
    if not points_dir.exists():
        warnings.warn(f"Points directory not found: {points_dir}. Outcomes will be empty.")
        return pd.DataFrame(columns=OUTCOME_COLS)

    point_files = sorted(p for p in points_dir.iterdir() if p.suffix == ".csv" and p.name.endswith(".points.csv"))
    if not point_files:
        warnings.warn(f"No *.points.csv files found in {points_dir}. Outcomes will be empty.")
        return pd.DataFrame(columns=OUTCOME_COLS)

    all_outcomes: List[pd.DataFrame] = []

    for file_path in point_files:
        # Parse match_id from filename: "<match_id>_<...>.points.csv" -> "<match_id>"
        match_id = file_path.name.split("_")[0]

        df = pd.read_csv(file_path, low_memory=False)

        # Ensure 'index' column exists (maps to point_index in shots)
        if "index" not in df.columns:
            warnings.warn(f"Skipping {file_path.name}: missing 'index' column")
            continue

        # Build outcome row - create with proper length
        n_rows = len(df)
        outcome_df = pd.DataFrame()
        outcome_df["match_id"] = [match_id] * n_rows  # Repeat match_id for all rows
        outcome_df["point_index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")

        # Map columns from points CSV to outcome columns
        col_mapping = {
            "name": "point_name",
            "result": "result",
            "reason": "reason",
            "is_match_point": "is_match_point",
            "is_set_point": "is_set_point",
            "is_break_point": "is_break_point",
            "is_deciding_point": "is_deciding_point",
            "tie_break": "tie_break",
            "final_set": "final_set",
            "final_set_tie_break": "final_set_tie_break",
            "conclusion_stroke": "conclusion_stroke",
            "conclusion_result": "conclusion_result",
            "conclusion_event_ID": "conclusion_event_ID",
            "conclusion_action": "conclusion_action",
            "num_shots": "num_shots",
            "duration": "duration",
        }

        for src_col, dst_col in col_mapping.items():
            if src_col in df.columns:
                outcome_df[dst_col] = df[src_col].values
            else:
                outcome_df[dst_col] = pd.NA

        all_outcomes.append(outcome_df)

    if not all_outcomes:
        return pd.DataFrame(columns=OUTCOME_COLS)

    outcomes = pd.concat(all_outcomes, ignore_index=True)

    # Validate uniqueness on (match_id, point_index)
    key_cols = ["match_id", "point_index"]
    duplicates = outcomes.duplicated(subset=key_cols, keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        warnings.warn(
            f"Found {dup_count} duplicate (match_id, point_index) rows in outcomes. "
            "Keeping first occurrence only."
        )
        outcomes = outcomes.drop_duplicates(subset=key_cols, keep="first")

    # Ensure all expected columns present (fill missing with NA)
    for col in OUTCOME_COLS:
        if col not in outcomes.columns:
            outcomes[col] = pd.NA

    # Reorder to match OUTCOME_COLS
    outcomes = outcomes[OUTCOME_COLS]

    return outcomes


def fit_stroke_imputer(train_df: pd.DataFrame, *, random_state: int) -> Pipeline | None:
    """
    Impute missing non-serve `stroke` labels (forehand/backhand):
      - a RandomForestClassifier fitted on the training matches only, to prevent leakage.
    """
    # We only need to infer non-serve strokes.
    known = train_df["stroke"].isin(["forehand", "backhand"])
    if known.sum() == 0:
        return None

    feature_cols = [
        "player_name",
        "start_speed",
        "spin_rpm",
        "spin_type",
        "speed",
        "ball_acceleration",
        "event_shot",
        "dominant_hand",
    ]
    categorical_cols = ["player_name", "dominant_hand", "spin_type", "event_shot"]
    numeric_cols = ["start_speed", "spin_rpm", "speed", "ball_acceleration"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ]
    )

    # Use single-process RF by default for portability (some environments restrict joblib multiprocessing).
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
    model = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    model.fit(train_df.loc[known, feature_cols], train_df.loc[known, "stroke"])
    return model


def apply_stroke_imputer(df: pd.DataFrame, stroke_model: Pipeline | None) -> pd.DataFrame:
    df = df.copy()
    if stroke_model is None:
        return df

    unknown = df["stroke"].isna() & (df["is_serve"] == False)
    if unknown.sum() == 0:
        return df

    feature_cols = [
        "player_name",
        "start_speed",
        "spin_rpm",
        "spin_type",
        "speed",
        "ball_acceleration",
        "event_shot",
        "dominant_hand",
    ]
    df.loc[unknown, "stroke"] = stroke_model.predict(df.loc[unknown, feature_cols])
    return df


def one_hot_fixed(series: pd.Series, *, prefix: str, categories: list[str]) -> pd.DataFrame:
    s = series.astype("string").str.strip().str.lower()
    s = s.fillna(categories[-1])
    s = s.where(s.isin(categories), categories[-1])
    cat = pd.Series(pd.Categorical(s, categories=categories), index=series.index)
    dummies = pd.get_dummies(cat, prefix=prefix)
    expected_cols = [f"{prefix}_{c}" for c in categories]
    return dummies.reindex(columns=expected_cols, fill_value=0).astype(int)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 39-column shot feature matrix. Player identifiers are excluded.

    Player IDs are kept as metadata in META_COLS but are NOT included in the
    feature matrix used for embedding/clustering. This ensures communities
    represent rally patterns/archetypes rather than "who played".
    """
    df = df.copy()

    # Ensure base fields are normalized.
    df["spin_type"] = df["spin_type"].astype("string").str.strip().str.lower().fillna("unknown")
    df.loc[~df["spin_type"].isin(SPIN_TYPE_CATEGORIES), "spin_type"] = "unknown"

    df["stroke"] = df["stroke"].astype("string").str.strip().str.lower()
    df.loc[~df["stroke"].isin(STROKE_CATEGORIES), "stroke"] = pd.NA

    # One-hot features.
    stroke_oh = one_hot_fixed(df["stroke"], prefix="stroke", categories=STROKE_CATEGORIES)
    spin_oh = one_hot_fixed(df["spin_type"], prefix="spin_type", categories=SPIN_TYPE_CATEGORIES)

    dominant_hand = df["dominant_hand"].astype("string").str.strip().str.upper()
    dominant_hand = dominant_hand.where(dominant_hand.isin(["R", "L"]), "R")

    out = pd.DataFrame(index=df.index)

    # Numeric/base columns (same as before, minus player IDs).
    for col in [
        "speed",
        "spin_rpm",
        "hit_peak_z",
        "bounce_peak_z",
        "start_pos_x",
        "start_pos_y",
        "start_pos_z",
        "start_speed",
        "end_pos_x",
        "end_pos_y",
        "ball_acceleration",
        "bounce_coordinate_x",
        "bounce_coordinate_y",
        "player_position_at_hit_x",
        "player_position_at_hit_y",
        "player_opponent_position_at_hit_x",
        "player_opponent_position_at_hit_y",
        "player_position_at_bounce_x",
        "player_position_at_bounce_y",
        "player_opponent_position_at_bounce_x",
        "player_opponent_position_at_bounce_y",
        "duration",
        "net_clearance_z_debug",
        "distance_travelled_by_player",
    ]:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # Flags.
    out["is_return"] = df["is_return"].astype(int)
    out["is_rally"] = df["is_rally"].astype(int)
    out["is_volley"] = df["is_volley"].astype(int)
    out["is_groundstroke"] = df["is_groundstroke"].astype(int)
    out["bounce"] = df["bounce"].astype(int)
    out["rebound_net"] = df["rebound_net"].astype(int)
    out["is_overhead_smash"] = df["is_overhead_smash"].astype(int)

    out["dominant_hand_R"] = (dominant_hand == "R").astype(int)

    out = pd.concat([out, stroke_oh, spin_oh], axis=1)

    # Ensure strict order and presence (39 features, no player IDs).
    out = out.reindex(columns=FEATURE_COLS)
    if out.isna().any().any():
        # leave NaNs for numeric imputer; but non-numeric must not have NaNs
        non_numeric = [c for c in FEATURE_COLS if c not in NUMERIC_TO_SCALE]
        bad = out[non_numeric].isna().sum()
        bad = bad[bad > 0]
        if not bad.empty:
            raise ValueError(f"Non-numeric features contain NaNs (should be fully deterministic):\n{bad}")
    return out


def validate_outputs(
    train_shots: pd.DataFrame,
    test_shots: pd.DataFrame,
    train_outcomes: pd.DataFrame,
    test_outcomes: pd.DataFrame,
    train_matches: List[str],
    test_matches: List[str],
) -> None:
    """
    Run validation checks before writing outputs.
    Raises ValueError on critical errors, warns on non-critical issues.
    """
    # 1. No match_id overlap between train and test
    overlap = set(train_matches) & set(test_matches)
    if overlap:
        raise ValueError(f"Train/test match overlap detected! Overlapping matches: {overlap}")

    # 2. Validate shot-level outputs
    for name, df in [("train_shots", train_shots), ("test_shots", test_shots)]:
        # Check all META_COLS present
        missing_meta = [c for c in META_COLS if c not in df.columns]
        if missing_meta:
            raise ValueError(f"{name}: Missing metadata columns: {missing_meta}")

        # Check all FEATURE_COLS present and in order
        missing_feat = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_feat:
            raise ValueError(f"{name}: Missing feature columns: {missing_feat}")

        # Check no NaNs in FEATURE_COLS after preprocessing
        nan_counts = df[FEATURE_COLS].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            raise ValueError(f"{name}: NaNs found in feature columns after preprocessing:\n{nan_cols}")

        # Check shot_index is integer and monotonic within each rally
        for (match_id, point_idx), group in df.groupby(["match_id", "point_index"]):
            shot_indices = group["shot_index"].tolist()
            expected = list(range(len(shot_indices)))
            if shot_indices != expected:
                warnings.warn(
                    f"{name}: shot_index not monotonic 0..N-1 for ({match_id}, {point_idx}): {shot_indices}"
                )

    # 3. Validate outcome outputs
    for name, df, matches in [
        ("train_outcomes", train_outcomes, train_matches),
        ("test_outcomes", test_outcomes, test_matches),
    ]:
        # Check uniqueness on (match_id, point_index)
        key_cols = ["match_id", "point_index"]
        if df.duplicated(subset=key_cols).any():
            raise ValueError(f"{name}: Duplicate (match_id, point_index) rows found!")

        # Check all outcomes belong to correct split
        invalid_matches = set(df["match_id"].unique()) - set(matches)
        if invalid_matches:
            raise ValueError(f"{name}: Contains match_ids not in split: {invalid_matches}")

    # 4. Coverage check: warn if some rallies missing outcomes
    train_rally_keys = set(zip(train_shots["match_id"], train_shots["point_index"]))
    train_outcome_keys = set(zip(train_outcomes["match_id"], train_outcomes["point_index"]))
    missing_train = train_rally_keys - train_outcome_keys
    if missing_train:
        warnings.warn(
            f"Train: {len(missing_train)} rallies have no matching outcome (missing points files?). "
            f"Sample: {list(missing_train)[:5]}"
        )

    test_rally_keys = set(zip(test_shots["match_id"], test_shots["point_index"]))
    test_outcome_keys = set(zip(test_outcomes["match_id"], test_outcomes["point_index"]))
    missing_test = test_rally_keys - test_outcome_keys
    if missing_test:
        warnings.warn(
            f"Test: {len(missing_test)} rallies have no matching outcome (missing points files?). "
            f"Sample: {list(missing_test)[:5]}"
        )

    print("Validation passed.")


def write_train_val_split(
    train_df: pd.DataFrame,
    out_path: Path,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Split the training matches into encoder training and validation matches.

    Whole matches are held out (``GroupShuffleSplit`` on ``match_id``), so no match
    contributes rallies to both partitions. The match lists are written to
    ``split_info.json`` and read by the rally trainer and discovery.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["match_id"].values))
    fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_match_ids = list(fit_df["match_id"].unique())
    val_match_ids = list(val_df["match_id"].unique())
    overlap = set(train_match_ids) & set(val_match_ids)
    if overlap:
        raise RuntimeError(f"Match ID overlap between train/val: {overlap}")

    info: Dict[str, object] = {
        "train_match_ids": [str(mid) for mid in train_match_ids],
        "val_match_ids": [str(mid) for mid in val_match_ids],
        "val_fraction": val_fraction,
        "random_state": random_state,
        "n_train_matches": len(train_match_ids),
        "n_val_matches": len(val_match_ids),
        "n_train_shots": len(fit_df),
        "n_val_shots": len(val_df),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess CourtStyle rally shots. Player identifiers are kept as metadata, never as "
            "features, and matches are split before any imputer or scaler is fitted."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input shots CSV (shots_merged.csv).")
    parser.add_argument(
        "--points-dir",
        type=Path,
        default=DEFAULT_POINTS_DIR,
        help="Directory containing per-match points CSVs (for dominant hand + outcomes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train/test processed CSVs and outcomes.",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=DEFAULT_PREPROCESSOR_PATH,
        help="Path to save preprocessor.pkl (fitted on TRAIN only).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of matches to hold out for test.")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Fraction of training matches held out for encoder validation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split and stroke imputer.")
    args = parser.parse_args()

    print("=" * 70)
    print("RALLY DATA PREPARATION - Pattern-first clustering")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Points dir: {args.points_dir}")
    print(f"Output dir: {args.output_dir}")

    # =========================================================================
    # Step 1: Load shots data
    # =========================================================================
    df = pd.read_csv(args.input, low_memory=False)
    required = [
        "match_id",
        "point_index",
        "shot_index",
        "player",
        "player_opponent",
        "player_name",
        "player_opponent_name",
        "event_shot",
        "stroke",
        "spin_type",
        "spin_rpm",
        "speed",
        "start_speed",
        "ball_acceleration",
        "distance_travelled",
        "is_serve",
        "is_return",
        "is_rally",
        "is_volley",
        "is_groundstroke",
        "hit_peak_z",
        "bounce_peak_z",
        "start_pos_x",
        "start_pos_y",
        "start_pos_z",
        "end_pos_x",
        "end_pos_y",
        "bounce_coordinate_x",
        "bounce_coordinate_y",
        "bounce",
        "rebound_net",
        "player_position_at_hit_x",
        "player_position_at_hit_y",
        "player_opponent_position_at_hit_x",
        "player_opponent_position_at_hit_y",
        "player_position_at_bounce_x",
        "player_position_at_bounce_y",
        "player_opponent_position_at_bounce_x",
        "player_opponent_position_at_bounce_y",
        "duration",
        "net_clearance_z_debug",
    ]
    _require_columns(df, required)
    print(f"Loaded {len(df)} shots from {len(df['match_id'].unique())} matches")

    # =========================================================================
    # Step 2: Load rally-level outcomes (Sidecar Strategy)
    # =========================================================================
    print("\nLoading rally-level outcomes from points CSVs...")
    outcomes_df = load_points_outcomes(args.points_dir)
    print(f"Loaded outcomes for {len(outcomes_df)} rallies")

    # =========================================================================
    # Step 3: Deterministic cleaning
    # =========================================================================
    df = df.dropna(subset=["match_id", "point_index", "shot_index", "player", "player_opponent", "player_name", "player_opponent_name"])
    df = df.copy()

    # Convert player tuples "(2, 1)" -> 2
    df["player"] = df["player"].apply(_extract_player_id)
    df["player_opponent"] = df["player_opponent"].apply(_extract_player_id)

    # Label every serve event as stroke "serve" (serve strokes are often missing or mislabelled in the raw logs).
    is_serve_event = df["event_shot"].astype("string").str.strip().str.lower() == "serve"
    is_serve_flag = df["is_serve"] == True
    df.loc[is_serve_event | is_serve_flag, "stroke"] = "serve"

    # Parse distance_travelled dict-string -> distance_travelled_by_player.
    df["distance_travelled_by_player"] = parse_distance_travelled_by_player(df["distance_travelled"], df["player"])

    # Additional deterministic derived fields required by the feature contract.
    df["bounce"] = _coerce_bool(df["bounce"]).fillna(False).astype(bool)
    df["rebound_net"] = _coerce_bool(df["rebound_net"]).fillna(False).astype(bool)
    df["is_overhead_smash"] = df["event_shot"].astype("string").str.strip().str.lower() == "overhead-smash"

    dominant_hand_map = load_dominant_hand_map(args.points_dir)
    df["dominant_hand"] = df["player_name"].map(dominant_hand_map).fillna("R")

    # Sort and re-index shots within each rally AFTER drops (deterministic; no medians/scalers).
    df = df.sort_values(["match_id", "point_index", "shot_index"]).reset_index(drop=True)
    df["shot_index"] = df.groupby(["match_id", "point_index"]).cumcount()

    # =========================================================================
    # Step 4: Split FIRST (strict requirement - no leakage)
    # =========================================================================
    print("\nSplitting by match_id (strict no-leakage)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_matches = sorted(train_df["match_id"].unique().tolist())
    test_matches = sorted(test_df["match_id"].unique().tolist())
    print(f"Train matches: {len(train_matches)}  Test matches: {len(test_matches)}  Overlap: {len(set(train_matches)&set(test_matches))}")

    # Split outcomes by match_id to align with shots
    train_outcomes = outcomes_df[outcomes_df["match_id"].isin(train_matches)].copy()
    test_outcomes = outcomes_df[outcomes_df["match_id"].isin(test_matches)].copy()
    print(f"Train outcomes: {len(train_outcomes)}  Test outcomes: {len(test_outcomes)}")

    # =========================================================================
    # Step 5: Fit stroke imputer on the training matches only
    # =========================================================================
    print("\nFitting stroke imputer on train only...")
    stroke_model = fit_stroke_imputer(train_df, random_state=args.random_state)
    train_df = apply_stroke_imputer(train_df, stroke_model)
    test_df = apply_stroke_imputer(test_df, stroke_model)

    # =========================================================================
    # Step 6: Build feature matrices (39 features, no player IDs)
    # =========================================================================
    print("\nBuilding feature matrices (39 columns, no player identifiers)...")
    train_X = build_feature_matrix(train_df)
    test_X = build_feature_matrix(test_df)

    # =========================================================================
    # Step 7: Fit imputer+scaler on TRAIN numeric columns only
    # =========================================================================
    print("Fitting numeric pipeline (imputer + scaler) on train only...")
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipeline.fit(train_X[NUMERIC_TO_SCALE])

    train_scaled = numeric_pipeline.transform(train_X[NUMERIC_TO_SCALE])
    test_scaled = numeric_pipeline.transform(test_X[NUMERIC_TO_SCALE])
    train_X.loc[:, NUMERIC_TO_SCALE] = train_scaled
    test_X.loc[:, NUMERIC_TO_SCALE] = test_scaled

    # =========================================================================
    # Step 8: Build metadata columns
    # =========================================================================
    # Metadata includes rally keys + player IDs (for attribution analysis, not features)
    train_meta = train_df[["match_id", "point_index", "shot_index"]].reset_index(drop=True)
    train_meta["player"] = pd.to_numeric(train_df["player"], errors="coerce").astype(int).values
    train_meta["player_opponent"] = pd.to_numeric(train_df["player_opponent"], errors="coerce").astype(int).values

    test_meta = test_df[["match_id", "point_index", "shot_index"]].reset_index(drop=True)
    test_meta["player"] = pd.to_numeric(test_df["player"], errors="coerce").astype(int).values
    test_meta["player_opponent"] = pd.to_numeric(test_df["player_opponent"], errors="coerce").astype(int).values

    # Combine metadata + features
    train_out = pd.concat([train_meta, train_X.reset_index(drop=True)], axis=1)
    test_out = pd.concat([test_meta, test_X.reset_index(drop=True)], axis=1)

    # Ensure correct column order: META_COLS + FEATURE_COLS
    expected_cols = META_COLS + FEATURE_COLS
    train_out = train_out.reindex(columns=expected_cols)
    test_out = test_out.reindex(columns=expected_cols)

    # =========================================================================
    # Step 9: Validate outputs
    # =========================================================================
    print("\nValidating outputs...")
    validate_outputs(
        train_shots=train_out,
        test_shots=test_out,
        train_outcomes=train_outcomes,
        test_outcomes=test_outcomes,
        train_matches=train_matches,
        test_matches=test_matches,
    )

    # =========================================================================
    # Step 10: Save outputs
    # =========================================================================
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    # Shot-level processed data
    train_path = args.output_dir / "train_processed.csv"
    test_path = args.output_dir / "test_processed.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Rally-level outcomes (sidecar files)
    train_outcomes_path = args.output_dir / "train_outcomes.csv"
    test_outcomes_path = args.output_dir / "test_outcomes.csv"
    train_outcomes.to_csv(train_outcomes_path, index=False)
    test_outcomes.to_csv(test_outcomes_path, index=False)

    # Encoder training / validation matches within the training partition
    split_info_path = args.output_dir / "split_info.json"
    split_info = write_train_val_split(
        train_out, split_info_path, val_fraction=args.val_size, random_state=args.random_state,
    )

    # Preprocessor artifacts (includes meta_cols and feature_cols for downstream use)
    artifacts = {
        # Column contracts
        "meta_cols": META_COLS,
        "feature_cols": FEATURE_COLS,
        "numeric_to_scale": NUMERIC_TO_SCALE,
        "outcome_cols": OUTCOME_COLS,
        # Fitted preprocessors (TRAIN only)
        "numeric_pipeline": numeric_pipeline,
        "stroke_imputer": stroke_model,
        # Fixed category spaces
        "stroke_categories": STROKE_CATEGORIES,
        "spin_type_categories": SPIN_TYPE_CATEGORIES,
    }
    joblib.dump(artifacts, args.preprocessor_path)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Shot-level outputs (columns = META_COLS + FEATURE_COLS = {len(META_COLS)} + {len(FEATURE_COLS)} = {len(expected_cols)}):")
    print(f"  {train_path}  rows={len(train_out)}")
    print(f"  {test_path}  rows={len(test_out)}")
    print(f"\nRally-level outcomes (sidecar, {len(OUTCOME_COLS)} columns):")
    print(f"  {train_outcomes_path}  rows={len(train_outcomes)}")
    print(f"  {test_outcomes_path}  rows={len(test_outcomes)}")
    print(f"\nEncoder train/val split ({split_info['n_train_matches']} / {split_info['n_val_matches']} matches):")
    print(f"  {split_info_path}")
    print(f"\nPreprocessor artifacts:")
    print(f"  {args.preprocessor_path}")
    print(f"\nKey design notes:")
    print(f"  - Player IDs ({META_COLS[3]}, {META_COLS[4]}) are METADATA only, not model features")
    print(f"  - Communities will represent rally patterns, not 'who played'")
    print(f"  - Outcome columns are SIDECAR only, not model features")
    print("=" * 70)


if __name__ == "__main__":
    main()

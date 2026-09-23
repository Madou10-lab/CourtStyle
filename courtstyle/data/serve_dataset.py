#!/usr/bin/env python3
"""Serve record preprocessing.

Input is one row of serve statistics per server and match (41 columns). The
pipeline:

1. Moves the nine match-outcome columns to sidecar files. Outcomes are never
   encoder inputs and are used only for profiling and validation.
2. Reparameterises correlated speed features into interpretable deltas:
   headroom (maximum minus average speed) and directional speed differences
   toward the T and wide targets. This removes the most collinear columns.
3. Splits records 80/20 into training and validation, stratified by surface.
4. Fits median imputation and standardisation on the training split only and
   applies them to validation and, if given, to the synthetic records.

Output: ``train_data.csv`` and ``val_data.csv`` (29 columns), outcome sidecars,
``synthetic_processed.csv`` when synthetic records are supplied, and the fitted
preprocessor.

Usage:
    python -m courtstyle.data.serve_dataset \\
        --real-input data/serve/fingerprint_serve.csv \\
        --synthetic-input data/serve/synthetic_serve_50k.csv \\
        --output-dir data/serve/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_REAL_INPUT = Path("data/serve/fingerprint_serve.csv")
DEFAULT_OUTPUT_DIR = Path("data/serve/processed")
DEFAULT_PREPROCESSOR_PATH = Path("data/serve/processed/preprocessor.pkl")

# Columns to drop from features but preserve in sidecar "outcomes" files
# These are performance metrics that would let the model "cheat"
OUTCOME_COLS = [
    "player_ace",
    "player_df",
    "player_svpt",
    "player_1stIn",
    "player_1stWon",
    "player_2ndWon",
    "player_SvGms",
    "player_bpSaved",
    "player_bpFaced",
]

# Columns dropped before reparameterisation (not encoder inputs)
DROP_COLS = [
    "player_age",
    "avg_rally_len_first_serve",
    "avg_rally_len_second_serve",
    "%_short_rally_wins_after_serve",
    # Surface columns are kept for stratification and balanced sampling.
    # hand_U is kept so that an all-zero handedness row stays well defined
    # When hand_L=0 and hand_R=0, argmax incorrectly returns 0 (Left)
]

# Columns to drop AFTER reparameterization
# - Raw directional averages: replaced by delta features (speed tax)
# - Max speeds: replaced by headroom features (ceiling above typical pace)
# - serve_speed_drop: redundant with keeping both 1st Avg and 2nd Avg
DROP_AFTER_REPARAM = [
    "1st T Avg",
    "1st Wide Avg",
    "2nd T Avg",
    "2nd Wide Avg",
    "Max 1st",   # Replaced by 1st_headroom = Max 1st - 1st Avg
    "Max 2nd",   # Replaced by 2nd_headroom = Max 2nd - 2nd Avg
]


def _reparameterize_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reparameterize speed features to remove multicollinearity and add interpretability.

    Instead of keeping raw directional averages (1st T Avg, 1st Wide Avg, etc.) which
    are highly correlated with 1st Avg (r > 0.85), we create:

    1. Headroom features: Max - Avg (capability usage gap)
       - "How much faster CAN you serve vs how fast you DO serve"
       - Captures capability vs strategy tradeoff

    2. Directional delta features: (Direction Avg - Overall Avg)
       - "Speed tax to hit T vs Wide"
       - Physically interpretable: negative = slower to that zone

    This reduces multicollinearity substantially; removes the worst offenders;
    remaining VIF typically < ~5 depending on feature subset.

    Args:
        df: DataFrame with raw speed columns

    Returns:
        DataFrame with new derived features added
    """
    df = df.copy()

    # ==========================================================================
    # First Serve Reparameterization
    # ==========================================================================
    if all(c in df.columns for c in ["Max 1st", "1st Avg"]):
        # Headroom: capability gap (how much faster could they serve?)
        df["1st_headroom"] = df["Max 1st"] - df["1st Avg"]

    if all(c in df.columns for c in ["1st T Avg", "1st Avg"]):
        # Speed delta to T (center): negative = slower to T
        df["1st_delta_T"] = df["1st T Avg"] - df["1st Avg"]

    if all(c in df.columns for c in ["1st Wide Avg", "1st Avg"]):
        # Speed delta to Wide: negative = slower to wide
        df["1st_delta_Wide"] = df["1st Wide Avg"] - df["1st Avg"]

    # ==========================================================================
    # Second Serve Reparameterization
    # ==========================================================================
    if all(c in df.columns for c in ["Max 2nd", "2nd Avg"]):
        # Headroom on second serve
        df["2nd_headroom"] = df["Max 2nd"] - df["2nd Avg"]

    if all(c in df.columns for c in ["2nd T Avg", "2nd Avg"]):
        # Speed delta to T on second serve
        df["2nd_delta_T"] = df["2nd T Avg"] - df["2nd Avg"]

    if all(c in df.columns for c in ["2nd Wide Avg", "2nd Avg"]):
        # Speed delta to Wide on second serve
        df["2nd_delta_Wide"] = df["2nd Wide Avg"] - df["2nd Avg"]

    # No first-to-second serve speed drop feature: both averages are kept.
    # The model can infer this relationship directly from the raw averages

    return df


def _clean_one_hot(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    """Ensure one-hot encoded columns are clean integers (0/1)."""
    df = df.copy()
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return df


def _enforce_one_hot(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    """Ensure one-hot groups have sum==1; fallback to *_U when invalid."""
    df = df.copy()
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue

        unknown_col = f"{prefix}U"
        if unknown_col not in df.columns:
            df[unknown_col] = 0
            cols = cols + [unknown_col]

        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        row_sum = df[cols].sum(axis=1)
        invalid = row_sum != 1
        if invalid.any():
            df.loc[invalid, cols] = 0
            df.loc[invalid, unknown_col] = 1
            print(f"  Fixed {int(invalid.sum())} invalid rows for {prefix} -> {unknown_col}=1")

    return df


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into continuous and categorical (one-hot) groups."""
    cat_cols = [c for c in df.columns if c.startswith("surface_") or c.startswith("hand_")]
    cont_cols = [c for c in df.columns if c not in cat_cols]
    return cont_cols, cat_cols


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce specified columns to numeric, replacing errors with NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _surface_stratify(df: pd.DataFrame, surface_cols: List[str]) -> np.ndarray | None:
    """Create stratification labels from surface one-hot columns."""
    if not surface_cols:
        return None
    arr = df[surface_cols].fillna(0).to_numpy()
    labels = arr.argmax(axis=1)
    if np.unique(labels).size < 2:
        return None
    return labels


def _extract_outcomes(df: pd.DataFrame, outcome_cols: List[str]) -> pd.DataFrame:
    """
    Extract outcome columns from dataframe.
    Returns a dataframe with only the outcome columns that exist.
    """
    existing_cols = [c for c in outcome_cols if c in df.columns]
    if not existing_cols:
        # Return empty dataframe with expected columns (all NaN)
        return pd.DataFrame({c: [np.nan] * len(df) for c in outcome_cols}, index=df.index)

    outcomes = df[existing_cols].copy()

    # Add missing outcome columns as NaN for schema consistency
    for c in outcome_cols:
        if c not in outcomes.columns:
            outcomes[c] = np.nan

    # Ensure consistent column order
    return outcomes[outcome_cols]


def _align_schema(
    df: pd.DataFrame,
    columns: List[str],
    fill_missing: bool = False,
    continuous_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Align dataframe to expected schema columns.

    Args:
        df: Input dataframe
        columns: Expected column order
        fill_missing: If True, fill missing columns; if False, raise error
        continuous_cols: Optional list to fill missing continuous columns with NaN
    """
    df = df.copy()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        if fill_missing:
            for c in missing:
                if continuous_cols is not None and c in continuous_cols:
                    df[c] = np.nan
                else:
                    df[c] = 0
        else:
            raise ValueError(f"Missing columns in input: {missing}")

    extra = [c for c in df.columns if c not in columns]
    if extra:
        df = df.drop(columns=extra)

    return df[columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess CourtStyle serve records. The scaler is fitted on the "
            "training split only, and outcomes are written to sidecar files."
        )
    )
    parser.add_argument(
        "--real-input", type=Path, default=DEFAULT_REAL_INPUT,
        help="Path to fingerprint_serve.csv."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for processed CSVs."
    )
    parser.add_argument(
        "--preprocessor-path", type=Path, default=DEFAULT_PREPROCESSOR_PATH,
        help="Path to save preprocessor.pkl (fit on real-train only).",
    )
    parser.add_argument(
        "--synthetic-input", type=Path, default=None,
        help="Optional synthetic CSV to scale using the real-train preprocessor.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of real data for validation split."
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for train/val split."
    )
    args = parser.parse_args()

    # =========================================================================
    # Step 1: Load and clean raw data
    # =========================================================================
    print(f"Loading real data from {args.real_input}")
    raw_df = pd.read_csv(args.real_input)
    raw_df = _clean_one_hot(raw_df, ["surface_", "hand_"])

    # =========================================================================
    # Step 2: Extract outcomes before dropping them
    # =========================================================================
    print("Extracting outcome columns for sidecar files...")
    all_outcomes = _extract_outcomes(raw_df, OUTCOME_COLS)

    # =========================================================================
    # Step 3: Feature Reparameterization (before dropping)
    # =========================================================================
    print("Applying feature reparameterization (headroom + directional deltas)...")
    raw_df = _reparameterize_speed_features(raw_df)

    # Log new features created
    new_features = [c for c in raw_df.columns if c.endswith(("_headroom", "_delta_T", "_delta_Wide"))]
    print(f"  Created {len(new_features)} derived features: {new_features}")

    # =========================================================================
    # Step 4: Drop outcome and unnecessary columns from features
    # =========================================================================
    # Drop outcomes, base drop cols, AND redundant directional averages (after reparameterization)
    all_drop_cols = OUTCOME_COLS + DROP_COLS + DROP_AFTER_REPARAM
    feature_df = raw_df.drop(columns=all_drop_cols, errors="ignore")

    # Enforce one-hot validity before splitting schema
    feature_df = _enforce_one_hot(feature_df, ["surface_", "hand_"])

    # Identify column types
    cont_cols, cat_cols = _split_columns(feature_df)
    feature_df = _coerce_numeric(feature_df, cont_cols)

    # =========================================================================
    # Step 4: Train/Val split with stratification
    # =========================================================================
    surface_cols = [c for c in feature_df.columns if c.startswith("surface_")]
    stratify = _surface_stratify(feature_df, surface_cols)

    # Get indices for splitting both features and outcomes consistently
    indices = np.arange(len(feature_df))

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        # Fallback without stratification
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
            stratify=None,
        )

    # Split features
    train_df = feature_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = feature_df.iloc[val_idx].copy().reset_index(drop=True)

    # Split outcomes (preserving row alignment)
    train_outcomes = all_outcomes.iloc[train_idx].copy().reset_index(drop=True)
    val_outcomes = all_outcomes.iloc[val_idx].copy().reset_index(drop=True)

    print(f"Train set: {len(train_df)} rows")
    print(f"Val set: {len(val_df)} rows")

    # =========================================================================
    # Step 5: Fit imputer and scaler on the training split only
    # =========================================================================
    print("Fitting imputer and scaler on the training split only...")
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # FIT only on train
    numeric_pipeline.fit(train_df[cont_cols])

    # TRANSFORM train and val using train-fitted scaler
    train_df.loc[:, cont_cols] = numeric_pipeline.transform(train_df[cont_cols])
    val_df.loc[:, cont_cols] = numeric_pipeline.transform(val_df[cont_cols])

    # Cast continuous columns to float32 for consistent dtypes
    train_df[cont_cols] = train_df[cont_cols].astype("float32")
    val_df[cont_cols] = val_df[cont_cols].astype("float32")

    # Store the feature schema for alignment
    feature_schema = train_df.columns.tolist()

    # =========================================================================
    # Step 6: Save processed data and outcomes
    # =========================================================================
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Features (scaled)
    train_path = args.output_dir / "train_data.csv"
    val_path = args.output_dir / "val_data.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Outcomes (unscaled, sidecar files)
    train_outcomes_path = args.output_dir / "train_outcomes.csv"
    val_outcomes_path = args.output_dir / "val_outcomes.csv"
    train_outcomes.to_csv(train_outcomes_path, index=False)
    val_outcomes.to_csv(val_outcomes_path, index=False)

    print(f"Wrote {train_path} (features) rows={len(train_df)} cols={len(train_df.columns)}")
    print(f"Wrote {train_outcomes_path} (outcomes) rows={len(train_outcomes)} cols={len(train_outcomes.columns)}")
    print(f"Wrote {val_path} (features) rows={len(val_df)} cols={len(val_df.columns)}")
    print(f"Wrote {val_outcomes_path} (outcomes) rows={len(val_outcomes)} cols={len(val_outcomes.columns)}")

    # =========================================================================
    # Step 7: Save preprocessor artifacts
    # =========================================================================
    args.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "numeric_pipeline": numeric_pipeline,
        "cont_cols": cont_cols,
        "cat_cols": cat_cols,
        "outcome_cols": OUTCOME_COLS,
        "drop_cols": DROP_COLS,
        "drop_after_reparam": DROP_AFTER_REPARAM,
        "schema_cols": feature_schema,
        "derived_features": new_features,  # Track what reparameterization created
    }
    joblib.dump(artifacts, args.preprocessor_path)
    print(f"Wrote {args.preprocessor_path}")

    # =========================================================================
    # Step 8: Process synthetic data (if provided)
    # =========================================================================
    if args.synthetic_input is not None:
        print(f"\nProcessing synthetic data from {args.synthetic_input}")
        synthetic_raw = pd.read_csv(args.synthetic_input)
        synthetic_raw = _clean_one_hot(synthetic_raw, ["surface_", "hand_"])

        # Extract synthetic outcomes (may be missing or placeholder)
        synthetic_outcomes = _extract_outcomes(synthetic_raw, OUTCOME_COLS)
        synthetic_outcomes = synthetic_outcomes.reset_index(drop=True)

        # Apply same reparameterization as real data
        synthetic_raw = _reparameterize_speed_features(synthetic_raw)

        # Drop outcome and unnecessary columns (including post-reparam redundant cols)
        synthetic_features = synthetic_raw.drop(columns=all_drop_cols, errors="ignore")

        # Align to train schema (categorical=0, continuous=NaN)
        synthetic_features = _align_schema(
            synthetic_features,
            feature_schema,
            fill_missing=True,
            continuous_cols=cont_cols,
        )
        synthetic_features = _coerce_numeric(synthetic_features, cont_cols)
        synthetic_features = _enforce_one_hot(synthetic_features, ["surface_", "hand_"])

        # TRANSFORM using train-fitted scaler (DO NOT RE-FIT)
        synthetic_features.loc[:, cont_cols] = numeric_pipeline.transform(synthetic_features[cont_cols])

        # Cast continuous columns to float32 for consistent dtypes
        synthetic_features[cont_cols] = synthetic_features[cont_cols].astype("float32")
        synthetic_features = synthetic_features.reset_index(drop=True)

        # Save synthetic processed data
        synthetic_path = args.output_dir / "synthetic_processed.csv"
        synthetic_outcomes_path = args.output_dir / "synthetic_outcomes.csv"

        synthetic_features.to_csv(synthetic_path, index=False)
        synthetic_outcomes.to_csv(synthetic_outcomes_path, index=False)

        print(f"Wrote {synthetic_path} (features) rows={len(synthetic_features)} cols={len(synthetic_features.columns)}")
        print(f"Wrote {synthetic_outcomes_path} (outcomes) rows={len(synthetic_outcomes)} cols={len(synthetic_outcomes.columns)}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Feature columns: {len(feature_schema)}")
    print(f"Outcome columns (sidecar): {len(OUTCOME_COLS)}")
    print("Scaler fitted on train_data.csv only")
    print("=" * 60)


if __name__ == "__main__":
    main()

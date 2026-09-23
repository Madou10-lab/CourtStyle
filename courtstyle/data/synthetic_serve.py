"""Synthetic serve records for encoder pre-training.

A tabular variational autoencoder (TVAE, SDV) is fitted to the real serve
records and sampled to produce the synthetic records used in stage-1
pre-training. The synthetic records are never used for validation, style
discovery or evaluation.

Before fitting, surface and handedness one-hot columns are collapsed into
categorical fields, and differential speed features (maximum minus average,
average minus minimum, for first and second serves) are added so that the
ordering between maximum, average and minimum speed is enforced through
reconstruction. Column types are declared explicitly so that low-cardinality
numeric columns are not treated as categorical. After sampling, the speed
columns are reconstructed from the differentials, one-hot columns are restored,
and the three serve-placement shares are projected back onto the simplex.

Fidelity is reported by the two-sample Kolmogorov-Smirnov complement on average
speed and width variance, and by the correlation between player height and
average serve speed in real and synthetic records.

The generator is fitted on all real records. TVAE training is stochastic, so a
new run produces a different synthetic set with the same distributional
properties.

Usage:
    python -m courtstyle.data.synthetic_serve \\
        --input data/serve/fingerprint_serve.csv \\
        --output data/serve/synthetic_serve_50k.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer


WIDTH_COLS = ["%_serves_T", "%_serves_Wide", "%_serves_Body"]


def collapse_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hand_* and surface_* one-hots into categorical columns."""
    df = df.copy()
    surface_cols = ["surface_Grass", "surface_Hard", "surface_Clay"]
    hand_cols = ["hand_L", "hand_R", "hand_U"]

    def collapse(cols, labels, default):
        sub = df[cols].fillna(0)
        arr = sub.to_numpy()
        idx = arr.argmax(axis=1)
        mask_all_zero = (arr.sum(axis=1) == 0)
        cat = pd.Categorical([labels[i] for i in idx], categories=labels)
        cat = cat.astype(object)
        cat = np.where(mask_all_zero, default, cat)
        return pd.Series(cat, index=df.index)

    df["Surface_Cat"] = collapse(surface_cols, ["Grass", "Hard", "Clay"], "Hard")
    df["Hand_Cat"] = collapse(hand_cols, ["L", "R", "U"], "R")
    df = df.drop(columns=surface_cols + hand_cols)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce differential speed features to model physics without clipping."""
    df = df.copy()
    # 1st serve
    df["Diff_1st_Max_Avg"] = df["Max 1st"] - df["1st Avg"]
    df["Diff_1st_Avg_Min"] = df["1st Avg"] - df["Min 1st"]
    # 2nd serve
    df["Diff_2nd_Max_Avg"] = df["Max 2nd"] - df["2nd Avg"]
    df["Diff_2nd_Avg_Min"] = df["2nd Avg"] - df["Min 2nd"]
    df = df.drop(columns=["Max 1st", "Min 1st", "Max 2nd", "Min 2nd"])
    return df


def reconstruct_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild Max/Min speeds from Avg + diffs while enforcing ordering."""
    df = df.copy()
    for col in ["Diff_1st_Max_Avg", "Diff_1st_Avg_Min", "Diff_2nd_Max_Avg", "Diff_2nd_Avg_Min"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df["Max 1st"] = df["1st Avg"] + df.pop("Diff_1st_Max_Avg")
    df["Min 1st"] = df["1st Avg"] - df.pop("Diff_1st_Avg_Min")
    df["Max 2nd"] = df["2nd Avg"] + df.pop("Diff_2nd_Max_Avg")
    df["Min 2nd"] = df["2nd Avg"] - df.pop("Diff_2nd_Avg_Min")
    return df


def reencode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Re-expand categorical hand/surface back to one-hot columns."""
    df = df.copy()
    surface_dummies = pd.get_dummies(df["Surface_Cat"], prefix="surface")
    hand_dummies = pd.get_dummies(df["Hand_Cat"], prefix="hand")
    for col in ["surface_Clay", "surface_Grass", "surface_Hard", "hand_L", "hand_R", "hand_U"]:
        if col not in surface_dummies.columns:
            surface_dummies[col] = 0
        if col not in hand_dummies.columns:
            hand_dummies[col] = 0
    df = df.drop(columns=["Surface_Cat", "Hand_Cat"])
    df = pd.concat(
        [df, surface_dummies[["surface_Grass", "surface_Hard", "surface_Clay"]],
         hand_dummies[["hand_L", "hand_R", "hand_U"]]],
        axis=1,
    )
    return df


def project_serve_width(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enforce serve-width proportions to be within a simplex with slack (sum <= 1).
    Uses empirical slack from real data: u = 1 - sum_width_real.
    """
    df = synth_df.copy()

    for col in WIDTH_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing width column in synthetic data: {col}")
        if col not in real_df.columns:
            raise ValueError(f"Missing width column in real data: {col}")

    df[WIDTH_COLS] = df[WIDTH_COLS].apply(pd.to_numeric, errors="coerce")
    df[WIDTH_COLS] = df[WIDTH_COLS].clip(lower=0.0, upper=1.0)
    sum_width = df[WIDTH_COLS].sum(axis=1)

    real_comp = real_df[WIDTH_COLS].apply(pd.to_numeric, errors="coerce")
    real_sum = real_comp.sum(axis=1)
    real_slack = (1.0 - real_sum).clip(lower=0.0, upper=1.0).dropna().to_numpy()
    rng = np.random.default_rng(seed)

    mask = sum_width > 1.0
    if mask.any():
        df.loc[mask, WIDTH_COLS] = df.loc[mask, WIDTH_COLS].div(sum_width[mask], axis=0)
        if real_slack.size:
            slack_samples = rng.choice(real_slack, size=int(mask.sum()), replace=True)
        else:
            slack_samples = np.zeros(int(mask.sum()))
        df.loc[mask, WIDTH_COLS] = df.loc[mask, WIDTH_COLS].mul(1.0 - slack_samples, axis=0)

    df[WIDTH_COLS] = df[WIDTH_COLS].clip(lower=0.0, upper=1.0)
    return df


def ks_complement(real: pd.Series, synth: pd.Series) -> float:
    """1 - KS statistic (higher is better similarity)."""
    stat, _ = ks_2samp(real.dropna(), synth.dropna())
    return 1.0 - stat


def build_explicit_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """
    Build explicit metadata to ensure all numeric columns are treated as numerical,
    especially player_ht which has low cardinality (16 unique values) but is ordinal.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Force all numeric columns to be treated as numerical (not categorical)
    # This is critical for player_ht which has only 16 unique values
    numerical_columns = [
        # Core speed metrics
        "Avg Speed", "1st Avg", "2nd Avg",
        "1st StDev", "2nd StDev",
        "1st T Avg", "1st Wide Avg", "2nd T Avg", "2nd Wide Avg",
        # Differential features (created by add_derived_features)
        "Diff_1st_Max_Avg", "Diff_1st_Avg_Min",
        "Diff_2nd_Max_Avg", "Diff_2nd_Avg_Min",
        # Serve placement
        "%_serves_T", "%_serves_Wide", "%_serves_Body",
        "serve_width_var", "serve_depth_var",
        "change_T", "change_Wide", "change_Body",
        # Rally stats
        "avg_rally_len_first_serve", "avg_rally_len_second_serve",
        "%_short_rally_wins_after_serve",
        # Player physical/stats - CRITICAL: player_ht must be numerical
        "player_ht", "player_age",
        "player_ace", "player_df", "player_svpt",
        "player_1stIn", "player_1stWon", "player_2ndWon",
        "player_SvGms", "player_bpSaved", "player_bpFaced",
    ]

    for col in numerical_columns:
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype="numerical")

    # Categorical columns (collapsed from one-hot)
    categorical_columns = ["Surface_Cat", "Hand_Cat"]
    for col in categorical_columns:
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype="categorical")

    return metadata



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic serve records with TVAE.")
    parser.add_argument("--input", type=Path, default=Path("data/serve/fingerprint_serve.csv"),
                        help="Real serve records (one row per server and match).")
    parser.add_argument("--output", type=Path, default=Path("data/serve/synthetic_serve_50k.csv"))
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=500, help="TVAE training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the placement-share projection.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input file: {args.input}")

    print(f"Loading real records from {args.input}")
    real_df = pd.read_csv(args.input)
    training_df = add_derived_features(collapse_one_hot(real_df))

    metadata = build_explicit_metadata(training_df)
    print(f"Training TVAE (epochs={args.epochs})...")
    model = TVAESynthesizer(metadata, epochs=args.epochs, verbose=True)
    model.fit(training_df)

    print(f"Sampling {args.n_samples} synthetic records...")
    synth = reencode_one_hot(reconstruct_physics(model.sample(args.n_samples)))
    for col in real_df.columns:
        if col not in synth.columns:
            synth[col] = 0
    synth = project_serve_width(synth[real_df.columns], real_df, seed=args.seed)

    print("Fidelity:")
    print(f"  KS complement, average speed:        {ks_complement(real_df['Avg Speed'], synth['Avg Speed']):.3f}")
    print(f"  KS complement, width variance:       "
          f"{ks_complement(real_df['serve_width_var'], synth['serve_width_var']):.3f}")
    print(f"  Height vs average speed correlation: real {real_df['player_ht'].corr(real_df['Avg Speed']):.3f}, "
          f"synthetic {synth['player_ht'].corr(synth['Avg Speed']):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

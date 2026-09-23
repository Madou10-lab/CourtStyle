"""Held-out transfer of the rally styles.

Rallies from matches never seen by the encoder or by K-Means are assigned to
the nearest training centroid by ``rally_discovery`` (run with ``--splits
train_all,test``). This module compares the styles across the two partitions:

* test assignment: style sizes, cosine silhouette of the test embeddings, the
  margin between the closest and second-closest centroid, and how many styles
  occur in every test match;
* outcome association on the test rallies: chi-square test of style against
  point result (Fault / Let / Played) and Cramer's V;
* profile transfer: per-style means of fault rate, played rate, serve speed and
  return net clearance on train and test, with bootstrap 95% confidence
  intervals, interval overlap and Spearman correlation of the style rankings.

Usage:
    python -m courtstyle.evaluation.rally_transfer --config configs/rally.yaml \\
        --discovery-dir outputs/rally/discovery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from courtstyle.data.rally_transitions import _safe_int

RANDOM_SEED = 0

METRIC_LABELS = {
    "fault_rate": "Fault Rate",
    "played_rate": "Played Rate",
    "serve_speed": "Serve\nSpeed",
    "return_clearance": "Return Net\nClearance",
}


def summarize_rallies(labels_df: pd.DataFrame, shots_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> pd.DataFrame:
    labels = labels_df.copy()
    labels["match_id"] = labels["match_id"].astype(str)
    labels["point_index"] = labels["point_index"].apply(_safe_int)

    shots = shots_df.copy()
    shots["match_id"] = shots["match_id"].astype(str)
    shots["point_index"] = shots["point_index"].apply(_safe_int)
    shots["shot_index"] = shots["shot_index"].apply(_safe_int)

    outcomes = outcomes_df.copy()
    outcomes["match_id"] = outcomes["match_id"].astype(str)
    outcomes["point_index"] = outcomes["point_index"].apply(_safe_int)

    serve = (
        shots[shots["shot_index"] == 0][["match_id", "point_index", "speed"]]
        .rename(columns={"speed": "serve_speed"})
    )
    ret = (
        shots[shots["shot_index"] == 1][["match_id", "point_index", "speed", "net_clearance_z_debug"]]
        .rename(columns={"speed": "return_speed", "net_clearance_z_debug": "return_net_clearance"})
    )
    rally_means = (
        shots.groupby(["match_id", "point_index"], as_index=False)
        .agg(
            avg_shot_speed=("speed", "mean"),
            avg_shot_duration=("duration", "mean"),
            avg_player_distance=("distance_travelled_by_player", "mean"),
        )
    )

    merged = labels.merge(serve, on=["match_id", "point_index"], how="left")
    merged = merged.merge(ret, on=["match_id", "point_index"], how="left")
    merged = merged.merge(rally_means, on=["match_id", "point_index"], how="left")
    out_cols = [
        "match_id",
        "point_index",
        "result",
        "num_shots",
    ]
    merged = merged.merge(outcomes[out_cols], on=["match_id", "point_index"], how="left")
    merged["n_shots"] = merged["num_shots"]
    return merged


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_SEED) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        val = float(arr[0])
        return val, val, val
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    means = samples.mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compute_profile_transfer_ci(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_communities: int,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Per-style mean and bootstrap 95% interval of each transfer metric on train and test."""
    metric_extractors = {
        "fault_rate": lambda df: (df["result"].fillna("") == "Fault").astype(float).to_numpy(),
        "played_rate": lambda df: (df["result"].fillna("") == "Played").astype(float).to_numpy(),
        "serve_speed": lambda df: df["serve_speed"].to_numpy(),
        "return_clearance": lambda df: df["return_net_clearance"].to_numpy(),
    }

    rows = []
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        for community_id in range(n_communities):
            cdf = split_df[split_df["community_id"].astype(int) == community_id]
            for metric_name, extractor in metric_extractors.items():
                mean, lower, upper = bootstrap_mean_ci(
                    extractor(cdf), n_boot=n_boot, seed=RANDOM_SEED + community_id
                )
                rows.append(
                    {
                        "split": split_name,
                        "community_id": community_id,
                        "metric": metric_name,
                        "mean": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n": int(len(cdf)),
                    }
                )
    return pd.DataFrame(rows)


def transfer_rankings(ci_df: pd.DataFrame) -> Dict[str, Any]:
    """Interval overlap and Spearman correlation of style means between train and test."""
    out: Dict[str, Any] = {}
    for metric in ci_df["metric"].unique():
        tr = ci_df[(ci_df["split"] == "train") & (ci_df["metric"] == metric)].sort_values("community_id")
        te = ci_df[(ci_df["split"] == "test") & (ci_df["metric"] == metric)].sort_values("community_id")
        tr_mean, te_mean = tr["mean"].to_numpy(), te["mean"].to_numpy()
        overlap = [
            bool(not (a_hi < b_lo or b_hi < a_lo))
            for a_lo, a_hi, b_lo, b_hi in zip(
                tr["ci_lower"], tr["ci_upper"], te["ci_lower"], te["ci_upper"]
            )
        ]
        rho, p = spearmanr(tr_mean, te_mean)
        train_order = [int(c) for c in np.argsort(-tr_mean, kind="stable")]
        test_order = [int(c) for c in np.argsort(-te_mean, kind="stable")]
        out[str(metric)] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "train_order_high_to_low": train_order,
            "test_order_high_to_low": test_order,
            "exact_order_match": train_order == test_order,
            "ci_overlap_per_style": overlap,
            "n_styles_ci_overlap": int(sum(overlap)),
            "train_values": {str(c): float(v) for c, v in enumerate(tr_mean)},
            "test_values": {str(c): float(v) for c, v in enumerate(te_mean)},
        }
    return out


def outcome_association(labels_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    """Chi-square test of style against point result, with Cramer's V."""
    labels = labels_df[["match_id", "point_index", "community_id"]].copy()
    labels["match_id"] = labels["match_id"].astype(str)
    labels["point_index"] = labels["point_index"].apply(_safe_int)
    outcomes = outcomes_df[["match_id", "point_index", "result"]].copy()
    outcomes["match_id"] = outcomes["match_id"].astype(str)
    outcomes["point_index"] = outcomes["point_index"].apply(_safe_int)
    merged = labels.merge(outcomes, on=["match_id", "point_index"], how="left")
    merged = merged.dropna(subset=["result"])

    table = pd.crosstab(merged["community_id"], merged["result"])
    chi2, p, dof, _ = chi2_contingency(table)
    n = int(table.to_numpy().sum())
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    rates = table.div(table.sum(axis=1), axis=0)
    return {
        "n_rallies": n,
        "result_chi2": float(chi2),
        "result_pvalue": float(p),
        "result_dof": int(dof),
        "result_cramers_v": cramers_v,
        "contingency_table": {str(int(c)): {str(k): int(v) for k, v in row.items()} for c, row in table.iterrows()},
        "community_outcome_rates": {str(int(c)): {str(k): float(v) for k, v in row.items()} for c, row in rates.iterrows()},
    }


def test_assignment_summary(
    test_emb: np.ndarray,
    test_labels_df: pd.DataFrame,
    centroids: np.ndarray,
) -> Dict[str, Any]:
    """Sizes, cosine silhouette and centroid margin of the test assignment."""
    labels = test_labels_df["community_id"].astype(int).to_numpy()
    sizes = pd.Series(labels).value_counts().sort_index()
    sim = normalize(test_emb) @ normalize(centroids).T
    top2 = np.sort(sim, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    per_match = test_labels_df.groupby(test_labels_df["match_id"].astype(str))["community_id"].nunique()
    per_style = test_labels_df.groupby("community_id")["match_id"].nunique()
    return {
        "n_test_rallies": int(len(labels)),
        "test_silhouette_cosine": float(silhouette_score(normalize(test_emb), labels, metric="cosine")),
        "cluster_sizes": {str(int(c)): int(v) for c, v in sizes.items()},
        "cluster_fracs": {str(int(c)): float(v / len(labels)) for c, v in sizes.items()},
        "centroid_margin": {
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "p10": float(np.quantile(margin, 0.1)),
            "p90": float(np.quantile(margin, 0.9)),
        },
        "communities_per_test_match": {str(m): int(v) for m, v in per_match.items()},
        "matches_per_community": {str(int(c)): int(v) for c, v in per_style.items()},
    }


def plot_profile_transfer(ci_df: pd.DataFrame, metrics: Sequence[str], out_path: Path) -> None:
    """Train vs test style means with bootstrap 95% intervals, one panel per metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_comm = int(ci_df["community_id"].max()) + 1
    x = np.arange(n_comm)
    bar_width = 0.35
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.9 * len(metrics), 3.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, metrics):
        for split, offset, color in [("train", -bar_width / 2, "#4878CF"), ("test", bar_width / 2, "#E87D72")]:
            rows = ci_df[(ci_df["split"] == split) & (ci_df["metric"] == key)].sort_values("community_id")
            vals = rows["mean"].to_numpy()
            yerr = np.vstack([vals - rows["ci_lower"].to_numpy(), rows["ci_upper"].to_numpy() - vals])
            ax.bar(
                x + offset, vals, bar_width, label=split.capitalize(), color=color, alpha=0.85,
                edgecolor="white", yerr=yerr, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#2c3e50"},
            )
        ax.set_xlabel(METRIC_LABELS.get(key, key), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in range(n_comm)], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out transfer of the rally styles.")
    p.add_argument("--config", type=Path, default=Path("configs/rally.yaml"))
    p.add_argument("--discovery-dir", type=Path, default=Path("outputs/rally/discovery"),
                   help="Rally discovery run that includes the test split.")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <discovery-dir>/transfer.")
    p.add_argument("--figure-metrics", type=str, default="played_rate,serve_speed",
                   help=f"Comma-separated panels for the transfer figure, from {sorted(METRIC_LABELS)}.")
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    disc = args.discovery_dir
    out_dir = args.out_dir or disc / "transfer"

    fit_labels_path = disc / "communities" / "community_labels_train_all.csv"
    test_labels_path = disc / "communities" / "community_labels_test.csv"
    needed = [
        fit_labels_path, test_labels_path,
        disc / "embeddings" / "rally_embeddings_test.npy",
        disc / "communities" / "community_centroids.npy",
        Path(data_cfg["train_shots"]), Path(data_cfg["train_outcomes"]),
        Path(data_cfg["test_shots"]), Path(data_cfg["test_outcomes"]),
    ]
    for path in needed:
        if not path.exists():
            sys.exit(
                f"Missing input: {path}. Run courtstyle.discovery.rally_discovery with the test split "
                "(--splits train_all,val,test) first."
            )

    train_labels = pd.read_csv(fit_labels_path)
    test_labels = pd.read_csv(test_labels_path)
    train_shots = pd.read_csv(data_cfg["train_shots"], low_memory=False)
    train_outcomes = pd.read_csv(data_cfg["train_outcomes"], low_memory=False)
    test_shots = pd.read_csv(data_cfg["test_shots"], low_memory=False)
    test_outcomes = pd.read_csv(data_cfg["test_outcomes"], low_memory=False)
    n_comm = int(train_labels["community_id"].max()) + 1

    cols = ["match_id", "point_index", "community_id"]
    train_df = summarize_rallies(train_labels[cols], train_shots, train_outcomes)
    test_df = summarize_rallies(test_labels[cols], test_shots, test_outcomes)
    ci_df = compute_profile_transfer_ci(train_df, test_df, n_comm, n_boot=args.n_boot)

    out_dir.mkdir(parents=True, exist_ok=True)
    ci_df.to_csv(out_dir / "rally_profile_transfer_ci.csv", index=False)
    summary = {
        "test_assignment": test_assignment_summary(
            np.load(disc / "embeddings" / "rally_embeddings_test.npy"),
            test_labels,
            np.load(disc / "communities" / "community_centroids.npy"),
        ),
        "test_outcome_association": outcome_association(test_labels, test_outcomes),
        "profile_transfer": transfer_rankings(ci_df),
    }
    (out_dir / "transfer_summary.json").write_text(json.dumps(summary, indent=2))

    metrics: List[str] = [m.strip() for m in args.figure_metrics.split(",") if m.strip()]
    plot_profile_transfer(ci_df, metrics, out_dir / "style_profile_transfer.png")

    assoc = summary["test_outcome_association"]
    print(f"Test rallies: {summary['test_assignment']['n_test_rallies']}")
    print(f"Outcome association: chi2={assoc['result_chi2']:.2f}  p={assoc['result_pvalue']:.2e}  "
          f"Cramer's V={assoc['result_cramers_v']:.4f}")
    for metric, r in summary["profile_transfer"].items():
        print(f"  {metric:17s} rho={r['spearman_rho']:.2f}  CI overlap {r['n_styles_ci_overlap']}/{n_comm}")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()

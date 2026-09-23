"""Serve style profiles and structured evidence statements.

Reads a serve style-discovery run and the processed records, and produces:

* a standardised profile for every style over the encoder inputs and over the
  held-out outcome variables (style means as z-scores over all records);
* the structured evidence statements supplied to the language model, written as
  ``coach_facts_v2.json`` with the grounded prompt ``coach_analysis_v2.txt``;
* profile figures and ``interpretation_report.json``.

Every statement carries an identifier (``C<k>_F<nn>`` or ``XR_F<nn>``) and
states one measured quantity as a level relative to the other styles. Report
generation from these statements is handled by
``courtstyle.interpretation.serve_reports``.

Usage:
    python -m courtstyle.interpretation.serve_interpretation \\
        --community-run-dir outputs/serve/community_run_<timestamp> \\
        --processed-dir data/serve/processed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances


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


FEATURE_DISPLAY_NAMES = {
    "Avg Speed": "Average Serve Speed",
    "1st Avg": "1st Serve Average Speed",
    "1st StDev": "1st Serve Speed Variability",
    "Min 1st": "Minimum 1st Serve Speed",
    "2nd Avg": "2nd Serve Average Speed",
    "2nd StDev": "2nd Serve Speed Variability",
    "Min 2nd": "Minimum 2nd Serve Speed",
    "%_serves_T": "T-Placement Frequency",
    "%_serves_Wide": "Wide Placement Frequency",
    "%_serves_Body": "Body Placement Frequency",
    "serve_width_var": "Placement Width Variance",
    "serve_depth_var": "Placement Depth Variance",
    "change_T": "T-Placement Adaptability",
    "change_Wide": "Wide Placement Adaptability",
    "change_Body": "Body Placement Adaptability",
    "player_ht": "Player Height",
    "1st_headroom": "1st Serve Height Margin",
    "1st_delta_T": "1st Serve T-Placement Accuracy",
    "1st_delta_Wide": "1st Serve Wide Accuracy",
    "2nd_headroom": "2nd Serve Height Margin",
    "2nd_delta_T": "2nd Serve T-Placement Accuracy",
    "2nd_delta_Wide": "2nd Serve Wide Accuracy",
    "player_ace": "Aces",
    "player_df": "Double Faults",
    "player_svpt": "Service Points Played",
    "player_1stIn": "1st Serve In %",
    "player_1stWon": "1st Serve Points Won",
    "player_2ndWon": "2nd Serve Points Won",
    "player_SvGms": "Service Games",
    "player_bpSaved": "Break Points Saved",
    "player_bpFaced": "Break Points Faced",
}


@dataclass
class CommunityProfile:
    """Structured profile for a single community."""
    community_id: int
    count: int
    pct_of_total: float

    # Surface association
    hard_pct: float
    grass_pct: float

    # Physics profile (z-scores)
    physics_z: Dict[str, float] = field(default_factory=dict)

    # Outcome profile (z-scores)
    outcome_z: Dict[str, float] = field(default_factory=dict)

    # Top drivers (sorted by |z|)
    top_physics_drivers: List[Tuple[str, float]] = field(default_factory=list)
    top_outcome_correlates: List[Tuple[str, float]] = field(default_factory=list)

    # Assignment confidence (from centroid distances)
    mean_distance_to_centroid: float = 0.0
    std_distance_to_centroid: float = 0.0

    # Derived name (not hardcoded)
    derived_name: str = ""

    # Merge info (which fine-grained communities merged into this)
    merged_from: List[int] = field(default_factory=list)


@dataclass
class RunLevelMetrics:
    """Run-level validation metrics (not per-community)."""
    n_communities: int
    n_samples: int
    modularity: float
    silhouette: float
    surface_nmi: float
    stability_ari_mean: float
    stability_ari_std: float
    stability_nmi_mean: float
    stability_nmi_std: float
    n_outcomes_tested: int
    n_significant_outcomes: int
    pct_significant: float


def load_community_discovery_outputs(run_dir: Path) -> Dict[str, Any]:
    """Load all outputs from a community discovery run."""
    outputs = {}

    # Required files
    required = {
        "community_profiles": run_dir / "communities" / "community_profiles.csv",
        "community_labels_train": run_dir / "communities" / "community_labels_train.csv",
        "community_labels_val": run_dir / "communities" / "community_labels_val.csv",
        "community_metrics": run_dir / "validation" / "community_metrics.json",
    }

    # Optional files
    optional = {
        "community_outcome_profiles": run_dir / "communities" / "community_outcome_profiles.csv",
        "community_surface_proportions": run_dir / "validation" / "community_surface_proportions.csv",
        "community_centroids": run_dir / "communities" / "community_centroids.npy",
        "merge_map": run_dir / "communities" / "merge_map.json",
        "original_labels": run_dir / "communities" / "original_community_labels.npy",
        "statistical_tests": run_dir / "validation" / "statistical_tests.csv",
        "run_config": run_dir / "run_config.json",
    }

    # Load required
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        if path.suffix == ".csv":
            outputs[name] = pd.read_csv(path)
        elif path.suffix == ".json":
            with open(path) as f:
                outputs[name] = json.load(f)
        elif path.suffix == ".npy":
            outputs[name] = np.load(path)

    # Load optional
    for name, path in optional.items():
        if path.exists():
            if path.suffix == ".csv":
                outputs[name] = pd.read_csv(path)
            elif path.suffix == ".json":
                with open(path) as f:
                    outputs[name] = json.load(f)
            elif path.suffix == ".npy":
                outputs[name] = np.load(path)
        else:
            outputs[name] = None
            print(f"Optional file not found (will skip): {path}")

    return outputs


def load_original_data(
    train_path: Path,
    val_path: Path,
    train_outcomes_path: Path,
    val_outcomes_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load original processed data and outcomes."""
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    train_outcomes = pd.read_csv(train_outcomes_path)
    val_outcomes = pd.read_csv(val_outcomes_path)
    return train_data, val_data, train_outcomes, val_outcomes


def extract_run_level_metrics(metrics: Dict) -> RunLevelMetrics:
    """Extract run-level metrics from community_metrics.json."""
    stability = metrics.get("stability", {})
    outcome_tests = metrics.get("outcome_tests", {})

    return RunLevelMetrics(
        n_communities=metrics.get("n_communities", 0),
        n_samples=metrics.get("n_samples", 0),
        modularity=metrics.get("modularity", 0.0),
        silhouette=metrics.get("silhouette_cosine", 0.0),
        surface_nmi=metrics.get("surface_nmi", 0.0),
        stability_ari_mean=stability.get("ari_mean", 0.0),
        stability_ari_std=stability.get("ari_std", 0.0),
        stability_nmi_mean=stability.get("nmi_mean", 0.0),
        stability_nmi_std=stability.get("nmi_std", 0.0),
        n_outcomes_tested=outcome_tests.get("n_outcomes_tested", 0),
        n_significant_outcomes=outcome_tests.get("n_significant_005", 0),
        pct_significant=outcome_tests.get("pct_significant", 0.0),
    )


def extract_z_scores(profiles_df: pd.DataFrame, community_id: int, prefix: str = "z_") -> Dict[str, float]:
    """Extract z-score columns for a community."""
    row = profiles_df[profiles_df["community_id"] == community_id]
    if row.empty:
        return {}

    z_cols = [c for c in profiles_df.columns if c.startswith(prefix)]
    z_scores = {}
    for col in z_cols:
        feature_name = col[len(prefix):]  # Remove "z_" prefix
        z_scores[feature_name] = float(row[col].iloc[0])
    return z_scores


def get_top_drivers(z_scores: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
    """Get top-k features by absolute z-score."""
    sorted_features = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_features[:top_k]


def compute_assignment_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    community_ids: List[int],
) -> Dict[int, Tuple[float, float]]:
    """Compute mean and std of distances to centroid for each community."""
    confidence = {}

    for i, cid in enumerate(community_ids):
        mask = labels == cid
        if mask.sum() == 0:
            confidence[cid] = (0.0, 0.0)
            continue

        community_embeddings = embeddings[mask]
        centroid = centroids[i]

        # Cosine distance to centroid
        distances = cosine_distances(community_embeddings, centroid.reshape(1, -1)).flatten()
        confidence[cid] = (float(np.mean(distances)), float(np.std(distances)))

    return confidence


def derive_archetype_name(physics_z: Dict[str, float], outcome_z: Dict[str, float]) -> str:
    """
    Derive a descriptive archetype name from the profile data.

    This ensures names are data-driven, not dependent on arbitrary community IDs.
    """
    # Key discriminative features
    speed_z = physics_z.get("Avg Speed", 0)
    stdev_z = physics_z.get("1st StDev", 0)
    ace_z = outcome_z.get("player_ace", 0)
    t_pct_z = physics_z.get("%_serves_T", 0)
    body_pct_z = physics_z.get("%_serves_Body", 0)

    # Primary characteristic: Speed
    if speed_z > 0.8:
        speed_desc = "Power"
    elif speed_z > 0.2:
        speed_desc = "Medium-Pace"
    elif speed_z < -0.5:
        speed_desc = "Slower"
    else:
        speed_desc = "Average-Speed"

    # Secondary characteristic: Consistency
    if stdev_z < -0.5:
        consistency_desc = "Consistent"
    elif stdev_z > 0.5:
        consistency_desc = "Variable"
    else:
        consistency_desc = ""

    # Tertiary: Placement tendency
    if t_pct_z > 0.3:
        placement_desc = "T-Focused"
    elif body_pct_z > 0.3:
        placement_desc = "Body-Focused"
    else:
        placement_desc = ""

    # Combine
    parts = [p for p in [consistency_desc, speed_desc, placement_desc] if p]
    name = " ".join(parts) if parts else "Balanced"

    # Add outcome hint
    if ace_z > 0.3:
        name += " (High Ace)"
    elif ace_z < -0.3:
        name += " (Low Ace)"

    return name


def build_community_profiles(
    community_outputs: Dict[str, Any],
    embeddings: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> List[CommunityProfile]:
    """Build structured profiles for all communities."""
    profiles_df = community_outputs["community_profiles"]
    outcome_profiles_df = community_outputs.get("community_outcome_profiles")
    surface_props = community_outputs.get("community_surface_proportions")
    centroids = community_outputs.get("community_centroids")
    merge_map = community_outputs.get("merge_map")

    community_ids = profiles_df["community_id"].unique()
    total_count = profiles_df["count"].sum()

    # Compute assignment confidence if we have embeddings and centroids
    confidence_map = {}
    if embeddings is not None and labels is not None and centroids is not None:
        confidence_map = compute_assignment_confidence(
            embeddings, labels, centroids, sorted(community_ids)
        )

    profiles = []
    for cid in sorted(community_ids):
        row = profiles_df[profiles_df["community_id"] == cid].iloc[0]
        count = int(row["count"])

        # Physics z-scores
        physics_z = extract_z_scores(profiles_df, cid, prefix="z_")

        # Outcome z-scores (if available)
        outcome_z = {}
        if outcome_profiles_df is not None:
            outcome_z = extract_z_scores(outcome_profiles_df, cid, prefix="z_")

        # Surface proportions
        hard_pct = 0.5
        grass_pct = 0.5
        if surface_props is not None:
            # Surface props CSV format: row_0 (community_id), 0 (Hard %), 1 (Grass %)
            try:
                # Check if there's an index column like 'row_0'
                if "row_0" in surface_props.columns:
                    surf_row = surface_props[surface_props["row_0"] == cid]
                    if not surf_row.empty:
                        hard_pct = float(surf_row["0"].iloc[0]) if "0" in surf_row.columns else 0.5
                        grass_pct = float(surf_row["1"].iloc[0]) if "1" in surf_row.columns else 0.5
                else:
                    # Try using index directly
                    if cid in surface_props.index:
                        surf_row = surface_props.loc[cid]
                        hard_pct = float(surf_row.get(0, surf_row.get("0", 0.5)))
                        grass_pct = float(surf_row.get(1, surf_row.get("1", 0.5)))
            except Exception as e:
                print(f"Warning: Could not parse surface props for community {cid}: {e}")

        # Top drivers
        top_physics = get_top_drivers(physics_z, top_k=5)
        top_outcomes = get_top_drivers(outcome_z, top_k=5) if outcome_z else []

        # Assignment confidence
        mean_dist, std_dist = confidence_map.get(cid, (0.0, 0.0))

        # Merge info
        merged_from = []
        if merge_map:
            merged_from = merge_map.get(str(cid), [])

        # Derived name
        derived_name = derive_archetype_name(physics_z, outcome_z)

        profile = CommunityProfile(
            community_id=cid,
            count=count,
            pct_of_total=100.0 * count / total_count,
            hard_pct=hard_pct * 100,
            grass_pct=grass_pct * 100,
            physics_z=physics_z,
            outcome_z=outcome_z,
            top_physics_drivers=top_physics,
            top_outcome_correlates=top_outcomes,
            mean_distance_to_centroid=mean_dist,
            std_distance_to_centroid=std_dist,
            derived_name=derived_name,
            merged_from=merged_from,
        )
        profiles.append(profile)

    return profiles


def _rank_by_metric(profiles: List[CommunityProfile], metric_fn) -> Tuple[Dict[int, float], Dict[int, int], List[int]]:
    """Return metric values, rank map (1=highest), and ordered community ids."""
    values = {int(p.community_id): float(metric_fn(p)) for p in profiles}
    ordered = sorted(values.keys(), key=lambda cid: values[cid], reverse=True)
    rank = {cid: i + 1 for i, cid in enumerate(ordered)}
    return values, rank, ordered


def _physics_support(profile: CommunityProfile, feature_key: str) -> Dict[str, float]:
    return {
        "feature_display_name": FEATURE_DISPLAY_NAMES.get(feature_key, feature_key),
        "z": round(float(profile.physics_z.get(feature_key, 0.0)), 4),
    }


def _outcome_support(profile: CommunityProfile, outcome_key: str) -> Dict[str, float]:
    return {
        "outcome_display_name": FEATURE_DISPLAY_NAMES.get(outcome_key, outcome_key),
        "z": round(float(profile.outcome_z.get(outcome_key, 0.0)), 4),
    }


def build_coach_facts_v2(
    profiles: List[CommunityProfile],
    run_metrics: RunLevelMetrics,
) -> Dict[str, Any]:
    """
    Build deterministic coach-facing facts from validated community profiles.

    All claims are generated from existing z-score fields only.
    """
    if not profiles:
        return {
            "schema_version": "coach_facts_v2",
            "community_ids": [],
            "communities": [],
            "cross_community_facts": [],
            "valid_fact_ids": [],
        }

    profiles_sorted = sorted(profiles, key=lambda p: p.community_id)
    n_communities = len(profiles_sorted)

    speed_vals, speed_rank, speed_order = _rank_by_metric(profiles_sorted, lambda p: p.physics_z.get("Avg Speed", 0.0))
    pace_drop_vals, pace_drop_rank, _ = _rank_by_metric(
        profiles_sorted,
        lambda p: p.physics_z.get("1st Avg", 0.0) - p.physics_z.get("2nd Avg", 0.0),
    )
    t_vals, t_rank, t_order = _rank_by_metric(profiles_sorted, lambda p: p.physics_z.get("%_serves_T", 0.0))
    wide_vals, wide_rank, wide_order = _rank_by_metric(profiles_sorted, lambda p: p.physics_z.get("%_serves_Wide", 0.0))
    body_vals, body_rank, body_order = _rank_by_metric(profiles_sorted, lambda p: p.physics_z.get("%_serves_Body", 0.0))
    variation_vals, variation_rank, _ = _rank_by_metric(profiles_sorted, lambda p: p.physics_z.get("serve_width_var", 0.0))
    adaptability_vals, adaptability_rank, _ = _rank_by_metric(
        profiles_sorted,
        lambda p: (
            p.physics_z.get("change_T", 0.0)
            + p.physics_z.get("change_Wide", 0.0)
            + p.physics_z.get("change_Body", 0.0)
        ) / 3.0,
    )
    ace_vals, ace_rank, ace_order = _rank_by_metric(profiles_sorted, lambda p: p.outcome_z.get("player_ace", 0.0))
    first_reward_vals, first_reward_rank, _ = _rank_by_metric(
        profiles_sorted,
        lambda p: (
            p.outcome_z.get("player_ace", 0.0)
            + p.outcome_z.get("player_1stWon", 0.0)
            + p.outcome_z.get("player_1stIn", 0.0)
        ) / 3.0,
    )
    second_risk_vals, second_risk_rank, _ = _rank_by_metric(
        profiles_sorted,
        lambda p: p.outcome_z.get("player_df", 0.0) - p.outcome_z.get("player_2ndWon", 0.0),
    )
    pressure_vals, pressure_rank, _ = _rank_by_metric(
        profiles_sorted,
        lambda p: (
            p.outcome_z.get("player_bpSaved", 0.0)
            - p.outcome_z.get("player_bpFaced", 0.0)
            + p.outcome_z.get("player_SvGms", 0.0)
        ) / 3.0,
    )

    communities_payload: List[Dict[str, Any]] = []
    valid_fact_ids: List[str] = []

    for p in profiles_sorted:
        cid = int(p.community_id)
        fact_idx = 1
        facts: List[Dict[str, Any]] = []

        def add_fact(topic: str, coach_claim: str, support: List[Dict[str, Any]]) -> None:
            nonlocal fact_idx
            fact_id = f"C{cid}_F{fact_idx:02d}"
            facts.append(
                {
                    "fact_id": fact_id,
                    "community_id": int(cid),
                    "topic": topic,
                    "coach_claim": coach_claim,
                    "support": support,
                }
            )
            valid_fact_ids.append(fact_id)
            fact_idx += 1

        # F01: pace profile
        if speed_rank[cid] == 1:
            pace_claim = "This is the highest-pace community relative to the other communities, with speed pressure on both first and second serves."
        elif speed_rank[cid] == n_communities:
            pace_claim = "This is the lowest-pace community relative to the other communities, so opponents should expect less raw speed pressure."
        else:
            pace_claim = "This community sits in the middle on overall pace relative to the other communities, blending speed threat with control phases."
        add_fact(
            topic="pace",
            coach_claim=pace_claim,
            support=[
                _physics_support(p, "Avg Speed"),
                _physics_support(p, "1st Avg"),
                _physics_support(p, "2nd Avg"),
            ],
        )

        # F02: first vs second tempo change
        if pace_drop_rank[cid] == 1:
            first_second_claim = "The first-to-second serve pace drop is the largest among communities, giving returners a clearer tempo shift after missed first serves."
        elif pace_drop_rank[cid] == n_communities:
            first_second_claim = "The first-to-second serve pace drop is the smallest among communities, which reduces tempo clues between deliveries."
        else:
            first_second_claim = "The first-to-second serve pace drop is moderate relative to peers, so returners still need to read location cues."
        add_fact(
            topic="first_vs_second",
            coach_claim=first_second_claim,
            support=[
                _physics_support(p, "1st Avg"),
                _physics_support(p, "2nd Avg"),
                {"feature_display_name": "First-Second Pace Gap Index", "z": round(float(pace_drop_vals[cid]), 4)},
            ],
        )

        # F03: primary lane usage
        lane_values = {
            "T": p.physics_z.get("%_serves_T", 0.0),
            "Wide": p.physics_z.get("%_serves_Wide", 0.0),
            "Body": p.physics_z.get("%_serves_Body", 0.0),
        }
        dominant_lane = max(lane_values.items(), key=lambda item: item[1])[0]
        if lane_values[dominant_lane] > 0.35:
            lane_claim = f"Targeting leans most toward {dominant_lane} lanes relative to baseline, so anticipation patterns can key on that lane first."
        elif all(abs(v) < 0.25 for v in lane_values.values()):
            lane_claim = "Lane usage is relatively balanced across T, wide, and body, which lowers simple pre-positioning cues."
        else:
            lane_claim = f"Lane usage is mixed but still tilts toward {dominant_lane}, so return positioning should stay flexible until toss and contact."
        add_fact(
            topic="location",
            coach_claim=lane_claim,
            support=[
                _physics_support(p, "%_serves_T"),
                _physics_support(p, "%_serves_Wide"),
                _physics_support(p, "%_serves_Body"),
            ],
        )

        # F04: body-lane pressure
        if body_rank[cid] == 1:
            body_claim = "Body-lane usage is the highest among communities, so a body-jam scenario should be treated as a primary expectation."
        elif body_rank[cid] == n_communities:
            body_claim = "Body-lane usage is the lowest among communities, so anticipation should center more on T and wide lanes."
        else:
            body_claim = "Body-lane usage is present but not dominant relative to peers, creating selective jam pressure rather than constant body targeting."
        add_fact(
            topic="body_jam",
            coach_claim=body_claim,
            support=[
                _physics_support(p, "%_serves_Body"),
                _physics_support(p, "%_serves_T"),
                _physics_support(p, "%_serves_Wide"),
            ],
        )

        # F05: variation / predictability
        if variation_rank[cid] == 1:
            variation_claim = "Placement variation across serve width is the highest among communities, which makes lane prediction harder for returners."
        elif variation_rank[cid] == n_communities:
            variation_claim = "Placement variation across serve width is the lowest among communities, which can make serving patterns easier to anticipate."
        else:
            variation_claim = "Placement variation is middle-range relative to peers, so opponents can form tendencies but not lock in one lane."
        add_fact(
            topic="variation",
            coach_claim=variation_claim,
            support=[
                _physics_support(p, "serve_width_var"),
                _physics_support(p, "serve_depth_var"),
            ],
        )

        # F06: adaptability
        if adaptability_rank[cid] == 1:
            adapt_claim = "Lane adaptability is strongest among communities, suggesting this group can shift location plans more actively across contexts."
        elif adaptability_rank[cid] == n_communities:
            adapt_claim = "Lane adaptability is weakest among communities, suggesting a more fixed location script that opponents may decode over time."
        else:
            adapt_claim = "Lane adaptability is moderate relative to peers, with some but not constant location re-patterning."
        add_fact(
            topic="adaptability",
            coach_claim=adapt_claim,
            support=[
                _physics_support(p, "change_T"),
                _physics_support(p, "change_Wide"),
                _physics_support(p, "change_Body"),
            ],
        )

        # F07: first-serve reward profile
        if first_reward_rank[cid] == 1:
            first_reward_claim = "First-serve reward is strongest among communities, combining free-point potential with stronger first-ball outcomes."
        elif first_reward_rank[cid] == n_communities:
            first_reward_claim = "First-serve reward is weakest among communities, so this profile relies less on immediate first-ball advantage."
        else:
            first_reward_claim = "First-serve reward is middle-tier relative to peers, so match leverage comes from sequencing rather than only first-ball dominance."
        add_fact(
            topic="first_serve_reward",
            coach_claim=first_reward_claim,
            support=[
                _outcome_support(p, "player_ace"),
                _outcome_support(p, "player_1stWon"),
                _outcome_support(p, "player_1stIn"),
            ],
        )

        # F08: second-serve risk profile
        if second_risk_rank[cid] == 1:
            second_risk_claim = "Second-serve risk is highest among communities, combining more fault pressure with weaker second-serve point outcomes."
        elif second_risk_rank[cid] == n_communities:
            second_risk_claim = "Second-serve risk is lowest among communities, indicating a safer second-delivery profile under pressure."
        else:
            second_risk_claim = "Second-serve risk is mid-range relative to peers, with trade-offs between aggression and containment."
        add_fact(
            topic="second_serve_risk",
            coach_claim=second_risk_claim,
            support=[
                _outcome_support(p, "player_df"),
                _outcome_support(p, "player_2ndWon"),
                _physics_support(p, "Min 2nd"),
            ],
        )

        # F09: pressure management
        if pressure_rank[cid] == 1:
            pressure_claim = "Scoreboard-pressure outcomes are strongest among communities, with better service-game protection and pressure recovery signals."
        elif pressure_rank[cid] == n_communities:
            pressure_claim = "Scoreboard-pressure outcomes are weakest among communities, leaving this profile more exposed during return pressure phases."
        else:
            pressure_claim = "Scoreboard-pressure profile is mixed relative to peers, with partial protection but visible stress points."
        add_fact(
            topic="pressure",
            coach_claim=pressure_claim,
            support=[
                _outcome_support(p, "player_bpSaved"),
                _outcome_support(p, "player_bpFaced"),
                _outcome_support(p, "player_SvGms"),
            ],
        )

        # F10: identity anchor
        top_physics_feat, top_physics_val = (p.top_physics_drivers[0] if p.top_physics_drivers else ("Avg Speed", p.physics_z.get("Avg Speed", 0.0)))
        top_outcome_feat, top_outcome_val = (p.top_outcome_correlates[0] if p.top_outcome_correlates else ("player_ace", p.outcome_z.get("player_ace", 0.0)))
        identity_claim = (
            f"The style identity is anchored by {FEATURE_DISPLAY_NAMES.get(top_physics_feat, top_physics_feat)} "
            f"and {FEATURE_DISPLAY_NAMES.get(top_outcome_feat, top_outcome_feat)}, which should frame scouting priorities."
        )
        add_fact(
            topic="identity",
            coach_claim=identity_claim,
            support=[
                {"feature_display_name": FEATURE_DISPLAY_NAMES.get(top_physics_feat, top_physics_feat), "z": round(float(top_physics_val), 4)},
                {"outcome_display_name": FEATURE_DISPLAY_NAMES.get(top_outcome_feat, top_outcome_feat), "z": round(float(top_outcome_val), 4)},
            ],
        )

        communities_payload.append(
            {
                "community_id": cid,
                "community_label": f"C{cid}",
                "facts": facts,
                "rank_snapshot": {
                    "speed_rank": int(speed_rank[cid]),
                    "t_rank": int(t_rank[cid]),
                    "wide_rank": int(wide_rank[cid]),
                    "body_rank": int(body_rank[cid]),
                    "ace_rank": int(ace_rank[cid]),
                    "second_serve_risk_rank": int(second_risk_rank[cid]),
                },
            }
        )

    cross_facts: List[Dict[str, Any]] = []
    cross_idx = 1

    def add_cross_fact(topic: str, claim: str, support: List[Dict[str, Any]], ranking: List[int]) -> None:
        nonlocal cross_idx
        fact_id = f"XR_F{cross_idx:02d}"
        cross_facts.append(
            {
                "fact_id": fact_id,
                "community_id": None,
                "topic": topic,
                "coach_claim": claim,
                "ranking": [int(cid) for cid in ranking],
                "support": support,
            }
        )
        valid_fact_ids.append(fact_id)
        cross_idx += 1

    add_cross_fact(
        topic="speed_rank",
        claim="Speed ranking from highest to lowest is " + " > ".join(f"C{cid}" for cid in speed_order) + ".",
        ranking=speed_order,
        support=[
            {
                "community_id": int(p.community_id),
                "feature_display_name": FEATURE_DISPLAY_NAMES.get("Avg Speed", "Avg Speed"),
                "z": round(float(speed_vals[int(p.community_id)]), 4),
            }
            for p in profiles_sorted
        ],
    )
    add_cross_fact(
        topic="t_rank",
        claim="T-lane ranking from highest to lowest is " + " > ".join(f"C{cid}" for cid in t_order) + ".",
        ranking=t_order,
        support=[
            {
                "community_id": int(p.community_id),
                "feature_display_name": FEATURE_DISPLAY_NAMES.get("%_serves_T", "%_serves_T"),
                "z": round(float(t_vals[int(p.community_id)]), 4),
            }
            for p in profiles_sorted
        ],
    )
    add_cross_fact(
        topic="wide_rank",
        claim="Wide-lane ranking from highest to lowest is " + " > ".join(f"C{cid}" for cid in wide_order) + ".",
        ranking=wide_order,
        support=[
            {
                "community_id": int(p.community_id),
                "feature_display_name": FEATURE_DISPLAY_NAMES.get("%_serves_Wide", "%_serves_Wide"),
                "z": round(float(wide_vals[int(p.community_id)]), 4),
            }
            for p in profiles_sorted
        ],
    )
    add_cross_fact(
        topic="body_rank",
        claim="Body-lane ranking from highest to lowest is " + " > ".join(f"C{cid}" for cid in body_order) + ".",
        ranking=body_order,
        support=[
            {
                "community_id": int(p.community_id),
                "feature_display_name": FEATURE_DISPLAY_NAMES.get("%_serves_Body", "%_serves_Body"),
                "z": round(float(body_vals[int(p.community_id)]), 4),
            }
            for p in profiles_sorted
        ],
    )
    add_cross_fact(
        topic="ace_rank",
        claim="Ace-association ranking from highest to lowest is " + " > ".join(f"C{cid}" for cid in ace_order) + ".",
        ranking=ace_order,
        support=[
            {
                "community_id": int(p.community_id),
                "outcome_display_name": FEATURE_DISPLAY_NAMES.get("player_ace", "player_ace"),
                "z": round(float(ace_vals[int(p.community_id)]), 4),
            }
            for p in profiles_sorted
        ],
    )

    return {
        "schema_version": "coach_facts_v2",
        "run_context": {
            "n_communities": int(run_metrics.n_communities),
            "n_samples": int(run_metrics.n_samples),
            "modularity": float(run_metrics.modularity),
            "stability_nmi_mean": float(run_metrics.stability_nmi_mean),
            "stability_nmi_std": float(run_metrics.stability_nmi_std),
            "n_significant_outcomes": int(run_metrics.n_significant_outcomes),
            "n_outcomes_tested": int(run_metrics.n_outcomes_tested),
        },
        "community_ids": [int(p.community_id) for p in profiles_sorted],
        "communities": communities_payload,
        "cross_community_facts": cross_facts,
        "valid_fact_ids": sorted(valid_fact_ids),
        "derived_indices": {
            "pace_drop_index": {f"C{cid}": round(float(v), 4) for cid, v in pace_drop_vals.items()},
            "adaptability_index": {f"C{cid}": round(float(v), 4) for cid, v in adaptability_vals.items()},
            "first_serve_reward_index": {f"C{cid}": round(float(v), 4) for cid, v in first_reward_vals.items()},
            "second_serve_risk_index": {f"C{cid}": round(float(v), 4) for cid, v in second_risk_vals.items()},
            "pressure_index": {f"C{cid}": round(float(v), 4) for cid, v in pressure_vals.items()},
            "variation_index": {f"C{cid}": round(float(v), 4) for cid, v in variation_vals.items()},
        },
    }


def generate_coach_analysis_prompt_v2(coach_facts: Dict[str, Any]) -> str:
    """Generate grounded coach prompt that references deterministic fact IDs."""
    lines: List[str] = []
    community_ids = coach_facts.get("community_ids", [])
    valid_fact_ids = coach_facts.get("valid_fact_ids", [])

    lines.append("=" * 80)
    lines.append("GROUND-TRUTH FACT PACK: COACH REPORT V2")
    lines.append("=" * 80)
    lines.append("")
    lines.append("You are writing a coach-facing technical serve report from deterministic facts.")
    lines.append("You MUST use only the fact IDs and claims listed below.")
    lines.append("")
    lines.append("OUTPUT FORMAT (MANDATORY):")
    lines.append("- Markdown only.")
    lines.append("- Keep each paragraph or bullet on a single line.")
    lines.append("- End EVERY paragraph or bullet with a citation comment:")
    lines.append("  <!--FACTS: C0_F01,C0_F04 -->")
    lines.append("- Use only fact IDs from this prompt. Do not invent IDs.")
    lines.append("")
    lines.append("HARD CONSTRAINTS (MANDATORY):")
    lines.append("- Do NOT include percentages, z-scores, decimals, or made-up numbers.")
    lines.append("- Do NOT include surfaces.")
    lines.append("- Do NOT include player names.")
    lines.append("- Do NOT include unsupported serve-type labels: kick, flat, slice, topspin.")
    lines.append("- Do NOT include biomechanics language.")
    lines.append("- Do NOT include any line beginning with EVIDENCE_LOCK|.")
    lines.append("")
    lines.append("REQUIRED STYLE:")
    lines.append("- Use coach-technical language grounded in facts (pace-first vs location-first, body-jam bias, predictability, second-serve risk, adaptability).")
    lines.append("- Avoid generic archetype labels without mechanism-level explanation (for example: balanced, aggressive, defensive by themselves).")
    lines.append("- Every key claim should connect mechanism to match implication using only cited facts.")
    lines.append("- For EACH community section, include at least two sentences that start exactly with:")
    lines.append("  Compared to the other communities,")
    lines.append("- Include opponent positioning/anticipation hypotheses in each community section.")
    lines.append("")
    lines.append("REQUIRED TECHNICAL COVERAGE PER COMMUNITY:")
    lines.append("- Pace profile and first-versus-second tempo shift.")
    lines.append("- Location geometry (T, wide, body) plus predictability/variation or adaptability.")
    lines.append("- Outcome reward-risk profile (first-serve reward and second-serve risk).")
    lines.append("- Scoreboard pressure profile (break-point exposure and protection signals).")
    lines.append("- Opponent positioning and anticipation hypotheses framed as testable tendencies.")
    lines.append("")
    lines.append("REQUIRED REPORT STRUCTURE:")
    lines.append("1. Executive summary (differences only)")
    for cid in community_ids:
        lines.append(f"2. Community C{cid}: identity, patterns, match implications, opponent positioning hypotheses")
    lines.append("3. Cross-community trade-offs (who wins what and why, qualitatively)")
    lines.append("4. Caveats")
    lines.append("5. Keep the full report compact and information-dense (about twenty to thirty single-line bullets/paragraphs).")
    lines.append("")
    lines.append("VALID FACT IDS:")
    lines.append(", ".join(valid_fact_ids))
    lines.append("")
    lines.append("=" * 80)
    lines.append("FACT CATALOG")
    lines.append("=" * 80)
    lines.append("")

    for community in coach_facts.get("communities", []):
        cid = community.get("community_id")
        lines.append(f"### Community C{cid} facts")
        for fact in community.get("facts", []):
            lines.append(
                f"- {fact.get('fact_id')} | topic={fact.get('topic')} | claim={fact.get('coach_claim')}"
            )
        lines.append("")

    lines.append("### Cross-community ranking facts")
    for fact in coach_facts.get("cross_community_facts", []):
        ranking = fact.get("ranking", [])
        rank_txt = " > ".join(f"C{cid}" for cid in ranking) if ranking else "N/A"
        lines.append(
            f"- {fact.get('fact_id')} | topic={fact.get('topic')} | claim={fact.get('coach_claim')} | ranking={rank_txt}"
        )
    lines.append("")
    lines.append("Now write the report in Markdown with the required citation comments.")
    lines.append("")

    return "\n".join(lines)


def plot_community_comparison_radar(
    profiles: List[CommunityProfile],
    features: List[str],
    out_path: Path,
    title: str = "Community Physics Profiles (Z-Scores)",
) -> None:
    """Radar plot comparing community profiles."""
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    for i, profile in enumerate(profiles):
        values = [profile.physics_z.get(f, 0) for f in features]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=f"C{profile.community_id}: {profile.derived_name}", color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
    ax.set_ylim(-2, 2)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_outcome_heatmap(
    profiles: List[CommunityProfile],
    out_path: Path,
) -> None:
    """Heatmap of outcome z-scores by community."""
    data = {}
    for p in profiles:
        data[f"C{p.community_id}"] = p.outcome_z

    df = pd.DataFrame(data).T
    if df.empty:
        print("No outcome data for heatmap")
        return

    # Reorder columns by importance
    col_order = [c for c in OUTCOME_COLS if c in df.columns]
    df = df[col_order]

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df,
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


def plot_surface_association(
    profiles: List[CommunityProfile],
    out_path: Path,
) -> None:
    """Stacked bar chart showing surface association per community."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(profiles))
    width = 0.6

    hard_pcts = [p.hard_pct for p in profiles]
    grass_pcts = [p.grass_pct for p in profiles]

    ax.bar(x, hard_pcts, width, label="Hard Court", color="coral")
    ax.bar(x, grass_pcts, width, bottom=hard_pcts, label="Grass Court", color="mediumseagreen")

    ax.axhline(y=50, color="black", linestyle="--", linewidth=1, alpha=0.5, label="50% baseline")

    ax.set_xlabel("Community", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_title("Surface Association by Community\n(Not Independence - Enrichment Exists)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{p.community_id}\n{p.derived_name}" for p in profiles], fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_top_drivers_comparison(
    profiles: List[CommunityProfile],
    out_path: Path,
) -> None:
    """Bar chart comparing top physics drivers across communities."""
    # Collect all unique top features
    all_features = set()
    for p in profiles:
        for feat, _ in p.top_physics_drivers[:3]:
            all_features.add(feat)

    features = sorted(all_features)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(features))
    width = 0.8 / len(profiles)
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    for i, p in enumerate(profiles):
        values = [p.physics_z.get(f, 0) for f in features]
        offset = (i - len(profiles) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f"C{p.community_id}: {p.derived_name}", color=colors[i])

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Z-Score", fontsize=12)
    ax.set_title("Top Physics Drivers by Community", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=9, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def _to_json(obj: Any) -> Any:
    """Convert numpy scalars and arrays to JSON-serialisable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(x) for x in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve style profiles and structured evidence statements.")
    parser.add_argument("--community-run-dir", type=Path, required=True,
                        help="Run directory written by courtstyle.discovery.serve_discovery.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/serve/processed"),
                        help="Directory with train/val data and outcome sidecars.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <community-run-dir>/interpretation.")
    parser.add_argument("--prompts-subdir", type=str, default="prompts_coach_v2")
    args = parser.parse_args()
    for path in (args.community_run_dir, args.processed_dir):
        if not path.exists():
            parser.exit(1, f"Missing input: {path}\n")
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or (args.community_run_dir / "interpretation")
    figures_dir = out_dir / "figures"
    prompts_dir = out_dir / args.prompts_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    community_outputs = load_community_discovery_outputs(args.community_run_dir)
    load_original_data(
        args.processed_dir / "train_data.csv", args.processed_dir / "val_data.csv",
        args.processed_dir / "train_outcomes.csv", args.processed_dir / "val_outcomes.csv",
    )
    all_labels = np.concatenate([
        community_outputs["community_labels_train"]["community_id"].values,
        community_outputs["community_labels_val"]["community_id"].values,
    ])
    emb_path = args.community_run_dir / "embeddings" / "all_embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None

    run_metrics = extract_run_level_metrics(community_outputs["community_metrics"])
    profiles = build_community_profiles(community_outputs, embeddings=embeddings, labels=all_labels)

    # Structured evidence statements and the grounded prompt.
    coach_facts = build_coach_facts_v2(profiles, run_metrics)
    (prompts_dir / "coach_analysis_v2.txt").write_text(generate_coach_analysis_prompt_v2(coach_facts))
    with open(prompts_dir / "coach_facts_v2.json", "w") as f:
        json.dump(coach_facts, f, indent=2)

    radar_features = [f for f in ["Avg Speed", "1st Avg", "1st StDev", "%_serves_T", "%_serves_Wide",
                                  "%_serves_Body", "serve_width_var", "2nd Avg"] if f in profiles[0].physics_z]
    plot_community_comparison_radar(profiles, radar_features, figures_dir / "community_radar.png")
    plot_outcome_heatmap(profiles, figures_dir / "outcome_heatmap.png")
    plot_surface_association(profiles, figures_dir / "surface_association.png")
    plot_top_drivers_comparison(profiles, figures_dir / "top_drivers.png")

    report = {
        "run_metrics": {
            "n_communities": int(run_metrics.n_communities),
            "n_samples": int(run_metrics.n_samples),
            "modularity": float(run_metrics.modularity),
            "silhouette": float(run_metrics.silhouette),
            "surface_nmi": float(run_metrics.surface_nmi),
            "stability_nmi": f"{run_metrics.stability_nmi_mean:.3f} +/- {run_metrics.stability_nmi_std:.3f}",
            "significant_outcomes": f"{run_metrics.n_significant_outcomes}/{run_metrics.n_outcomes_tested}",
        },
        "communities": [
            {
                "id": int(p.community_id),
                "count": int(p.count),
                "pct": round(float(p.pct_of_total), 1),
                "hard_pct": round(float(p.hard_pct), 1),
                "grass_pct": round(float(p.grass_pct), 1),
                "top_physics_drivers": [(feat, float(val)) for feat, val in p.top_physics_drivers],
                "top_outcome_correlates": [(feat, float(val)) for feat, val in p.top_outcome_correlates],
                "merged_from": [int(x) for x in p.merged_from] if p.merged_from else [],
            }
            for p in profiles
        ],
    }
    with open(out_dir / "interpretation_report.json", "w") as f:
        json.dump(_to_json(report), f, indent=2)
    print(f"Evidence statements and prompt: {prompts_dir}")
    print(f"Figures and report: {out_dir}")


if __name__ == "__main__":
    main()

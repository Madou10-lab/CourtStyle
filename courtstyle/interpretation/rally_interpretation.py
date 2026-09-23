"""Evidence statements and report prompt for the discovered rally styles.

Reads a rally discovery run directory and writes

  1. ``specialist_facts_v1.json``: a deterministic fact pack. Every statement
     carries an ID (``C<k>_F<nn>`` for one style, ``XR_F<nn>`` for cross-style
     comparisons) and the profile values that support it.
  2. ``specialist_report_prompt_v1.txt``: the report prompt. The language model
     may only use these statements and must cite their IDs in
     ``<!--FACTS: ...-->`` comments, which ``rally_reports`` validates.
  3. ``interpretation_report.json``: run-level metrics and a per-style summary.

Usage:
    python -m courtstyle.interpretation.rally_interpretation \\
        --run-dir outputs/rally/discovery
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature display names for rally state / action / delta features
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    # Continuous shot features
    "speed": "Shot Speed",
    "spin_rpm": "Spin Rate",
    "net_clearance_z_debug": "Net Clearance",
    "bounce_coordinate_y": "Bounce Depth",
    "bounce_coordinate_x": "Bounce Width",
    "player_position_at_hit_y": "Hitter Depth at Contact",
    "player_position_at_hit_x": "Hitter Width at Contact",
    "player_opponent_position_at_hit_y": "Opponent Depth at Contact",
    "player_opponent_position_at_hit_x": "Opponent Width at Contact",
    "player_position_at_bounce_y": "Hitter Depth at Bounce",
    "player_position_at_bounce_x": "Hitter Width at Bounce",
    "player_opponent_position_at_bounce_y": "Opponent Depth at Bounce",
    "player_opponent_position_at_bounce_x": "Opponent Width at Bounce",
    "start_pos_x": "Ball Start Width",
    "start_pos_y": "Ball Start Depth",
    "end_pos_x": "Ball End Width",
    "end_pos_y": "Ball End Depth",
    "duration": "Shot Duration",
    "distance_travelled_by_player": "Player Distance Covered",
    # Categorical shot features
    "is_volley": "Volley Shot",
    "is_return": "Return Shot",
    "is_groundstroke": "Groundstroke",
    "is_rally": "Rally Shot",
    "is_overhead_smash": "Overhead Smash",
    "stroke_backhand": "Backhand Stroke",
    "stroke_forehand": "Forehand Stroke",
    "stroke_serve": "Serve Stroke",
    "spin_type_back": "Backspin",
    "spin_type_none": "No Spin",
    "spin_type_top": "Topspin",
    "spin_type_unknown": "Unknown Spin",
    # Outcome columns
    "Fault": "Fault Rate",
    "Let": "Let Rate",
    "Played": "Played Rate",
}

# Features used for serve/return contrast summaries
CONTRAST_FEATURES = ["speed", "spin_rpm", "net_clearance_z_debug", "bounce_coordinate_y", "end_pos_y"]

# Delta dynamics features of coaching interest
DELTA_CURATED = [
    "speed", "spin_rpm", "net_clearance_z_debug", "bounce_coordinate_y",
    "duration", "distance_travelled_by_player",
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CommunityProfile:
    community_id: int
    count: int
    pct_of_total: float
    mean_n_transitions: float
    # Outcome rates
    fault_rate: float = 0.0
    let_rate: float = 0.0
    played_rate: float = 0.0
    # Z-score profiles  {raw_col: z}
    state_z: Dict[str, float] = field(default_factory=dict)
    action_z: Dict[str, float] = field(default_factory=dict)
    delta_z: Dict[str, float] = field(default_factory=dict)
    # Serve / return shot summaries  {col: mean}
    serve_summary: Dict[str, float] = field(default_factory=dict)
    return_summary: Dict[str, float] = field(default_factory=dict)
    serve_return_contrast: Dict[str, float] = field(default_factory=dict)
    # Derived name (auto)
    derived_name: str = ""


@dataclass
class RunLevelMetrics:
    k: int
    n_samples: int
    silhouette: float
    stability_ari_mean: float
    stability_ari_std: float
    stability_nmi_mean: float
    stability_nmi_std: float
    confound_nmi_match: float = 0.0
    confound_nmi_rally_len: float = 0.0
    passed: bool = True


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------

def _rank_by_metric(profiles: List[CommunityProfile], metric_fn) -> Tuple[Dict[int, float], Dict[int, int], List[int]]:
    """Return (values, rank_map 1=highest, ordered_cids)."""
    values = {p.community_id: float(metric_fn(p)) for p in profiles}
    ordered = sorted(values.keys(), key=lambda cid: values[cid], reverse=True)
    rank = {cid: i + 1 for i, cid in enumerate(ordered)}
    return values, rank, ordered


def _display(col: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_profile_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def resolve_fit_labels_path(run_dir: Path, run_config: Dict[str, Any]) -> Path:
    """Resolve the rally label CSV for the discovery fit split."""
    com_dir = run_dir / "communities"
    fit_split = str(run_config.get("fit_split") or run_config.get("args", {}).get("fit_split") or "").strip()

    candidates: List[Path] = []
    if fit_split:
        candidates.append(com_dir / f"community_labels_{fit_split}.csv")
    candidates.extend(
        [
            com_dir / "community_labels_train.csv",
            com_dir / "community_labels_train_all.csv",
        ]
    )

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not resolve fit label CSV. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def extract_z_dict(row: pd.Series, raw_columns: List[str]) -> Dict[str, float]:
    """Extract z_<col> values from a profile row into {col: z}."""
    d: Dict[str, float] = {}
    for col in raw_columns:
        z_col = f"z_{col}"
        if z_col in row.index:
            d[col] = float(row[z_col])
    return d


# ---------------------------------------------------------------------------
# Serve / return shot summaries
# ---------------------------------------------------------------------------

def compute_serve_return_summaries(
    run_config: Dict[str, Any],
    labels_df: pd.DataFrame,
    contrast_cols: List[str],
) -> Dict[int, Dict[str, Any]]:
    """
    Join raw shots with community labels on (match_id, point_index).
    shot_index=0 -> serve shot,  shot_index=1 -> return shot.
    Returns per-community summaries for serve, return, and contrast.
    """
    args = run_config.get("args", {})
    shots_path = args.get("train_shots_csv", "")
    if not shots_path or not Path(shots_path).exists():
        print(f"  [WARN] train_shots_csv not found: {shots_path}")
        return {}

    shots = pd.read_csv(shots_path)

    # Keep only serve (shot_index=0) and return (shot_index=1)
    serve_shots = shots[shots["shot_index"] == 0][["match_id", "point_index"] + contrast_cols].copy()
    return_shots = shots[shots["shot_index"] == 1][["match_id", "point_index"] + contrast_cols].copy()

    # Join with labels
    merge_keys = ["match_id", "point_index"]
    labels_merge = labels_df[["match_id", "point_index", "community_id"]].drop_duplicates(subset=merge_keys)

    serve_merged = serve_shots.merge(labels_merge, on=merge_keys, how="inner")
    return_merged = return_shots.merge(labels_merge, on=merge_keys, how="inner")

    result: Dict[int, Dict[str, Any]] = {}
    for cid in sorted(labels_df["community_id"].unique()):
        s = serve_merged[serve_merged["community_id"] == cid]
        r = return_merged[return_merged["community_id"] == cid]
        serve_means = {c: float(s[c].mean()) if len(s) > 0 else 0.0 for c in contrast_cols}
        return_means = {c: float(r[c].mean()) if len(r) > 0 else 0.0 for c in contrast_cols}
        contrast = {c: serve_means[c] - return_means[c] for c in contrast_cols}
        result[cid] = {
            "serve": serve_means,
            "return": return_means,
            "contrast": contrast,
            "n_serve": len(s),
            "n_return": len(r),
        }
    return result


# ---------------------------------------------------------------------------
# Build community profiles
# ---------------------------------------------------------------------------

def build_profiles(
    run_dir: Path,
    run_config: Dict[str, Any],
    raw_columns: List[str],
) -> List[CommunityProfile]:
    """Load all community data and build profile objects."""
    com_dir = run_dir / "communities"

    # Labels
    labels_train = pd.read_csv(resolve_fit_labels_path(run_dir, run_config))
    total = len(labels_train)

    # Profiles
    state_df = load_profile_csv(com_dir / "community_state_profiles_raw.csv")
    action_df = load_profile_csv(com_dir / "community_action_profiles_raw.csv")
    delta_df = load_profile_csv(com_dir / "community_delta_profiles_raw.csv")

    # Outcomes
    outcome_df = load_profile_csv(com_dir / "community_outcome_profiles_train.csv")
    outcome_dict: Dict[int, Dict[str, float]] = {}
    if not outcome_df.empty:
        for _, row in outcome_df.iterrows():
            cid = int(row["community_id"])
            outcome_dict[cid] = {
                "Fault": float(row.get("Fault", 0)),
                "Let": float(row.get("Let", 0)),
                "Played": float(row.get("Played", 0)),
            }

    # Mean transitions per community
    trans_means: Dict[int, float] = {}
    for cid, grp in labels_train.groupby("community_id"):
        trans_means[int(cid)] = float(grp["n_transitions"].mean())

    # Serve/return summaries
    sr_summaries = compute_serve_return_summaries(run_config, labels_train, CONTRAST_FEATURES)

    profiles: List[CommunityProfile] = []
    community_ids = sorted(state_df["community_id"].astype(int).tolist()) if not state_df.empty else sorted(labels_train["community_id"].unique())

    for cid in community_ids:
        count = int((labels_train["community_id"] == cid).sum())
        p = CommunityProfile(
            community_id=cid,
            count=count,
            pct_of_total=round(100.0 * count / total, 2) if total > 0 else 0.0,
            mean_n_transitions=round(trans_means.get(cid, 0.0), 2),
        )
        # Z-scores
        if not state_df.empty:
            row = state_df[state_df["community_id"] == cid]
            if len(row) == 1:
                p.state_z = extract_z_dict(row.iloc[0], raw_columns)
        if not action_df.empty:
            row = action_df[action_df["community_id"] == cid]
            if len(row) == 1:
                p.action_z = extract_z_dict(row.iloc[0], raw_columns)
        if not delta_df.empty:
            row = delta_df[delta_df["community_id"] == cid]
            if len(row) == 1:
                p.delta_z = extract_z_dict(row.iloc[0], raw_columns)

        # Outcomes
        oc = outcome_dict.get(cid, {})
        p.fault_rate = oc.get("Fault", 0.0)
        p.let_rate = oc.get("Let", 0.0)
        p.played_rate = oc.get("Played", 0.0)

        # Serve/return
        sr = sr_summaries.get(cid, {})
        p.serve_summary = sr.get("serve", {})
        p.return_summary = sr.get("return", {})
        p.serve_return_contrast = sr.get("contrast", {})

        profiles.append(p)

    return profiles


# ---------------------------------------------------------------------------
# Derive archetype name (data-driven, not hardcoded)
# ---------------------------------------------------------------------------

def derive_archetype_name(p: CommunityProfile) -> str:
    """Derive a short rally-archetype name from profile z-scores."""
    parts: List[str] = []

    # Primary: serve speed from state profile
    speed_z = p.state_z.get("speed", 0.0)
    if speed_z > 0.10:
        parts.append("Fast-Entry")
    elif speed_z < -0.05:
        parts.append("Slow-Entry")
    else:
        parts.append("Mid-Entry")

    # Secondary: rally length proxy (mean_n_transitions)
    if p.mean_n_transitions > 4.5:
        parts.append("Extended")
    elif p.mean_n_transitions < 3.0:
        parts.append("Short")

    # Tertiary: delta dynamics
    delta_speed = p.delta_z.get("speed", 0.0)
    delta_dist = p.delta_z.get("distance_travelled_by_player", 0.0)
    if delta_speed > 0.05:
        parts.append("Escalating")
    elif delta_speed < -0.05:
        parts.append("Decelerating")

    if delta_dist > 0.06:
        parts.append("High-Move")

    # Outcome hint
    if p.fault_rate > 0.18:
        parts.append("(High-Fault)")
    elif p.played_rate > 0.87:
        parts.append("(High-Played)")

    return " ".join(parts) if parts else f"Schema-{p.community_id}"


# ---------------------------------------------------------------------------
# Specialist facts V1
# ---------------------------------------------------------------------------

def _z_support(z_dict: Dict[str, float], col: str, prefix: str = "") -> Dict[str, Any]:
    label = f"{prefix}{_display(col)}" if prefix else _display(col)
    return {"feature": label, "z": round(z_dict.get(col, 0.0), 4)}


def _contrast_support(contrast: Dict[str, float], col: str) -> Dict[str, Any]:
    return {"feature": f"Serve-Return {_display(col)} Gap", "value": round(contrast.get(col, 0.0), 4)}


def _top_z(z_dict: Dict[str, float], n: int = 3, positive: bool = True) -> List[Tuple[str, float]]:
    """Return top-n features by z magnitude (positive or negative direction)."""
    items = [(k, v) for k, v in z_dict.items() if (v > 0 if positive else v < 0)]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:n]


def build_specialist_facts_v1(
    profiles: List[CommunityProfile],
    run_metrics: RunLevelMetrics,
    run_dir: str,
    raw_columns: List[str],
) -> Dict[str, Any]:
    """Build deterministic specialist fact pack for rally schemas."""
    profiles_sorted = sorted(profiles, key=lambda p: p.community_id)
    k = len(profiles_sorted)

    # ---- size / outcomes ranks (for number-free coach claims) ----
    count_vals, count_rank, count_order = _rank_by_metric(
        profiles_sorted, lambda p: p.count
    )
    trans_vals, trans_rank, trans_order = _rank_by_metric(
        profiles_sorted, lambda p: p.mean_n_transitions
    )
    played_vals, played_rank, played_order = _rank_by_metric(
        profiles_sorted, lambda p: p.played_rate
    )
    let_vals, let_rank, let_order = _rank_by_metric(
        profiles_sorted, lambda p: p.let_rate
    )

    # ---- cross-community rankings ----
    speed_vals, speed_rank, speed_order = _rank_by_metric(
        profiles_sorted, lambda p: p.state_z.get("speed", 0.0)
    )
    spin_vals, spin_rank, spin_order = _rank_by_metric(
        profiles_sorted, lambda p: p.state_z.get("spin_rpm", 0.0)
    )
    net_vals, net_rank, net_order = _rank_by_metric(
        profiles_sorted, lambda p: p.state_z.get("net_clearance_z_debug", 0.0)
    )
    fault_vals, fault_rank, fault_order = _rank_by_metric(
        profiles_sorted, lambda p: p.fault_rate
    )
    movement_vals, movement_rank, movement_order = _rank_by_metric(
        profiles_sorted, lambda p: p.delta_z.get("distance_travelled_by_player", 0.0)
    )
    # Serve speed from serve_summary
    serve_speed_vals, serve_speed_rank, serve_speed_order = _rank_by_metric(
        profiles_sorted, lambda p: p.serve_summary.get("speed", 0.0)
    )
    return_clearance_vals, return_clearance_rank, return_clearance_order = _rank_by_metric(
        profiles_sorted, lambda p: p.return_summary.get("net_clearance_z_debug", 0.0)
    )
    contrast_speed_vals, contrast_speed_rank, contrast_speed_order = _rank_by_metric(
        profiles_sorted, lambda p: p.serve_return_contrast.get("speed", 0.0)
    )
    duration_inflation_vals, duration_inflation_rank, _ = _rank_by_metric(
        profiles_sorted, lambda p: p.delta_z.get("duration", 0.0)
    )

    communities_payload: List[Dict[str, Any]] = []
    valid_fact_ids: List[str] = []

    for p in profiles_sorted:
        cid = p.community_id
        facts: List[Dict[str, Any]] = []
        fidx = 1

        def add(topic: str, claim: str, support: List[Dict[str, Any]]) -> None:
            nonlocal fidx
            fid = f"C{cid}_F{fidx:02d}"
            facts.append({"fact_id": fid, "topic": topic, "coach_claim": claim, "support": support})
            valid_fact_ids.append(fid)
            fidx += 1

        # F01: size / rally length
        if count_rank[cid] == 1:
            size_clause = "the largest community by frequency"
        elif count_rank[cid] == k:
            size_clause = "the smallest community by frequency"
        else:
            size_clause = "a mid-sized community by frequency"

        if trans_rank[cid] == 1:
            trans_clause = "the most transitions per rally (more extended exchanges)"
        elif trans_rank[cid] == k:
            trans_clause = "the fewest transitions per rally (shorter exchanges)"
        else:
            trans_clause = "a mid-range transitions-per-rally level"

        add(
            "size",
            f"Community C{cid} is {size_clause} and tends to have {trans_clause}.",
            [{"metric": "count", "value": p.count},
             {"metric": "pct", "value": p.pct_of_total},
             {"metric": "mean_n_transitions", "value": p.mean_n_transitions}],
        )

        # F02: outcome rates
        def _rank_phrase(metric_label: str, rank_map: Dict[int, int]) -> str:
            r = rank_map[cid]
            if r == 1:
                return f"{metric_label} is the highest among communities"
            if r == k:
                return f"{metric_label} is the lowest among communities"
            return f"{metric_label} is mid-range relative to peers"

        outcome_claim = (
            "Outcome profile: "
            + _rank_phrase("played rate", played_rank)
            + "; "
            + _rank_phrase("fault rate", fault_rank)
            + "; "
            + _rank_phrase("let rate", let_rank)
            + "."
        )
        add(
            "outcome_rates",
            outcome_claim,
            [{"outcome": "Played", "rate": round(p.played_rate, 4)},
             {"outcome": "Fault", "rate": round(p.fault_rate, 4)},
             {"outcome": "Let", "rate": round(p.let_rate, 4)}],
        )

        # F03: serve profile (state-level speed/spin at entry)
        serve_claim_parts = []
        if serve_speed_rank[cid] == 1:
            serve_claim_parts.append("highest serve-shot speed among communities")
        elif serve_speed_rank[cid] == k:
            serve_claim_parts.append("lowest serve-shot speed among communities")
        else:
            serve_claim_parts.append("mid-range serve-shot speed")
        if speed_rank[cid] == 1:
            serve_claim_parts.append("highest overall state speed among communities")
        elif speed_rank[cid] == k:
            serve_claim_parts.append("lowest overall state speed among communities")
        add(
            "serve_profile",
            "Serve entry shows " + ", ".join(serve_claim_parts) + ".",
            [_z_support(p.state_z, "speed", "State "),
             _z_support(p.state_z, "spin_rpm", "State "),
             _z_support(p.state_z, "net_clearance_z_debug", "State ")],
        )

        # F04: return profile
        ret_parts = []
        if return_clearance_rank[cid] == 1:
            ret_parts.append("highest return net-clearance among communities")
        elif return_clearance_rank[cid] == k:
            ret_parts.append("lowest return net-clearance among communities")
        else:
            ret_parts.append("mid-range return net-clearance")
        add(
            "return_profile",
            "Return response shows " + ", ".join(ret_parts) + ".",
            [_z_support(p.action_z, "speed", "Action "),
             _z_support(p.action_z, "net_clearance_z_debug", "Action "),
             _z_support(p.action_z, "bounce_coordinate_y", "Action ")],
        )

        # F05: serve->return contrast
        contrast_parts = []
        if contrast_speed_rank[cid] == 1:
            contrast_parts.append("largest serve-to-return speed drop")
        elif contrast_speed_rank[cid] == k:
            contrast_parts.append("smallest serve-to-return speed drop (or return faster)")
        else:
            contrast_parts.append("moderate serve-to-return speed change")
        add(
            "serve_return_contrast",
            "Serve-to-return contrast shows " + ", ".join(contrast_parts) + ".",
            [_contrast_support(p.serve_return_contrast, c) for c in CONTRAST_FEATURES],
        )

        # F06: delta dynamics (top +/- drivers)
        top_pos = _top_z(p.delta_z, 3, positive=True)
        top_neg = _top_z(p.delta_z, 3, positive=False)
        delta_parts = []
        if top_pos:
            delta_parts.append("increasing " + ", ".join(_display(c) for c, _ in top_pos))
        if top_neg:
            delta_parts.append("decreasing " + ", ".join(_display(c) for c, _ in top_neg))
        delta_claim = "Delta dynamics drivers: " + "; ".join(delta_parts) + "." if delta_parts else "No strong delta dynamics drivers."
        delta_support = [_z_support(p.delta_z, c, "Delta ") for c, _ in top_pos + top_neg]
        add("delta_dynamics", delta_claim, delta_support)

        # F07: movement / tempo escalation
        dist_z = p.delta_z.get("distance_travelled_by_player", 0.0)
        dur_z = p.delta_z.get("duration", 0.0)
        if movement_rank[cid] == 1:
            move_claim = "Movement escalation is the highest among communities, indicating growing displacement through the rally."
        elif movement_rank[cid] == k:
            move_claim = "Movement escalation is the lowest among communities, suggesting compact rallies with limited displacement growth."
        else:
            move_claim = "Movement escalation is mid-range relative to peers."
        if duration_inflation_rank[cid] == 1:
            move_claim += " Tempo also inflates the most (longest shot-to-shot duration growth)."
        elif duration_inflation_rank[cid] == k:
            move_claim += " Tempo inflation is the smallest among communities."
        add(
            "movement_tempo",
            move_claim,
            [_z_support(p.delta_z, "distance_travelled_by_player", "Delta "),
             _z_support(p.delta_z, "duration", "Delta ")],
        )

        # F08: identity anchor (top state + top action feature)
        top_state = _top_z(p.state_z, 1, positive=True)
        top_action = _top_z(p.action_z, 1, positive=True)
        ts_feat = top_state[0] if top_state else ("speed", 0.0)
        ta_feat = top_action[0] if top_action else ("speed", 0.0)
        identity_claim = (
            f"Rally identity is anchored by state-level {_display(ts_feat[0])} "
            f"and action-level {_display(ta_feat[0])}, which frame scouting priorities."
        )
        add(
            "identity",
            identity_claim,
            [_z_support(p.state_z, ts_feat[0], "State "),
             _z_support(p.action_z, ta_feat[0], "Action ")],
        )

        communities_payload.append({
            "community_id": cid,
            "community_label": f"C{cid}",
            "derived_name": p.derived_name,
            "facts": facts,
            "rank_snapshot": {
                "serve_speed_rank": int(serve_speed_rank[cid]),
                "return_clearance_rank": int(return_clearance_rank[cid]),
                "contrast_speed_rank": int(contrast_speed_rank[cid]),
                "movement_rank": int(movement_rank[cid]),
                "fault_rank": int(fault_rank[cid]),
                "state_speed_rank": int(speed_rank[cid]),
            },
        })

    # ---- Cross-community facts ----
    cross_facts: List[Dict[str, Any]] = []
    xidx = 1

    def add_cross(topic: str, claim: str, ranking: List[int], support: List[Dict[str, Any]]) -> None:
        nonlocal xidx
        fid = f"XR_F{xidx:02d}"
        cross_facts.append({
            "fact_id": fid, "topic": topic,
            "coach_claim": claim, "ranking": ranking, "support": support,
        })
        valid_fact_ids.append(fid)
        xidx += 1

    def _ranking_claim(label: str, order: List[int]) -> str:
        return f"{label} ranking (highest to lowest): " + " > ".join(f"C{c}" for c in order) + "."

    def _ranking_support(vals: Dict[int, float], label: str) -> List[Dict[str, Any]]:
        return [{"community_id": cid, "feature": label, "value": round(v, 4)} for cid, v in sorted(vals.items())]

    add_cross("serve_pace", _ranking_claim("Serve-shot pace", serve_speed_order),
              serve_speed_order, _ranking_support(serve_speed_vals, "Serve Speed"))
    add_cross("return_clearance", _ranking_claim("Return net-clearance", return_clearance_order),
              return_clearance_order, _ranking_support(return_clearance_vals, "Return Net Clearance"))
    add_cross("contrast_speed_gap", _ranking_claim("Serve-to-return speed gap", contrast_speed_order),
              contrast_speed_order, _ranking_support(contrast_speed_vals, "Serve-Return Speed Gap"))
    add_cross("movement_inflation", _ranking_claim("Movement inflation (delta distance)", movement_order),
              list(movement_order), _ranking_support(movement_vals, "Delta Distance"))
    add_cross("fault_rate", _ranking_claim("Fault rate", fault_order),
              fault_order, _ranking_support(fault_vals, "Fault Rate"))
    add_cross("state_speed", _ranking_claim("Overall state speed", speed_order),
              speed_order, _ranking_support(speed_vals, "State Speed z"))

    return {
        "schema_version": "specialist_facts_v1",
        "run_dir": str(run_dir),
        "k": k,
        "run_validity": {
            "silhouette": run_metrics.silhouette,
            "stability_ari_mean": run_metrics.stability_ari_mean,
            "stability_ari_std": run_metrics.stability_ari_std,
            "stability_nmi_mean": run_metrics.stability_nmi_mean,
            "stability_nmi_std": run_metrics.stability_nmi_std,
            "confound_nmi_match": run_metrics.confound_nmi_match,
            "confound_nmi_rally_len": run_metrics.confound_nmi_rally_len,
            "passed": run_metrics.passed,
        },
        "community_ids": [p.community_id for p in profiles_sorted],
        "communities": communities_payload,
        "cross_community_facts": cross_facts,
        "valid_fact_ids": sorted(valid_fact_ids),
    }


# ---------------------------------------------------------------------------
# Grounded prompt generation
# ---------------------------------------------------------------------------

def generate_specialist_prompt_v1(facts: Dict[str, Any]) -> str:
    """Generate grounded specialist report prompt referencing fact IDs."""
    lines: List[str] = []
    community_ids = facts.get("community_ids", [])
    valid_ids = facts.get("valid_fact_ids", [])
    k = facts.get("k", len(community_ids))

    lines.append("=" * 80)
    lines.append("GROUND-TRUTH FACT PACK: RALLY SCHEMA SPECIALIST REPORT V1")
    lines.append("=" * 80)
    lines.append("")
    lines.append("You are writing a specialist rally-schema report from deterministic facts.")
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
    lines.append("- Do NOT include z-scores, decimals, or raw column names in the report text.")
    lines.append("- Do NOT include player names.")
    lines.append("- Do NOT include unsupported biomechanics language.")
    lines.append("- Do NOT include any line beginning with EVIDENCE_LOCK|.")
    lines.append("- Use uncertainty language: 'tends to', 'may', 'suggests', 'appears to'.")
    lines.append("")

    lines.append("REQUIRED STYLE:")
    lines.append("- Use coach-technical language grounded in facts.")
    lines.append("- For EACH community section, include at least two sentences starting with:")
    lines.append("  Compared to the other communities,")
    lines.append("- Connect mechanism to match implication using only cited facts.")
    lines.append("")

    lines.append("REQUIRED TECHNICAL COVERAGE PER COMMUNITY:")
    lines.append("- Serve entry: how the rally begins (speed, spin, clearance).")
    lines.append("- Return response: first response characteristics.")
    lines.append("- Serve-to-return contrast: what changes between serve and return.")
    lines.append("- Early exchange delta signature: what escalates or decays.")
    lines.append("- Displacement and tempo: movement inflation and duration growth.")
    lines.append("- Opponent preparation: anticipation and positioning hypotheses.")
    lines.append("")

    lines.append("REQUIRED REPORT STRUCTURE:")
    lines.append("1. Executive summary (key differentiators only)")
    for i, cid in enumerate(community_ids, start=2):
        lines.append(f"{i}. Community C{cid}: identity, patterns, match implications, opponent positioning hypotheses")
    lines.append(f"{len(community_ids) + 2}. Cross-community trade-offs (who wins what and why, qualitatively)")
    lines.append(f"{len(community_ids) + 3}. Caveats")
    lines.append(f"Keep the full report compact and information-dense (about twenty to forty single-line bullets/paragraphs).")
    lines.append("")

    lines.append("VALID FACT IDS:")
    lines.append(", ".join(valid_ids))
    lines.append("")

    lines.append("=" * 80)
    lines.append("FACT CATALOG")
    lines.append("=" * 80)
    lines.append("")

    for community in facts.get("communities", []):
        cid = community.get("community_id")
        name = community.get("derived_name", "")
        lines.append(f"### Community C{cid} ({name}) facts")
        for fact in community.get("facts", []):
            lines.append(
                f"- {fact['fact_id']} | topic={fact['topic']} | claim={fact['coach_claim']}"
            )
        lines.append("")

    lines.append("### Cross-community ranking facts")
    for fact in facts.get("cross_community_facts", []):
        ranking = fact.get("ranking", [])
        rank_txt = " > ".join(f"C{c}" for c in ranking) if ranking else "N/A"
        lines.append(
            f"- {fact['fact_id']} | topic={fact['topic']} | claim={fact['coach_claim']} | ranking={rank_txt}"
        )
    lines.append("")
    lines.append("Now write the report in Markdown with the required citation comments.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interpretation report
# ---------------------------------------------------------------------------

def build_interpretation_report(
    profiles: List[CommunityProfile],
    run_metrics: RunLevelMetrics,
    run_dir: Path,
    facts_path: Optional[str] = None,
    prompt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build interpretation_report.json."""
    fig_dir = run_dir / "figures"
    figures = {}
    for fig_name in [
        "profile_heatmap_state.png", "profile_heatmap_action.png",
        "profile_heatmap_delta.png", "radar_profiles.png",
        "outcome_heatmap.png", "umap_communities.png",
        "profile_heatmap_state_tpp3.png", "profile_heatmap_action_tpp3.png",
        "profile_heatmap_delta_tpp3.png",
    ]:
        fig_path = fig_dir / fig_name
        if fig_path.exists():
            figures[fig_name.replace(".png", "")] = str(fig_path)

    communities = []
    for p in sorted(profiles, key=lambda x: x.community_id):
        top_state = _top_z(p.state_z, 3, positive=True)
        top_delta = _top_z(p.delta_z, 3, positive=True)
        communities.append({
            "community_id": p.community_id,
            "derived_name": p.derived_name,
            "count": p.count,
            "pct": p.pct_of_total,
            "mean_n_transitions": p.mean_n_transitions,
            "fault_rate": round(p.fault_rate, 4),
            "played_rate": round(p.played_rate, 4),
            "top_state_drivers": [
                {"feature": _display(c), "z": round(z, 4)} for c, z in top_state
            ],
            "top_delta_drivers": [
                {"feature": _display(c), "z": round(z, 4)} for c, z in top_delta
            ],
        })

    report: Dict[str, Any] = {
        "schema_version": "interpretation_v1",
        "run_dir": str(run_dir),
        "k": run_metrics.k,
        "n_samples": run_metrics.n_samples,
        "silhouette": run_metrics.silhouette,
        "stability_ari_mean": run_metrics.stability_ari_mean,
        "stability_nmi_mean": run_metrics.stability_nmi_mean,
        "passed": run_metrics.passed,
        "communities": communities,
        "figures": figures,
    }
    if facts_path:
        report["specialist_facts_path"] = facts_path
    if prompt_path:
        report["specialist_prompt_path"] = prompt_path

    return report


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence statements and report prompt for the discovered rally styles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", default="outputs/rally/discovery", type=str,
        help="Rally discovery run directory (courtstyle.discovery.rally_discovery --work-dir).",
    )
    parser.add_argument(
        "--output-dir", default=None, type=str,
        help="Output directory. Default: <run-dir>/interpretation.",
    )
    parser.add_argument(
        "--specialist-v1-subdir", default="prompts_specialist_v1", type=str,
        help="Subdirectory under output-dir for the fact pack and prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "run_config.json").exists():
        sys.exit(f"No discovery run found at {run_dir}. Run courtstyle.discovery.rally_discovery first.")
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "interpretation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Rally style interpretation")
    print(f"  Run dir : {run_dir}")
    print(f"  Out dir : {out_dir}")

    # ---- Load run config ----
    run_config = load_run_config(run_dir)
    raw_columns = run_config.get("raw_columns", [])
    print(f"  Raw columns: {len(raw_columns)}")

    # ---- Load validation metrics ----
    cm = load_json(run_dir / "validation" / "community_metrics.json")
    stab = load_json(run_dir / "validation" / "stability.json")
    confound = load_json(run_dir / "validation" / "confound_metrics.json")
    decision = load_json(run_dir / "validation" / "decision_summary.json")

    # Confounds: use train split if available
    confound_split = confound.get("train", confound)

    labels_train = pd.read_csv(resolve_fit_labels_path(run_dir, run_config))
    n_samples = len(labels_train)

    run_metrics = RunLevelMetrics(
        k=int(cm.get("k_selected", cm.get("n_clusters", 0))),
        n_samples=n_samples,
        silhouette=float(cm.get("silhouette", 0.0)),
        stability_ari_mean=float(stab.get("ari_mean", 0.0)),
        stability_ari_std=float(stab.get("ari_std", 0.0)),
        stability_nmi_mean=float(stab.get("nmi_mean", 0.0)),
        stability_nmi_std=float(stab.get("nmi_std", 0.0)),
        confound_nmi_match=float(confound_split.get("nmi_match_id", 0.0)),
        confound_nmi_rally_len=float(confound_split.get("nmi_rally_len_bin", 0.0)),
        passed=bool(decision.get("pass", False)),
    )
    print(f"  K={run_metrics.k}, n={run_metrics.n_samples}, sil={run_metrics.silhouette:.4f}, "
          f"ari={run_metrics.stability_ari_mean:.4f}, passed={run_metrics.passed}")

    # ---- Build profiles ----
    profiles = build_profiles(run_dir, run_config, raw_columns)
    for p in profiles:
        p.derived_name = derive_archetype_name(p)
    print(f"  Built {len(profiles)} community profiles")
    for p in profiles:
        print(f"    C{p.community_id}: {p.derived_name} (n={p.count}, {p.pct_of_total}%)")

    # ---- Specialist V1 outputs ----
    facts_path_str: Optional[str] = None
    prompt_path_str: Optional[str] = None

    spec_dir = out_dir / args.specialist_v1_subdir
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Facts
    facts = build_specialist_facts_v1(profiles, run_metrics, str(run_dir), raw_columns)
    facts_file = spec_dir / "specialist_facts_v1.json"
    with open(facts_file, "w") as f:
        json.dump(facts, f, indent=2)
    facts_path_str = str(facts_file)
    print(f"  Wrote {facts_file}")

    # Prompt
    prompt_text = generate_specialist_prompt_v1(facts)
    prompt_file = spec_dir / "specialist_report_prompt_v1.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt_text)
    prompt_path_str = str(prompt_file)
    print(f"  Wrote {prompt_file}")

    # ---- Interpretation report ----
    report = build_interpretation_report(
        profiles, run_metrics, run_dir,
        facts_path=facts_path_str,
        prompt_path=prompt_path_str,
    )
    report_file = out_dir / "interpretation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_file}")

    print("Done.")


if __name__ == "__main__":
    main()

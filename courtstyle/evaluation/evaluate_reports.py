#!/usr/bin/env python3
"""Evaluation of generated coach notes against coach-annotated reference notes.

For every held-out serve record, a serve fact card is derived from the record's
measured statistics, and the language model rewrites only those facts into a
coach-facing note. Each generated note is then compared with the
coach-annotated reference note of the same record.

Metrics:
  - Prometheus 2 (Kim et al., EMNLP 2024): semantic equivalence to the reference
    note on a five-point rubric scoped to the fact card, normalised by 5.
    A score of at least 4/5 is counted as a rubric pass.
  - MiniCheck (Tang et al., EMNLP 2024) and AlignScore (Zha et al., ACL 2023):
    factual consistency of the note with its fact card, reported as AUC against
    shuffled fact cards.
  - Compliance: rule-based checks for raw numbers, surface and shot terms.

Language models are served locally through Ollama. Prometheus 2 is pulled as
``tensortemplar/prometheus2:7b-fp16``. MiniCheck and AlignScore are optional.

Usage:
    python -m courtstyle.evaluation.evaluate_reports \\
        --community-run-dir outputs/serve/community_run_<timestamp> \\
        --annotations-csv data/serve/test_set/australian_open_2017_annotations.csv \\
        --predictions-csv outputs/serve/community_run_<timestamp>/test_set_eval/inference/predictions.csv \\
        --raw-stats-csv data/serve/test_set/australian_open_2017.csv \\
        --train-stats-csv data/serve/fingerprint_serve.csv

Optional dependencies:
    pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    pip install git+https://github.com/yuh-zha/AlignScore.git
    ollama pull tensortemplar/prometheus2:7b-fp16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Configuration
# =============================================================================


OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_ENDPOINT = "http://localhost:11434/api/generate"

# Model name mapping
MODEL_TAG_MAP = {
    "llama3.3": "llama3.3:latest",
    "qwen2.5": "qwen2.5:32b",
    "gemma3": "gemma3:latest",
}

PROMETHEUS_MODEL = "tensortemplar/prometheus2:7b-fp16"

# AlignScore defaults
ALIGNSCORE_EVAL_MODE = "nli_sp"
ALIGNSCORE_BATCH_SIZE = 16
ALIGNSCORE_LARGE_DEFAULT = Path("checkpoints/AlignScore-large.ckpt")
ALIGNSCORE_BASE_DEFAULT = Path("checkpoints/AlignScore-base.ckpt")
ALIGNSCORE_URLS = {
    "base": "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt",
    "large": "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
}

# Forbidden shot words for compliance
FORBIDDEN_SHOT_WORDS = [
    "slice", "chip", "lob", "topspin", "kick", "flat",
    "drop shot", "dropshot", "backhand slice", "forehand slice",
]

RANDOM_SEED = 42

# Global model instances
_minicheck_model = None
_alignscore_model = None


# =============================================================================
# Validation & Utilities
# =============================================================================

def compute_row_hash(row: pd.Series) -> str:
    """Deterministic row key shared with the annotation file (see docs/data_format.md)."""
    key_fields = [
        str(row.get("Player", "")),
        str(row.get("Match", "")),
        str(row.get("Result", "")),
        str(row.get("Avg Speed", "")),
        str(row.get("player_ace", "")),
    ]
    hash_input = "|".join(key_fields)
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"ERROR: {name} missing columns: {missing}")
        sys.exit(1)


def require_any_column(df: pd.DataFrame, options: List[str], name: str, label: str) -> str:
    for col in options:
        if col in df.columns:
            return col
    print(f"ERROR: {name} missing required column for {label}. Tried: {options}")
    sys.exit(1)


def validate_key_integrity(df: pd.DataFrame, name: str) -> None:
    required = ["instance_id", "row_index"]
    require_columns(df, required, name)

    if df[required].isna().any().any():
        print(f"ERROR: {name} has missing instance_id/row_index values.")
        sys.exit(1)

    dup_mask = df.duplicated(subset=required)
    if dup_mask.any():
        examples = df.loc[dup_mask, required].head(5).to_dict(orient="records")
        print(f"ERROR: {name} has duplicate (instance_id,row_index). Examples: {examples}")
        sys.exit(1)

    row_index = pd.to_numeric(df["row_index"], errors="coerce")
    if row_index.isna().any():
        print(f"ERROR: {name} has non-numeric row_index values.")
        sys.exit(1)

    unique_idx = sorted(row_index.astype(int).unique().tolist())
    if not unique_idx:
        print(f"ERROR: {name} has empty row_index.")
        sys.exit(1)
    expected = list(range(min(unique_idx), min(unique_idx) + len(unique_idx)))
    if unique_idx != expected or min(unique_idx) != 0:
        missing = sorted(set(expected) - set(unique_idx))[:10]
        print(f"ERROR: {name} row_index not contiguous from 0. Missing examples: {missing}")
        sys.exit(1)


def validate_key_sets_match(pred_df: pd.DataFrame, ann_df: pd.DataFrame) -> None:
    pred_keys = set(zip(pred_df["instance_id"], pred_df["row_index"]))
    ann_keys = set(zip(ann_df["instance_id"], ann_df["row_index"]))
    if pred_keys != ann_keys:
        missing_in_pred = list(ann_keys - pred_keys)[:5]
        missing_in_ann = list(pred_keys - ann_keys)[:5]
        print("ERROR: (instance_id,row_index) mismatch between predictions and annotations.")
        if missing_in_pred:
            print(f"  Missing in predictions (examples): {missing_in_pred}")
        if missing_in_ann:
            print(f"  Missing in annotations (examples): {missing_in_ann}")
        sys.exit(1)


def get_alignscore_url(preset: str) -> str:
    """Official download URL of an AlignScore checkpoint."""
    return ALIGNSCORE_URLS.get(preset, "")


# =============================================================================
# Training Thresholds Computation
# =============================================================================

def compute_training_thresholds(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute percentile-based thresholds from training data.

    Uses 33rd and 67th percentiles for 3-way splits (low/neutral/high).
    """
    thresholds = {}

    # Speed threshold (1st Avg)
    thresholds["speed"] = {
        "p33": train_df["1st Avg"].quantile(0.33),
        "p67": train_df["1st Avg"].quantile(0.67),
        "mean": train_df["1st Avg"].mean(),
        "std": train_df["1st Avg"].std(),
    }

    # Variability threshold (1st StDev)
    thresholds["variability"] = {
        "p33": train_df["1st StDev"].quantile(0.33),
        "p67": train_df["1st StDev"].quantile(0.67),
        "mean": train_df["1st StDev"].mean(),
        "std": train_df["1st StDev"].std(),
    }

    # Second serve speed
    thresholds["2nd_speed"] = {
        "p33": train_df["2nd Avg"].quantile(0.33),
        "p67": train_df["2nd Avg"].quantile(0.67),
        "mean": train_df["2nd Avg"].mean(),
    }

    # Placement dominance threshold (when one area > 40%)
    thresholds["placement_dominance"] = 0.40

    return thresholds


def derive_speed_descriptor(first_avg: float, thresholds: Dict) -> str:
    """Derive speed descriptor from 1st serve average."""
    t = thresholds["speed"]
    if first_avg >= t["p67"]:
        return "faster than average"
    elif first_avg <= t["p33"]:
        return "slower than average"
    else:
        return "moderate pace"


def derive_placement_descriptor(pct_t: float, pct_wide: float, pct_body: float,
                                dominance_threshold: float) -> str:
    """Derive placement descriptor from serve distribution."""
    placements = {"T": pct_t, "Wide": pct_wide, "Body": pct_body}
    dominant = max(placements, key=placements.get)
    dominant_pct = placements[dominant]

    if dominant_pct >= dominance_threshold:
        if dominant == "T":
            return "tends to favor the T area"
        elif dominant == "Wide":
            return "tends to favor wide placement"
        else:
            return "tends to favor body serves"
    else:
        return "shows mixed placement patterns"


def derive_variability_descriptor(first_stdev: float, thresholds: Dict) -> str:
    """Derive variability descriptor from 1st serve StDev."""
    t = thresholds["variability"]
    if first_stdev >= t["p67"]:
        return "high serve speed variability"
    elif first_stdev <= t["p33"]:
        return "consistent serve speed"
    else:
        return "moderate serve variability"


def build_serve_fact_card(row: pd.Series, thresholds: Dict) -> Dict[str, str]:
    """
    Build a Serve Fact Card from raw serve statistics.

    Returns a dictionary of facts that can be directly rewritten.
    """
    facts = {}

    # Speed fact
    first_avg = row.get("1st Avg", row.get("1st_Avg", None))
    if first_avg is not None and not pd.isna(first_avg):
        facts["speed"] = derive_speed_descriptor(float(first_avg), thresholds)

    # Placement fact
    pct_t = row.get("%_serves_T", 0.33)
    pct_wide = row.get("%_serves_Wide", 0.33)
    pct_body = row.get("%_serves_Body", 0.33)
    if not any(pd.isna([pct_t, pct_wide, pct_body])):
        facts["placement"] = derive_placement_descriptor(
            float(pct_t), float(pct_wide), float(pct_body),
            thresholds["placement_dominance"]
        )

    # Variability fact
    first_stdev = row.get("1st StDev", row.get("1st_StDev", None))
    if first_stdev is not None and not pd.isna(first_stdev):
        facts["variability"] = derive_variability_descriptor(float(first_stdev), thresholds)

    # Second serve fact (if notably different)
    second_avg = row.get("2nd Avg", row.get("2nd_Avg", None))
    if second_avg is not None and not pd.isna(second_avg):
        t = thresholds["2nd_speed"]
        if second_avg >= t["p67"]:
            facts["second_serve"] = "second serve tends to be relatively fast"
        elif second_avg <= t["p33"]:
            facts["second_serve"] = "second serve may be vulnerable to attack"

    return facts


def format_fact_card_for_prompt(facts: Dict[str, str], community_id: int,
                                 confidence: float) -> str:
    """Format the fact card as bullet points for the prompt."""
    lines = ["Serve Facts (derived from match statistics):"]

    if "speed" in facts:
        lines.append(f"- First serve speed: {facts['speed']}")
    if "placement" in facts:
        lines.append(f"- Placement pattern: {facts['placement']}")
    if "variability" in facts:
        lines.append(f"- Speed consistency: {facts['variability']}")
    if "second_serve" in facts:
        lines.append(f"- Second serve: {facts['second_serve']}")

    # Add confidence qualifier
    conf_str = "high" if confidence > 0.8 else "moderate" if confidence > 0.5 else "lower"
    lines.append(f"\nPattern confidence: {conf_str}")

    return "\n".join(lines)


def format_fact_card_as_evidence(facts: Dict[str, str]) -> str:
    """Render fact card in a fixed order for evaluation evidence."""
    lines = []
    if "speed" in facts:
        lines.append(f"First serve speed: {facts['speed']}.")
    if "placement" in facts:
        lines.append(f"Placement pattern: {facts['placement']}.")
    if "variability" in facts:
        lines.append(f"Speed consistency: {facts['variability']}.")
    if "second_serve" in facts:
        lines.append(f"Second serve: {facts['second_serve']}.")
    return " ".join(lines).strip()


# =============================================================================
# Prometheus 2 - Semantic Equivalence (EMNLP 2024)
# =============================================================================

PROMETHEUS_RUBRIC = """###Task Description:
You are evaluating whether a generated coach note matches a reference note, given the FACT_CARD
that the model was allowed to use. This is a facts-to-rewrite setting.

###Evaluation Criteria:
[Semantic Equivalence]
Score 1: The generated note contradicts the reference or invents unsupported serve characteristics.
Score 2: The generated note captures less than half of the reference's key serve characteristics,
         with major mismatches or contradictions.
Score 3: The generated note captures about half of the reference's key serve characteristics,
         with some mismatches.
Score 4: The generated note captures most key serve characteristics (>75%) with minor differences.
Score 5: The generated note captures all key serve characteristics from the reference and stays
         consistent with the FACT_CARD evidence.

IMPORTANT:
- Only score overlap on characteristics inferable from the FACT_CARD (speed, placement pattern,
  variability/consistency, second-serve vulnerability/fastness, uncertainty language if present).
- Do NOT penalize missing concepts that are not in the FACT_CARD.
- Penalize contradictions relative to the FACT_CARD.

###FACT_CARD (evidence available to the model):
{fact_card}

###REFERENCE_COACH_NOTE:
{reference}

###GENERATED_NOTE:
{generated}

###Feedback:
Provide your score as: [RESULT] X (where X is 1-5)."""


def call_ollama_raw(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 180,
) -> str:
    """Call Ollama generate API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = requests.post(OLLAMA_GENERATE_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


def compute_prometheus_score(reference: str, generated: str, fact_card: str) -> Tuple[float, str]:
    """Compute semantic equivalence score using Prometheus 2."""
    prompt = PROMETHEUS_RUBRIC.format(
        fact_card=fact_card,
        reference=reference,
        generated=generated,
    )

    try:
        response = call_ollama_raw(PROMETHEUS_MODEL, prompt, temperature=0.0)

        match = re.search(r'\[RESULT\]\s*(\d)', response)
        if match:
            raw_score = int(match.group(1))
            raw_score = max(1, min(5, raw_score))
            normalized = raw_score / 5.0
            return normalized, response
        else:
            match = re.search(r'(\d)\s*$', response.strip())
            if match:
                raw_score = int(match.group(1))
                raw_score = max(1, min(5, raw_score))
                return raw_score / 5.0, response
            return 0.6, response

    except Exception as e:
        return 0.0, f"Error: {e}"


def compute_prometheus_scores(
    references: List[str],
    generated: List[str],
    fact_cards: List[str],
    cache_path: Optional[Path] = None,
) -> Tuple[List[float], List[str]]:
    """Compute Prometheus scores for all pairs."""
    n = len(references)

    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
            if len(cache.get("scores", [])) == n:
                print("      Using cached Prometheus scores")
                return cache["scores"], cache["feedbacks"]

    scores = []
    feedbacks = []

    print(f"      Evaluating {n} pairs with Prometheus 2...")

    for i, (ref, gen, fact_card) in enumerate(zip(references, generated, fact_cards)):
        if not gen.strip():
            scores.append(0.0)
            feedbacks.append("Empty generation")
        else:
            score, feedback = compute_prometheus_score(ref, gen, fact_card)
            scores.append(score)
            feedbacks.append(feedback)

        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"        Progress: {i + 1}/{n}")

    if cache_path:
        cache_data = {"scores": scores, "feedbacks": feedbacks}
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    return scores, feedbacks


# =============================================================================
# MiniCheck - Factual Consistency (EMNLP 2024)
# =============================================================================

def get_minicheck_model():
    """Lazy load MiniCheck model."""
    global _minicheck_model
    if _minicheck_model is None:
        try:
            from minicheck.minicheck import MiniCheck
        except ImportError:
            print("ERROR: minicheck not installed.")
            print('  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"')
            sys.exit(1)

        print("  Loading MiniCheck model (flan-t5-large)...")
        try:
            _minicheck_model = MiniCheck(model_name='flan-t5-large')
            print("  MiniCheck model loaded.")
        except Exception as e:
            print(f"ERROR loading MiniCheck: {e}")
            sys.exit(1)
    return _minicheck_model


def compute_minicheck(docs: List[str], claims: List[str]) -> Tuple[List[int], List[float]]:
    """Compute MiniCheck for factual consistency."""
    checker = get_minicheck_model()

    valid_indices = [i for i, g in enumerate(claims) if g.strip()]
    valid_refs = [docs[i] for i in valid_indices]
    valid_gens = [claims[i] for i in valid_indices]

    if valid_gens:
        pred_labels, raw_scores, _, _ = checker.score(docs=valid_refs, claims=valid_gens)
    else:
        pred_labels, raw_scores = [], []

    full_labels = []
    full_scores = []
    valid_idx = 0
    for i in range(len(claims)):
        if i in valid_indices:
            full_labels.append(int(pred_labels[valid_idx]))
            full_scores.append(float(raw_scores[valid_idx]))
            valid_idx += 1
        else:
            full_labels.append(0)
            full_scores.append(0.0)

    return full_labels, full_scores


# =============================================================================
# AlignScore - Factual Consistency (ACL 2023)
# =============================================================================

def get_alignscore_model(
    ckpt_path: Optional[Path],
    model_name: str,
    batch_size: int,
    evaluation_mode: str,
) -> Optional["AlignScore"]:
    """Lazy load AlignScore model."""
    global _alignscore_model
    if ckpt_path is None:
        return None
    if _alignscore_model is not None:
        return _alignscore_model

    if not ckpt_path.exists():
        print(f"ERROR: AlignScore checkpoint not found: {ckpt_path}")
        sys.exit(1)

    try:
        from alignscore import AlignScore
    except ImportError:
        print("ERROR: alignscore not installed.")
        sys.exit(1)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Loading AlignScore model ({model_name}) on {device}...")
    _alignscore_model = AlignScore(
        model=model_name,
        batch_size=batch_size,
        device=device,
        ckpt_path=str(ckpt_path),
        evaluation_mode=evaluation_mode,
        verbose=False,
    )
    print("  AlignScore model loaded.")
    return _alignscore_model


def compute_alignscore(
    contexts: List[str],
    claims: List[str],
    cache_path: Optional[Path],
    alignscore_model: Optional["AlignScore"],
) -> List[float]:
    """Compute AlignScore factual consistency scores."""
    if alignscore_model is None:
        return []

    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
            if len(cache.get("scores", [])) == len(contexts):
                print("      Using cached AlignScore scores")
                return cache["scores"]

    valid_indices = [i for i, g in enumerate(claims) if g.strip()]
    valid_refs = [contexts[i] for i in valid_indices]
    valid_gens = [claims[i] for i in valid_indices]

    if valid_gens:
        scores = alignscore_model.score(contexts=valid_refs, claims=valid_gens)
    else:
        scores = []

    full_scores = []
    valid_idx = 0
    for i in range(len(claims)):
        if i in valid_indices:
            full_scores.append(float(scores[valid_idx]))
            valid_idx += 1
        else:
            full_scores.append(0.0)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump({"scores": full_scores}, f, indent=2)

    return full_scores


def run_alignscore_sanity_check(alignscore_model: "AlignScore") -> None:
    """Light sanity check for AlignScore behavior."""
    contexts = [
        "First serve speed is faster than average. Placement favors the T.",
        "Serve speed is consistent. Placement is mixed.",
        "Second serve may be vulnerable to attack.",
    ]
    claims_same = contexts[:]
    claims_mismatch = contexts[::-1]

    try:
        scores_same = alignscore_model.score(contexts=contexts, claims=claims_same)
        scores_mismatch = alignscore_model.score(contexts=contexts, claims=claims_mismatch)
        mean_same = float(np.mean(scores_same))
        mean_mismatch = float(np.mean(scores_mismatch))
        if mean_same <= 0.3 or mean_mismatch >= mean_same:
            print("WARNING: AlignScore sanity check failed—interpret with caution.")
    except Exception as e:
        print(f"WARNING: AlignScore sanity check failed with error: {e}")


# =============================================================================
# Compliance Checking
# =============================================================================

def check_digit_violation(text: str) -> bool:
    """Check if text contains digits."""
    return bool(re.search(r'\d', text))


def check_surface_violation(text: str) -> bool:
    """Check if text mentions 'surface'."""
    return bool(re.search(r'\bsurface\b', text, re.IGNORECASE))


def check_shot_violation(text: str) -> bool:
    """Check if text contains forbidden shot words."""
    text_lower = text.lower()
    for word in FORBIDDEN_SHOT_WORDS:
        if word.lower() in text_lower:
            return True
    return False


def compute_compliance(text: str) -> Dict[str, bool]:
    """Compute all compliance violations."""
    return {
        "digit_violation": check_digit_violation(text),
        "surface_violation": check_surface_violation(text),
        "shot_violation": check_shot_violation(text),
    }


# =============================================================================
# Ollama API
# =============================================================================

def call_ollama(
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    timeout: int = 120,
) -> str:
    """Call Ollama chat API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


# =============================================================================
# Prompt Templates - Facts to Rewrite Approach
# =============================================================================

SYSTEM_PROMPT = """You are a tennis coach assistant. Your task is to REWRITE given serve facts into a strict two-block format.

Output EXACTLY this format (plain text, no bullets, no markdown):

FACTS:
<2 sentences, each sentence must restate only evidence from the fact card>

HYPOTHESES:
<0 or 1 sentence; if present it MUST start with 'Hypothesis:' and be explicitly speculative>

CRITICAL RULES:
- FACTS: only what appears in the fact card (paraphrase allowed), no extra claims
- HYPOTHESES: may be empty; if present use hedging ("may", "might", "could")
- NO exact numbers or percentages
- NO mention of court surface
- NO specific shot prescriptions (slice, chip, lob, topspin, kick, etc.)
- NO "best" or definitive prescriptions"""


def build_facts_rewrite_prompt(fact_card_str: str) -> str:
    """Build the facts-to-rewrite prompt."""
    return f"""Rewrite ONLY the following serve facts into the required two-block format.
Do NOT add any claims not present in the facts below.

{fact_card_str}

Remember the exact format:
FACTS:
<2 sentences, only evidence from the fact card>

HYPOTHESES:
<0 or 1 sentence starting with 'Hypothesis:' if present>"""


def split_facts_hypotheses(text: str) -> Tuple[str, str]:
    """Split generated text into FACTS and HYPOTHESES blocks."""
    if not text:
        return "", ""
    raw = text.strip()
    facts_block = ""
    hyp_block = ""

    facts_match = re.search(r'(?i)\bfacts\s*:\s*', raw)
    if facts_match:
        facts_start = facts_match.end()
        hyp_match = re.search(r'(?i)\bhypotheses\s*:\s*', raw[facts_start:])
        if hyp_match:
            hyp_start = facts_start + hyp_match.start()
            facts_block = raw[facts_start:hyp_start]
            hyp_block = raw[hyp_start + hyp_match.end() - hyp_match.start():]
        else:
            facts_block = raw[facts_start:]
    else:
        facts_block = raw

    def clean_block(block: str) -> str:
        block = re.sub(r'\s+', ' ', block).strip()
        return block

    facts_text = clean_block(facts_block)
    hypotheses_text = clean_block(hyp_block)
    return facts_text, hypotheses_text


# =============================================================================
# Generation and Caching
# =============================================================================

def get_cache_path(out_dir: Path, model_short: str) -> Path:
    """Get cache file path."""
    return out_dir / f"generated_{model_short}_mode_a.json"

def get_metrics_cache_path(out_dir: Path, model_short: str) -> Path:
    """Get metrics cache file path."""
    return out_dir / f"metrics_cache_{model_short}_mode_a.json"


def load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load generation cache."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"entries": []}


def save_cache(cache_path: Path, cache_data: Dict[str, Any]) -> None:
    """Save generation cache."""
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    os.rename(tmp_path, cache_path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_metrics_cache(out_dir: Path, model_short: str, metrics: Dict[str, Any]) -> None:
    """Save metrics cache for plot-only regeneration."""
    cache_path = get_metrics_cache_path(out_dir, model_short)
    payload = {k: _json_ready(v) for k, v in metrics.items()}
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_metrics_cache(out_dir: Path, model_short: str) -> Dict[str, Any]:
    """Load metrics cache for plot-only regeneration."""
    cache_path = get_metrics_cache_path(out_dir, model_short)
    if not cache_path.exists():
        print(f"ERROR: Metrics cache not found: {cache_path}")
        print("Run the full evaluation once to populate metrics caches, then use --plot-only.")
        sys.exit(1)
    with open(cache_path, "r") as f:
        return json.load(f)


def generate_notes(
    model_tag: str, model_short: str,
    df: pd.DataFrame, thresholds: Dict,
    out_dir: Path, force_regenerate: bool = False
) -> pd.DataFrame:
    """Generate LLM notes using facts-to-rewrite approach."""
    cache_path = get_cache_path(out_dir, model_short)

    if not force_regenerate:
        cache_data = load_cache(cache_path)
        if cache_data.get("entries") and len(cache_data["entries"]) == len(df):
            print(f"    Using cached generations from {cache_path.name}")
            cached_df = pd.DataFrame(cache_data["entries"])
            if "fact_card_text" not in cached_df.columns and "fact_card" in cached_df.columns:
                cached_df["fact_card_text"] = cached_df["fact_card"].apply(format_fact_card_as_evidence)
            if "facts_text" not in cached_df.columns:
                facts_hyp = cached_df["generated_text"].apply(split_facts_hypotheses)
                cached_df["facts_text"] = facts_hyp.apply(lambda x: x[0])
                cached_df["hypotheses_text"] = facts_hyp.apply(lambda x: x[1])
            cache_data["entries"] = cached_df.to_dict(orient="records")
            save_cache(cache_path, cache_data)
            return cached_df

    entries = []
    total = len(df)
    print(f"    Generating {total} notes with {model_tag} (facts-to-rewrite)...")

    for idx, row in df.iterrows():
        instance_id = row["instance_id"]
        row_index = int(row["row_index"])

        # Build fact card from raw statistics
        facts = build_serve_fact_card(row, thresholds)
        fact_card_text = format_fact_card_as_evidence(facts)

        community_id = int(row.get("community_id", 0))
        confidence = float(row.get("max_confidence", 0.5))

        fact_card_str = format_fact_card_for_prompt(facts, community_id, confidence)
        prompt = build_facts_rewrite_prompt(fact_card_str)

        try:
            generated_text = call_ollama(
                model=model_tag, prompt=prompt,
                system_prompt=SYSTEM_PROMPT, temperature=0.0
            )
        except Exception as e:
            print(f"      Warning: Generation failed for {instance_id}: {e}")
            generated_text = ""

        facts_text, hypotheses_text = split_facts_hypotheses(generated_text)

        entries.append({
            "instance_id": instance_id,
            "row_index": row_index,
            "generated_text": generated_text,
            "fact_card": facts,
            "fact_card_text": fact_card_text,
            "facts_text": facts_text,
            "hypotheses_text": hypotheses_text,
        })

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"      Progress: {idx + 1}/{total}")

    cache_data = {
        "model": model_tag,
        "mode": "mode_a_facts_rewrite",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": entries,
    }
    save_cache(cache_path, cache_data)
    print(f"    Saved cache to {cache_path.name}")

    return pd.DataFrame(entries)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    references: List[str],
    generated_raw: List[str],
    facts_texts: List[str],
    fact_card_texts: List[str],
    out_dir: Path,
    model_short: str,
    alignscore_model: Optional["AlignScore"],
) -> Dict[str, Any]:
    """Compute all evaluation metrics."""
    n = len(references)
    if not (len(generated_raw) == n and len(facts_texts) == n and len(fact_card_texts) == n):
        print("ERROR: Length mismatch among references/generated/facts/fact_card_texts.")
        sys.exit(1)

    # Prometheus
    print("      Computing Prometheus 2 scores...")
    prometheus_cache = out_dir / f"prometheus_cache_{model_short}_mode_a.json"
    prometheus_scores, prometheus_feedbacks = compute_prometheus_scores(
        references, facts_texts, fact_card_texts, cache_path=prometheus_cache
    )

    # MiniCheck
    print("      Computing MiniCheck...")
    minicheck_labels, minicheck_scores = compute_minicheck(fact_card_texts, facts_texts)

    # AlignScore
    alignscore_scores = []
    if alignscore_model is not None:
        print("      Computing AlignScore...")
        alignscore_cache = out_dir / f"alignscore_cache_{model_short}_mode_a.json"
        alignscore_scores = compute_alignscore(
            fact_card_texts, facts_texts, cache_path=alignscore_cache,
            alignscore_model=alignscore_model,
        )

    # Shuffled baseline for faithfulness
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    shuffled_fact_cards = [fact_card_texts[i] for i in perm]

    print("      Computing MiniCheck (shuffled baseline)...")
    minicheck_labels_shuf, minicheck_scores_shuf = compute_minicheck(shuffled_fact_cards, facts_texts)

    alignscore_scores_shuf = []
    if alignscore_model is not None:
        print("      Computing AlignScore (shuffled baseline)...")
        alignscore_cache_shuf = out_dir / f"alignscore_cache_{model_short}_mode_a_shuf.json"
        alignscore_scores_shuf = compute_alignscore(
            shuffled_fact_cards, facts_texts, cache_path=alignscore_cache_shuf,
            alignscore_model=alignscore_model,
        )

    def paired_auc(true_scores: List[float], shuf_scores: List[float]) -> float:
        true_arr = np.asarray(true_scores)
        shuf_arr = np.asarray(shuf_scores)
        if len(true_arr) == 0:
            return float("nan")
        return float(np.mean((true_arr > shuf_arr) + 0.5 * (true_arr == shuf_arr)))

    # Compliance (raw output)
    digit_violations = 0
    surface_violations = 0
    shot_violations = 0

    for gen in generated_raw:
        compliance = compute_compliance(gen)
        if compliance["digit_violation"]:
            digit_violations += 1
        if compliance["surface_violation"]:
            surface_violations += 1
        if compliance["shot_violation"]:
            shot_violations += 1

    # Aggregates
    prometheus_mean = float(np.mean(prometheus_scores)) if prometheus_scores else 0.0
    prometheus_median = float(np.median(prometheus_scores)) if prometheus_scores else 0.0
    minicheck_support_rate = float(np.mean(minicheck_labels)) if minicheck_labels else 0.0
    minicheck_support_rate_shuf = float(np.mean(minicheck_labels_shuf)) if minicheck_labels_shuf else 0.0
    minicheck_score_mean = float(np.mean(minicheck_scores)) if minicheck_scores else 0.0
    minicheck_score_mean_shuf = float(np.mean(minicheck_scores_shuf)) if minicheck_scores_shuf else 0.0
    minicheck_delta_mean = minicheck_score_mean - minicheck_score_mean_shuf
    minicheck_auc = paired_auc(minicheck_scores, minicheck_scores_shuf)

    alignscore_mean = float(np.mean(alignscore_scores)) if alignscore_scores else float("nan")
    alignscore_mean_shuf = float(np.mean(alignscore_scores_shuf)) if alignscore_scores_shuf else float("nan")
    alignscore_delta_mean = alignscore_mean - alignscore_mean_shuf if not np.isnan(alignscore_mean_shuf) else float("nan")
    alignscore_auc = paired_auc(alignscore_scores, alignscore_scores_shuf) if alignscore_scores_shuf else float("nan")

    return {
        "prometheus_scores": prometheus_scores,
        "prometheus_feedbacks": prometheus_feedbacks,
        "minicheck_labels": minicheck_labels,
        "minicheck_scores": minicheck_scores,
        "minicheck_labels_shuf": minicheck_labels_shuf,
        "minicheck_scores_shuf": minicheck_scores_shuf,
        "alignscore_scores": alignscore_scores,
        "alignscore_scores_shuf": alignscore_scores_shuf,
        "prometheus_mean": prometheus_mean,
        "prometheus_median": prometheus_median,
        "minicheck_support_rate": minicheck_support_rate,
        "minicheck_support_rate_shuf": minicheck_support_rate_shuf,
        "minicheck_auc": minicheck_auc,
        "minicheck_score_mean": minicheck_score_mean,
        "minicheck_score_mean_shuf": minicheck_score_mean_shuf,
        "minicheck_delta_mean": minicheck_delta_mean,
        "alignscore_mean": alignscore_mean,
        "alignscore_mean_shuf": alignscore_mean_shuf,
        "alignscore_auc": alignscore_auc,
        "alignscore_delta_mean": alignscore_delta_mean,
        "digit_violations": digit_violations,
        "surface_violations": surface_violations,
        "shot_violations": shot_violations,
        "compliance_rate_raw": 1.0 - (digit_violations + surface_violations + shot_violations) / max(n * 3, 1),
        "digit_violation_rate_raw": digit_violations / max(n, 1),
        "surface_violation_rate_raw": surface_violations / max(n, 1),
        "shot_violation_rate_raw": shot_violations / max(n, 1),
        "n": n,
    }


# =============================================================================
# Figure Generation
# =============================================================================

def set_publication_style():
    """Set matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_metrics_boxplot(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Plot boxplots for Prometheus, MiniCheck, and AlignScore."""
    set_publication_style()

    models = list(results.keys())
    n_models = len(models)

    has_alignscore = any(len(results[m]["alignscore_scores"]) > 0 for m in models)
    n_rows = 3 if has_alignscore else 2
    fig_height = 8 if has_alignscore else 6
    fig_width = 6 if n_models == 1 else 10

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height))
    palette = sns.color_palette("Set2", max(n_models, 1))

    # Prometheus
    ax = axes[0]
    data_for_box = [results[m]["prometheus_scores"] for m in models]
    bp = ax.boxplot(data_for_box, labels=models, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='>=4/5 rubric')
    ax.set_ylabel("Prometheus Score")
    ax.set_title("Semantic Equivalence vs Coach Note (Fact-Card Grounded)")
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=8)

    # MiniCheck
    ax = axes[1]
    data_for_box = []
    labels = []
    colors = []
    for i, m in enumerate(models):
        data_for_box.append(results[m]["minicheck_scores"])
        labels.append(f"{m}\ntrue")
        colors.append(palette[i])
        data_for_box.append(results[m]["minicheck_scores_shuf"])
        labels.append(f"{m}\nshuf")
        colors.append(palette[i])
    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("MiniCheck Score")
    ax.set_title("Faithfulness vs Fact Card (True vs Shuffled)")
    ax.set_ylim(0, 1)

    # AlignScore
    if has_alignscore:
        ax = axes[2]
        data_for_box = []
        labels = []
        colors = []
        for i, m in enumerate(models):
            data_for_box.append(results[m]["alignscore_scores"])
            labels.append(f"{m}\ntrue")
            colors.append(palette[i])
            data_for_box.append(results[m]["alignscore_scores_shuf"])
            labels.append(f"{m}\nshuf")
            colors.append(palette[i])
        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel("AlignScore")
        ax.set_title("Faithfulness vs Fact Card (True vs Shuffled)")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(out_dir / "llm_note_semantic_boxplot.png")
    fig.savefig(out_dir / "llm_note_semantic_boxplot.pdf")
    plt.close(fig)


def plot_prometheus_histogram(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Plot histogram of Prometheus scores."""
    set_publication_style()

    models = list(results.keys())
    n_models = len(models)

    fig_width = 5 if n_models == 1 else 4 * n_models
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 13)
    bar_color = "#1f77b4"

    for idx, model in enumerate(models):
        ax = axes[idx]
        scores = results[model]["prometheus_scores"]

        ax.hist(scores, bins=bins, color=bar_color, edgecolor="#bdbdbd", linewidth=0.6, alpha=0.6)
        mean_score = float(np.mean(scores)) if scores else 0.0
        ax.axvline(x=mean_score, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"Mean={mean_score:.3f}")
        ax.axvline(x=0.8, color="green", linestyle=":", linewidth=1.2,
                   label="Rubric pass (score ≥ 4/5)")

        ax.set_xlabel("Prometheus score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.text(0.98, 0.95, f"n={len(scores)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "llm_note_prometheus_hist.png")
    fig.savefig(out_dir / "llm_note_prometheus_hist.pdf")
    plt.close(fig)


def plot_faithfulness_histogram(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Plot true vs shuffled histograms for MiniCheck and AlignScore."""
    set_publication_style()

    models = list(results.keys())
    n_models = len(models)
    has_alignscore = any(len(results[m]["alignscore_scores"]) > 0 for m in models)
    n_rows = 2 if has_alignscore else 1

    fig_width = 4 * n_models
    fig_height = 3.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_models, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    if n_rows == 1 and n_models == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_models == 1:
        axes = [[ax] for ax in axes]

    bins = np.linspace(0, 1, 13)
    true_color = "#1f77b4"
    shuf_color = "#ff7f0e"

    for col_idx, model in enumerate(models):
        # MiniCheck
        ax = axes[0][col_idx]
        true_scores = results[model]["minicheck_scores"]
        shuf_scores = results[model]["minicheck_scores_shuf"]
        auc = results[model]["minicheck_auc"]
        delta = results[model]["minicheck_delta_mean"]

        counts_true, _ = np.histogram(true_scores, bins=bins)
        counts_shuf, _ = np.histogram(shuf_scores, bins=bins)
        y_max = max(counts_true.max(initial=0), counts_shuf.max(initial=0))
        y_max = max(y_max, 1)

        ax.hist(true_scores, bins=bins, alpha=0.65, color=true_color, label=f"True (n={len(true_scores)})")
        ax.hist(shuf_scores, bins=bins, alpha=0.65, color=shuf_color, label=f"Shuffled (n={len(shuf_scores)})")
        mean_true = float(np.mean(true_scores)) if true_scores else 0.0
        mean_shuf = float(np.mean(shuf_scores)) if shuf_scores else 0.0
        ax.axvline(mean_true, color=true_color, linestyle="--", linewidth=1.0,
                   zorder=3)
        ax.axvline(mean_shuf, color=shuf_color, linestyle="--", linewidth=1.0,
                   zorder=3)
        ax.text(mean_true, y_max * 0.95, f"μ_true={mean_true:.2f}", color=true_color,
                fontsize=9, rotation=90, va="top", ha="right")
        ax.text(mean_shuf, y_max * 0.85, f"μ_shuf={mean_shuf:.2f}", color=shuf_color,
                fontsize=9, rotation=90, va="top", ha="right")

        if n_models > 1:
            ax.set_title(f"{model} — MiniCheck", fontsize=11)
        else:
            ax.set_title("MiniCheck", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_ylim(0, y_max * 1.3)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend_handles = [
            Line2D([0], [0], color=true_color, lw=6, alpha=0.65, label=f"True (n={len(true_scores)})"),
            Line2D([0], [0], color=shuf_color, lw=6, alpha=0.65, label=f"Shuffled (n={len(shuf_scores)})"),
            Line2D([], [], color="none", label=f"AUC={auc:.2f}, Δμ={delta:.2f}"),
        ]
        ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc="upper right")

        # AlignScore
        if has_alignscore:
            ax = axes[1][col_idx]
            true_scores = results[model]["alignscore_scores"]
            shuf_scores = results[model]["alignscore_scores_shuf"]
            auc = results[model]["alignscore_auc"]
            delta = results[model]["alignscore_delta_mean"]

            counts_true, _ = np.histogram(true_scores, bins=bins)
            counts_shuf, _ = np.histogram(shuf_scores, bins=bins)
            y_max = max(counts_true.max(initial=0), counts_shuf.max(initial=0))
            y_max = max(y_max, 1)

            ax.hist(true_scores, bins=bins, alpha=0.65, color=true_color, label=f"True (n={len(true_scores)})")
            ax.hist(shuf_scores, bins=bins, alpha=0.65, color=shuf_color, label=f"Shuffled (n={len(shuf_scores)})")
            mean_true = float(np.mean(true_scores)) if true_scores else 0.0
            mean_shuf = float(np.mean(shuf_scores)) if shuf_scores else 0.0
            ax.axvline(mean_true, color=true_color, linestyle="--", linewidth=1.0,
                       zorder=3)
            ax.axvline(mean_shuf, color=shuf_color, linestyle="--", linewidth=1.0,
                       zorder=3)
            ax.text(mean_true, y_max * 0.95, f"μ_true={mean_true:.2f}", color=true_color,
                    fontsize=9, rotation=90, va="top", ha="right")
            ax.text(mean_shuf, y_max * 0.85, f"μ_shuf={mean_shuf:.2f}", color=shuf_color,
                    fontsize=9, rotation=90, va="top", ha="right")

            if n_models > 1:
                ax.set_title(f"{model} — AlignScore", fontsize=11)
            else:
                ax.set_title("AlignScore", fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_ylim(0, y_max * 1.3)
            ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
            ax.grid(False, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            legend_handles = [
                Line2D([0], [0], color=true_color, lw=6, alpha=0.65, label=f"True (n={len(true_scores)})"),
                Line2D([0], [0], color=shuf_color, lw=6, alpha=0.65, label=f"Shuffled (n={len(shuf_scores)})"),
                Line2D([], [], color="none", label=f"AUC={auc:.2f}, Δμ={delta:.2f}"),
            ]
            ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc="upper right")

    fig.suptitle("Faithfulness vs Fact Card (True vs Shuffled)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "llm_note_faithfulness_hist.png")
    fig.savefig(out_dir / "llm_note_faithfulness_hist.pdf")
    plt.close(fig)


def plot_compliance_bar(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Plot compliance violation rates."""
    set_publication_style()

    models = list(results.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(models))
    width = 0.25

    digit_rates = [results[m]["digit_violation_rate_raw"] for m in models]
    surface_rates = [results[m]["surface_violation_rate_raw"] for m in models]
    shot_rates = [results[m]["shot_violation_rate_raw"] for m in models]

    ax.bar(x - width, digit_rates, width, label='Digit', color='#e41a1c')
    ax.bar(x, surface_rates, width, label='Surface', color='#377eb8')
    ax.bar(x + width, shot_rates, width, label='Shot words', color='#4daf4a')

    ax.set_xlabel("Model")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Compliance Violations (Raw Output)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(0.5, max(digit_rates + surface_rates + shot_rates) * 1.2) if digit_rates else 0.5)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "llm_note_compliance_bar.png")
    fig.savefig(out_dir / "llm_note_compliance_bar.pdf")
    plt.close(fig)


# =============================================================================
# Table Generation
# =============================================================================

def save_metrics_csv(results: Dict[str, Dict[str, Any]], out_path: Path) -> pd.DataFrame:
    """Save metrics as CSV."""
    rows = []

    for model in results:
        m = results[model]
        rows.append({
            "model": model,
            "mode": "mode_a",
            "n": m["n"],
            "prometheus_mean": m["prometheus_mean"],
            "prometheus_median": m["prometheus_median"],
            "minicheck_support_rate_true": m["minicheck_support_rate"],
            "minicheck_support_rate_shuf": m["minicheck_support_rate_shuf"],
            "minicheck_auc": m["minicheck_auc"],
            "minicheck_delta_mean": m["minicheck_delta_mean"],
            "alignscore_mean_true": m["alignscore_mean"],
            "alignscore_mean_shuf": m["alignscore_mean_shuf"],
            "alignscore_auc": m["alignscore_auc"],
            "alignscore_delta_mean": m["alignscore_delta_mean"],
            "compliance_rate_raw": m["compliance_rate_raw"],
            "digit_violation_raw": m["digit_violation_rate_raw"],
            "surface_violation_raw": m["surface_violation_rate_raw"],
            "shot_violation_raw": m["shot_violation_rate_raw"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.4f")
    return df


def save_metrics_latex(results: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Save metrics as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LLM Coach Note Evaluation (Facts-to-Rewrite, Mode A)}",
        r"\label{tab:llm_coach_note_eval}",
        r"\small",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Model & Prometheus & MiniCheck$_{true}$ & MiniCheck$_{shuf}$ & MiniCheck AUC & MiniCheck $\Delta$ & AlignScore$_{true}$ & AlignScore$_{shuf}$ & AlignScore AUC & AlignScore $\Delta$ & Compliance$_{raw}$ \\",
        r"\midrule",
    ]

    for model in results:
        m = results[model]
        alignscore_delta = f"{m['alignscore_delta_mean']:.3f}" if not np.isnan(m["alignscore_delta_mean"]) else "--"
        alignscore_auc = f"{m['alignscore_auc']:.3f}" if not np.isnan(m["alignscore_auc"]) else "--"
        alignscore_true = f"{m['alignscore_mean']:.3f}" if not np.isnan(m["alignscore_mean"]) else "--"
        alignscore_shuf = f"{m['alignscore_mean_shuf']:.3f}" if not np.isnan(m["alignscore_mean_shuf"]) else "--"
        line = (
            f"{model} & "
            f"{m['prometheus_mean']:.3f} & "
            f"{m['minicheck_support_rate']:.3f} & "
            f"{m['minicheck_support_rate_shuf']:.3f} & "
            f"{m['minicheck_auc']:.3f} & "
            f"{m['minicheck_delta_mean']:.3f} & "
            f"{alignscore_true} & "
            f"{alignscore_shuf} & "
            f"{alignscore_auc} & "
            f"{alignscore_delta} & "
            f"{m['compliance_rate_raw']:.3f} \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_eval_summary(out_dir: Path, model: str, metrics: Dict[str, Any]) -> None:
    """Write a short markdown summary for paper audit."""
    lines = [
        f"- Model evaluated: {model}",
        "- Evaluation uses FACTS block only (HYPOTHESES excluded from scoring).",
        f"- Prometheus (FACTS vs coach_note): mean={metrics['prometheus_mean']:.3f}, median={metrics['prometheus_median']:.3f}",
        f"- MiniCheck faithfulness (FACTS vs fact card): mean_true={metrics['minicheck_score_mean']:.3f}, mean_shuf={metrics['minicheck_score_mean_shuf']:.3f}, delta={metrics['minicheck_delta_mean']:.3f}, auc={metrics['minicheck_auc']:.3f}",
    ]
    if not np.isnan(metrics["alignscore_mean"]):
        lines.append(
            f"- AlignScore faithfulness (FACTS vs fact card): mean_true={metrics['alignscore_mean']:.3f}, "
            f"mean_shuf={metrics['alignscore_mean_shuf']:.3f}, "
            f"delta={metrics['alignscore_delta_mean']:.3f}, "
            f"auc={metrics['alignscore_auc']:.3f}"
        )
    lines.append(f"- Compliance (raw output): {metrics['compliance_rate_raw']:.3f}")

    summary_path = out_dir / "llm_note_eval_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# CLI and Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM Coach Notes (Prometheus 2 + MiniCheck + AlignScore) - Facts-to-Rewrite Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--community-run-dir", type=Path, required=True,
                        help="Community discovery run directory")
    parser.add_argument("--annotations-csv", type=Path, required=True,
                        help="Annotations CSV with coach_note column")
    parser.add_argument("--predictions-csv", type=Path, required=True,
                        help="Predictions CSV with community_id and max_confidence")
    parser.add_argument("--raw-stats-csv", type=Path, required=True,
                        help="Raw serve statistics CSV for test set")
    parser.add_argument("--train-stats-csv", type=Path, required=True,
                        help="Training statistics CSV for threshold computation")
    parser.add_argument("--models", nargs="+", default=["llama3.3"],
                        help="Models to evaluate (default: llama3.3)")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regeneration even if cache exists")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only regenerate plots from cached metrics")
    parser.add_argument("--alignscore-preset", choices=["none", "base", "large"], default=None,
                        help="AlignScore preset (default: large if local ckpt exists, else none)")
    parser.add_argument("--alignscore-ckpt", type=Path, default=None,
                        help="AlignScore checkpoint path (overrides preset)")
    parser.add_argument("--alignscore-model", type=str, default=None,
                        help="HF model name for AlignScore (optional)")
    parser.add_argument("--dump-examples", type=int, default=0,
                        help="Dump N debug examples to JSON (default: 0)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    out_dir = args.out_dir or (args.community_run_dir / "test_set_eval" / "llm_coach_note_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        print("[Plot Only] Regenerating plots from cached metrics...")
        results = {}
        for model_short in args.models:
            results[model_short] = load_metrics_cache(out_dir, model_short)
        plot_metrics_boxplot(results, out_dir)
        plot_prometheus_histogram(results, out_dir)
        plot_faithfulness_histogram(results, out_dir)
        plot_compliance_bar(results, out_dir)
        print("  Saved: llm_note_semantic_boxplot.png/pdf")
        print("  Saved: llm_note_prometheus_hist.png/pdf")
        print("  Saved: llm_note_faithfulness_hist.png/pdf")
        print("  Saved: llm_note_compliance_bar.png/pdf")
        return

    # Validate inputs
    for path, name in [
        (args.community_run_dir, "Community run directory"),
        (args.annotations_csv, "Annotations CSV"),
        (args.predictions_csv, "Predictions CSV"),
        (args.raw_stats_csv, "Raw stats CSV"),
        (args.train_stats_csv, "Train stats CSV"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    alignscore_preset = args.alignscore_preset
    if alignscore_preset is None:
        alignscore_preset = "large" if ALIGNSCORE_LARGE_DEFAULT.exists() else "none"

    alignscore_ckpt = None
    alignscore_model_name = None
    if args.alignscore_ckpt is not None:
        alignscore_ckpt = args.alignscore_ckpt
        if args.alignscore_model:
            alignscore_model_name = args.alignscore_model
        else:
            ckpt_lower = alignscore_ckpt.name.lower()
            alignscore_model_name = "roberta-large" if "large" in ckpt_lower else "roberta-base"
    elif alignscore_preset != "none":
        if alignscore_preset == "large":
            alignscore_ckpt = ALIGNSCORE_LARGE_DEFAULT
            alignscore_model_name = args.alignscore_model or "roberta-large"
        else:
            alignscore_ckpt = ALIGNSCORE_BASE_DEFAULT
            alignscore_model_name = args.alignscore_model or "roberta-base"
        if not alignscore_ckpt.exists():
            url = get_alignscore_url(alignscore_preset)
            print("ERROR: AlignScore preset selected but checkpoint is missing.")
            print(f"  Expected: {alignscore_ckpt}")
            print(f"  URL: {url}")
            sys.exit(1)

    print("=" * 70)
    print("LLM COACH NOTE EVALUATION (Facts-to-Rewrite Mode)")
    print("Metrics: Prometheus 2 + MiniCheck + AlignScore")
    print("=" * 70)
    print(f"  Annotations: {args.annotations_csv}")
    print(f"  Predictions: {args.predictions_csv}")
    print(f"  Raw Stats: {args.raw_stats_csv}")
    print(f"  Train Stats: {args.train_stats_csv}")
    print(f"  Models: {args.models}")
    print(f"  AlignScore preset: {alignscore_preset}")
    if alignscore_ckpt is not None:
        print(f"  AlignScore ckpt: {alignscore_ckpt}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # Load Data
    print("\n[Step 1] Loading data...")
    annotations_df = pd.read_csv(args.annotations_csv)
    predictions_df = pd.read_csv(args.predictions_csv)
    raw_stats_df = pd.read_csv(args.raw_stats_csv)
    train_stats_df = pd.read_csv(args.train_stats_csv)

    print(f"  Annotations: {len(annotations_df)} rows")
    print(f"  Predictions: {len(predictions_df)} rows")
    print(f"  Raw stats: {len(raw_stats_df)} rows")
    print(f"  Train stats: {len(train_stats_df)} rows")

    # Validate schemas and keys
    print("\n[Step 2] Validating schemas and keys...")
    require_columns(predictions_df, ["instance_id", "row_index", "community_id", "max_confidence"], "Predictions")
    require_columns(annotations_df, ["instance_id", "row_index", "row_hash", "coach_note"], "Annotations")
    require_columns(raw_stats_df, ["Player", "Match", "Result", "Avg Speed", "player_ace"], "Raw stats")
    require_columns(raw_stats_df, ["%_serves_T", "%_serves_Wide", "%_serves_Body"], "Raw stats")

    first_avg_col = require_any_column(raw_stats_df, ["1st Avg", "1st_Avg"], "Raw stats", "1st Avg")
    first_stdev_col = require_any_column(raw_stats_df, ["1st StDev", "1st_StDev"], "Raw stats", "1st StDev")
    second_avg_col = require_any_column(raw_stats_df, ["2nd Avg", "2nd_Avg"], "Raw stats", "2nd Avg")

    require_columns(train_stats_df, ["1st Avg", "1st StDev", "2nd Avg"], "Train stats")

    validate_key_integrity(predictions_df, "Predictions")
    validate_key_integrity(annotations_df, "Annotations")
    validate_key_sets_match(predictions_df, annotations_df)

    # Raw stats alignment via row_hash
    if "row_index" not in raw_stats_df.columns:
        raw_stats_df["row_index"] = range(len(raw_stats_df))
    raw_stats_df["row_index"] = pd.to_numeric(raw_stats_df["row_index"], errors="coerce").astype(int)
    raw_stats_df["computed_row_hash"] = raw_stats_df.apply(compute_row_hash, axis=1)
    hash_check = raw_stats_df[["row_index", "computed_row_hash", "Player"]].merge(
        annotations_df[["row_index", "row_hash"]],
        on="row_index",
        how="left",
    )
    if hash_check["row_hash"].isna().any():
        missing_rows = hash_check[hash_check["row_hash"].isna()].head(5)
        print("ERROR: Raw stats row_index does not align with annotations.")
        print(missing_rows[["row_index", "Player"]].to_string(index=False))
        sys.exit(1)
    mismatch = hash_check[hash_check["computed_row_hash"] != hash_check["row_hash"]]
    if not mismatch.empty:
        print("ERROR: Raw stats row_hash mismatch with annotations (showing first 5).")
        print(mismatch[["row_index", "Player", "computed_row_hash", "row_hash"]].head(5).to_string(index=False))
        sys.exit(1)

    # Compute training thresholds
    print("\n[Step 3] Computing training thresholds...")
    thresholds = compute_training_thresholds(train_stats_df)
    print(f"  Speed thresholds: p33={thresholds['speed']['p33']:.1f}, p67={thresholds['speed']['p67']:.1f}")
    print(f"  Variability thresholds: p33={thresholds['variability']['p33']:.2f}, p67={thresholds['variability']['p67']:.2f}")

    # Merge data
    print("\n[Step 4] Merging data...")

    merged_df = predictions_df.merge(
        annotations_df[["instance_id", "row_index", "row_hash", "coach_note"]],
        on=["instance_id", "row_index"],
        how="left",
    )

    # Merge raw statistics
    merged_df = merged_df.merge(
        raw_stats_df,
        left_on="row_index",
        right_on="row_index",
        how="left",
    )

    valid_mask = merged_df["coach_note"].notna() & (merged_df["coach_note"].str.strip() != "")
    merged_df = merged_df[valid_mask].reset_index(drop=True)
    print(f"  Valid instances: {len(merged_df)}")

    critical_cols = [
        first_avg_col,
        first_stdev_col,
        second_avg_col,
        "%_serves_T",
        "%_serves_Wide",
        "%_serves_Body",
    ]
    missing_any = merged_df[critical_cols].isna().any(axis=1)
    missing_count = int(missing_any.sum())
    if missing_count > 0:
        per_col = merged_df[critical_cols].isna().sum().to_dict()
        print(f"WARNING: {missing_count} rows have missing critical raw columns.")
        print(f"  Missing by column: {per_col}")
        severe_threshold = max(1, int(0.05 * len(merged_df)))
        if missing_count > severe_threshold:
            print("ERROR: Missingness exceeds tolerance for evaluation.")
            sys.exit(1)
    for col in critical_cols:
        if merged_df[col].isna().all():
            print(f"ERROR: Column '{col}' is entirely missing after merge.")
            sys.exit(1)

    references = merged_df["coach_note"].tolist()

    # Load evaluation models
    print("\n[Step 5] Loading evaluation models...")
    _ = get_minicheck_model()

    alignscore_model = None
    if alignscore_ckpt is not None:
        alignscore_model = get_alignscore_model(
            ckpt_path=alignscore_ckpt,
            model_name=alignscore_model_name or "roberta-base",
            batch_size=ALIGNSCORE_BATCH_SIZE,
            evaluation_mode=ALIGNSCORE_EVAL_MODE,
        )
        run_alignscore_sanity_check(alignscore_model)
    else:
        print("  AlignScore: skipped (preset=none)")

    # Generate and Evaluate
    print("\n[Step 6] Generating and evaluating LLM notes...")

    results = {}
    model_generations: Dict[str, pd.DataFrame] = {}

    for model_short in args.models:
        model_tag = MODEL_TAG_MAP.get(model_short, f"{model_short}:latest")
        print(f"\n  Model: {model_short} ({model_tag})")

        gen_df = generate_notes(
            model_tag=model_tag,
            model_short=model_short,
            df=merged_df,
            thresholds=thresholds,
            out_dir=out_dir,
            force_regenerate=args.force_regenerate,
        )
        model_generations[model_short] = gen_df

        generated = gen_df["generated_text"].fillna("").tolist()
        facts_texts = gen_df["facts_text"].fillna("").tolist()
        fact_card_texts = gen_df["fact_card_text"].fillna("").tolist()
        metrics = compute_metrics(
            references, generated, facts_texts, fact_card_texts, out_dir, model_short, alignscore_model
        )
        results[model_short] = metrics
        save_metrics_cache(out_dir, model_short, metrics)

        print(f"      Prometheus mean: {metrics['prometheus_mean']:.4f}")
        print(f"      MiniCheck support rate (true): {metrics['minicheck_support_rate']:.4f}")
        print(f"      MiniCheck AUC (true vs shuf): {metrics['minicheck_auc']:.4f}")
        if not np.isnan(metrics["alignscore_mean"]):
            print(f"      AlignScore mean (true): {metrics['alignscore_mean']:.4f}")
            print(f"      AlignScore AUC (true vs shuf): {metrics['alignscore_auc']:.4f}")
        print(f"      Compliance rate (raw): {metrics['compliance_rate_raw']:.4f}")

    # Generate Outputs
    print("\n[Step 7] Generating outputs...")

    save_metrics_csv(results, out_dir / "llm_note_metrics.csv")
    print(f"  Saved: llm_note_metrics.csv")

    save_metrics_latex(results, out_dir / "llm_note_metrics.tex")
    print(f"  Saved: llm_note_metrics.tex")

    primary_model = args.models[0]
    write_eval_summary(out_dir, primary_model, results[primary_model])
    print(f"  Saved: llm_note_eval_summary.md")

    print("\n[Step 8] Generating figures...")

    plot_metrics_boxplot(results, out_dir)
    print(f"  Saved: llm_note_semantic_boxplot.png/pdf")

    plot_prometheus_histogram(results, out_dir)
    print(f"  Saved: llm_note_prometheus_hist.png/pdf")

    plot_faithfulness_histogram(results, out_dir)
    print(f"  Saved: llm_note_faithfulness_hist.png/pdf")

    plot_compliance_bar(results, out_dir)
    print(f"  Saved: llm_note_compliance_bar.png/pdf")

    # Debug examples
    if args.dump_examples > 0:
        primary_model = args.models[0]
        gen_df = model_generations[primary_model]
        m = results[primary_model]
        n_dump = min(args.dump_examples, len(gen_df))
        indices = np.random.choice(len(gen_df), size=n_dump, replace=False).tolist()
        debug_rows = []
        for idx in indices:
            debug_rows.append({
                "instance_id": gen_df.iloc[idx]["instance_id"],
                "row_index": int(gen_df.iloc[idx]["row_index"]),
                "fact_card_text": gen_df.iloc[idx]["fact_card_text"],
                "coach_note": references[idx],
                "facts_text": gen_df.iloc[idx].get("facts_text", ""),
                "hypotheses_text": gen_df.iloc[idx].get("hypotheses_text", ""),
                "generated_text": gen_df.iloc[idx]["generated_text"],
                "prometheus_score": m["prometheus_scores"][idx],
                "prometheus_feedback": m["prometheus_feedbacks"][idx],
                "minicheck_label_true": m["minicheck_labels"][idx],
                "minicheck_score_true": m["minicheck_scores"][idx],
                "minicheck_label_shuf": m["minicheck_labels_shuf"][idx],
                "minicheck_score_shuf": m["minicheck_scores_shuf"][idx],
                "alignscore_score_true": m["alignscore_scores"][idx] if m["alignscore_scores"] else None,
                "alignscore_score_shuf": m["alignscore_scores_shuf"][idx] if m["alignscore_scores_shuf"] else None,
            })
        debug_path = out_dir / "debug_examples.json"
        with open(debug_path, "w") as f:
            json.dump(debug_rows, f, indent=2)
        print(f"  Saved: {debug_path.name}")

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE (Facts-to-Rewrite Mode)")
    print("=" * 70)

    primary_model = args.models[0]
    m = results[primary_model]

    print(f"\n  {primary_model} Results:")
    print(f"    Prometheus:    {m['prometheus_mean']:.3f} (vs coach_note)")
    print(f"    MiniCheck:     {m['minicheck_support_rate']:.3f} (vs fact card)")
    print(f"    MiniCheck AUC: {m['minicheck_auc']:.3f} (true vs shuffled)")
    if not np.isnan(m["alignscore_mean"]):
        print(f"    AlignScore:    {m['alignscore_mean']:.3f} (vs fact card)")
        print(f"    AlignScore AUC:{m['alignscore_auc']:.3f} (true vs shuffled)")
    print(f"    Compliance:    {m['compliance_rate_raw']:.3f} (raw output)")

    print(f"\n  Interpretation:")
    print(f"    Prometheus >=4/5 rubric = strong equivalence to coach_note (within fact-card scope)")
    print(f"    Faithfulness AUC > 0.5 = evidence of grounding beyond shuffle baseline")

    print("=" * 70)


if __name__ == "__main__":
    main()

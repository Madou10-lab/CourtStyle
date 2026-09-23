"""Language-model report generation for rally styles.

The prompt written by ``rally_interpretation`` supplies the fact pack of every
rally style. A fact carries an identifier (``C<k>_F<nn>`` for one style,
``XR_F<nn>`` for a cross-style ranking) and the profile values behind it. The
model is instructed to use only these facts and to end every line with the
identifiers it draws on, as an HTML comment ``<!--FACTS: ... -->``.

Every generated report is validated against the fact pack:

* every cited identifier must exist;
* every style must have its own section citing at least seven identifiers and
  covering the required topics;
* identity anchors, delta directions and relative levels must agree with the
  cited facts, and pairwise comparisons must cite a cross-style fact;
* decimal numbers, percentages, z-score notation, snake_case column names and
  out-of-scope terms (surfaces, biomechanics, serve-type labels) are rejected.

A failed report is re-prompted with the validation errors and the previous
draft, for up to two repair rounds.

Models are served locally through Ollama (http://localhost:11434). The paper
uses Qwen 2.5 32B.

Usage:
    python -m courtstyle.interpretation.rally_reports \\
        --prompts-dir outputs/rally/discovery/interpretation/prompts_specialist_v1 --model qwen2.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


# =============================================================================
# Configuration
# =============================================================================

OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Allowed Ollama models (only these models are supported)
ALLOWED_MODELS = {
    "llama3.3": "llama3.3:latest",
    "qwen2.5": "qwen2.5:32b",
    "gemma3": "gemma3:latest",
}

SPECIALIST_GROUNDED_SYSTEM_PROMPT = """You are a tennis analyst writing a specialist rally-schema report.
You MUST stay grounded in the provided deterministic fact pack.

OUTPUT FORMAT:
- Output Markdown only.
- Keep each paragraph or bullet on a single line.
- Keep each line compact and information-dense; avoid long multi-clause sentences.
- End every paragraph or bullet with an HTML fact citation comment:
  <!--FACTS: C0_F01,C0_F04 -->
- The citation comment must be the final token on the line; do not place punctuation after `-->`.
- Use only fact IDs provided in the prompt/fact pack.

HARD CONSTRAINTS:
- Use only provided facts; do not invent numbers.
- Do NOT output any line beginning with EVIDENCE_LOCK|.
- Do NOT use percentages, z-scores, decimals, or raw column names.
- Do NOT mention surfaces, player names, or biomechanics.
- When a cited fact names specific anchor features or direction words, preserve them exactly.
- Do not rename identity anchors into broader labels, and do not flip increasing/decreasing polarity in delta lines.
- CRITICAL: For EVERY delta_dynamics fact (topic=delta_dynamics, fact IDs ending in _F06), you MUST transcribe the EXACT feature names and their directions (increasing/decreasing) from the claim text. Never summarize, omit, or guess delta features. Never say features are "not provided" or "not explicitly stated" — they ARE stated in the fact pack. This is the most common and most serious error.
- Use uncertainty language: 'tends to', 'may', 'suggests', 'appears to', 'likely'.

COMMUNITY SECTION REQUIREMENTS:
- Use community headings exactly in the form:
  ### Community C0
- Keep each community section to seven concise lines.
- For each community section, include at least two lines that start exactly with:
  Compared to the other communities,
- Cover all required community topics from the prompt.
- Cite at least seven distinct local community fact IDs inside each community section.
"""

SPECIALIST_GROUNDED_REPAIR_PROMPT_TEMPLATE = """Your previous Markdown output failed validation.

Validation errors:
{errors}

Return a corrected Markdown report that fixes every error.
Mandatory reminders:
- Keep each paragraph or bullet on ONE line and end it with <!--FACTS: ...-->.
- Keep each line concise; do not let one community section run long.
- The citation comment must be the final token on the line; do not place punctuation after `-->`.
- Use ONLY these valid fact IDs: {valid_fact_ids}
- Every cited fact ID must exist in that list.
- Keep community sections contrastive and grounded.
- Use headings exactly like `### Community C0`.
- Include at least two lines per community that start with:
  Compared to the other communities,
- Cover every required community topic from the fact pack.
- Cite at least seven distinct local fact IDs inside each community section.
- Preserve exact identity-anchor features and delta-direction words from the cited claims.
- No %, no z=, no decimal numbers, no raw column names, no surfaces, no player names, no biomechanics.
- Output only the corrected report.
"""


# =============================================================================
# Ollama API Functions
# =============================================================================

def check_ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_available_models() -> List[str]:
    """List models available in local Ollama installation."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def send_prompt_to_ollama(
    prompt_text: str,
    model: str,
    system_prompt: str,
    timeout: int = 600,
    num_predict: int = 1200,
) -> str:
    """
    Send a prompt to local Ollama and return the response.

    Args:
        prompt_text: The user prompt to send
        model: Ollama model name (e.g., 'llama3.3:latest', 'qwen2.5:32b')
        system_prompt: System instructions for the model
        timeout: Request timeout in seconds
        num_predict: Maximum output tokens

    Returns:
        The model's response text
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": int(num_predict),
        },
    }

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "")
        raise RuntimeError(f"Ollama request failed ({resp.status_code}): {resp.text}")

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout}s. Try a smaller model or increase timeout.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Is it running? Start with: ollama serve")


# =============================================================================
# Fact Catalog Loading + Validation Helpers
# =============================================================================

FACT_ID_RE = re.compile(r"\b(?:C\d+_F\d{2}|XR_F\d{2})\b")
FACTS_COMMENT_RE = re.compile(r"<!--\s*FACTS\s*:\s*([A-Za-z0-9_,\s]+?)\s*-->", re.IGNORECASE)
FACTS_COMMENT_AT_END_RE = re.compile(r"<!--\s*FACTS\s*:\s*([A-Za-z0-9_,\s]+?)\s*-->\s*$", re.IGNORECASE)
COMMUNITY_HEADING_RE = re.compile(r"^#{1,6}\s+Community\s+(?:C)?(\d+)\b", re.IGNORECASE)


def _split_fact_ids(raw_ids: str) -> List[str]:
    return [part.strip() for part in raw_ids.split(",") if part.strip()]


def extract_fact_ids_from_prompt_text(prompt_text: str) -> Set[str]:
    """Extract valid fact IDs from prompt text as fallback."""
    return set(FACT_ID_RE.findall(prompt_text))


def load_specialist_grounded_fact_catalog(
    prompts_dir: Path,
    prompt_text: str,
) -> Tuple[Set[str], List[int], str]:
    """
    Load valid fact IDs and expected communities for specialist grounded mode.

    Priority:
      1) prompts_dir/specialist_facts_v1.json
      2) parse IDs from prompt text
    """
    json_path = prompts_dir / "specialist_facts_v1.json"
    valid_fact_ids: Set[str] = set()
    community_ids: Set[int] = set()

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())

            for fid in data.get("valid_fact_ids", []):
                if isinstance(fid, str):
                    valid_fact_ids.add(fid.strip())

            for community in data.get("communities", []):
                cid = community.get("community_id")
                if isinstance(cid, int):
                    community_ids.add(cid)
                for fact in community.get("facts", []):
                    fact_id = fact.get("fact_id")
                    if isinstance(fact_id, str):
                        valid_fact_ids.add(fact_id.strip())

            for fact in data.get("cross_community_facts", []):
                fact_id = fact.get("fact_id")
                if isinstance(fact_id, str):
                    valid_fact_ids.add(fact_id.strip())

            for cid in data.get("community_ids", []):
                if isinstance(cid, int):
                    community_ids.add(cid)

            if valid_fact_ids:
                return valid_fact_ids, sorted(community_ids), str(json_path.name)
        except Exception:
            pass

    inferred_ids = extract_fact_ids_from_prompt_text(prompt_text)
    for fid in inferred_ids:
        match = re.match(r"C(\d+)_F\d{2}$", fid)
        if match:
            community_ids.add(int(match.group(1)))
    return inferred_ids, sorted(community_ids), "prompt_text"


def load_specialist_topic_index(prompts_dir: Path) -> Dict[int, Dict[str, Set[str]]]:
    """
    Load community topic -> fact-id mapping from specialist_facts_v1.json.

    Returns:
        {community_id: {topic: {fact_id, ...}}}
    """
    json_path = prompts_dir / "specialist_facts_v1.json"
    topic_index: Dict[int, Dict[str, Set[str]]] = {}
    if not json_path.exists():
        return topic_index

    try:
        data = json.loads(json_path.read_text())
        for community in data.get("communities", []):
            cid = community.get("community_id")
            if not isinstance(cid, int):
                continue
            cid_topics = topic_index.setdefault(cid, {})
            for fact in community.get("facts", []):
                fid = fact.get("fact_id")
                topic = fact.get("topic")
                if not isinstance(fid, str) or not isinstance(topic, str):
                    continue
                cid_topics.setdefault(topic.strip(), set()).add(fid.strip())
    except Exception:
        return {}

    return topic_index


def load_specialist_fact_index(prompts_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load fact_id -> fact record mapping from specialist_facts_v1.json.

    Each record contains:
      - fact_id
      - topic
      - coach_claim
      - community_id (or None for cross-community facts)
    """
    json_path = prompts_dir / "specialist_facts_v1.json"
    fact_index: Dict[str, Dict[str, Any]] = {}
    if not json_path.exists():
        return fact_index

    try:
        data = json.loads(json_path.read_text())
        for community in data.get("communities", []):
            cid = community.get("community_id")
            for fact in community.get("facts", []):
                fid = fact.get("fact_id")
                if not isinstance(fid, str):
                    continue
                fact_index[fid.strip()] = {
                    "fact_id": fid.strip(),
                    "topic": fact.get("topic"),
                    "coach_claim": fact.get("coach_claim"),
                    "community_id": cid if isinstance(cid, int) else None,
                }
        for fact in data.get("cross_community_facts", []):
            fid = fact.get("fact_id")
            if not isinstance(fid, str):
                continue
            fact_index[fid.strip()] = {
                "fact_id": fid.strip(),
                "topic": fact.get("topic"),
                "coach_claim": fact.get("coach_claim"),
                "community_id": None,
            }
    except Exception:
        return {}

    return fact_index


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _split_feature_list(raw_text: str) -> List[str]:
    parts = re.split(r",|\band\b", raw_text)
    cleaned = []
    for part in parts:
        item = part.strip(" .;:-")
        if item:
            cleaned.append(item)
    return cleaned


def _extract_identity_anchors(claim: str) -> List[str]:
    match = re.search(r"anchored by (.+?), which", claim, re.IGNORECASE)
    if not match:
        return []
    return _split_feature_list(match.group(1))


def _extract_delta_features(text: str) -> Dict[str, List[str]]:
    result = {"increasing": [], "decreasing": []}
    inc_match = re.search(
        r"increasing\s+(.+?)(?:;\s*decreasing|,\s*and\s*decreasing|\s+and\s+decreasing|$)",
        text,
        re.IGNORECASE,
    )
    dec_match = re.search(r"decreasing\s+(.+?)(?:[.]|$)", text, re.IGNORECASE)
    if inc_match:
        result["increasing"] = _split_feature_list(inc_match.group(1))
    if dec_match:
        result["decreasing"] = _split_feature_list(dec_match.group(1))
    return result


def _extract_rank_tokens(text: str) -> Set[str]:
    patterns = {
        "highest": r"\bhighest\b",
        "lowest": r"\blowest\b",
        "largest": r"\blargest\b",
        "smallest": r"\bsmallest\b",
        "mid-range": r"\bmid-?range\b",
        "moderate": r"\bmoderate\b",
        "most": r"\bmost\b",
        "fewest": r"\bfewest\b",
        "mid-sized": r"\bmid-?sized\b",
    }
    # Synonym groups: if one is present, the other is also allowed
    SYNONYMS = {
        "mid-range": "moderate",
        "moderate": "mid-range",
    }
    found: Set[str] = set()
    lower = text.lower()
    for label, pattern in patterns.items():
        if re.search(pattern, lower):
            found.add(label)
            if label in SYNONYMS:
                found.add(SYNONYMS[label])
    return found


def validate_specialist_semantics(
    markdown_text: str,
    fact_index: Dict[str, Dict[str, Any]],
    expected_community_ids: List[int],
) -> List[str]:
    """
    Validate semantic faithfulness of each cited line against its cited facts.

    Intent:
      - preserve exact identity anchors
      - preserve exact delta-feature polarity
      - reject unsupported disclaimers / invented missing-information claims
      - reject local fact citations from the wrong community section
      - keep relative-level words aligned with the cited claims
    """
    errors: List[str] = []
    current_cid: Optional[int] = None
    unsupported_re = re.compile(
        r"\b(not explicitly stated|not provided in the facts|not provided in the cited facts|"
        r"not in the provided facts|not in the facts|however,\s+it does mention)\b",
        re.IGNORECASE,
    )

    for line_no, raw_line in enumerate(markdown_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        heading_match = COMMUNITY_HEADING_RE.match(stripped)
        if heading_match:
            current_cid = int(heading_match.group(1))
            continue

        if stripped.startswith("#") or stripped.startswith("<!--"):
            # Reset current_cid for non-community headings (e.g. Cross-community, Caveats)
            if stripped.startswith("#"):
                current_cid = None
            continue

        end_comment_match = FACTS_COMMENT_AT_END_RE.search(stripped)
        if not end_comment_match:
            continue

        text_part = FACTS_COMMENT_AT_END_RE.sub("", stripped).strip()
        fact_ids = _split_fact_ids(end_comment_match.group(1))
        cited_facts = [fact_index[fid] for fid in fact_ids if fid in fact_index]
        local_facts = [fact for fact in cited_facts if isinstance(fact.get("community_id"), int)]
        cross_facts = [fact for fact in cited_facts if fact.get("community_id") is None]

        if current_cid is not None:
            wrong_local = [
                fact["fact_id"]
                for fact in local_facts
                if fact.get("community_id") != current_cid
            ]
            if wrong_local:
                errors.append(
                    f"Line {line_no}: cites local fact(s) from another community inside Community C{current_cid}: {', '.join(wrong_local)}"
                )

        if unsupported_re.search(text_part):
            errors.append(f"Line {line_no}: contains unsupported missing-information disclaimer")

        lower_text = _norm_text(text_part)

        # Identity-anchor preservation
        identity_like_line = lower_text.startswith("- identity") or lower_text.startswith("identity ") or " anchored by " in lower_text
        for fact in local_facts:
            if fact.get("topic") != "identity" or not identity_like_line:
                continue
            claim = str(fact.get("coach_claim") or "")
            anchors = _extract_identity_anchors(claim)
            missing = [anchor for anchor in anchors if _norm_text(anchor) not in lower_text]
            if missing:
                errors.append(
                    f"Line {line_no}: identity line misses cited anchor(s) from {fact['fact_id']}: {', '.join(missing)}"
                )

        # Delta-polarity preservation
        for fact in local_facts:
            if fact.get("topic") != "delta_dynamics":
                continue
            claim = str(fact.get("coach_claim") or "")
            expected = _extract_delta_features(claim)
            observed = _extract_delta_features(text_part)

            if not observed["increasing"] and not observed["decreasing"]:
                errors.append(f"Line {line_no}: delta line does not preserve explicit increasing/decreasing features from {fact['fact_id']}")
                continue

            for direction in ("increasing", "decreasing"):
                expected_feats = expected[direction]
                observed_feats = observed[direction]
                missing = [feat for feat in expected_feats if _norm_text(feat) not in {_norm_text(x) for x in observed_feats}]
                if missing:
                    errors.append(
                        f"Line {line_no}: delta line misses {direction} feature(s) from {fact['fact_id']}: {', '.join(missing)}"
                    )

            opposite_expected = {
                feat: "decreasing"
                for feat in expected["decreasing"]
            }
            opposite_expected.update({feat: "increasing" for feat in expected["increasing"]})
            for direction in ("increasing", "decreasing"):
                for feat in observed[direction]:
                    norm_feat = _norm_text(feat)
                    for expected_feat, expected_direction in opposite_expected.items():
                        if norm_feat == _norm_text(expected_feat) and direction != expected_direction:
                            errors.append(
                                f"Line {line_no}: delta line flips polarity for '{expected_feat}' relative to {fact['fact_id']}"
                            )

        # Relative-level token preservation for local facts
        # Allow tokens from ALL facts in the current community section (not just cited),
        # since synthesis lines may combine language from multiple facts.
        allowed_rank_tokens: Set[str] = set()
        for fact in cited_facts:
            allowed_rank_tokens.update(_extract_rank_tokens(str(fact.get("coach_claim") or "")))
        # Also allow tokens from ALL facts in the current community + all XR facts.
        # When outside any community section (e.g. executive summary), allow all facts.
        for fid, fact in fact_index.items():
            fc = fact.get("community_id")
            if current_cid is None or fc == current_cid or fc is None:
                allowed_rank_tokens.update(_extract_rank_tokens(str(fact.get("coach_claim") or "")))
        observed_rank_tokens = _extract_rank_tokens(text_part)
        disallowed_rank_tokens = sorted(tok for tok in observed_rank_tokens if tok not in allowed_rank_tokens)
        if disallowed_rank_tokens:
            errors.append(
                f"Line {line_no}: uses unsupported relative-level token(s) for cited facts: {', '.join(disallowed_rank_tokens)}"
            )

        # Pairwise comparisons should cite at least one cross-community fact.
        mentioned_communities = {
            int(match.group(1))
            for match in re.finditer(r"\bcommunity\s+c?([0-9]+)\b", text_part, re.IGNORECASE)
        }
        mentions_other_community = any(cid != current_cid for cid in mentioned_communities)
        if mentions_other_community and not cross_facts:
            errors.append(f"Line {line_no}: mentions another community without any cited cross-community fact")

    return errors


def _first_fact_id(topic_ids: Dict[str, Set[str]], topic: str) -> Optional[str]:
    ids = sorted(topic_ids.get(topic, set()))
    return ids[0] if ids else None


def _join_fact_ids(*fact_ids: Optional[str]) -> str:
    seen: Set[str] = set()
    ordered: List[str] = []
    for fid in fact_ids:
        if not fid or fid in seen:
            continue
        seen.add(fid)
        ordered.append(fid)
    return ",".join(ordered)


def build_specialist_structure_contract(
    expected_community_ids: List[int],
    topic_index: Dict[int, Dict[str, Set[str]]],
) -> str:
    """
    Build a strict output scaffold that makes the validator contract explicit.

    This does not invent facts; it only maps required topics to known fact IDs so the
    model can satisfy the coverage constraints deterministically.
    """
    lines: List[str] = []
    lines.append("MANDATORY OUTPUT CONTRACT (rewrite the wording, but keep the heading labels and citation placement):")
    lines.append("- Use `## Executive Summary`, then one section per community using headings exactly `### Community C<ID>`.")
    lines.append("- In each community section, produce seven single-line bullets or paragraphs covering: identity/frequency, serve entry, return response, serve-to-return contrast, delta signature, displacement/tempo, and opponent preparation/identity.")
    lines.append("- At least two lines in each community section must start exactly with `Compared to the other communities,`.")
    lines.append("- Put any sentence period before the citation comment, not after it.")
    lines.append("")
    lines.append("Suggested compact template:")
    lines.append("## Executive Summary")
    lines.append("- One to three lines on the main schema differences. <!--FACTS: XR_F01,XR_F02,XR_F05 -->")
    lines.append("")

    for cid in expected_community_ids:
        cid_topics = topic_index.get(cid, {})
        size_id = _first_fact_id(cid_topics, "size")
        outcome_id = _first_fact_id(cid_topics, "outcome_rates")
        serve_id = _first_fact_id(cid_topics, "serve_profile")
        return_id = _first_fact_id(cid_topics, "return_profile")
        contrast_id = _first_fact_id(cid_topics, "serve_return_contrast")
        delta_id = _first_fact_id(cid_topics, "delta_dynamics")
        move_id = _first_fact_id(cid_topics, "movement_tempo")
        identity_id = _first_fact_id(cid_topics, "identity")

        lines.append(f"### Community C{cid}")
        lines.append(
            f"- Identity and frequency: preserve the exact anchor features from the cited identity claim. <!--FACTS: {_join_fact_ids(size_id, outcome_id, identity_id)} -->"
        )
        lines.append(
            f"- Serve entry: keep the cited relative level exactly. <!--FACTS: {_join_fact_ids(serve_id, 'XR_F01', 'XR_F06')} -->"
        )
        lines.append(
            f"- Compared to the other communities, return response: keep the cited relative level exactly. <!--FACTS: {_join_fact_ids(return_id, 'XR_F02')} -->"
        )
        lines.append(
            f"- Compared to the other communities, serve-to-return contrast: keep the cited relative level exactly. <!--FACTS: {_join_fact_ids(contrast_id, 'XR_F03')} -->"
        )
        lines.append(
            f"- Delta signature: preserve the exact increasing and decreasing features from the cited claim. <!--FACTS: {_join_fact_ids(delta_id)} -->"
        )
        lines.append(
            f"- Displacement and tempo: keep the cited relative level exactly. <!--FACTS: {_join_fact_ids(move_id, 'XR_F04')} -->"
        )
        lines.append(
            f"- Opponent preparation and match implication: base the line on the cited outcome and identity claims without renaming the anchors. <!--FACTS: {_join_fact_ids(outcome_id, identity_id, 'XR_F05')} -->"
        )
        lines.append("")

    lines.append("## Cross-community Trade-offs")
    lines.append("- Qualitative ranking trade-offs only. <!--FACTS: XR_F01,XR_F02,XR_F03,XR_F04,XR_F05,XR_F06 -->")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Keep caveats factual and modest. <!--FACTS: XR_F05,XR_F06 -->")
    return "\n".join(lines)


def validate_specialist_grounded_markdown(
    markdown_text: str,
    valid_fact_ids: Set[str],
    expected_community_ids: List[int],
    min_fact_ids_per_community: int = 6,
    topic_index: Optional[Dict[int, Dict[str, Set[str]]]] = None,
    fact_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], Dict[int, Set[str]]]:
    """Validate specialist grounded markdown output and return errors + fact usage map."""
    errors: List[str] = []
    per_community_fact_usage: Dict[int, Set[str]] = {cid: set() for cid in expected_community_ids}

    if re.search(r"(?m)^EVIDENCE_LOCK\|", markdown_text):
        errors.append("Output contains forbidden EVIDENCE_LOCK| lines")

    if "```" in markdown_text:
        errors.append("Output contains forbidden code-fence ``` blocks")

    text_wo_comments = FACTS_COMMENT_RE.sub("", markdown_text)

    if "%" in text_wo_comments:
        errors.append("Output contains forbidden '%' symbol")
    if re.search(r"z\s*=", text_wo_comments, re.IGNORECASE):
        errors.append("Output contains forbidden z-score notation (z=)")
    if re.search(r"\d+\.\d+", text_wo_comments):
        errors.append("Output contains forbidden decimal numbers")

    # Likely raw column names: snake_case tokens (ignore fact IDs if they appear in text)
    underscore_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9_]+\b", text_wo_comments)
    for tok in underscore_tokens:
        if FACT_ID_RE.fullmatch(tok):
            continue
        errors.append(f"Output contains forbidden raw-style token with underscore: '{tok}'")
        break

    forbidden_patterns = [
        (r"\bgrass\b", "word 'grass'"),
        (r"\bclay\b", "word 'clay'"),
        (r"\bhard\s+court\b", "phrase 'hard court'"),
        (r"\bgrip\b", "biomechanics term 'grip'"),
        (r"\btoss\b", "biomechanics term 'toss'"),
        (r"\bpronation\b", "biomechanics term 'pronation'"),
        (r"\bracket\s+angle\b", "biomechanics phrase 'racket angle'"),
        (r"\bkick\s+serve\b", "serve-type label 'kick serve'"),
        (r"\bflat\s+serve\b", "serve-type label 'flat serve'"),
        (r"\bslice\s+serve\b", "serve-type label 'slice serve'"),
    ]
    for pattern, desc in forbidden_patterns:
        if re.search(pattern, text_wo_comments, re.IGNORECASE):
            errors.append(f"Output contains forbidden {desc}")

    lines = markdown_text.splitlines()
    in_code_block = False
    any_content_line = False
    globally_referenced_fact_ids: Set[str] = set()

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if re.match(r"^<!--.*-->$", stripped):
            continue
        if re.match(r"^-{3,}$", stripped):
            continue

        any_content_line = True

        end_comment_match = FACTS_COMMENT_AT_END_RE.search(stripped)
        if not end_comment_match:
            errors.append(f"Line {line_no}: paragraph/bullet missing trailing <!--FACTS: ...--> comment")
            continue

        fact_ids_raw = end_comment_match.group(1)
        fact_ids = _split_fact_ids(fact_ids_raw)
        if not fact_ids:
            errors.append(f"Line {line_no}: empty FACTS citation list")
            continue

        content_text = FACTS_COMMENT_AT_END_RE.sub("", stripped).strip()
        if not content_text:
            errors.append(f"Line {line_no}: citation exists but content text is empty")

        for fid in fact_ids:
            globally_referenced_fact_ids.add(fid)
            if valid_fact_ids and fid not in valid_fact_ids:
                errors.append(f"Line {line_no}: references unknown fact id '{fid}'")

    if not any_content_line:
        errors.append("Output has no content paragraphs or bullets")
    if not globally_referenced_fact_ids:
        errors.append("No fact IDs were cited in output")

    # Community section checks
    section_starts: List[Tuple[int, int]] = []
    for idx, raw_line in enumerate(lines):
        match = COMMUNITY_HEADING_RE.match(raw_line.strip())
        if match:
            section_starts.append((int(match.group(1)), idx))

    found_section_ids = {cid for cid, _ in section_starts}
    for cid in expected_community_ids:
        if cid not in found_section_ids:
            errors.append(f"Missing required community section heading for C{cid} (use heading like '### Community C{cid}')")

    section_starts_sorted = sorted(section_starts, key=lambda x: x[1])
    for i, (cid, start_idx) in enumerate(section_starts_sorted):
        end_idx = section_starts_sorted[i + 1][1] if i + 1 < len(section_starts_sorted) else len(lines)
        section_text = "\n".join(lines[start_idx:end_idx])

        section_fact_ids: Set[str] = set()
        for match in FACTS_COMMENT_RE.finditer(section_text):
            section_fact_ids.update(_split_fact_ids(match.group(1)))

        if cid in per_community_fact_usage:
            per_community_fact_usage[cid].update(section_fact_ids)
            if len(per_community_fact_usage[cid]) < min_fact_ids_per_community:
                errors.append(
                    f"Community C{cid} references {len(per_community_fact_usage[cid])} distinct fact IDs; require at least {min_fact_ids_per_community}"
                )

        compared_lines = len(re.findall(r"(?im)^.*Compared to the other communities,.*$", section_text))
        if cid in expected_community_ids and compared_lines < 2:
            errors.append(
                f"Community C{cid} has {compared_lines} lines starting with 'Compared to the other communities,'; require at least 2"
            )

    # Required topic coverage per community (when topic index is available)
    if topic_index:
        required_topics = {
            "serve entry": {"serve_profile"},
            "return response": {"return_profile"},
            "serve-to-return contrast": {"serve_return_contrast"},
            "delta signature": {"delta_dynamics"},
            "displacement/tempo": {"movement_tempo"},
            "identity": {"identity"},
        }
        for cid in expected_community_ids:
            cited_ids = per_community_fact_usage.get(cid, set())
            cid_topics = topic_index.get(cid, {})
            for label, topics in required_topics.items():
                allowed_ids: Set[str] = set()
                for t in topics:
                    allowed_ids.update(cid_topics.get(t, set()))
                if allowed_ids and cited_ids.isdisjoint(allowed_ids):
                    errors.append(f"Community C{cid} missing topic coverage: {label}")

    if fact_index:
        errors.extend(
            validate_specialist_semantics(
                markdown_text=markdown_text,
                fact_index=fact_index,
                expected_community_ids=expected_community_ids,
            )
        )

    return errors, per_community_fact_usage


# =============================================================================
# Prompt Processing
# =============================================================================

def _normalize_specialist_markdown(markdown_text: str) -> str:
    """
    Normalize small, non-semantic format deviations that commonly break validation.

    Intent: keep the content intact while enforcing a consistent required phrasing
    for contrast lines.
    """
    txt = markdown_text
    # Some models drop "the" in the required prefix; standardize it.
    txt = re.sub(
        r"(?im)^Compared to other communities\s*,?\s*",
        "Compared to the other communities, ",
        txt,
    )
    txt = re.sub(
        r"(?im)^Compared to the other communities\s*,?\s*",
        "Compared to the other communities, ",
        txt,
    )
    txt = re.sub(
        r"(?im)^#{1,6}\s+Community\s+(?:C)?(\d+)(?:\s*\([^)]*\))?\s*$",
        lambda match: f"### Community C{int(match.group(1))}",
        txt,
    )
    txt = re.sub(
        r"(?im)^(.*?)(\s*)(<!--\s*FACTS\s*:\s*[A-Za-z0-9_,\s]+?\s*-->)([.!?])\s*$",
        lambda match: f"{match.group(1).rstrip()}{match.group(4)} {match.group(3)}",
        txt,
    )
    return txt


def generate_reports(
    prompts_dir: Path,
    reports_dir: Path,
    model: str,
    timeout: int,
    max_retries: int,
    num_predict: int,
) -> Dict[str, Any]:
    """Process all .txt prompts in prompts_dir and generate reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(prompts_dir.glob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt prompt files found in: {prompts_dir}")

    print()
    print("Found prompt files:")
    for p in prompt_files:
        print(f"  - {p.name}")
    print()

    results: Dict[str, Any] = {
        "model": model,
        "mode": "specialist_grounded",
        "prompts_processed": 0,
        "prompts_failed": 0,
        "reports": [],
    }

    for idx, prompt_path in enumerate(prompt_files, start=1):
        prompt_text = prompt_path.read_text()
        prompt_name = prompt_path.stem

        print(f"[{idx}/{len(prompt_files)}] Processing {prompt_path.name}...")

        valid_fact_ids, fact_community_ids, fact_source = load_specialist_grounded_fact_catalog(
            prompts_dir=prompts_dir,
            prompt_text=prompt_text,
        )
        topic_index = load_specialist_topic_index(prompts_dir=prompts_dir)
        fact_index = load_specialist_fact_index(prompts_dir=prompts_dir)
        expected_ids = fact_community_ids
        structure_contract = build_specialist_structure_contract(
            expected_community_ids=expected_ids,
            topic_index=topic_index,
        )

        print(f"    Fact catalog source: {fact_source}")
        print(f"    Valid fact IDs: {len(valid_fact_ids)}")
        if expected_ids:
            print(f"    Expected community sections: {expected_ids}")
        else:
            print("    WARNING: Could not infer expected community IDs; section coverage checks may be weaker")

        if not valid_fact_ids:
            print("    FAILED: No valid fact IDs were found in prompt or specialist_facts_v1.json")
            results["prompts_failed"] += 1
            results["reports"].append(
                {
                    "prompt": prompt_path.name,
                    "error": "No valid fact IDs found for specialist-grounded validation",
                }
            )
            if idx < len(prompt_files):
                time.sleep(1)
            continue

        attempt = 0
        success = False
        raw_response = ""
        validation_errors: List[str] = []
        per_community_usage: Dict[int, Set[str]] = {}
        current_prompt = prompt_text + "\n\n" + structure_contract

        while attempt <= max_retries and not success:
            try:
                start_time = time.time()

                if attempt > 0:
                    print(f"    Retry {attempt}/{max_retries}...")
                    error_list = "\n".join(f"  - {e}" for e in validation_errors[:20])
                    repair_prompt = SPECIALIST_GROUNDED_REPAIR_PROMPT_TEMPLATE.format(
                        errors=error_list,
                        valid_fact_ids=", ".join(sorted(valid_fact_ids)),
                    )
                    current_prompt = (
                        prompt_text
                        + "\n\n"
                        + structure_contract
                        + "\n\n"
                        + repair_prompt
                        + "\n\nPrevious invalid draft:\n"
                        + raw_response
                    )

                raw_response = send_prompt_to_ollama(
                    prompt_text=current_prompt,
                    model=model,
                    system_prompt=SPECIALIST_GROUNDED_SYSTEM_PROMPT,
                    timeout=timeout,
                    num_predict=num_predict,
                )
                elapsed = time.time() - start_time

                normalized_response = _normalize_specialist_markdown(raw_response)
                validation_errors, per_community_usage = validate_specialist_grounded_markdown(
                    markdown_text=normalized_response,
                    valid_fact_ids=valid_fact_ids,
                    expected_community_ids=expected_ids,
                    min_fact_ids_per_community=7,
                    topic_index=topic_index,
                    fact_index=fact_index,
                )
                if validation_errors:
                    print(f"    Validation failed ({len(validation_errors)} errors):")
                    for err in validation_errors[:6]:
                        print(f"      - {err}")
                    if len(validation_errors) > 6:
                        print(f"      ... and {len(validation_errors) - 6} more")
                    attempt += 1
                    continue

                header = f"""<!--
Generated by: {model}
Source prompt: {prompt_path.name}
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
Mode: specialist_grounded
-->

"""
                report_content = header + normalized_response
                report_path = reports_dir / f"report_{prompt_name}.md"
                report_path.write_text(report_content)
                print(f"    Saved: {report_path.name} ({elapsed:.1f}s)")

                usage_summary = {
                    f"C{cid}": sorted(list(fids))
                    for cid, fids in sorted(per_community_usage.items(), key=lambda x: x[0])
                }

                results["prompts_processed"] += 1
                results["reports"].append(
                    {
                        "prompt": prompt_path.name,
                        "report": report_path.name,
                        "elapsed_seconds": round(elapsed, 1),
                        "attempts": attempt + 1,
                        "fact_catalog_source": fact_source,
                        "community_fact_usage": usage_summary,
                    }
                )
                success = True

            except Exception as e:
                print(f"    Error: {e}")
                validation_errors = [str(e)]
                attempt += 1

        if not success:
            print(f"    FAILED after {max_retries + 1} attempts")
            failed_path = reports_dir / f"failed_{prompt_name}.txt"
            failed_content = f"""VALIDATION ERRORS:
{chr(10).join(validation_errors)}

RAW RESPONSE:
{raw_response}
"""
            failed_path.write_text(failed_content)
            print(f"    Saved failed output: {failed_path.name}")

            results["prompts_failed"] += 1
            results["reports"].append(
                {
                    "prompt": prompt_path.name,
                    "error": "; ".join(validation_errors[:5]),
                    "failed_file": failed_path.name,
                }
            )

        if idx < len(prompt_files):
            time.sleep(1)

    return results


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated rally style reports with a local LLM.")
    parser.add_argument("--prompts-dir", type=Path, required=False,
                        help="Directory written by courtstyle.interpretation.rally_interpretation "
                             "(prompt .txt and specialist_facts_v1.json).")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="Output directory. Defaults to <prompts-dir>/../reports_<model>_specialist_v1.")
    parser.add_argument("--model", type=str, default="qwen2.5",
                        help=f"Model key {sorted(ALLOWED_MODELS)} or a full Ollama tag.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=2200, help="Maximum output tokens.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--list-models", action="store_true", help="List locally available Ollama models.")
    args = parser.parse_args()

    if not check_ollama_available():
        sys.exit("Ollama is not reachable at http://localhost:11434. Start it with `ollama serve`.")
    if args.list_models:
        for name in list_available_models():
            print(name)
        return
    if args.prompts_dir is None or not args.prompts_dir.exists():
        sys.exit("Provide an existing --prompts-dir.")

    model = ALLOWED_MODELS.get(args.model, args.model)
    short = args.model.split(":")[0].split(".")[0].split("-")[0].lower()
    reports_dir = args.reports_dir or args.prompts_dir.parent / f"reports_{short}_specialist_v1"
    summary = generate_reports(
        prompts_dir=args.prompts_dir,
        reports_dir=reports_dir,
        model=model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        num_predict=args.num_predict,
    )
    (reports_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Processed {summary['prompts_processed']}, failed {summary['prompts_failed']}. Reports in {reports_dir}")


if __name__ == "__main__":
    main()

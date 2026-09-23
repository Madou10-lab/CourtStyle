"""Language-model report generation for serve styles.

Each prompt supplies the structured evidence statements of every serve style.
A statement carries an identifier (``C<k>_F<nn>`` for one style, ``XR_F<nn>``
for a cross-style comparison) and expresses one measured quantity as a level
relative to the other styles. The model is instructed to use only these
statements and to close every paragraph with the identifiers it draws on, as
an HTML comment ``<!--FACTS: ... -->``.

Every generated report is validated against the statements:

* every cited identifier must exist;
* every style must have its own section citing at least seven identifiers;
* comparative claims must agree in direction with the cited statements;
* raw values, player names and unsupported terms are rejected.

A failed report is re-prompted with the list of validation errors and the
previous draft, for up to three repair rounds.

Models are served locally through Ollama (http://localhost:11434). The paper
uses Qwen 2.5 32B.

Usage:
    python -m courtstyle.interpretation.serve_reports \\
        --prompts-dir outputs/serve/community_run_<timestamp>/interpretation/prompts_coach_v2 --model qwen2.5
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


OLLAMA_API_URL = "http://localhost:11434/api/chat"


ALLOWED_MODELS = {
    "llama3.3": "llama3.3:latest",
    "qwen2.5": "qwen2.5:32b",
    "gemma3": "gemma3:latest",
}


COACH_GROUNDED_SYSTEM_PROMPT = """You are a tennis analyst writing a coach-usable serve report.
You MUST stay grounded in the provided deterministic fact pack.

OUTPUT FORMAT:
- Output Markdown only.
- Keep each paragraph or bullet on a single line.
- Keep the report compact and dense: roughly twenty to thirty total lines.
- End every paragraph or bullet with an HTML fact citation:
  <!--FACTS: C0_F01,C0_F04 -->
- Use only fact IDs provided in the prompt/fact pack.

REQUIRED STRUCTURE:
1. Executive summary focused on differences.
2. Community-by-community sections for each listed community.
3. Cross-community trade-offs.
4. Caveats.

COMMUNITY SECTION REQUIREMENTS:
- Include identity, serve-pattern interpretation, match implications.
- Include opponent positioning/anticipation hypotheses.
- Include at least two sentences that start exactly with:
  Compared to the other communities,

HARD CONSTRAINTS:
- Use only provided facts; do not invent features or numbers.
- Do NOT output EVIDENCE_LOCK lines.
- Do NOT use percentages, z-scores, decimals, or made-up numbers.
- Do NOT mention surfaces, surface conditions, court surfaces, player names, biomechanics, or unsupported serve types. The word "surface" must never appear.
- Unsupported serve types: kick, flat, slice, topspin.
- Avoid generic labels without mechanism-level explanation.
- Cover per community: pace, tempo shift, location geometry, reward-risk outcomes, pressure profile, and opponent positioning hypotheses.
- Prefer contrastive coach language over restating metric names.
- Prefer short, information-dense lines over long narrative paragraphs.
"""


COACH_GROUNDED_REPAIR_PROMPT_TEMPLATE = """Your previous markdown output failed validation.

Validation errors:
{errors}

Return a corrected Markdown report that fixes every error.
Mandatory reminders:
- Keep each paragraph or bullet on ONE line and end it with <!--FACTS: ...-->.
- Use ONLY these valid fact IDs: {valid_fact_ids}
- Every cited fact ID must exist in that list.
- Keep community sections contrastive and grounded.
- Include at least two lines per community that start with:
  Compared to the other communities,
- No %, no z=, no decimal numbers, no biomechanics, no unsupported serve-type labels.
- The word "surface" must NOT appear anywhere in the output.
- Every <!--FACTS: ...--> citation must list at least one valid fact ID.
- Output only the corrected report.
"""


FACT_ID_RE = re.compile(r"\b(?:C\d+_F\d{2}|XR_F\d{2})\b")


FACTS_COMMENT_RE = re.compile(r"<!--\s*FACTS\s*:\s*([A-Za-z0-9_,\s]+?)\s*-->", re.IGNORECASE)


FACTS_COMMENT_AT_END_RE = re.compile(r"<!--\s*FACTS\s*:\s*([A-Za-z0-9_,\s]+?)\s*-->[.\s]*$", re.IGNORECASE)


COMMUNITY_HEADING_RE = re.compile(r"^#{2,6}\s+Community\s+(?:C)?(\d+)\b", re.IGNORECASE)


COMMUNITY_HEADING_RE_SERVE = re.compile(r"^#{1,4}\s+Community\s+C(\d+)", re.IGNORECASE)


FACTS_COMMENT_AT_END_RE_SERVE = re.compile(r"<!--\s*FACTS:\s*([\w\s,_]+?)\s*-->[.\s]*$")


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
) -> str:
    """
    Send a prompt to local Ollama and return the response.

    Args:
        prompt_text: The user prompt to send
        model: Ollama model name (e.g., 'llama3.3', 'qwen2.5:32b')
        system_prompt: System instructions for the model
        timeout: Request timeout in seconds (default 10 min for large models)

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
            "temperature": 0.0,  # Deterministic output to reduce drift
            "num_predict": 1500,  # Keep reports compact but allow room for fact citations
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
        else:
            raise RuntimeError(f"Ollama request failed ({resp.status_code}): {resp.text}")

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout}s. Try a smaller model or increase timeout.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Is it running? Start with: ollama serve")


def extract_expected_community_ids(prompt_text: str) -> List[int]:
    """Extract expected community IDs from prompt text."""
    # Look for EVIDENCE_LOCK lines to find community IDs
    ids = set()
    for line in prompt_text.split("\n"):
        match = re.match(r"EVIDENCE_LOCK\|C(\d+)\|", line)
        if match:
            ids.add(int(match.group(1)))

    # Fallback: look for "### COMMUNITY X" headers
    if not ids:
        for match in re.finditer(r"### COMMUNITY (\d+)", prompt_text):
            ids.add(int(match.group(1)))

    return sorted(ids)


def _split_fact_ids(raw_ids: str) -> List[str]:
    return [part.strip() for part in raw_ids.split(",") if part.strip()]


def extract_fact_ids_from_prompt_text(prompt_text: str) -> Set[str]:
    """Extract valid fact IDs from prompt text as fallback."""
    return set(FACT_ID_RE.findall(prompt_text))


def load_coach_grounded_fact_catalog(
    prompts_dir: Path,
    prompt_text: str,
) -> Tuple[Set[str], List[int], str]:
    """
    Load valid fact IDs and expected communities for grounded coach mode.

    Priority:
      1) prompts_dir/coach_facts_v2.json
      2) parse IDs from prompt text
    """
    json_path = prompts_dir / "coach_facts_v2.json"
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

    # Fallback: infer from prompt text
    inferred_ids = extract_fact_ids_from_prompt_text(prompt_text)
    for fid in inferred_ids:
        match = re.match(r"C(\d+)_F\d{2}$", fid)
        if match:
            community_ids.add(int(match.group(1)))
    return inferred_ids, sorted(community_ids), "prompt_text"


def load_coach_grounded_topic_index(
    prompts_dir: Path,
) -> Dict[int, Dict[str, Set[str]]]:
    """
    Load community topic -> fact-id mapping from coach_facts_v2.json.

    Returns:
        {community_id: {topic: {fact_id, ...}}}
    """
    json_path = prompts_dir / "coach_facts_v2.json"
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


def load_coach_grounded_fact_index(prompts_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load fact_id -> fact record mapping from coach_facts_v2.json.

    Each record contains: fact_id, topic, coach_claim, community_id.
    Cross-community facts have community_id = None.
    """
    json_path = prompts_dir / "coach_facts_v2.json"
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


def _extract_rank_tokens(text: str) -> Set[str]:
    """Extract relative-level tokens from text."""
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
        "weakest": r"\bweakest\b",
        "strongest": r"\bstrongest\b",
        "mixed": r"\bmixed\b",
        "middle-tier": r"\bmiddle-?tier\b",
        "middle-range": r"\bmiddle-?range\b",
    }
    SYNONYMS = {
        "mid-range": "moderate",
        "moderate": "mid-range",
        "middle-range": "moderate",
        "middle-tier": "mid-range",
    }
    found: Set[str] = set()
    lower = text.lower()
    for label, pattern in patterns.items():
        if re.search(pattern, lower):
            found.add(label)
            if label in SYNONYMS:
                found.add(SYNONYMS[label])
    return found


def _split_fact_ids_serve(raw: str) -> List[str]:
    return [fid.strip() for fid in raw.split(",") if fid.strip()]


def validate_coach_grounded_semantics(
    markdown_text: str,
    fact_index: Dict[str, Dict[str, Any]],
    expected_community_ids: List[int],
) -> List[str]:
    """
    Validate semantic faithfulness of each cited line against its cited facts.

    Checks:
    1. Rank token preservation: relative-level tokens (highest, lowest, moderate, etc.)
       must appear in the fact claims for the current community or cross-community facts.
    2. Identity anchor check: lines citing identity facts must mention the anchor features.
    """
    errors: List[str] = []
    current_cid: Optional[int] = None

    for line_no, raw_line in enumerate(markdown_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        heading_match = COMMUNITY_HEADING_RE_SERVE.match(stripped)
        if heading_match:
            current_cid = int(heading_match.group(1))
            continue

        if stripped.startswith("#") or stripped.startswith("<!--"):
            # Reset current_cid for non-community headings
            if stripped.startswith("#"):
                current_cid = None
            continue

        end_comment_match = FACTS_COMMENT_AT_END_RE_SERVE.search(stripped)
        if not end_comment_match:
            continue

        text_part = FACTS_COMMENT_AT_END_RE_SERVE.sub("", stripped).strip()
        fact_ids = _split_fact_ids_serve(end_comment_match.group(1))
        cited_facts = [fact_index[fid] for fid in fact_ids if fid in fact_index]

        # Identity anchor check: lines citing identity facts within a community section
        # must mention the anchor features. Outside community sections (exec summary,
        # cross-community, caveats), identity facts may be cited for other purposes.
        if current_cid is not None:
            for fact in cited_facts:
                if fact.get("topic") == "identity" and fact.get("community_id") == current_cid:
                    claim = str(fact.get("coach_claim") or "")
                    anchored_match = re.search(r"anchored by (.+?)(?:,\s*which|$)", claim)
                    if anchored_match:
                        expected_anchors = [a.strip() for a in anchored_match.group(1).split(" and ")]
                        text_lower = text_part.lower()
                        for anchor in expected_anchors:
                            if anchor.lower() not in text_lower:
                                errors.append(
                                    f"Line {line_no}: identity line misses cited anchor '{anchor}' from {fact['fact_id']}"
                                )

        # Rank token preservation: allow tokens from ALL facts in current community + XR facts
        allowed_rank_tokens: Set[str] = set()
        for fact in cited_facts:
            allowed_rank_tokens.update(_extract_rank_tokens(str(fact.get("coach_claim") or "")))
        for fid, fact in fact_index.items():
            fc = fact.get("community_id")
            if current_cid is None or fc == current_cid or fc is None:
                allowed_rank_tokens.update(_extract_rank_tokens(str(fact.get("coach_claim") or "")))
        observed_rank_tokens = _extract_rank_tokens(text_part)
        disallowed_rank_tokens = sorted(tok for tok in observed_rank_tokens if tok not in allowed_rank_tokens)
        if disallowed_rank_tokens:
            errors.append(
                f"Line {line_no}: uses unsupported relative-level token(s): {', '.join(disallowed_rank_tokens)}"
            )

        # Cross-community mention check: mentioning another community requires a cross-community fact
        if current_cid is not None:
            mentioned_communities = {
                int(match.group(1))
                for match in re.finditer(r"\bcommunity\s+c?([0-9]+)\b", text_part, re.IGNORECASE)
            }
            cross_facts = [f for f in cited_facts if f.get("community_id") is None]
            mentions_other = any(cid != current_cid for cid in mentioned_communities)
            if mentions_other and not cross_facts:
                errors.append(f"Line {line_no}: mentions another community without any cited cross-community fact")

    return errors


def validate_coach_grounded_markdown(
    markdown_text: str,
    valid_fact_ids: Set[str],
    expected_community_ids: List[int],
    min_fact_ids_per_community: int = 6,
    fact_topic_index: Optional[Dict[int, Dict[str, Set[str]]]] = None,
    fact_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], Dict[int, Set[str]]]:
    """Validate grounded coach markdown output and return errors + fact usage map."""
    errors: List[str] = []
    per_community_fact_usage: Dict[int, Set[str]] = {cid: set() for cid in expected_community_ids}

    if "EVIDENCE_LOCK|" in markdown_text:
        errors.append("Output contains forbidden EVIDENCE_LOCK| lines")

    text_wo_comments = FACTS_COMMENT_RE.sub("", markdown_text)

    if "%" in text_wo_comments:
        errors.append("Output contains forbidden '%' symbol")
    if re.search(r"z\s*=", text_wo_comments, re.IGNORECASE):
        errors.append("Output contains forbidden z-score notation (z=)")
    if re.search(r"\d+\.\d+", text_wo_comments):
        errors.append("Output contains forbidden decimal numbers")

    forbidden_patterns = [
        (r"\bsurface\b", "word 'surface'"),
        (r"\bgrass\b", "word 'grass'"),
        (r"\bclay\b", "word 'clay'"),
        (r"\bhard\s+court\b", "phrase 'hard court'"),
        (r"\bgrip\b", "biomechanics term 'grip'"),
        (r"\btoss\b", "biomechanics term 'toss'"),
        (r"\bpronation\b", "biomechanics term 'pronation'"),
        (r"\bracket\s+angle\b", "biomechanics phrase 'racket angle'"),
        (r"\bkick\s+serve\b", "unsupported serve type 'kick serve'"),
        (r"\bflat\s+serve\b", "unsupported serve type 'flat serve'"),
        (r"\bslice\s+serve\b", "unsupported serve type 'slice serve'"),
        (r"\btopspin\b", "unsupported serve type 'topspin'"),
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

        compared_lines = len(re.findall(r"(?im)^.*Compared to (?:the )?other communities,.*$", section_text))
        if cid in expected_community_ids and compared_lines < 2:
            errors.append(
                f"Community C{cid} has {compared_lines} lines starting with 'Compared to the other communities,'; require at least 2"
            )

    if fact_topic_index:
        required_topic_groups = {
            "pace": {"pace"},
            "tempo shift": {"first_vs_second"},
            "location geometry": {"location", "body_jam", "variation", "adaptability"},
            "reward-risk outcomes": {"first_serve_reward", "second_serve_risk"},
            "pressure profile": {"pressure"},
        }
        for cid in expected_community_ids:
            cited_ids = per_community_fact_usage.get(cid, set())
            topic_map = fact_topic_index.get(cid, {})
            for label, topic_names in required_topic_groups.items():
                allowed_ids: Set[str] = set()
                for topic in topic_names:
                    allowed_ids.update(topic_map.get(topic, set()))
                if allowed_ids and cited_ids.isdisjoint(allowed_ids):
                    errors.append(f"Community C{cid} missing topic coverage: {label}")

    if fact_index:
        errors.extend(
            validate_coach_grounded_semantics(
                markdown_text=markdown_text,
                fact_index=fact_index,
                expected_community_ids=expected_community_ids,
            )
        )

    return errors, per_community_fact_usage



def generate_reports(
    prompts_dir: Path,
    reports_dir: Path,
    model: str,
    timeout: int = 600,
    max_retries: int = 3,
    min_fact_ids_per_community: int = 7,
) -> Dict[str, Any]:
    """Generate and validate one report per prompt file.

    Args:
        prompts_dir: Directory with ``*.txt`` prompts and the statement catalogue
            written by the interpretation stage.
        reports_dir: Output directory for reports and ``generation_summary.json``.
        model: Ollama model tag.
        timeout: Request timeout in seconds.
        max_retries: Number of repair rounds after a failed validation.
        min_fact_ids_per_community: Minimum cited identifiers per style section.

    Returns:
        Summary with one entry per prompt, recording attempts or errors.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    prompt_files = sorted(prompts_dir.glob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt prompt files found in {prompts_dir}")

    results: Dict[str, Any] = {
        "model": model, "mode": "coach_grounded",
        "prompts_processed": 0, "prompts_failed": 0, "reports": [],
    }
    for idx, prompt_path in enumerate(prompt_files, start=1):
        prompt_text = prompt_path.read_text()
        print(f"[{idx}/{len(prompt_files)}] {prompt_path.name}")

        valid_fact_ids, fact_community_ids, fact_source = load_coach_grounded_fact_catalog(
            prompts_dir=prompts_dir, prompt_text=prompt_text)
        topic_index = load_coach_grounded_topic_index(prompts_dir=prompts_dir)
        fact_index = load_coach_grounded_fact_index(prompts_dir=prompts_dir)
        expected_ids = extract_expected_community_ids(prompt_text) or fact_community_ids
        if not valid_fact_ids:
            results["prompts_failed"] += 1
            results["reports"].append({"prompt": prompt_path.name, "error": "No statement identifiers found"})
            continue

        attempt, success, raw_response = 0, False, ""
        validation_errors: List[str] = []
        per_community_usage: Dict[int, Set[str]] = {}
        current_prompt = prompt_text
        while attempt <= max_retries and not success:
            try:
                start_time = time.time()
                if attempt > 0:
                    print(f"    repair round {attempt}/{max_retries}")
                    repair_prompt = COACH_GROUNDED_REPAIR_PROMPT_TEMPLATE.format(
                        errors="\n".join(f"  - {e}" for e in validation_errors[:20]),
                        valid_fact_ids=", ".join(sorted(valid_fact_ids)),
                    )
                    current_prompt = (prompt_text + "\n\n" + repair_prompt
                                      + "\n\nPrevious invalid draft:\n" + raw_response)
                raw_response = send_prompt_to_ollama(prompt_text=current_prompt, model=model,
                                                     system_prompt=COACH_GROUNDED_SYSTEM_PROMPT, timeout=timeout)
                elapsed = time.time() - start_time
                validation_errors, per_community_usage = validate_coach_grounded_markdown(
                    markdown_text=raw_response,
                    valid_fact_ids=valid_fact_ids,
                    expected_community_ids=expected_ids,
                    min_fact_ids_per_community=min_fact_ids_per_community,
                    fact_topic_index=topic_index,
                    fact_index=fact_index,
                )
                if validation_errors:
                    print(f"    {len(validation_errors)} validation errors")
                    attempt += 1
                    continue

                report_path = reports_dir / f"report_{prompt_path.stem}.md"
                report_path.write_text(
                    f"<!--\nGenerated by: {model}\nSource prompt: {prompt_path.name}\n-->\n\n" + raw_response)
                results["prompts_processed"] += 1
                results["reports"].append({
                    "prompt": prompt_path.name,
                    "report": report_path.name,
                    "elapsed_seconds": round(elapsed, 1),
                    "attempts": attempt + 1,
                    "fact_catalog_source": fact_source,
                    "community_fact_usage": {f"C{c}": sorted(f) for c, f in sorted(per_community_usage.items())},
                })
                success = True
                print(f"    saved {report_path.name} after {attempt + 1} attempt(s)")
            except Exception as exc:
                validation_errors = [str(exc)]
                attempt += 1

        if not success:
            failed_path = reports_dir / f"failed_{prompt_path.stem}.txt"
            failed_path.write_text("VALIDATION ERRORS:\n" + "\n".join(validation_errors)
                                   + "\n\nRAW RESPONSE:\n" + raw_response)
            results["prompts_failed"] += 1
            results["reports"].append({"prompt": prompt_path.name, "error": "; ".join(validation_errors),
                                       "failed_file": failed_path.name})
            print(f"    failed after {max_retries + 1} attempts")
        if idx < len(prompt_files):
            time.sleep(1)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated serve style reports with a local LLM.")
    parser.add_argument("--prompts-dir", type=Path, required=False,
                        help="Directory written by courtstyle.interpretation.serve_interpretation.")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="Output directory. Defaults to <prompts-dir>/../reports_<model>.")
    parser.add_argument("--model", type=str, default="qwen2.5",
                        help=f"Model key {sorted(ALLOWED_MODELS)} or a full Ollama tag.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
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
    short = args.model.split(":")[0].split(".")[0].lower()
    reports_dir = args.reports_dir or args.prompts_dir.parent / f"reports_{short}"
    summary = generate_reports(args.prompts_dir, reports_dir, model, args.timeout, args.max_retries)
    (reports_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Processed {summary['prompts_processed']}, failed {summary['prompts_failed']}. Reports in {reports_dir}")


if __name__ == "__main__":
    main()

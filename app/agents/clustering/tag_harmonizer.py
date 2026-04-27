"""
Tag harmonization: maps a raw demographic tag value to a canonical
vocabulary key.

Three-tier resolution:
  1. Static lookup — lowercased exact match across all known synonyms
  2. Fuzzy match — RapidFuzz with score_cutoff=85
  3. LLM fallback — single LLM call per (field, raw_value), cached

Public function: `harmonize_tag(field, raw_value) -> (canonical, source)`
where source ∈ {"static", "fuzzy", "llm", "cache", "unknown"}.

Cache hits do NOT count as LLM calls — caller can sum source=='llm'
to surface LLM usage in the response (`tags_via_llm_fallback`).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Tuple

from app.agents.clustering import cache as tag_cache
from app.agents.clustering.taxonomy import (
    TAG_FIELDS,
    all_synonyms_for,
    synonym_to_canonical,
)

logger = logging.getLogger(__name__)


# Pre-build per-field reverse lookup tables (synonym → canonical) once
# at module load. Cheap, ~hundreds of entries total.
_REVERSE_LOOKUP = {field: synonym_to_canonical(field) for field in TAG_FIELDS}


def harmonize_tag(field: str, raw_value: str) -> Tuple[str, str]:
    """Resolve a raw tag value to a canonical key.

    Returns (canonical, source).
      - canonical: a key from TAG_FIELDS[field] or "unknown"
      - source: "static" | "fuzzy" | "cache" | "llm" | "unknown"
    """
    if not field or not isinstance(raw_value, str):
        return "unknown", "unknown"

    field_norm = field.lower().strip()
    if field_norm not in TAG_FIELDS:
        # Field outside known taxonomy — return raw lowered value as-is
        return (raw_value.lower().strip() or "unknown"), "static"

    raw_norm = raw_value.lower().strip()
    if not raw_norm:
        return "unknown", "unknown"

    # Tier 1: static exact match
    rev = _REVERSE_LOOKUP[field_norm]
    if raw_norm in rev:
        return rev[raw_norm], "static"

    # Cache check before fuzzy/LLM (LLM-resolved values cached as-is)
    cached = tag_cache.get(field_norm, raw_norm)
    if cached is not None:
        return cached, "cache"

    # Tier 2: fuzzy match
    fuzzy = _fuzzy_match(field_norm, raw_norm)
    if fuzzy is not None:
        # Cache the fuzzy resolution too — saves the rapidfuzz pass next time
        tag_cache.put(field_norm, raw_norm, fuzzy)
        return fuzzy, "fuzzy"

    # Tier 3: LLM fallback
    canonical = _llm_resolve(field_norm, raw_norm)
    tag_cache.put(field_norm, raw_norm, canonical)
    if canonical == "unknown":
        return "unknown", "unknown"
    return canonical, "llm"


def _fuzzy_match(field: str, raw_norm: str) -> str | None:
    """RapidFuzz fuzzy match against the field's synonym list."""
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed; skipping fuzzy match")
        return None

    synonyms = all_synonyms_for(field)
    if not synonyms:
        return None

    match = process.extractOne(
        raw_norm, synonyms, scorer=fuzz.WRatio, score_cutoff=85,
    )
    if match is None:
        return None
    matched_synonym = match[0]
    return _REVERSE_LOOKUP[field].get(matched_synonym.lower().strip())


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _llm_resolve(field: str, raw_norm: str) -> str:
    """Single LLM call to resolve an unseen tag. Returns canonical or 'unknown'.

    Failures (rate limit, parse error) → return "unknown" rather than raise,
    so clustering can proceed.
    """
    try:
        from langchain_core.output_parsers import StrOutputParser

        from app.agents.clustering.taxonomy import TAG_FIELDS
        from app.llm_config import get_llm
        from app.prompts.tag_normalization_prompts import TAG_NORMALIZATION_PROMPT
    except ImportError as e:
        logger.warning("LLM tag fallback unavailable: %s", e)
        return "unknown"

    canonical_keys = sorted(TAG_FIELDS.get(field, {}).keys())
    if not canonical_keys:
        return "unknown"

    try:
        llm = get_llm("context_analysis")
        chain = TAG_NORMALIZATION_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({
            "field": field,
            "raw_value": raw_norm,
            "canonical_keys": "\n".join(f"- {k}" for k in canonical_keys),
        })
    except Exception as e:
        logger.warning("LLM tag fallback failed for (%s, %s): %s", field, raw_norm, e)
        return "unknown"

    parsed = _parse_json_loose(raw)
    canonical = (parsed.get("canonical") or "").strip().lower()

    if canonical == "unknown":
        return "unknown"
    if canonical in canonical_keys:
        return canonical
    # LLM returned a value not in our taxonomy — fail closed
    logger.debug("LLM returned out-of-taxonomy value %r for %s; using 'unknown'", canonical, field)
    return "unknown"


def _parse_json_loose(raw: str) -> dict:
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        m = _JSON_BLOCK_RE.search(s)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}

"""LLM prompt for tag-normalization fallback (when static taxonomy + fuzzy match miss)."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


TAG_NORMALIZATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a taxonomy normalization assistant. Given a tag value and a "
        "list of canonical keys, pick the single closest canonical key. If "
        "none reasonably fit, return 'unknown'. Return ONLY a JSON object "
        "with one field — no prose, no markdown.\n\n"
        "Output format:\n"
        '{{"canonical": "<one of the canonical keys, or \\"unknown\\">"}}',
    ),
    (
        "human",
        "Field: {field}\n"
        "Raw value: {raw_value}\n\n"
        "Canonical keys (pick exactly one, or 'unknown'):\n"
        "{canonical_keys}\n\n"
        "Return JSON only.",
    ),
])

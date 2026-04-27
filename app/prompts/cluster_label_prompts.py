"""Prompt for the per-cluster human-readable labeler (one LLM call per cluster)."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CLUSTER_LABEL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research analyst naming a cluster of survey respondents. "
        "You receive a deterministic characterisation of the cluster (defining "
        "tags, top terms, dominant traits, sentiment) and produce a short, "
        "human-readable label and 1-sentence description.\n\n"
        "Rules:\n"
        "- Label MUST be 2-4 words, Title Case, evocative but specific. "
        "Examples: 'Pricing-Sensitive SaaS Leaders', 'Risk-Averse Enterprise "
        "Buyers', 'Growth-Focused Founders'.\n"
        "- Description MUST be ONE sentence describing what unifies the cluster.\n"
        "- Use ONLY the input characterisation — do NOT invent traits.\n"
        "- Avoid generic words (Cluster, Group, Segment) unless you genuinely "
        "have nothing concrete.\n"
        "- Return ONLY a JSON object — no prose, no markdown.\n\n"
        "Output schema:\n"
        '{{"label": "...", "description": "..."}}',
    ),
    (
        "human",
        "Cluster characterisation:\n"
        "Defining tags: {defining_tags}\n"
        "Top terms: {top_terms}\n"
        "Dominant traits: {dominant_traits}\n"
        "Mean VADER compound (sentiment): {mean_vader}\n"
        "Cluster size: {size}\n"
        "{focus_section}"
        "\n"
        "Return JSON only.",
    ),
])

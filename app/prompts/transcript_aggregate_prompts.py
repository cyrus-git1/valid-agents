"""Prompt for cross-session aggregator synthesis."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CROSS_SESSION_AGGREGATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior research synthesist. You will be given the analysis "
        "results from multiple individual sessions of the SAME research survey. "
        "Your job is to synthesise patterns ACROSS sessions — not re-analyse a "
        "single session.\n\n"
        "Rules:\n"
        "- Every shared theme MUST cite session_ids that mentioned it.\n"
        "- Contradictions are valuable — surface places where sessions disagree.\n"
        "- Persona patterns: respondent-side traits/behaviours that recur.\n"
        "- Aggregate sentiment: compute overall sentiment across sessions and "
        "report per-session compound score so trends are visible.\n"
        "- notable_quotes_top_n: pull 5-10 of the strongest quotes drawn from "
        "the per-session quote outputs, each with source session_id.\n"
        "- Skip a section if the input doesn't support it. Don't pad.\n"
        "- Return ONLY valid JSON. No prose, no markdown fences.\n\n"
        "Output schema:\n"
        '{{\n'
        '  "shared_themes": [\n'
        '    {{\n'
        '      "theme": "short noun phrase",\n'
        '      "sessions_mentioning": ["<session_id>"],\n'
        '      "frequency": "high|medium|low",\n'
        '      "supporting_quotes": [\n'
        '        {{"session_id": "...", "speaker": "...", "text": "verbatim quote"}}\n'
        '      ]\n'
        '    }}\n'
        '  ],\n'
        '  "contradictions": [\n'
        '    {{\n'
        '      "topic": "...",\n'
        '      "session_a": "<session_id>", "session_b": "<session_id>",\n'
        '      "summary": "1-2 sentence description of the disagreement"\n'
        '    }}\n'
        '  ],\n'
        '  "persona_patterns": [\n'
        '    {{"pattern": "...", "sessions_observed": ["<session_id>"]}}\n'
        '  ],\n'
        '  "aggregate_sentiment": {{\n'
        '    "overall_compound": 0.42,\n'
        '    "per_session": [{{"session_id": "...", "compound": 0.6, "label": "positive"}}]\n'
        '  }},\n'
        '  "notable_quotes_top_n": [\n'
        '    {{"quote": "verbatim", "speaker": "...", "session_id": "...", "why_notable": "..."}}\n'
        '  ],\n'
        '  "per_session_summary": [\n'
        '    {{"session_id": "...", "headline": "1-line takeaway", "sentiment_label": "positive|negative|mixed", "top_theme": "..."}}\n'
        '  ]\n'
        '}}',
    ),
    (
        "human",
        "Optional focus: {focus}\n"
        "Survey ID: {survey_id}\n"
        "Number of sessions: {session_count}\n\n"
        "Per-session analysis results (compact JSON, one object per session):\n"
        "{sessions_json}\n\n"
        "Synthesise across sessions now.",
    ),
])

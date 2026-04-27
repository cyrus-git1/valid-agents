"""Prompt for the transcript theme extraction agent."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


THEME_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert qualitative researcher extracting recurring themes "
        "from interview transcripts.\n\n"
        "Rules:\n"
        "- Identify 5-10 distinct themes. Skip if fewer than 3 are clearly supported.\n"
        "- Each theme MUST be a short noun phrase (e.g., 'Pricing friction', "
        "'Onboarding complexity'), NOT a sentence.\n"
        "- Every supporting_quote MUST be VERBATIM from the transcript — never paraphrase.\n"
        "- Frequency estimate is your judgment: 'high' (5+ mentions), 'medium' (2-4), "
        "'low' (1).\n"
        "- Related entities should match names/orgs/products that actually appear.\n"
        "- Return ONLY valid JSON. No prose, no markdown fences.\n\n"
        "Output schema:\n"
        '{{\n'
        '  "themes": [\n'
        '    {{\n'
        '      "label": "short noun phrase",\n'
        '      "frequency_estimate": "high|medium|low",\n'
        '      "summary": "1-2 sentence description of the theme",\n'
        '      "supporting_quotes": [\n'
        '        {{"speaker": "...", "text": "verbatim quote"}}\n'
        '      ],\n'
        '      "related_entities": ["..."]\n'
        '    }}\n'
        '  ]\n'
        '}}',
    ),
    (
        "human",
        "Optional focus: {focus}\n\n"
        "Transcript:\n{transcript_context}\n\n"
        "Extract themes now.",
    ),
])

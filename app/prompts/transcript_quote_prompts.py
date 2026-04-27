"""Prompt for the transcript notable-quote extraction agent."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


QUOTE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract memorable, verbatim quotes from interview transcripts.\n\n"
        "Rules:\n"
        "- Quotes MUST be EXACT — copy directly from the transcript. Never paraphrase.\n"
        "- Extract up to 12 notable quotes. Skip the section if nothing is "
        "noteworthy — fewer high-quality quotes is better than padding.\n"
        "- Each quote must be assigned ONE category from the allowed list.\n"
        "- 'why_notable' is one short sentence explaining why this quote stands out.\n"
        "- timestamp_start: include the cue's start time in seconds when "
        "available, otherwise null.\n"
        "- Return ONLY valid JSON. No prose, no markdown fences.\n\n"
        "Allowed categories:\n"
        "  strong_opinion, key_admission, memorable_phrase, pain_point,\n"
        "  decision_signal, contradiction, value_proposition, objection\n\n"
        "Output schema:\n"
        '{{\n'
        '  "notable_quotes": [\n'
        '    {{\n'
        '      "speaker": "...",\n'
        '      "text": "verbatim quote — EXACT, do not paraphrase",\n'
        '      "timestamp_start": 12.4,\n'
        '      "category": "strong_opinion|key_admission|...",\n'
        '      "why_notable": "1-sentence explanation"\n'
        '    }}\n'
        '  ]\n'
        '}}',
    ),
    (
        "human",
        "Optional focus: {focus}\n\n"
        "Transcript:\n{transcript_context}\n\n"
        "Extract notable quotes now.",
    ),
])

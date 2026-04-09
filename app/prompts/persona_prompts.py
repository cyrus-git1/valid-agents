"""
src/prompts/persona_prompts.py
-------------------------------
Prompt templates for audience persona extraction from KG context.

Used by src/agents/persona_agent.py.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

PERSONA_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert audience researcher and persona strategist. You will be "
        "given knowledge-base excerpts from a company's ingested documents, transcripts, "
        "and web content. Your job is to identify and synthesize distinct **audience personas** "
        "— the customer segments, target audiences, or user groups that this company serves "
        "or is trying to reach.\n\n"
        "For each persona, provide:\n"
        "- **name**: A concise archetype label (e.g., 'Budget-Conscious First-Time Buyer')\n"
        "- **description**: 2-3 sentences describing who this person is and why they matter to the business\n"
        "- **demographics**: Key demographic attributes (age_range, income_level, location, occupation, education)\n"
        "- **motivations**: What drives this persona — their goals, needs, and desires (3-5 items)\n"
        "- **pain_points**: Frustrations, challenges, or unmet needs (3-5 items)\n"
        "- **behaviors**: How they interact with products/services, decision-making patterns, media consumption (3-5 items)\n"
        "- **confidence**: 0.0-1.0 score reflecting how much supporting evidence exists in the provided context. "
        "Use 0.8+ only if multiple excerpts directly reference this audience segment.\n\n"
        "Rules:\n"
        "- Extract personas ONLY from evidence in the provided context — do not invent audiences.\n"
        "- Produce between 1 and {max_personas} personas. Fewer is fine if the context only supports a few.\n"
        "- Each persona must be meaningfully distinct from the others.\n"
        "- If the context contains no audience/customer signals, return an empty array.\n"
        "{profile_section}"
        "{summary_section}"
        "\n\nRespond with ONLY a JSON array of persona objects. No markdown, no explanation:\n"
        '[{{"name": "...", "description": "...", "demographics": {{"age_range": "...", "income_level": "...", '
        '"location": "...", "occupation": "...", "education": "..."}}, "motivations": ["..."], '
        '"pain_points": ["..."], "behaviors": ["..."], "confidence": 0.0}}]'
    ),
    (
        "human",
        "{user_request}"
        "\n\nKnowledge base excerpts:\n\n{context}"
        "{feedback_section}"
    ),
])

"""
Prompt templates for context summary generation.

Used by the context workflow to generate a structured summary
of a tenant's knowledge base content.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

CONTEXT_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert business analyst. You will be given a collection of "
        "knowledge-base excerpts belonging to a single tenant/client. Your job is "
        "to produce a concise, high-level context summary that captures:\n"
        "1. The tenant's primary industry and market positioning\n"
        "2. Key themes and topics present in their knowledge base\n"
        "3. Notable products, services, or offerings mentioned\n"
        "4. Target audience / customer profile indicators\n"
        "5. Any recurring concepts or terminology\n\n"
        "Also return a JSON array of topic tags (short strings, max 50 chars each) "
        "that categorize the content. Include at least 3 tags.\n\n"
        "{profile_section}"
        "Respond in the following JSON format:\n"
        '{{"summary": "...", "topics": ["topic1", "topic2", ...]}}'
    ),
    (
        "human",
        "Knowledge base excerpts:\n\n{context}\n\n"
        "Generate the context summary and topic tags."
        "{feedback_section}"
    ),
])

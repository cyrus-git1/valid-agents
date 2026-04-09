"""
src/prompts/enrichment_prompts.py
-----------------------------------
Prompt templates for KG enrichment — gap analysis and URL ranking.

Used by src/agents/enrichment_agent.py.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ── Gap Analysis ─────────────────────────────────────────────────────────────

GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledge base analyst. You will be given excerpts from a "
        "company's internal knowledge graph along with a context summary describing "
        "their domain. Your job is to identify **knowledge gaps** — topics that are "
        "missing, underrepresented, or outdated in the knowledge base.\n\n"
        "For each gap, provide:\n"
        "- **topic**: What is missing or thin (concise label)\n"
        "- **reason**: Why this is a gap — what signals indicate it's missing or weak\n"
        "- **priority**: 'high', 'medium', or 'low' based on business importance\n"
        "- **search_queries**: 2-3 Google search queries that would find content to fill this gap\n\n"
        "Rules:\n"
        "- Identify 3-7 gaps, prioritized by business relevance.\n"
        "- Focus on gaps that external web content could realistically fill "
        "(industry reports, competitor analysis, best practices, market data, regulations).\n"
        "- Do NOT flag gaps that only internal data could fill (e.g., proprietary financials).\n"
        "- Search queries should be specific and actionable — not generic.\n"
        "{profile_section}"
        "{summary_section}"
        "\n\nRespond with ONLY valid JSON:\n"
        '{{"gaps": [{{"topic": "...", "reason": "...", "priority": "high|medium|low", '
        '"search_queries": ["query1", "query2"]}}]}}'
    ),
    (
        "human",
        "{user_request}"
        "\n\nKnowledge base excerpts:\n\n{context}"
        "{feedback_section}"
    ),
])

# ── URL Ranking ──────────────────────────────────────────────────────────────

URL_RANKING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content quality evaluator. You will be given a knowledge gap "
        "description and a list of web search results (title, URL, snippet). Your "
        "job is to rank and filter these results by relevance and quality.\n\n"
        "For each URL worth ingesting, provide:\n"
        "- **url**: The full URL\n"
        "- **title**: The page title\n"
        "- **relevance_reason**: Why this source is useful for filling the gap (1 sentence)\n"
        "- **priority**: 1 (best) to N (worst)\n\n"
        "Rules:\n"
        "- EXCLUDE results that are: paywalled, login-required, forums/reddit threads, "
        "social media posts, PDF download links, or clearly irrelevant.\n"
        "- PREFER: industry reports, authoritative blogs, news articles, documentation, "
        "research papers, and company pages.\n"
        "- Return at most {max_urls} URLs.\n"
        "- If none of the results are worth ingesting, return an empty array.\n\n"
        "Respond with ONLY valid JSON:\n"
        '{{"urls": [{{"url": "...", "title": "...", "relevance_reason": "...", "priority": 1}}]}}'
    ),
    (
        "human",
        "Gap to fill: {gap_topic}\n"
        "Reason: {gap_reason}\n\n"
        "Search results:\n{search_results}"
    ),
])

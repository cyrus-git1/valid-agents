"""
Tools for the enrichment workflow.

Covers the full enrichment pipeline: KB search, context summary,
gap analysis, web search, URL ranking, and web ingestion.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from app import core_client
from app.llm_config import get_llm
from app.prompts.enrichment_prompts import GAP_ANALYSIS_PROMPT, URL_RANKING_PROMPT
from app.analysis.base import BaseAnalysisService
from app.serper_service import SerperService

logger = logging.getLogger(__name__)


# ── Knowledge Base tools ────────────────────────────────────────────────────


@tool
def search_knowledge_base(
    tenant_id: str,
    client_id: str,
    query: str,
    top_k: int = 15,
    hop_limit: int = 1,
) -> List[Dict[str, Any]]:
    """Search the knowledge base for content related to a query.

    Returns chunks with similarity scores. Uses hybrid ranking to boost
    pinned/canonical content and exclude archived documents.
    Use this to sample what the knowledge base currently covers for gap analysis.
    """
    try:
        docs = core_client.search_graph(
            tenant_id=tenant_id,
            client_id=client_id,
            query=query,
            top_k=top_k,
            hop_limit=hop_limit,
            boost_pinned=True,
            exclude_status=["archived", "deprecated"],
        )
    except Exception as e:
        logger.warning("search_knowledge_base failed: %s", e)
        return []

    return [
        {
            "content": doc.page_content,
            "similarity_score": doc.metadata.get("similarity_score", 0.0),
            "node_id": doc.metadata.get("node_id"),
            "document_id": doc.metadata.get("document_id"),
        }
        for doc in docs
    ]


@tool
def get_context_summary(
    tenant_id: str,
    client_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch all context summaries for a tenant+client.

    Returns the tenant-wide summary (summary, topics) plus any
    document-level and topic-level summaries that exist.
    This gives gap analysis a complete view of what the memory layer covers.
    """
    result: Dict[str, Any] = {}
    try:
        tenant_summary = core_client.get_context_summary(
            tenant_id=tenant_id,
            client_id=client_id,
        )
        if tenant_summary:
            result["summary"] = tenant_summary.get("summary", "")
            result["topics"] = tenant_summary.get("topics", [])
    except Exception as e:
        logger.warning("get_context_summary (tenant) failed: %s", e)

    try:
        all_summaries = core_client.list_summaries(
            tenant_id=tenant_id,
            client_id=client_id,
        )
        summaries_list = all_summaries.get("summaries", [])
        doc_topics = [
            {"source_type": s.get("source_type"), "label": s.get("topic") or s.get("document_id"), "topics": s.get("topics", [])}
            for s in summaries_list
            if s.get("source_type") in ("DocumentSummary", "TopicSummary")
        ]
        if doc_topics:
            result["sub_summaries"] = doc_topics
    except Exception as e:
        logger.warning("get_context_summary (list) failed: %s", e)

    return result if result else None


# ── Gap analysis tool ───────────────────────────────────────────────────────


@tool
def analyze_knowledge_gaps(
    context: str,
    profile_section: str,
    summary_section: str,
    user_request: str,
) -> Dict[str, Any]:
    """Analyze the knowledge base to identify gaps that could be filled with web content.

    Takes KB context excerpts, profile, summary, and the user's request.
    Returns a dict with 'gaps' — each gap has topic, reason, priority, and search_queries.
    """
    llm = get_llm("enrichment")
    chain = GAP_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "context": context,
            "profile_section": profile_section,
            "summary_section": summary_section,
            "user_request": user_request,
            "feedback_section": "",
        })
        return _parse_json(raw)
    except Exception as e:
        logger.exception("analyze_knowledge_gaps failed")
        return {"gaps": [], "error": str(e)}


# ── Web search tools ───────────────────────────────────────────────────────


@tool
def web_search(
    query: str,
    num_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search the web using Serper API.

    Returns a list of results, each with 'title', 'link', and 'snippet'.
    Returns an empty list if Serper is not configured or the search fails.
    """
    serper = SerperService()
    if not serper.is_configured:
        logger.warning("web_search: Serper API key not configured")
        return []

    try:
        return serper.search(query, num_results=num_results)
    except Exception as e:
        logger.warning("web_search failed for query %r: %s", query[:50], e)
        return []


@tool
def rank_web_sources(
    gap_topic: str,
    gap_reason: str,
    search_results_text: str,
    max_urls: int = 3,
) -> Dict[str, Any]:
    """Rank web search results by relevance and quality for filling a knowledge gap.

    Takes the gap description and formatted search results.
    Returns a dict with 'urls' — each ranked by priority with relevance reasons.
    Filters out paywalled, login-required, forum, and social media content.
    """
    llm = get_llm("enrichment")
    chain = URL_RANKING_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "gap_topic": gap_topic,
            "gap_reason": gap_reason,
            "search_results": search_results_text,
            "max_urls": str(max_urls),
        })
        return _parse_json(raw)
    except Exception as e:
        logger.warning("rank_web_sources failed: %s", e)
        return {"urls": []}


# ── Ingestion tool ──────────────────────────────────────────────────────────


@tool
def ingest_web_url(
    tenant_id: str,
    client_id: str,
    url: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recommend ingestion of a web URL into the knowledge base.

    Returns a recommendation payload for a caller to send to the stateless
    ingest endpoint, or an error payload on failure.
    """
    try:
        # Send as a recommendation — the core API or user decides whether to ingest
        return {"status": "recommended", "url": url, "title": title, "message": "URL recommended for ingestion. Use /ingest/web to process."}
    except Exception as e:
        logger.warning("ingest_web_url failed for %s: %s", url, e)
        return {"error": str(e), "url": url}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code blocks."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON: %s", cleaned[:300])
        return {}

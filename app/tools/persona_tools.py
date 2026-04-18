"""
Tools for the persona discovery agent.

Uses a closure-based factory pattern — tenant/client context is baked into
each tool so the ReAct agent's tool signatures are clean (no UUIDs).

Usage
-----
    tools = create_persona_tools(tenant_id="...", client_id="...")
    # Each tool is a @tool-decorated callable with clean signatures:
    #   search_kb("enterprise SaaS buyer persona")
    #   get_summary()
    #   count_transcripts()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app import core_client
from app.serper_service import SerperService

logger = logging.getLogger(__name__)


def create_persona_tools(tenant_id: str, client_id: str) -> list:
    """Build persona tools with tenant/client context captured in closures.

    Returns a list of @tool-decorated callables for the ReAct agent.
    """

    @tool
    def search_kb(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Search the knowledge base for audience and persona evidence.

        Returns content chunks with similarity scores and document IDs.
        Uses hybrid ranking: boosts pinned/canonical documents and excludes
        archived content for higher-quality persona evidence.

        Use diverse queries to find different audience segments:
        - Customer demographics and segments
        - Pain points and challenges
        - Buyer behaviors and decision patterns
        - Product usage patterns
        """
        try:
            docs = core_client.search_graph(
                tenant_id=tenant_id,
                client_id=client_id,
                query=query,
                top_k=top_k,
                hop_limit=1,
                boost_pinned=True,
                exclude_status=["archived", "deprecated"],
            )
        except Exception as e:
            logger.warning("persona search_kb failed: %s", e)
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
    def get_summary() -> Optional[Dict[str, Any]]:
        """Fetch all available context summaries for this tenant.

        Returns the tenant-wide summary plus any document-level or topic-level
        summaries that exist. Check this first to understand what the KB covers
        before searching.
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
            logger.warning("persona get_summary (tenant) failed: %s", e)

        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            doc_summaries = [
                s for s in all_summaries.get("summaries", [])
                if s.get("source_type") in ("DocumentSummary", "TopicSummary")
            ]
            if doc_summaries:
                result["additional_summaries"] = doc_summaries
        except Exception as e:
            logger.warning("persona get_summary (list) failed: %s", e)

        return result if result else None

    @tool
    def count_transcripts() -> int:
        """Count how many transcript documents exist for this tenant.

        If transcripts exist, they may contain voice-of-customer data
        useful for building personas. Call get_transcripts() to fetch them.
        """
        try:
            return core_client.count_transcripts(
                tenant_id=tenant_id,
                client_id=client_id,
            )
        except Exception as e:
            logger.warning("persona count_transcripts failed: %s", e)
            return 0

    @tool
    def get_transcripts(limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch transcript chunks for voice-of-customer analysis.

        Returns transcript content that may reveal customer sentiments,
        pain points, and behaviors directly from conversations.
        """
        try:
            return core_client.get_transcript_chunks(
                tenant_id=tenant_id,
                client_id=client_id,
                limit=limit,
            )
        except Exception as e:
            logger.warning("persona get_transcripts failed: %s", e)
            return []

    @tool
    def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for external audience and demographic data.

        Use this when the KB lacks demographic or behavioral data for a
        segment. Search for industry-specific persona research, audience
        reports, or demographic studies.

        Returns list of {title, link, snippet}.
        """
        serper = SerperService()
        if not serper.is_configured:
            return []
        try:
            return serper.search(query, num_results=num_results)
        except Exception as e:
            logger.warning("persona search_web failed: %s", e)
            return []

    return [search_kb, get_summary, count_transcripts, get_transcripts, search_web]

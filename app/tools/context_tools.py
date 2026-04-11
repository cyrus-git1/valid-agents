"""
Tools for the context generation workflow.

Handles KG retrieval, existing summary lookup, and summary storage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app import core_client

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(
    tenant_id: str,
    client_id: str,
    query: str,
    top_k: int = 15,
    hop_limit: int = 1,
) -> List[Dict[str, Any]]:
    """Search the knowledge base for content to build a context summary.

    Uses graph-expanded vector search for broad coverage.
    Returns content chunks with similarity scores.
    """
    try:
        docs = core_client.search_graph(
            tenant_id=tenant_id,
            client_id=client_id,
            query=query,
            top_k=top_k,
            hop_limit=hop_limit,
        )
    except Exception as e:
        logger.warning("context search_knowledge_base failed: %s", e)
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
def get_existing_summary(
    tenant_id: str,
    client_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch the existing context summary for a tenant+client.

    Returns a dict with 'summary' and 'topics', or None if no summary exists.
    """
    try:
        return core_client.get_context_summary(
            tenant_id=tenant_id,
            client_id=client_id,
        )
    except Exception as e:
        logger.warning("get_existing_summary failed: %s", e)
        return None


@tool
def store_context_summary(
    tenant_id: str,
    client_id: str,
    summary: str,
    topics: List[str],
    source_stats: Optional[Dict[str, Any]] = None,
    client_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a validated context summary back to the memory layer.

    Sends the summary to the core API for storage.
    """
    try:
        result = core_client.upsert_context_summary(
            tenant_id=tenant_id,
            client_id=client_id,
            summary=summary,
            topics=topics,
            metadata=client_profile or {},
            source_stats=source_stats or {},
        )
        return {"status": "ok", **result}
    except Exception as e:
        logger.warning("store_context_summary failed: %s", e)
        return {"status": "error", "error": str(e)}

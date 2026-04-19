"""
Tools for the Valid docs chat agent.

Single tool — search — hitting the core API's /search/valid endpoint.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx
from langchain_core.tools import tool

from app.core_client import CORE_API_URL, _outbound_headers, _TIMEOUT

logger = logging.getLogger(__name__)


def create_valid_tools() -> list:
    """Build tools for the Valid docs KG. No tenant/client scoping."""

    @tool
    def search_valid(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Valid's internal knowledge base.

        Use this to answer any question about Valid — the product,
        features, pricing, team, customers, roadmap, positioning, etc.
        Returns content chunks with similarity scores.
        Always cite the content you find in your response.
        """
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(
                    f"{CORE_API_URL}/search/valid",
                    json={"query": query, "top_k": top_k, "node_types": ["Chunk"]},
                    headers=_outbound_headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            return [
                {
                    "content": r.get("content", ""),
                    "similarity_score": r.get("similarity_score", 0.0),
                    "document_id": r.get("document_id"),
                    "chunk_index": r.get("chunk_index"),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("search_valid failed: %s", e)
            return [{"error": str(e)}]

    return [search_valid]

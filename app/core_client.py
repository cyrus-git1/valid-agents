"""
HTTP client for the Core API (Data Plane).

Used for data operations that still live in the core API:
  - Vector search (semantic, graph)
  - Transcript data access
  - Document title resolution
  - Survey output persistence

Operations that are now local (no longer HTTP calls):
  - Ingest (use app.services.ingest_service.IngestService)
  - RAG/ask (use app.services.search_service.SearchService)
  - Context summaries (use app.services.context_summary_service.ContextSummaryService)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CORE_API_URL = os.environ.get("CORE_API_URL", "http://localhost:8000")
_TIMEOUT = 300


# -- Search (data plane queries) --


def search_graph(
    *,
    tenant_id: str,
    client_id: str,
    query: str,
    top_k: int = 5,
    hop_limit: int = 1,
    max_neighbours: int = 3,
    min_edge_weight: float = 0.75,
) -> List[Document]:
    """KG graph-expanded search via Core API -> LangChain Documents."""
    resp = _post("/search/graph", {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "query": query,
        "top_k": top_k,
        "hop_limit": hop_limit,
        "max_neighbours": max_neighbours,
        "min_edge_weight": min_edge_weight,
    })
    return _results_to_documents(resp.get("results", []))


def search_semantic(
    *,
    tenant_id: str,
    client_id: str,
    query: str,
    top_k: int = 5,
) -> List[Document]:
    """KG vector-only search via Core API -> LangChain Documents."""
    resp = _post("/search/semantic", {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "query": query,
        "top_k": top_k,
    })
    return _results_to_documents(resp.get("results", []))


# -- Context summaries --


def get_context_summary(
    *,
    tenant_id: str,
    client_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch context summary for a tenant+client."""
    try:
        resp = _get("/data/context/summary/get", {
            "tenant_id": tenant_id,
            "client_id": client_id,
        })
        return resp if resp.get("summary") else None
    except Exception as e:
        logger.warning("Failed to fetch context summary: %s", e)
        return None


# -- Survey outputs --


def get_survey_outputs(
    *,
    tenant_id: str,
    client_id: str,
    output_type: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Fetch prior survey outputs for a tenant+client."""
    try:
        params: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "client_id": client_id,
        }
        if output_type:
            params["output_type"] = output_type
        resp = _get("/data/survey-outputs", params)
        outputs = resp.get("outputs", [])
        return outputs[:limit]
    except Exception as e:
        logger.warning("Failed to fetch survey outputs: %s", e)
        return []


def save_survey_output(
    *,
    tenant_id: str,
    client_id: str,
    output_type: str,
    request: str,
    questions: List[Dict[str, Any]],
    reasoning: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a survey output via core API."""
    try:
        _post("/data/survey-outputs", {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "output_type": output_type,
            "request": request,
            "questions": questions,
            "reasoning": reasoning,
            "metadata": metadata or {},
        })
    except Exception as e:
        logger.warning("Failed to save survey output: %s", e)


# -- Transcript data --


def get_transcript_chunks(
    *,
    tenant_id: str,
    client_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch transcript chunks for analysis services."""
    try:
        return _get("/data/transcript-chunks", {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "limit": limit,
        }).get("chunks", [])
    except Exception as e:
        logger.warning("Failed to fetch transcript chunks: %s", e)
        return []


def count_transcripts(
    *,
    tenant_id: str,
    client_id: str,
) -> int:
    """Count transcript documents for a tenant+client."""
    try:
        return _get("/data/transcript-count", {
            "tenant_id": tenant_id,
            "client_id": client_id,
        }).get("count", 0)
    except Exception as e:
        logger.warning("Failed to count transcripts: %s", e)
        return 0


def get_document_titles(
    document_ids: List[str],
) -> Dict[str, str]:
    """Resolve document IDs to titles."""
    if not document_ids:
        return {}
    try:
        return _post("/data/document-titles", {
            "document_ids": document_ids,
        }).get("titles", {})
    except Exception as e:
        logger.warning("Failed to resolve document titles: %s", e)
        return {}


# -- Helpers --


def _results_to_documents(results: List[Dict[str, Any]]) -> List[Document]:
    """Convert core API search results to LangChain Documents."""
    docs = []
    for r in results:
        docs.append(Document(
            page_content=r.get("content", ""),
            metadata={
                "node_id": r.get("node_id"),
                "node_key": r.get("node_key"),
                "node_type": r.get("node_type"),
                "similarity_score": r.get("similarity_score"),
                "document_id": r.get("document_id"),
                "chunk_index": r.get("chunk_index"),
                "source": r.get("source", "vector"),
                "retrieval_reason": r.get("retrieval_reason"),
                "evidence_quote": r.get("evidence_quote"),
                "evidence_score": r.get("evidence_score"),
                "evidence_count": r.get("evidence_count", 0),
            },
        ))
    return docs


def _post(endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{CORE_API_URL}{endpoint}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        return resp.json()


def _get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{CORE_API_URL}{endpoint}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

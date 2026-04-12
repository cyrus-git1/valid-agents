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

CORE_API_URL = os.environ.get("CORE_API_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = 600  # 10 minutes — large payloads (web scrapes) need time
_CHUNK_BATCH_SIZE = 25  # send chunks in batches to avoid timeouts


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
    node_types: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
) -> List[Document]:
    """KG graph-expanded search via Core API -> LangChain Documents."""
    body: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "query": query,
        "top_k": top_k,
        "hop_limit": hop_limit,
        "max_neighbours": max_neighbours,
        "min_edge_weight": min_edge_weight,
    }
    if node_types:
        body["node_types"] = node_types
    if rel_types:
        body["rel_types"] = rel_types
    resp = _post("/search/graph", body)
    return _results_to_documents(resp.get("results", []))


def search_semantic(
    *,
    tenant_id: str,
    client_id: str,
    query: str,
    top_k: int = 5,
    node_types: Optional[List[str]] = None,
) -> List[Document]:
    """KG vector-only search via Core API -> LangChain Documents."""
    body: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "query": query,
        "top_k": top_k,
    }
    if node_types:
        body["node_types"] = node_types
    resp = _post("/search/semantic", body)
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


# -- Ingest (stateless — agent sends processed data, core API stores it) --


def ingest_document(
    *,
    tenant_id: str,
    client_id: str,
    file_name: str,
    file_bytes: bytes,
    source_type: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Send a processed document to the core API for storage.

    Sends chunks in batches to avoid timeouts on large documents.
    First call creates the document + first batch of chunks.
    Subsequent calls append chunks to the same document.
    """
    import base64
    all_chunks = chunks or []
    all_entities = entities or []

    # First batch — includes file bytes + document creation
    first_batch = all_chunks[:_CHUNK_BATCH_SIZE]
    payload = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "file_name": file_name,
        "file_bytes_b64": base64.b64encode(file_bytes).decode("utf-8"),
        "source_type": source_type,
        "title": title or file_name,
        "metadata": metadata or {},
        "chunks": first_batch,
        "entities": all_entities,
    }
    result = _post("/ingest/processed", payload)
    total_stored = len(first_batch)
    warnings = list(result.get("warnings", []))

    # Remaining batches — append chunks to the same document
    doc_id = result.get("document_id", "")
    for i in range(_CHUNK_BATCH_SIZE, len(all_chunks), _CHUNK_BATCH_SIZE):
        batch = all_chunks[i:i + _CHUNK_BATCH_SIZE]
        try:
            batch_result = _post("/ingest/processed", {
                "tenant_id": tenant_id,
                "client_id": client_id,
                "file_name": file_name,
                "source_type": source_type,
                "title": title or file_name,
                "metadata": {**(metadata or {}), "append_to_document": doc_id},
                "chunks": batch,
                "entities": [],  # entities only sent with first batch
            })
            total_stored += len(batch)
            warnings.extend(batch_result.get("warnings", []))
        except Exception as e:
            warnings.append(f"Chunk batch {i}-{i+len(batch)} failed: {e}")
            logger.warning("Chunk batch %d failed: %s", i, e)

    result["chunks_upserted"] = total_stored
    result["warnings"] = warnings
    return result


def ingest_web_scraped(
    *,
    tenant_id: str,
    client_id: str,
    url: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Send processed web scrape data to the core API for storage.

    Sends chunks in batches to avoid timeouts on large scrapes.
    """
    all_chunks = chunks or []
    all_entities = entities or []

    # First batch
    first_batch = all_chunks[:_CHUNK_BATCH_SIZE]
    result = _post("/ingest/processed-web", {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "url": url,
        "title": title or url,
        "metadata": metadata or {},
        "chunks": first_batch,
        "entities": all_entities,
    })
    total_stored = len(first_batch)
    warnings = list(result.get("warnings", []))

    # Remaining batches
    doc_id = result.get("document_id", "")
    for i in range(_CHUNK_BATCH_SIZE, len(all_chunks), _CHUNK_BATCH_SIZE):
        batch = all_chunks[i:i + _CHUNK_BATCH_SIZE]
        try:
            batch_result = _post("/ingest/processed-web", {
                "tenant_id": tenant_id,
                "client_id": client_id,
                "url": url,
                "title": title or url,
                "metadata": {**(metadata or {}), "append_to_document": doc_id},
                "chunks": batch,
                "entities": [],
            })
            total_stored += len(batch)
            warnings.extend(batch_result.get("warnings", []))
        except Exception as e:
            warnings.append(f"Chunk batch {i}-{i+len(batch)} failed: {e}")
            logger.warning("Chunk batch %d failed: %s", i, e)

    result["chunks_upserted"] = total_stored
    result["warnings"] = warnings
    return result


def upsert_context_summary(
    *,
    tenant_id: str,
    client_id: str,
    summary: str,
    topics: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    source_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a context summary via the core API."""
    return _post("/data/context/summary/upsert", {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "summary": summary,
        "topics": topics,
        "metadata": metadata or {},
        "source_stats": source_stats or {},
    })


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

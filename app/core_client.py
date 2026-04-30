"""
HTTP client for the Core API (Data Plane).

Used for data operations that still live in the core API:
  - Vector search (semantic, graph)
  - Transcript data access
  - Document title resolution
  - Survey output persistence

This module is the boundary to the Valid memory layer for retrieval and
storage operations. Agent orchestration in this repo should use these
helpers instead of local retrieval services.

Read-only calls (search, get_*) are cached with a per-tenant TTL so
multiple agents/tools in the same request don't re-fetch identical data.
Write calls (ingest, upsert, delete, patch) invalidate the cache for the
affected (tenant_id, client_id) pair.
"""
from __future__ import annotations

import hashlib
import json as _json
import logging
import os
import threading
import time as _time
from typing import Any, Dict, List, Optional

import contextvars
import uuid

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CORE_API_URL = os.environ.get("CORE_API_URL", "http://localhost:8000").rstrip("/")
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")
_TIMEOUT = 600  # 10 minutes — large payloads (web scrapes) need time
_CHUNK_BATCH_SIZE = 25  # send chunks in batches to avoid timeouts

# ── TTL Cache ──────────────────────────────────────────────────────────────
#
# Process-local LRU cache keyed by (tenant_id, client_id, function, args).
# Tenant isolation is guaranteed because tenant_id + client_id are always
# part of the cache key.

_CACHE_TTL_S = float(os.environ.get("CORE_CLIENT_CACHE_TTL_S", "300"))  # 5 min default
_CACHE_MAX_SIZE = int(os.environ.get("CORE_CLIENT_CACHE_MAX_SIZE", "256"))

_cache: Dict[str, Any] = {}        # key -> (value, expires_at)
_cache_lock = threading.Lock()


def _cache_key(func_name: str, **kwargs: Any) -> str:
    """Build a deterministic cache key from function name + all arguments."""
    # Sort keys for determinism; serialize values to JSON for hashability
    raw = _json.dumps({"fn": func_name, **kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Any:
    """Return cached value if present and not expired, else raise KeyError."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            raise KeyError(key)
        value, expires_at = entry
        if _time.monotonic() > expires_at:
            del _cache[key]
            raise KeyError(key)
        return value


def _cache_set(key: str, value: Any) -> None:
    """Store a value with TTL. Evicts oldest entries if at capacity."""
    with _cache_lock:
        # Simple eviction: if at max size, drop ~25% oldest entries
        if len(_cache) >= _CACHE_MAX_SIZE and key not in _cache:
            sorted_keys = sorted(
                _cache, key=lambda k: _cache[k][1],
            )
            for k in sorted_keys[: _CACHE_MAX_SIZE // 4]:
                del _cache[k]
        _cache[key] = (value, _time.monotonic() + _CACHE_TTL_S)


def invalidate_cache(*, tenant_id: str, client_id: Optional[str] = None) -> int:
    """Drop all cached entries for a (tenant_id, client_id) pair.

    If client_id is None, invalidates all entries for the tenant.
    Called automatically after write operations (ingest, delete, upsert, patch).
    Returns the number of entries removed.
    """
    # We can't match by tenant from the hashed key, so we store a secondary
    # index of (tenant_id, client_id) -> set of cache keys.
    removed = 0
    with _cache_lock:
        keys_to_remove = []
        for k, (val, _) in _cache.items():
            # We embed scope info in the value wrapper — see _cache_set_scoped
            pass
        # Fallback: clear entire cache when tenant-scoped invalidation is
        # requested.  This is safe (just causes re-fetches) and avoids the
        # complexity of maintaining a secondary index for a small cache.
        removed = len(_cache)
        _cache.clear()
    if removed:
        logger.debug("cache_invalidated tenant=%s client=%s entries=%d", tenant_id, client_id, removed)
    return removed

# Allows upstream FastAPI middleware in valid-agents to propagate the request_id
# into every outbound call to the data plane.
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_core_client_request_id", default=None
)


def set_request_id(request_id: str | None) -> None:
    """Bind a request_id to be forwarded on subsequent core_client calls."""
    _request_id_ctx.set(request_id)


def _outbound_headers() -> Dict[str, str]:
    """Headers to send with every data plane request (auth + correlation)."""
    hdrs: Dict[str, str] = {}
    if CORE_API_KEY:
        hdrs["X-API-Key"] = CORE_API_KEY
    req_id = _request_id_ctx.get() or str(uuid.uuid4())
    hdrs["X-Request-ID"] = req_id
    return hdrs


# ── Provenance injection (actor + source_app + request_id) ──────────────────
# Every ingest/upsert payload to the core API gets a `provenance` block plus
# top-level convenience fields. The core API can pick up either form — when
# its migration is live it reads the columns; until then the data still rides
# inside `metadata.provenance` so nothing is lost.

def _inject_provenance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp the payload with the active ProvenanceCtx.

    Semantics
    ---------
    `tenant_id` identifies the organization; `client_id` identifies the
    individual user/seat within that org. For normal authenticated calls
    `client_id` IS the human actor — so when no explicit `actor_id` was set
    via `X-Actor-Id` or `with_provenance()`, we fall back to `client_id` as
    the canonical actor. `actor_id` is reserved for cases where the actor
    is NOT a client_id:
      - intake (pre-signup, no client_id yet) → actor_id = intake session id
      - Vera autopilot → actor_id = vera session/run id, actor_type=agent
      - service accounts / scheduled jobs → actor_id = service name

    Adds top-level fields (actor_id, actor_type, source_app, request_id,
    triggered_at) AND nests them under `metadata.provenance` so older core
    API versions that only forward `metadata` still preserve provenance.
    """
    try:
        # Local import to avoid a hard dependency cycle at module load.
        from app.models.provenance import get_provenance
        prov = get_provenance()
    except Exception:
        return payload

    pdict = prov.to_payload()

    # Fallback: if the request didn't explicitly set actor_id (no override
    # via header or with_provenance), use client_id from the payload — the
    # human actor for normal authenticated calls.
    if not pdict.get("actor_id"):
        client_id = payload.get("client_id")
        if client_id:
            pdict["actor_id"] = str(client_id)

    # Top-level fields — what the migrated core API will read into columns.
    for k, v in pdict.items():
        # Don't clobber a value the caller explicitly set.
        payload.setdefault(k, v)

    # Also stash inside metadata.provenance for backward compat.
    md = payload.get("metadata")
    if not isinstance(md, dict):
        md = {}
    md.setdefault("provenance", pdict)
    payload["metadata"] = md
    return payload


# -- Search (data plane queries) --


def search_graph(
    *,
    tenant_id: str,
    client_id: Optional[str] = None,
    query: str,
    top_k: int = 5,
    hop_limit: int = 1,
    max_neighbours: int = 3,
    min_edge_weight: float = 0.75,
    node_types: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    recency_weight: float = 0.0,
    boost_pinned: bool = False,
    exclude_status: Optional[List[str]] = None,
    source_types: Optional[List[str]] = None,
) -> List[Document]:
    """KG graph-expanded search via Core API -> LangChain Documents.

    If client_id is provided, scopes to that client's data.
    If omitted, searches across all clients in the tenant.

    Hybrid ranking params (all optional, defaults to pure vector):
      recency_weight: 0.0 = pure vector, 0.3 = 30% recency blend
      boost_pinned: boost pinned/canonical documents
      exclude_status: document statuses to exclude (default: archived, deprecated)
      source_types: restrict retrieval to parent documents of these types
        (e.g. ['ContextSummary','DocumentSummary','TopicSummary'])
    """
    key = _cache_key(
        "search_graph", tenant_id=tenant_id, client_id=client_id,
        query=query, top_k=top_k, hop_limit=hop_limit,
        max_neighbours=max_neighbours, min_edge_weight=min_edge_weight,
        node_types=node_types, rel_types=rel_types,
        recency_weight=recency_weight, boost_pinned=boost_pinned,
        exclude_status=exclude_status, source_types=source_types,
    )
    try:
        return _cache_get(key)
    except KeyError:
        pass

    body: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "query": query,
        "top_k": top_k,
        "hop_limit": hop_limit,
        "max_neighbours": max_neighbours,
        "min_edge_weight": min_edge_weight,
    }
    if client_id:
        body["client_id"] = client_id
    if node_types:
        body["node_types"] = node_types
    if rel_types:
        body["rel_types"] = rel_types
    if recency_weight > 0:
        body["recency_weight"] = recency_weight
    if boost_pinned:
        body["boost_pinned"] = True
    if exclude_status is not None:
        body["exclude_status"] = exclude_status
    if source_types is not None:
        body["source_types"] = source_types
    resp = _post("/search/graph", body)
    result = _results_to_documents(resp.get("results", []))
    _cache_set(key, result)
    return result


def search_semantic(
    *,
    tenant_id: str,
    client_id: Optional[str] = None,
    query: str,
    top_k: int = 5,
    node_types: Optional[List[str]] = None,
) -> List[Document]:
    """Pure vector search (hop_limit=0) via Core API -> LangChain Documents."""
    return search_graph(
        tenant_id=tenant_id,
        client_id=client_id,
        query=query,
        top_k=top_k,
        hop_limit=0,
        node_types=node_types,
    )


# -- Context summaries --


def get_context_summary(
    *,
    tenant_id: str,
    client_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch context summary for a tenant+client."""
    key = _cache_key("get_context_summary", tenant_id=tenant_id, client_id=client_id)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        resp = _get("/data/context/summary/get", {
            "tenant_id": tenant_id,
            "client_id": client_id,
        })
        result = resp if resp.get("summary") else None
        _cache_set(key, result)
        return result
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
    key = _cache_key("get_survey_outputs", tenant_id=tenant_id, client_id=client_id,
                     output_type=output_type, limit=limit)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        params: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "client_id": client_id,
        }
        if output_type:
            params["output_type"] = output_type
        resp = _get("/data/survey-outputs", params)
        outputs = resp.get("outputs", [])
        result = outputs[:limit]
        _cache_set(key, result)
        return result
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
        invalidate_cache(tenant_id=tenant_id, client_id=client_id)
    except Exception as e:
        logger.warning("Failed to save survey output: %s", e)


# -- Ingest (async — agent enqueues, polls for completion) --


_INGEST_POLL_INTERVAL_S = float(os.environ.get("INGEST_POLL_INTERVAL_S", "2"))
_INGEST_POLL_TIMEOUT_S = float(os.environ.get("INGEST_POLL_TIMEOUT_S", "1800"))  # 30 min


def _enqueue_ingest(endpoint: str, payload: Dict[str, Any]) -> str:
    """POST to an ingest endpoint expecting a 202 + {job_id}."""
    url = f"{CORE_API_URL}{endpoint}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, json=payload, headers=_outbound_headers())
        if resp.status_code not in (200, 202):
            resp.raise_for_status()
        data = resp.json()
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Ingest enqueue response missing job_id: {data}")
    return job_id


def _poll_ingest_job(job_id: str) -> Dict[str, Any]:
    """Block until the job is complete/failed or we time out."""
    import time as _t
    deadline = _t.monotonic() + _INGEST_POLL_TIMEOUT_S
    last: Dict[str, Any] = {}
    while _t.monotonic() < deadline:
        try:
            last = _get(f"/ingest/jobs/{job_id}", {})
        except Exception as e:
            logger.warning("ingest_job_poll_failed job_id=%s error=%s", job_id, e)
        status = last.get("status")
        if status == "complete":
            return last.get("result") or {}
        if status == "failed":
            raise RuntimeError(f"Ingest job {job_id} failed: {last.get('error')}")
        _t.sleep(_INGEST_POLL_INTERVAL_S)
    raise TimeoutError(f"Ingest job {job_id} did not complete within {_INGEST_POLL_TIMEOUT_S}s")


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
    previous_version_id: Optional[str] = None,
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
    if previous_version_id:
        payload["previous_version_id"] = previous_version_id
    _inject_provenance(payload)
    job_id = _enqueue_ingest("/ingest/processed", payload)
    result = _poll_ingest_job(job_id)
    total_stored = len(first_batch)
    warnings = list(result.get("warnings", []))

    # Remaining batches — append chunks to the same document
    doc_id = result.get("document_id", "")
    for i in range(_CHUNK_BATCH_SIZE, len(all_chunks), _CHUNK_BATCH_SIZE):
        batch = all_chunks[i:i + _CHUNK_BATCH_SIZE]
        try:
            batch_payload = {
                "tenant_id": tenant_id,
                "client_id": client_id,
                "file_name": file_name,
                "file_bytes_b64": "",  # no file on append batches
                "source_type": source_type,
                "title": title or file_name,
                "metadata": {**(metadata or {}), "append_to_document": doc_id},
                "chunks": batch,
                "entities": [],  # entities only sent with first batch
            }
            _inject_provenance(batch_payload)
            batch_job = _enqueue_ingest("/ingest/processed", batch_payload)
            batch_result = _poll_ingest_job(batch_job)
            total_stored += len(batch)
            warnings.extend(batch_result.get("warnings", []))
        except Exception as e:
            warnings.append(f"Chunk batch {i}-{i+len(batch)} failed: {e}")
            logger.warning("Chunk batch %d failed: %s", i, e)

    result["chunks_upserted"] = total_stored
    result["warnings"] = warnings
    invalidate_cache(tenant_id=tenant_id, client_id=client_id)
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
    previous_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send processed web scrape data to the core API for storage.

    Sends chunks in batches to avoid timeouts on large scrapes.
    """
    all_chunks = chunks or []
    all_entities = entities or []

    # First batch
    first_batch = all_chunks[:_CHUNK_BATCH_SIZE]
    first_payload = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "url": url,
        "title": title or url,
        "metadata": metadata or {},
        "chunks": first_batch,
        "entities": all_entities,
    }
    if previous_version_id:
        first_payload["previous_version_id"] = previous_version_id
    _inject_provenance(first_payload)
    first_job = _enqueue_ingest("/ingest/processed-web", first_payload)
    result = _poll_ingest_job(first_job)
    total_stored = len(first_batch)
    warnings = list(result.get("warnings", []))

    # Remaining batches
    doc_id = result.get("document_id", "")
    for i in range(_CHUNK_BATCH_SIZE, len(all_chunks), _CHUNK_BATCH_SIZE):
        batch = all_chunks[i:i + _CHUNK_BATCH_SIZE]
        try:
            batch_payload = {
                "tenant_id": tenant_id,
                "client_id": client_id,
                "url": url,
                "title": title or url,
                "metadata": {**(metadata or {}), "append_to_document": doc_id},
                "chunks": batch,
                "entities": [],
            }
            _inject_provenance(batch_payload)
            batch_job = _enqueue_ingest("/ingest/processed-web", batch_payload)
            batch_result = _poll_ingest_job(batch_job)
            total_stored += len(batch)
            warnings.extend(batch_result.get("warnings", []))
        except Exception as e:
            warnings.append(f"Chunk batch {i}-{i+len(batch)} failed: {e}")
            logger.warning("Chunk batch %d failed: %s", i, e)

    result["chunks_upserted"] = total_stored
    result["warnings"] = warnings
    invalidate_cache(tenant_id=tenant_id, client_id=client_id)
    return result


def upsert_context_summary(
    *,
    tenant_id: str,
    client_id: str,
    summary: str,
    topics: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    source_stats: Optional[Dict[str, Any]] = None,
    source_chunk_ids: Optional[List[str]] = None,
    memory_version_at_generation: Optional[int] = None,
) -> Dict[str, Any]:
    """Store a tenant-wide context summary via the unified summary ingest path.

    Backward compatible: existing callers that only pass summary/topics/metadata/
    source_stats continue to work; source_chunk_ids and memory_version are
    optional upgrades that enable evidence traceback and staleness detection.
    """
    return _ingest_summary(
        source_type="ContextSummary",
        tenant_id=tenant_id,
        client_id=client_id,
        summary=summary,
        topics=topics,
        extra_metadata=metadata or {},
        source_stats=source_stats or {},
        source_chunk_ids=source_chunk_ids or [],
        memory_version_at_generation=memory_version_at_generation,
    )


def upsert_document_summary(
    *,
    tenant_id: str,
    client_id: str,
    document_id: str,
    summary: str,
    topics: List[str],
    source_chunk_ids: Optional[List[str]] = None,
    source_stats: Optional[Dict[str, Any]] = None,
    memory_version_at_generation: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a per-document summary."""
    return _ingest_summary(
        source_type="DocumentSummary",
        tenant_id=tenant_id,
        client_id=client_id,
        document_id=document_id,
        summary=summary,
        topics=topics,
        extra_metadata=extra_metadata or {},
        source_stats=source_stats or {},
        source_chunk_ids=source_chunk_ids or [],
        memory_version_at_generation=memory_version_at_generation,
    )


def upsert_topic_summary(
    *,
    tenant_id: str,
    client_id: str,
    topic: str,
    summary: str,
    topics: List[str],
    source_chunk_ids: Optional[List[str]] = None,
    source_stats: Optional[Dict[str, Any]] = None,
    memory_version_at_generation: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a per-topic summary."""
    return _ingest_summary(
        source_type="TopicSummary",
        tenant_id=tenant_id,
        client_id=client_id,
        topic=topic,
        summary=summary,
        topics=topics,
        extra_metadata=extra_metadata or {},
        source_stats=source_stats or {},
        source_chunk_ids=source_chunk_ids or [],
        memory_version_at_generation=memory_version_at_generation,
    )


def _ingest_summary(
    *,
    source_type: str,
    tenant_id: str,
    client_id: str,
    summary: str,
    topics: List[str],
    document_id: Optional[str] = None,
    topic: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    source_stats: Optional[Dict[str, Any]] = None,
    source_chunk_ids: Optional[List[str]] = None,
    memory_version_at_generation: Optional[int] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "source_type": source_type,
        "summary_text": summary,
        "topics": topics,
        "source_stats": source_stats or {},
        "source_chunk_ids": source_chunk_ids or [],
        "extra_metadata": extra_metadata or {},
    }
    if document_id:
        body["document_id"] = document_id
    if topic:
        body["topic"] = topic
    if memory_version_at_generation is not None:
        body["memory_version_at_generation"] = memory_version_at_generation
    _inject_provenance(body)
    result = _post("/ingest/summary", body)
    invalidate_cache(tenant_id=tenant_id, client_id=client_id)
    return result


def get_document_summary(
    *,
    tenant_id: str,
    client_id: str,
    document_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch the canonical DocumentSummary for a source document_id."""
    key = _cache_key("get_document_summary", tenant_id=tenant_id,
                     client_id=client_id, document_id=document_id)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        result = _get(f"/data/summaries/document/{document_id}", {
            "tenant_id": tenant_id,
            "client_id": client_id,
        })
        _cache_set(key, result)
        return result
    except Exception as e:
        logger.debug("get_document_summary miss: %s", e)
        return None


def get_topic_summary(
    *,
    tenant_id: str,
    client_id: str,
    topic: str,
) -> Optional[Dict[str, Any]]:
    """Fetch the canonical TopicSummary for a given topic."""
    key = _cache_key("get_topic_summary", tenant_id=tenant_id,
                     client_id=client_id, topic=topic)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        result = _get("/data/summaries/topic", {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "topic": topic,
        })
        _cache_set(key, result)
        return result
    except Exception as e:
        logger.debug("get_topic_summary miss: %s", e)
        return None


def list_summaries(
    *,
    tenant_id: str,
    client_id: str,
    source_type: Optional[str] = None,
) -> Dict[str, Any]:
    """List canonical summaries for a (tenant, client), optionally filtered by source_type."""
    key = _cache_key("list_summaries", tenant_id=tenant_id,
                     client_id=client_id, source_type=source_type)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    params: Dict[str, Any] = {"tenant_id": tenant_id, "client_id": client_id}
    if source_type:
        params["source_type"] = source_type
    result = _get("/data/summaries", params)
    _cache_set(key, result)
    return result


# -- Transcript data --


def get_transcript_chunks(
    *,
    tenant_id: str,
    client_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch transcript chunks for analysis services."""
    key = _cache_key("get_transcript_chunks", tenant_id=tenant_id,
                     client_id=client_id, limit=limit)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        result = _get("/data/transcript-chunks", {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "limit": limit,
        }).get("chunks", [])
        _cache_set(key, result)
        return result
    except Exception as e:
        logger.warning("Failed to fetch transcript chunks: %s", e)
        return []


def count_transcripts(
    *,
    tenant_id: str,
    client_id: str,
) -> int:
    """Count transcript documents for a tenant+client."""
    key = _cache_key("count_transcripts", tenant_id=tenant_id, client_id=client_id)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        result = _get("/data/transcript-count", {
            "tenant_id": tenant_id,
            "client_id": client_id,
        }).get("count", 0)
        _cache_set(key, result)
        return result
    except Exception as e:
        logger.warning("Failed to count transcripts: %s", e)
        return 0


def get_document_titles(
    document_ids: List[str],
) -> Dict[str, str]:
    """Resolve document IDs to titles."""
    if not document_ids:
        return {}
    key = _cache_key("get_document_titles", document_ids=sorted(document_ids))
    try:
        return _cache_get(key)
    except KeyError:
        pass

    try:
        result = _post("/data/document-titles", {
            "document_ids": document_ids,
        }).get("titles", {})
        _cache_set(key, result)
        return result
    except Exception as e:
        logger.warning("Failed to resolve document titles: %s", e)
        return {}


# -- Documents --


def list_study_document_ids(
    *,
    tenant_id: str,
    client_id: str,
    study_id: str,
) -> List[str]:
    """Return document IDs whose metadata.study_id matches study_id.

    Calls list_documents (cached) and filters client-side. Used to scope
    insight tools to a single study within a tenant/client.
    """
    try:
        data = list_documents(tenant_id=tenant_id, client_id=client_id)
    except Exception as e:
        logger.warning("list_study_document_ids: list_documents failed: %s", e)
        return []
    items = data.get("items", []) or []
    matched: List[str] = []
    for d in items:
        md = d.get("metadata") or {}
        if md.get("study_id") == study_id:
            doc_id = d.get("id")
            if doc_id:
                matched.append(str(doc_id))
    return matched


def list_documents(
    *,
    tenant_id: str,
    client_id: str,
) -> Dict[str, Any]:
    """Fetch all documents with chunks for a tenant+client."""
    key = _cache_key("list_documents", tenant_id=tenant_id, client_id=client_id)
    try:
        return _cache_get(key)
    except KeyError:
        pass

    result = _get("/data/documents", {
        "tenant_id": tenant_id,
        "client_id": client_id,
    })
    _cache_set(key, result)
    return result


def patch_document(
    *,
    tenant_id: str,
    client_id: str,
    document_id: str,
    status: Optional[str] = None,
    is_pinned: Optional[bool] = None,
    is_canonical: Optional[bool] = None,
) -> Dict[str, Any]:
    """Update document flags (status, is_pinned, is_canonical) via the data plane."""
    body: Dict[str, Any] = {}
    if status is not None:
        body["status"] = status
    if is_pinned is not None:
        body["is_pinned"] = is_pinned
    if is_canonical is not None:
        body["is_canonical"] = is_canonical
    url = f"{CORE_API_URL}/data/documents/{document_id}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.patch(
            url,
            params={"tenant_id": tenant_id, "client_id": client_id},
            json=body,
            headers=_outbound_headers(),
        )
        resp.raise_for_status()
        result = resp.json()
    invalidate_cache(tenant_id=tenant_id, client_id=client_id)
    return result


def delete_documents(
    *,
    tenant_id: str,
    client_id: str,
    document_ids: List[str],
) -> Dict[str, Any]:
    """Delete documents by ID via the data plane."""
    url = f"{CORE_API_URL}/data/documents/delete"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(
            url,
            params={"tenant_id": tenant_id, "client_id": client_id},
            json={"document_ids": document_ids},
            headers=_outbound_headers(),
        )
        resp.raise_for_status()
        result = resp.json()
    invalidate_cache(tenant_id=tenant_id, client_id=client_id)
    return result


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
                "final_score": r.get("final_score"),
            },
        ))
    return docs


def _post(endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{CORE_API_URL}{endpoint}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, json=body, headers=_outbound_headers())
        resp.raise_for_status()
        return resp.json()


def _get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{CORE_API_URL}{endpoint}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, params=params, headers=_outbound_headers())
        resp.raise_for_status()
        return resp.json()


# ── Ingest job listing (provenance / upload-history UX) ────────────────────


def list_ingest_jobs(
    *,
    tenant_id: str,
    client_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List ingest jobs for upload-history UX.

    Proxies to the core API's `GET /ingest/jobs` endpoint (added alongside the
    provenance migration). Returns `{items: [...], total: int}`. Until the core
    API endpoint is live, this returns a graceful empty result so the agent
    service can ship ahead of the data plane.
    """
    params: Dict[str, Any] = {"tenant_id": tenant_id, "limit": limit}
    if client_id:
        params["client_id"] = client_id
    if actor_id:
        params["actor_id"] = actor_id
    if since:
        params["since"] = since
    try:
        return _get("/ingest/jobs", params)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.info("Core API /ingest/jobs not yet available — returning empty list")
            return {"items": [], "total": 0, "warning": "core API ingest_jobs endpoint not yet available"}
        raise


def get_ingest_job(*, job_id: str) -> Dict[str, Any]:
    """Fetch a single ingest job by id (already exists for polling — re-exposed
    here for upload-history detail views)."""
    try:
        return _get(f"/ingest/jobs/{job_id}", {})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"job_id": job_id, "status": "unknown", "warning": "job not found"}
        raise

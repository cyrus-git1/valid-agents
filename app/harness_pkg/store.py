"""
Harness trace store — single persistence layer for all harness job records.

Storage:
  1. In-memory list (for /harness/traces API — fast, volatile)
  2. Redis (for /jobs/ API — durable across requests, 3-day TTL)
  3. Supabase JSONL bucket (for optimizer — long-term history)

Eviction: traces leave in-memory after 100 entries or 3 days.
Evicted traces are already in Redis + Supabase, so no data is lost.

Flush: redis_flush() dequeues ALL Redis entries and appends them
to the Supabase JSONL bucket as a batch, then deletes from Redis.
Use this for periodic archival or before shutting down.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TRACES = 100
_MAX_AGE_SECONDS = 86400 * 3  # 3 days
_lock = threading.Lock()
_traces: list[dict[str, Any]] = []

REDIS_PREFIX = "harness_jobs:"
REDIS_TTL = 86400 * 3  # 3 days


def _get_redis():
    """Get Redis client, or None if unavailable."""
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


# ── Write ───────────────────────────────────────────────────────────────────


def _evict_stale() -> None:
    """Remove traces over 3 days old or past the 100 limit.
    Must be called with _lock held.
    """
    now = time.time()

    # Remove by age
    _traces[:] = [t for t in _traces if now - t.get("_inserted_at", now) <= _MAX_AGE_SECONDS]

    # Remove by count (keep newest)
    if len(_traces) > _MAX_TRACES:
        _traces[:] = _traces[-_MAX_TRACES:]


def record_trace(result: Any) -> None:
    """Store a harness result trace (in-memory + Redis + Supabase JSONL)."""
    trace_dict = result.to_dict() if hasattr(result, "to_dict") else result

    # 1. In-memory
    trace_dict["_inserted_at"] = time.time()
    with _lock:
        _traces.append(trace_dict)
        _evict_stale()

    # 2. Redis
    try:
        r = _get_redis()
        if r:
            job_id = trace_dict.get("job_id", "")
            key = f"{REDIS_PREFIX}{job_id}"
            # Strip internal field before storing
            store_dict = {k: v for k, v in trace_dict.items() if not k.startswith("_")}
            r.set(key, json.dumps(store_dict, default=str), ex=REDIS_TTL)
    except Exception as e:
        logger.debug("Redis trace persistence skipped: %s", e)

    # 3. Supabase JSONL bucket
    try:
        from app.supabase_client import get_supabase
        from app.optimizer.trace_persistence import persist_trace
        sb = get_supabase()
        store_dict = {k: v for k, v in trace_dict.items() if not k.startswith("_")}
        persist_trace(sb, store_dict)
    except Exception as e:
        logger.debug("Supabase trace persistence skipped: %s", e)


# ── Redis flush ─────────────────────────────────────────────────────────────


def redis_flush() -> dict[str, Any]:
    """Dequeue ALL entries from Redis and append them to the Supabase JSONL bucket.

    Returns a summary of what was flushed. Call this for periodic archival
    or before shutting down to ensure nothing is lost when Redis TTLs expire.
    """
    r = _get_redis()
    if not r:
        return {"flushed": 0, "error": "Redis not available"}

    # Scan all harness job keys
    keys: list[str] = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor, match=f"{REDIS_PREFIX}*", count=100)
        keys.extend(batch)
        if cursor == 0:
            break

    if not keys:
        return {"flushed": 0}

    # Read all traces
    traces: list[dict[str, Any]] = []
    for key in keys:
        raw = r.get(key)
        if raw:
            try:
                traces.append(json.loads(raw))
            except json.JSONDecodeError:
                pass

    if not traces:
        return {"flushed": 0}

    # Group by step_name + date for JSONL files
    from app.optimizer.trace_persistence import persist_trace
    try:
        from app.supabase_client import get_supabase
        sb = get_supabase()

        for trace in traces:
            persist_trace(sb, trace)

        # Delete from Redis after successful flush
        r.delete(*keys)

        logger.info("Redis flush: archived %d traces to Supabase JSONL, deleted %d Redis keys", len(traces), len(keys))
        return {"flushed": len(traces), "keys_deleted": len(keys)}

    except Exception as e:
        logger.warning("Redis flush failed: %s — Redis keys NOT deleted", e)
        return {"flushed": 0, "error": str(e)}


# ── In-memory queries (for /harness/traces) ────────────────────────────────


def get_traces(
    limit: int = 20,
    step_name: str | None = None,
    status: str | None = None,
    tenant_id: str | None = None,
    client_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent traces from in-memory store, newest first."""
    with _lock:
        items = list(_traces)
    items.reverse()
    if step_name:
        items = [t for t in items if t.get("step_name") == step_name]
    if status:
        items = [t for t in items if t.get("status") == status]
    if tenant_id:
        items = [t for t in items if t.get("tenant_id") == tenant_id]
    if client_id:
        items = [t for t in items if t.get("client_id") == client_id]
    return [{k: v for k, v in t.items() if not k.startswith("_")} for t in items[:limit]]


def get_trace_by_id(job_id: str) -> dict[str, Any] | None:
    """Look up a trace by job_id — tries in-memory first, then Redis."""
    with _lock:
        for t in _traces:
            if t.get("job_id") == job_id:
                return {k: v for k, v in t.items() if not k.startswith("_")}

    try:
        r = _get_redis()
        if r:
            raw = r.get(f"{REDIS_PREFIX}{job_id}")
            if raw:
                return json.loads(raw)
    except Exception:
        pass

    return None


def clear_traces() -> int:
    """Clear in-memory traces."""
    with _lock:
        count = len(_traces)
        _traces.clear()
        return count


# ── Redis queries (for /jobs/) ──────────────────────────────────────────────


def list_jobs(
    limit: int = 50,
    step_name: str | None = None,
    status: str | None = None,
    tenant_id: str | None = None,
    client_id: str | None = None,
) -> list[dict[str, Any]]:
    """List harness jobs from Redis, newest first."""
    try:
        r = _get_redis()
        if not r:
            return get_traces(limit=limit, step_name=step_name, status=status,
                              tenant_id=tenant_id, client_id=client_id)

        keys = []
        cursor = 0
        while True:
            cursor, batch = r.scan(cursor, match=f"{REDIS_PREFIX}*", count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        jobs = []
        for key in keys:
            raw = r.get(key)
            if raw:
                try:
                    job = json.loads(raw)
                    if step_name and job.get("step_name") != step_name:
                        continue
                    if status and job.get("status") != status:
                        continue
                    if tenant_id and job.get("tenant_id") != tenant_id:
                        continue
                    if client_id and job.get("client_id") != client_id:
                        continue
                    jobs.append(job)
                except json.JSONDecodeError:
                    continue

        jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)
        return jobs[:limit]

    except Exception as e:
        logger.debug("Redis list_jobs failed: %s — using in-memory fallback", e)
        return get_traces(limit=limit, step_name=step_name, status=status,
                          tenant_id=tenant_id, client_id=client_id)


def get_job(job_id: str) -> dict[str, Any] | None:
    return get_trace_by_id(job_id)


def get_job_errors(job_id: str) -> list[dict[str, Any]]:
    """Extract errors from a job's attempt traces."""
    job = get_trace_by_id(job_id)
    if not job:
        return []

    errors = []
    for attempt in job.get("attempt_traces", []):
        if attempt.get("error"):
            errors.append({
                "attempt": attempt.get("attempt"),
                "error": attempt["error"],
                "started_at": attempt.get("started_at"),
                "outcome": attempt.get("outcome"),
            })
        if attempt.get("cheap_check_feedback"):
            errors.append({
                "attempt": attempt.get("attempt"),
                "error": f"cheap_check: {attempt['cheap_check_feedback']}",
                "started_at": attempt.get("started_at"),
                "outcome": attempt.get("outcome"),
            })
    return errors

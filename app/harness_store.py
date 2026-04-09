"""
In-memory trace store for harness job results + Supabase persistence.

Stores the last N harness traces in-memory for the /harness/traces endpoint.
Also persists to Supabase (table + bucket) for the optimizer to read history.
Not persistent in-memory — clears on restart. Supabase is the durable store.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TRACES = 100
_lock = threading.Lock()
_traces: deque[dict[str, Any]] = deque(maxlen=_MAX_TRACES)


def record_trace(result: Any) -> None:
    """Store a harness result trace (in-memory + Supabase)."""
    trace_dict = result.to_dict() if hasattr(result, "to_dict") else result

    with _lock:
        _traces.append(trace_dict)

    # Persist to Supabase (best-effort, never breaks requests)
    try:
        from app.supabase_client import get_supabase
        from app.optimizer.trace_persistence import persist_trace
        sb = get_supabase()
        persist_trace(sb, trace_dict)
    except Exception as e:
        logger.debug("Supabase trace persistence skipped: %s", e)


def get_traces(
    limit: int = 20,
    step_name: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent traces, optionally filtered."""
    with _lock:
        items = list(_traces)

    # Newest first
    items.reverse()

    if step_name:
        items = [t for t in items if t.get("step_name") == step_name]
    if status:
        items = [t for t in items if t.get("status") == status]

    return items[:limit]


def get_trace_by_id(job_id: str) -> dict[str, Any] | None:
    """Look up a single trace by job_id."""
    with _lock:
        for t in _traces:
            if t.get("job_id") == job_id:
                return t
    return None


def clear_traces() -> int:
    """Clear all stored traces. Returns count cleared."""
    with _lock:
        count = len(_traces)
        _traces.clear()
        return count

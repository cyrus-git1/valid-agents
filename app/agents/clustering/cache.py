"""
TTL cache for tag-normalization LLM fallback results.

Mirrors the pattern in app/agents/transcription/cache.py: thread-safe
dict with monotonic-time TTL and oldest-25% eviction on overflow.

Keyed by (field, raw_value_lowercased) → canonical_key. So an unseen
tag value gets resolved by LLM exactly once across the process lifetime.
"""
from __future__ import annotations

import threading
import time as _time
from typing import Dict, Optional, Tuple


_TTL_S = 24 * 60 * 60     # 24 hours
_MAX = 1024
_EVICT_PCT = 0.25

_store: Dict[Tuple[str, str], Tuple[str, float]] = {}
_lock = threading.Lock()


def _normalize(field: str, raw_value: str) -> Tuple[str, str]:
    return (field.lower().strip(), (raw_value or "").lower().strip())


def get(field: str, raw_value: str) -> Optional[str]:
    """Return cached canonical, or None if missing/expired."""
    key = _normalize(field, raw_value)
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        canonical, expires_at = entry
        if _time.monotonic() > expires_at:
            del _store[key]
            return None
        return canonical


def put(field: str, raw_value: str, canonical: str) -> None:
    """Insert a canonical mapping with TTL. Evicts oldest 25% if at capacity."""
    key = _normalize(field, raw_value)
    with _lock:
        if len(_store) >= _MAX and key not in _store:
            evict_n = max(1, int(_MAX * _EVICT_PCT))
            sorted_keys = sorted(_store, key=lambda k: _store[k][1])
            for k in sorted_keys[:evict_n]:
                _store.pop(k, None)
        _store[key] = (canonical, _time.monotonic() + _TTL_S)


def clear() -> int:
    with _lock:
        n = len(_store)
        _store.clear()
        return n

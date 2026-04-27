"""
Thread-safe TTL cache for the transcription orchestrator.

Keyed by (tenant, client, survey, session, sha256(vtt_content), sorted(scope)).
24hr TTL, max 512 entries. On overflow drops the 25% oldest by expiry time.
Mirrors the cache pattern in `app/core_client.py`.
"""
from __future__ import annotations

import hashlib
import threading
import time as _time
from typing import Any, Iterable, Optional, Tuple


_TTL_S = 24 * 60 * 60     # 24 hours
_MAX = 512                 # max entries before eviction kicks in
_EVICT_PCT = 0.25          # drop oldest 25% on overflow

_store: dict[str, tuple[Any, float]] = {}
_lock = threading.Lock()


# ── Key helpers ──────────────────────────────────────────────────────────


def vtt_hash(vtt_content: str) -> str:
    """SHA-256 of vtt_content. Deterministic — same content → same hash."""
    return hashlib.sha256(vtt_content.encode("utf-8")).hexdigest()


def make_key(
    tenant_id: str,
    client_id: str,
    survey_id: str,
    session_id: str,
    vtt_h: str,
    scope: Iterable[str],
) -> str:
    """Stable cache key. `scope` is an iterable of analysis names."""
    scope_sorted = ",".join(sorted(s for s in scope))
    return f"{tenant_id}|{client_id}|{survey_id}|{session_id}|{vtt_h}|{scope_sorted}"


# ── Get / put ────────────────────────────────────────────────────────────


def get(key: str) -> Optional[Any]:
    """Return cached value if present and not expired, else None."""
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if _time.monotonic() > expires_at:
            del _store[key]
            return None
        return value


def put(key: str, value: Any) -> None:
    """Insert/update an entry with TTL. Evicts oldest 25% if at capacity."""
    with _lock:
        if len(_store) >= _MAX and key not in _store:
            evict_n = max(1, int(_MAX * _EVICT_PCT))
            sorted_keys = sorted(_store, key=lambda k: _store[k][1])
            for k in sorted_keys[:evict_n]:
                _store.pop(k, None)
        _store[key] = (value, _time.monotonic() + _TTL_S)


def clear() -> int:
    """Clear all entries. Returns number cleared. Mostly for tests/admin."""
    with _lock:
        n = len(_store)
        _store.clear()
        return n

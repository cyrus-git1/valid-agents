"""
Thread-safe TTL cache for the insights diagnostic pipeline.

Keyed by (tenant, client, sorted(survey_ids), study_id, sha256(focus_query), critic_enabled).
12hr TTL, max 256 entries. On overflow drops the 25% oldest by expiry time.
"""
from __future__ import annotations

import hashlib
import threading
import time as _time
from typing import Any, Iterable, Optional


_TTL_S = 12 * 60 * 60
_MAX = 256
_EVICT_PCT = 0.25

_store: dict[str, tuple[Any, float]] = {}
_lock = threading.Lock()


def _hash_focus(focus_query: Optional[str]) -> str:
    if not focus_query:
        return "_"
    return hashlib.sha256(focus_query.encode("utf-8")).hexdigest()[:16]


def make_key(
    tenant_id: str,
    client_id: str,
    survey_ids: Optional[Iterable[str]],
    study_id: Optional[str],
    focus_query: Optional[str],
    critic_enabled: bool,
) -> str:
    sids = ",".join(sorted(s for s in (survey_ids or [])))
    sid = study_id or "_"
    fh = _hash_focus(focus_query)
    return f"{tenant_id}|{client_id}|{sids}|{sid}|{fh}|{int(bool(critic_enabled))}"


def get(key: str) -> Optional[Any]:
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
    with _lock:
        if len(_store) >= _MAX and key not in _store:
            evict_n = max(1, int(_MAX * _EVICT_PCT))
            sorted_keys = sorted(_store, key=lambda k: _store[k][1])
            for k in sorted_keys[:evict_n]:
                _store.pop(k, None)
        _store[key] = (value, _time.monotonic() + _TTL_S)


def clear() -> int:
    with _lock:
        n = len(_store)
        _store.clear()
        return n

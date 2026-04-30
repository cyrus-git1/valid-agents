"""Request-scoped provenance context.

Identity model
--------------
- `tenant_id` (in request body, NOT in this ctx) = the organization
- `client_id` (in request body, NOT in this ctx) = the individual user/seat
  under that organization. For normal authenticated calls, client_id IS the
  human actor.
- `actor_id` (in this ctx, OPTIONAL) = override for cases where the actor is
  NOT a client_id:
    - anonymous intake (pre-signup, no client_id yet) → intake session id
    - Vera autopilot acting on behalf of a client → vera session/run id
    - service accounts / scheduled jobs → service name
  When actor_id is None, core_client._inject_provenance falls back to using
  the payload's client_id, so persisted rows always have an actor_id.
- `actor_type` = user | service | agent | anonymous (always set)
- `source_app` = which surface triggered the call (console, vera, intake,
  valid_chat, api) — adds info client_id can't carry on its own.
- `request_id` = correlation id across logs.

Usage
-----
Reading the active provenance:

    from app.models.provenance import get_provenance
    prov = get_provenance()
    # prov.actor_id may be None for normal user calls — that's fine,
    # core_client will fall back to client_id when stamping outbound payloads.

Stamping a service-triggered job (no HTTP request, e.g. Vera autopilot):

    from app.models.provenance import with_provenance
    with with_provenance(actor_id="vera-session-abc", actor_type="agent",
                          source_app="vera"):
        core_client.ingest_document(...)
"""
from __future__ import annotations

import contextlib
import contextvars
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional


VALID_ACTOR_TYPES = {"user", "service", "agent", "anonymous"}
VALID_SOURCE_APPS = {"console", "intake", "vera", "valid_chat", "api", "service", "unknown"}


@dataclass
class ProvenanceCtx:
    """Per-request provenance — never None when read inside a request handler.

    All fields are strings (or None). The middleware fills in defaults so
    callers never need to null-check fields.
    """
    actor_id: Optional[str] = None
    actor_type: str = "anonymous"
    source_app: str = "unknown"
    request_id: str = ""
    triggered_at: str = ""

    def to_payload(self) -> Dict[str, Any]:
        """Render as a dict suitable for inclusion in core API request bodies."""
        return asdict(self)

    def to_metadata(self) -> Dict[str, Any]:
        """Render under a `provenance` namespace for the metadata dict.

        Useful for older core API endpoints that don't yet have first-class
        provenance columns — the data still rides through metadata.
        """
        return {"provenance": asdict(self)}


def _make_default() -> ProvenanceCtx:
    return ProvenanceCtx(
        actor_id=None,
        actor_type="anonymous",
        source_app="unknown",
        request_id=str(uuid.uuid4()),
        triggered_at=datetime.now(timezone.utc).isoformat(),
    )


_ctx: contextvars.ContextVar[ProvenanceCtx] = contextvars.ContextVar(
    "_provenance_ctx", default=_make_default()
)


def get_provenance() -> ProvenanceCtx:
    """Return the active ProvenanceCtx. Always non-None."""
    return _ctx.get()


def set_provenance(prov: ProvenanceCtx) -> contextvars.Token:
    """Bind a ProvenanceCtx for the current execution context.

    Returns a Token suitable for `_ctx.reset(token)` to restore prior state.
    """
    return _ctx.set(prov)


def reset_provenance(token: contextvars.Token) -> None:
    _ctx.reset(token)


@contextlib.contextmanager
def with_provenance(
    *,
    actor_id: Optional[str] = None,
    actor_type: str = "service",
    source_app: str = "service",
    request_id: Optional[str] = None,
) -> Iterator[ProvenanceCtx]:
    """Temporarily override the active ProvenanceCtx.

    Used for service-triggered work (e.g. Vera autopilot, scheduled jobs)
    where there's no inbound HTTP request to populate provenance from headers.
    """
    if actor_type not in VALID_ACTOR_TYPES:
        actor_type = "service"
    if source_app not in VALID_SOURCE_APPS:
        source_app = "service"
    prov = ProvenanceCtx(
        actor_id=actor_id,
        actor_type=actor_type,
        source_app=source_app,
        request_id=request_id or str(uuid.uuid4()),
        triggered_at=datetime.now(timezone.utc).isoformat(),
    )
    token = _ctx.set(prov)
    try:
        yield prov
    finally:
        _ctx.reset(token)


def build_from_headers(headers: Dict[str, str]) -> ProvenanceCtx:
    """Build a ProvenanceCtx from inbound HTTP headers.

    Tolerates missing values: actor_type defaults to "anonymous" when there
    is no X-Actor-Id, and "user" when there is. source_app defaults to "unknown".
    request_id is generated if absent so every request always has one.
    """
    def _h(name: str) -> Optional[str]:
        # Starlette headers are case-insensitive but we accept both casings to
        # tolerate proxies that normalise differently.
        return headers.get(name) or headers.get(name.lower())

    actor_id = _h("X-Actor-Id")
    actor_type_raw = _h("X-Actor-Type") or ("user" if actor_id else "anonymous")
    actor_type = actor_type_raw if actor_type_raw in VALID_ACTOR_TYPES else "user"

    source_app_raw = _h("X-Source-App") or "unknown"
    source_app = source_app_raw if source_app_raw in VALID_SOURCE_APPS else "unknown"

    request_id = _h("X-Request-Id") or str(uuid.uuid4())

    return ProvenanceCtx(
        actor_id=actor_id,
        actor_type=actor_type,
        source_app=source_app,
        request_id=request_id,
        triggered_at=datetime.now(timezone.utc).isoformat(),
    )

"""ProvenanceMiddleware — extract actor/source-app/request-id from inbound
headers and bind them to the request-scoped ProvenanceCtx.

Identity model
--------------
For normal authenticated calls, `client_id` (in the request body) IS the human
actor — so `X-Actor-Id` is OPTIONAL and only used to override actor_id with a
non-client identifier (Vera session id, intake session id, service account name).

Headers read:
    X-Actor-Id     — OPTIONAL override (non-client actors only); empty for normal user calls
    X-Actor-Type   — user | service | agent | anonymous (defaults to "user" if any
                     auth header is present, else "anonymous")
    X-Source-App   — console | intake | vera | valid_chat | api (defaults "unknown")
    X-Request-Id   — UUID; auto-generated if absent

Behaviour:
- Sets the ProvenanceCtx ContextVar so any downstream code (incl. core_client)
  can read it without threading it through call signatures.
- Also calls core_client.set_request_id() so the existing X-Request-ID header
  on outbound calls carries the same id (correlates client → agent → core).
- core_client._inject_provenance falls back actor_id = client_id when no
  explicit X-Actor-Id was sent — every persisted row gets an actor_id.
"""
from __future__ import annotations

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app import core_client
from app.models.provenance import (
    build_from_headers,
    reset_provenance,
    set_provenance,
)

logger = logging.getLogger(__name__)


class ProvenanceMiddleware(BaseHTTPMiddleware):
    """Bind ProvenanceCtx for the duration of each inbound HTTP request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Starlette's request.headers is a Mapping; normalise to dict[str,str]
        headers = {k: v for k, v in request.headers.items()}
        prov = build_from_headers(headers)

        token = set_provenance(prov)
        # Also bind the request_id into the existing core_client ContextVar so
        # outbound X-Request-ID matches inbound. Reuses the dead hook from
        # core_client.py:114-131.
        try:
            core_client.set_request_id(prov.request_id)
        except Exception:
            # Best-effort: never let provenance setup break the request.
            logger.debug("core_client.set_request_id failed; continuing")

        try:
            response = await call_next(request)
        finally:
            reset_provenance(token)

        # Echo the request id back so the client can log it for support.
        try:
            response.headers["X-Request-Id"] = prov.request_id
        except Exception:
            pass
        return response

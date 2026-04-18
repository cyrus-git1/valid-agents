"""
Redis-backed rate limiter middleware for FastAPI.

Limits requests per tenant using a sliding window counter in Redis.
Each route group has its own limit. Tenant is extracted from the request
body (tenant_id field) or X-Tenant-ID header.

If Redis is unavailable, requests pass through (fail-open).

Config via env vars:
  RATE_LIMIT_ENABLED=true          — toggle on/off (default: true)
  RATE_LIMIT_DEFAULT=60            — default requests/min (default: 60)

Per-group limits are defined in ROUTE_LIMITS below.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# ── Route group limits (requests per minute per tenant) ─────────────────────

ROUTE_LIMITS: dict[str, int] = {
    "/survey/generate": 10,
    "/survey/generate-whole": 10,
    "/survey/generate-question": 20,
    "/survey/generate-follow-up": 10,
    "/survey/generate-title": 30,
    "/survey/generate-description": 30,
    "/agent/query": 20,
    "/enrich/run": 5,
    "/persona/find": 10,
    "/optimizer/run": 1,            # 1 per minute (effectively 1 per hour in practice)
    "/ingest/file": 10,
    "/ingest/web": 10,
    "/ingest/batch": 5,
}

DEFAULT_LIMIT = int(os.environ.get("RATE_LIMIT_DEFAULT", "60"))
WINDOW_SECONDS = 60


def _get_redis():
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def _match_route_limit(path: str) -> int:
    """Find the rate limit for a request path."""
    for prefix, limit in ROUTE_LIMITS.items():
        if path.startswith(prefix):
            return limit
    return DEFAULT_LIMIT


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter per tenant per route group."""

    def __init__(self, app: Any):
        super().__init__(app)
        self.enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self._redis = _get_redis() if self.enabled else None

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not self.enabled or not self._redis:
            return await call_next(request)

        # Skip health check and docs
        path = request.url.path
        if path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Extract tenant_id
        tenant_id = await self._extract_tenant_id(request)
        if not tenant_id:
            # No tenant = no rate limit (let auth middleware handle this)
            return await call_next(request)

        # Check rate limit
        limit = _match_route_limit(path)
        key = f"ratelimit:{tenant_id}:{path.split('/')[1]}"  # group by first path segment

        try:
            allowed, remaining, reset_at = self._check_limit(key, limit)
        except Exception as e:
            logger.debug("Rate limiter Redis error: %s — allowing request", e)
            return await call_next(request)

        if not allowed:
            logger.warning(
                "Rate limit exceeded: tenant=%s path=%s limit=%d/min",
                tenant_id, path, limit,
            )
            return Response(
                content=json.dumps({
                    "detail": "Rate limit exceeded",
                    "limit": limit,
                    "window": "60s",
                    "retry_after": reset_at,
                }),
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(reset_at),
                },
            )

        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)

        return response

    def _check_limit(
        self,
        key: str,
        limit: int,
    ) -> tuple[bool, int, int]:
        """Check and increment the sliding window counter.

        Returns (allowed, remaining, seconds_until_reset).
        """
        now = time.time()
        window_start = now - WINDOW_SECONDS

        pipe = self._redis.pipeline()
        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current entries in window
        pipe.zcard(key)
        # Add this request
        pipe.zadd(key, {f"{now}:{id(pipe)}": now})
        # Set TTL so keys auto-clean
        pipe.expire(key, WINDOW_SECONDS + 1)
        results = pipe.execute()

        current_count = results[1]  # zcard result before adding

        remaining = max(0, limit - current_count - 1)
        reset_at = int(WINDOW_SECONDS - (now - window_start))

        if current_count >= limit:
            return False, 0, reset_at

        return True, remaining, reset_at

    async def _extract_tenant_id(self, request: Request) -> str | None:
        """Extract tenant_id from X-Tenant-ID header only.

        We NEVER read the request body in middleware — doing so consumes
        the stream and breaks FastAPI's body parsing for the endpoint.
        Callers should pass X-Tenant-ID header for rate limiting.
        If no header is present, rate limiting is skipped for this request.
        """
        return request.headers.get("X-Tenant-ID")

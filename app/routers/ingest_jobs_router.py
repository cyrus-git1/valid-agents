"""/ingest/jobs router — upload-history endpoints.

Proxies to the core API's ingest_jobs table (created alongside the provenance
migration). Lets the frontend show "all uploads by user X this week" or drill
into a single job for status + error info.

Until the core API endpoint is live, list_ingest_jobs returns an empty list
with a `warning` field so the frontend doesn't 500.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app import core_client
from app.models.api.ingest_jobs import IngestJob, IngestJobListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest/jobs", tags=["ingest"])


@router.get("", response_model=IngestJobListResponse)
def list_jobs(
    tenant_id: str = Query(..., description="Tenant UUID"),
    client_id: Optional[str] = Query(default=None, description="Client UUID (optional)"),
    actor_id: Optional[str] = Query(default=None, description="Filter to a single actor"),
    since: Optional[str] = Query(default=None, description="ISO8601 lower bound on started_at"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List ingest jobs. Most-recent first.

    Filters: by client_id, actor_id, and optional started_at lower bound.
    Returns a graceful empty list with a warning if the core API endpoint
    isn't available yet (pre-migration).
    """
    try:
        result = core_client.list_ingest_jobs(
            tenant_id=tenant_id,
            client_id=client_id,
            actor_id=actor_id,
            since=since,
            limit=limit,
        )
        return IngestJobListResponse(**result)
    except Exception as e:
        logger.exception("list_ingest_jobs failed")
        raise HTTPException(status_code=500, detail=f"list_ingest_jobs failed: {e}")


@router.get("/{job_id}", response_model=IngestJob)
def get_job(job_id: str):
    """Fetch a single ingest job by id."""
    try:
        result = core_client.get_ingest_job(job_id=job_id)
        return IngestJob(id=result.get("id", job_id), **{
            k: v for k, v in result.items() if k != "id"
        })
    except Exception as e:
        logger.exception("get_ingest_job failed")
        raise HTTPException(status_code=500, detail=f"get_ingest_job failed: {e}")

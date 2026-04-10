"""
Harness API — traces, jobs, and flush endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.harness_pkg import store

router = APIRouter(prefix="/harness", tags=["harness"])


# ── Traces (in-memory) ─────────────────────────────────────────────────────


@router.get("/traces", response_model=List[Dict[str, Any]])
def list_traces(
    limit: int = Query(20, ge=1, le=100),
    step_name: Optional[str] = Query(None, description="Filter by step name"),
    status: Optional[str] = Query(None, description="Filter by status: passed, exhausted, failed"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
):
    """List recent harness traces from in-memory store, newest first."""
    return store.get_traces(limit=limit, step_name=step_name, status=status,
                            tenant_id=tenant_id, client_id=client_id)


@router.get("/traces/{job_id}", response_model=Dict[str, Any])
def get_trace(job_id: str):
    """Get a single harness trace by job_id."""
    trace = store.get_trace_by_id(job_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {job_id} not found")
    return trace


@router.delete("/traces")
def clear_traces():
    """Clear in-memory traces."""
    count = store.clear_traces()
    return {"cleared": count}


# ── Jobs (Redis-backed) ────────────────────────────────────────────────────


@router.get("/jobs", response_model=List[Dict[str, Any]])
def list_jobs(
    step_name: Optional[str] = Query(None, description="Filter by step name"),
    status: Optional[str] = Query(None, description="Filter by status: passed, exhausted, failed"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    limit: int = Query(50, ge=1, le=200),
):
    """List harness jobs from Redis, newest first."""
    return store.list_jobs(limit=limit, step_name=step_name, status=status,
                           tenant_id=tenant_id, client_id=client_id)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get full job record including all attempt traces."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("/jobs/{job_id}/errors")
def get_job_errors(job_id: str):
    """Get errors and validation failures from a job's attempts."""
    errors = store.get_job_errors(job_id)
    job = store.get_job(job_id)
    return {
        "job_id": job_id,
        "step_name": job.get("step_name") if job else None,
        "status": job.get("status") if job else None,
        "error_count": len(errors),
        "errors": errors,
    }


# ── Flush ───────────────────────────────────────────────────────────────────


@router.post("/flush")
def flush_redis():
    """Flush all Redis harness entries to Supabase JSONL bucket.

    Dequeues every trace from Redis, appends to daily JSONL files,
    then deletes the Redis keys. Use for archival or before shutdown.
    """
    return store.redis_flush()

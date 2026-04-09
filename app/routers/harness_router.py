"""
Harness observability endpoints — view traces from harness runs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app import harness_store

router = APIRouter(prefix="/harness", tags=["harness"])


@router.get("/traces", response_model=List[Dict[str, Any]])
def list_traces(
    limit: int = Query(20, ge=1, le=100),
    step_name: Optional[str] = Query(None, description="Filter by step name"),
    status: Optional[str] = Query(None, description="Filter by status: passed, exhausted, failed"),
):
    """List recent harness traces, newest first."""
    return harness_store.get_traces(limit=limit, step_name=step_name, status=status)


@router.get("/traces/{job_id}", response_model=Dict[str, Any])
def get_trace(job_id: str):
    """Get a single harness trace by job_id."""
    trace = harness_store.get_trace_by_id(job_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {job_id} not found")
    return trace


@router.delete("/traces")
def clear_traces():
    """Clear all stored traces."""
    count = harness_store.clear_traces()
    return {"cleared": count}

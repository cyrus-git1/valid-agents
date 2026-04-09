"""
/panel router
--------------
Panel participant ingest and filtering.

POST /panel/ingest                 -- Ingest participant data (background task)
GET  /panel/ingest/status/{job_id} -- Poll ingest job status
POST /panel/filter                 -- Filter participants against business context
"""
from __future__ import annotations

import uuid
import logging
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.supabase_client import get_supabase
from app.models.api.panel_participants import (
    PanelFilterRequest,
    PanelFilterResponse,
    PanelIngestRequest,
    PanelIngestResponse,
)
from app.services.panel_participant_service import PanelParticipantService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/panel", tags=["panel"])

_jobs: Dict[str, Dict] = {}


def _run_panel_ingest(job_id: str, req: PanelIngestRequest) -> None:
    try:
        sb = get_supabase()
        svc = PanelParticipantService(sb)
        result = svc.ingest_participants(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            vendor_name=req.vendor_name,
            participants=req.participants,
            metadata=req.metadata,
            embed_model=req.embed_model,
            embed_batch_size=req.embed_batch_size,
            build_kg=req.build_kg,
        )
        _jobs[job_id] = {
            "status": "complete",
            "vendor_name": req.vendor_name,
            "total_participants": result["total_participants"],
            "completed": result["completed"],
            "failed": result["failed"],
            "results": result["results"],
            "warnings": result["warnings"],
        }
    except Exception as e:
        logger.exception("Panel ingest job %s failed", job_id)
        _jobs[job_id] = {
            "status": "failed",
            "vendor_name": req.vendor_name,
            "total_participants": len(req.participants),
            "completed": 0,
            "failed": len(req.participants),
            "results": [],
            "warnings": [str(e)],
        }


@router.post("/ingest", response_model=PanelIngestResponse)
async def ingest_panel_participants(
    req: PanelIngestRequest,
    background_tasks: BackgroundTasks,
) -> PanelIngestResponse:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "vendor_name": req.vendor_name,
        "total_participants": len(req.participants),
        "completed": 0, "failed": 0, "results": [], "warnings": [],
    }
    background_tasks.add_task(_run_panel_ingest, job_id, req)
    return PanelIngestResponse(
        job_id=job_id, vendor_name=req.vendor_name,
        total_participants=len(req.participants), status="running",
    )


@router.get("/ingest/status/{job_id}", response_model=PanelIngestResponse)
async def get_panel_ingest_status(job_id: str) -> PanelIngestResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return PanelIngestResponse(
        job_id=job_id, vendor_name=job["vendor_name"],
        total_participants=job["total_participants"], status=job["status"],
        completed=job.get("completed", 0), failed=job.get("failed", 0),
        results=job.get("results", []), warnings=job.get("warnings", []),
    )


@router.post("/filter", response_model=PanelFilterResponse)
async def filter_panel_participants(req: PanelFilterRequest) -> PanelFilterResponse:
    sb = get_supabase()
    svc = PanelParticipantService(sb)
    try:
        return svc.filter_participants(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            filter_mode=req.filter_mode,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            llm_model=req.llm_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Panel filtering failed")
        raise HTTPException(status_code=500, detail=f"Filtering failed: {e}")

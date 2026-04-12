"""
/ingest router
--------------
Handles getting content into the system.

POST /ingest/file    -- Upload PDF, DOCX, VTT, or XLSX (multipart), runs full pipeline in background
POST /ingest/web     -- Kick off a website crawl by URL, runs in background
GET  /ingest/status/{job_id} -- Poll job status
POST /ingest/batch/files  -- Batch upload multiple files
POST /ingest/batch/web    -- Batch scrape multiple URLs
GET  /ingest/batch/status/{batch_id} -- Poll batch progress
"""
from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
import json as _json

from app.models.api.ingest import (
    IngestEntity,
    IngestFileResponse,
    IngestWebRequest,
    IngestWebResponse,
    IngestStatusResponse,
    BatchWebRequest,
    BatchIngestResponse,
    BatchIngestStatusResponse,
    BatchItemStatus,
)
from app.models.api.ingest import IngestInput
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

# In-memory job store
_jobs: Dict[str, Dict[str, Any]] = {}


def _run_file_ingest(
    job_id: str,
    file_bytes: bytes,
    file_name: str,
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    title: str | None,
    prune_after_ingest: bool,
    entities: List[IngestEntity] | None = None,
    extract_entities: bool = True,
) -> None:
    """Background task: full PDF/DOCX ingest pipeline."""
    _jobs[job_id] = {"status": "running"}
    try:
        svc = IngestService()
        result = svc.ingest(IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            file_bytes=file_bytes,
            file_name=file_name,
            title=title,
            entities=entities or [],
            extract_entities=extract_entities,
            prune_after_ingest=prune_after_ingest,
        ))
        _jobs[job_id] = {
            "status": "complete",
            "document_id": str(result.document_id),
            "source_type": result.source_type,
            "source_uri": result.source_uri,
            "chunks_upserted": result.chunks_upserted,
            "entities_linked": result.entities_linked,
            "warnings": result.warnings,
            "prune_result": result.prune_result,
        }
        logger.info("Job %s complete — %d chunks, %d entities", job_id, result.chunks_upserted, result.entities_linked)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id] = {"status": "failed", "detail": str(e)}


def _run_web_ingest(
    job_id: str,
    url: str,
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    title: str | None,
    metadata: Dict[str, Any],
    prune_after_ingest: bool,
    entities: List[IngestEntity] | None = None,
    extract_entities: bool = True,
) -> None:
    """Background task: full web scrape + ingest pipeline."""
    _jobs[job_id] = {"status": "running"}
    try:
        svc = IngestService()
        result = svc.ingest(IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            web_url=url,
            title=title,
            metadata=metadata,
            entities=entities or [],
            extract_entities=extract_entities,
            prune_after_ingest=prune_after_ingest,
        ))
        _jobs[job_id] = {
            "status": "complete",
            "document_id": str(result.document_id),
            "source_type": result.source_type,
            "source_uri": result.source_uri,
            "chunks_upserted": result.chunks_upserted,
            "entities_linked": result.entities_linked,
            "warnings": result.warnings,
            "prune_result": result.prune_result,
        }
        logger.info("Job %s complete — %d chunks, %d entities", job_id, result.chunks_upserted, result.entities_linked)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id] = {"status": "failed", "detail": str(e)}


@router.post("/file", response_model=IngestFileResponse, status_code=202)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF, DOCX, VTT, or XLSX file to ingest"),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    title: str | None = Form(default=None),
    entities: str | None = Form(default=None, description='JSON array of entities: [{"name": "...", "type": "..."}]'),
    extract_entities: bool = Form(default=True, description="Run LLM-based NER to auto-extract entities from chunks"),
    prune_after_ingest: bool = Form(default=False),
) -> IngestFileResponse:
    _ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/x-pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "text/vtt",
        "text/plain",
        "application/octet-stream",
    }
    _ALLOWED_EXTENSIONS = {"pdf", "docx", "vtt", "xlsx", "xls"}

    file_bytes = await file.read()
    file_name = file.filename or f"upload_{uuid.uuid4().hex}.bin"
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()

    if file.content_type not in _ALLOWED_CONTENT_TYPES and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}' (ext='{ext}'). Send a PDF, DOCX, VTT, or XLSX.",
        )
    # Parse entities from JSON string
    parsed_entities: List[IngestEntity] = []
    if entities:
        try:
            raw = _json.loads(entities)
            parsed_entities = [IngestEntity(**e) for e in raw]
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid entities JSON: {e}")

    job_id = str(uuid.uuid4())

    _EXT_TO_TYPE = {"pdf": "pdf", "docx": "docx", "vtt": "vtt", "xlsx": "xlsx", "xls": "xlsx"}
    source_type = _EXT_TO_TYPE.get(ext, ext or "file")

    background_tasks.add_task(
        _run_file_ingest,
        job_id, file_bytes, file_name,
        tenant_id, client_id, title, prune_after_ingest,
        parsed_entities, extract_entities,
    )

    return IngestFileResponse(
        job_id=job_id,
        document_id="pending",
        source_type=source_type,
        source_uri="pending",
        chunks_upserted=0,
        warnings=[],
    )


@router.post("/web", response_model=IngestWebResponse, status_code=202)
def ingest_web(
    req: IngestWebRequest,
    background_tasks: BackgroundTasks,
) -> IngestWebResponse:
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    job_id = str(uuid.uuid4())

    background_tasks.add_task(
        _run_web_ingest,
        job_id, url,
        req.tenant_id, req.client_id,
        req.title, req.metadata, req.prune_after_ingest,
        req.entities, req.extract_entities,
    )

    return IngestWebResponse(
        job_id=job_id,
        document_id="pending",
        source_type="web",
        source_uri=req.url,
        chunks_upserted=0,
        warnings=[],
    )


@router.get("/status/{job_id}")
def ingest_status(job_id: str):
    """Get full job status including chunks, entities, and warnings."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {
        "job_id": job_id,
        **job,
    }




# -- Batch ingest --

_batches: Dict[str, Dict[str, Any]] = {}


def _run_batch_file_ingest(
    batch_id: str,
    files_data: List[Dict[str, Any]],
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    prune_after_ingest: bool,
) -> None:
    total_chunks = 0
    for i, item in enumerate(files_data):
        _batches[batch_id]["items"][i]["status"] = "running"
        try:
            svc = IngestService()
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                file_bytes=item["file_bytes"],
                file_name=item["file_name"],
                title=item.get("title"),
                prune_after_ingest=prune_after_ingest and (i == len(files_data) - 1),
                skip_context_generation=True,  # generate once at end
            ))
            total_chunks += result.chunks_upserted
            _batches[batch_id]["items"][i].update({
                "status": "complete",
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
                "warnings": result.warnings,
            })
        except Exception as e:
            _batches[batch_id]["items"][i].update({
                "status": "failed",
                "detail": str(e),
            })

    # Generate context summary once after all items ingested
    if total_chunks > 0:
        try:
            from app.agents.context_agent import run_context_agent
            run_context_agent(
                tenant_id=str(tenant_id), client_id=str(client_id),
                force_regenerate=True,
            )
        except Exception as e:
            logger.warning("Batch context generation failed: %s", e)

    _finalise_batch(batch_id)


def _run_batch_web_ingest(
    batch_id: str,
    items: List[Dict[str, Any]],
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    prune_after_ingest: bool,
) -> None:
    total_chunks = 0
    for i, item in enumerate(items):
        _batches[batch_id]["items"][i]["status"] = "running"
        try:
            svc = IngestService()
            raw_url = item["url"].strip()
            if not raw_url.startswith(("http://", "https://")):
                raw_url = f"https://{raw_url}"
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                web_url=raw_url,
                title=item.get("title"),
                metadata=item.get("metadata", {}),
                prune_after_ingest=prune_after_ingest and (i == len(items) - 1),
                skip_context_generation=True,
            ))
            total_chunks += result.chunks_upserted
            _batches[batch_id]["items"][i].update({
                "status": "complete",
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
                "warnings": result.warnings,
            })
        except Exception as e:
            _batches[batch_id]["items"][i].update({
                "status": "failed",
                "detail": str(e),
            })

    # Generate context summary once after all items ingested
    if total_chunks > 0:
        try:
            from app.agents.context_agent import run_context_agent
            run_context_agent(
                tenant_id=str(tenant_id), client_id=str(client_id),
                force_regenerate=True,
            )
        except Exception as e:
            logger.warning("Batch context generation failed: %s", e)

    _finalise_batch(batch_id)


def _finalise_batch(batch_id: str) -> None:
    items = _batches[batch_id]["items"]
    failed = sum(1 for it in items if it["status"] == "failed")
    completed = sum(1 for it in items if it["status"] == "complete")
    if failed == len(items):
        _batches[batch_id]["status"] = "failed"
    elif failed > 0:
        _batches[batch_id]["status"] = "partial_failure"
    else:
        _batches[batch_id]["status"] = "complete"
    _batches[batch_id]["completed"] = completed
    _batches[batch_id]["failed"] = failed
    _batches[batch_id]["running"] = 0


@router.post("/batch/files", response_model=BatchIngestResponse, status_code=202)
async def batch_ingest_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    prune_after_ingest: bool = Form(default=False),
) -> BatchIngestResponse:
    _ALLOWED_EXTENSIONS = {"pdf", "docx", "vtt", "xlsx", "xls"}
    files_data: List[Dict[str, Any]] = []
    for f in files:
        f_ext = (f.filename.rsplit(".", 1)[-1] if f.filename and "." in f.filename else "").lower()
        if f_ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file '{f.filename}'.")
        file_bytes = await f.read()
        files_data.append({"file_bytes": file_bytes, "file_name": f.filename or f"upload_{uuid.uuid4().hex}.bin"})

    batch_id = str(uuid.uuid4())
    items = [{"index": i, "source": fd["file_name"], "status": "pending", "document_id": None, "chunks_upserted": 0, "warnings": [], "detail": None} for i, fd in enumerate(files_data)]
    _batches[batch_id] = {"status": "running", "total": len(files_data), "completed": 0, "failed": 0, "running": len(files_data), "items": items}
    background_tasks.add_task(_run_batch_file_ingest, batch_id, files_data, tenant_id, client_id, prune_after_ingest)
    return BatchIngestResponse(batch_id=batch_id, total=len(files_data), status="running", items=[BatchItemStatus(**it) for it in items])


@router.post("/batch/web", response_model=BatchIngestResponse, status_code=202)
def batch_ingest_web(req: BatchWebRequest, background_tasks: BackgroundTasks) -> BatchIngestResponse:
    batch_id = str(uuid.uuid4())
    items_raw = [{"url": item.url, "title": item.title, "metadata": item.metadata} for item in req.items]
    items = [{"index": i, "source": it["url"], "status": "pending", "document_id": None, "chunks_upserted": 0, "warnings": [], "detail": None} for i, it in enumerate(items_raw)]
    _batches[batch_id] = {"status": "running", "total": len(items_raw), "completed": 0, "failed": 0, "running": len(items_raw), "items": items}
    background_tasks.add_task(_run_batch_web_ingest, batch_id, items_raw, req.tenant_id, req.client_id, req.prune_after_ingest)
    return BatchIngestResponse(batch_id=batch_id, total=len(items_raw), status="running", items=[BatchItemStatus(**it) for it in items])


@router.get("/batch/status/{batch_id}", response_model=BatchIngestStatusResponse)
def batch_ingest_status(batch_id: str) -> BatchIngestStatusResponse:
    batch = _batches.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    return BatchIngestStatusResponse(
        batch_id=batch_id, total=batch["total"], completed=batch["completed"],
        failed=batch["failed"], running=batch["running"], status=batch["status"],
        items=[BatchItemStatus(**it) for it in batch["items"]],
    )

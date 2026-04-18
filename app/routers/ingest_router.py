"""
/ingest router
--------------
Stateless ingest endpoints for file uploads, raw WebVTT text, and survey-results JSON.
"""
from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from app.models.ingest_api import (
    BatchIngestResponse,
    BatchWebRequest,
    IngestFileResponse,
    IngestWebRequest,
    IngestWebResponse,
    SURVEY_RESULTS_JSON_EXAMPLE,
)
from app.services.ingest import IngestService
from app.services.ingest.presenters import to_ingest_response

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file", response_model=IngestFileResponse)
async def ingest_file(
    file: UploadFile | None = File(default=None, description="PDF, DOCX, VTT, or XLSX file to ingest."),
    webvtt_content: str | None = Form(default=None, description="Raw WebVTT text beginning with WEBVTT."),
    survey_results_json: str | None = Form(
        default=None,
        description=(
            "Validated survey-results JSON payload. Example:\n"
            f"{SURVEY_RESULTS_JSON_EXAMPLE}"
        ),
    ),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    title: str | None = Form(default=None),
    entities: str | None = Form(default=None, description='JSON array of entities: [{"name": "...", "type": "..."}]'),
    extract_entities: bool = Form(default=True, description="Run LLM-based NER to auto-extract entities from chunks"),
    prune_after_ingest: bool = Form(default=False),
) -> IngestFileResponse:
    file_bytes = await file.read() if file is not None else None
    file_name = file.filename if file is not None else None
    content_type = file.content_type if file is not None else None
    result = IngestService().ingest_file_request(
        tenant_id=tenant_id,
        client_id=client_id,
        file_bytes=file_bytes,
        file_name=file_name,
        content_type=content_type,
        webvtt_content=webvtt_content,
        survey_results_json=survey_results_json,
        title=title,
        entities_json=entities,
        extract_entities=extract_entities,
        prune_after_ingest=prune_after_ingest,
    )
    return to_ingest_response(result)


@router.post("/web", response_model=IngestWebResponse)
def ingest_web(req: IngestWebRequest) -> IngestWebResponse:
    result = IngestService().ingest_web_request(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        url=req.url,
        title=req.title,
        metadata=req.metadata,
        entities=req.entities,
        extract_entities=req.extract_entities,
        prune_after_ingest=req.prune_after_ingest,
    )
    return IngestWebResponse(**to_ingest_response(result).model_dump())


@router.post("/batch/files", response_model=BatchIngestResponse)
async def batch_ingest_files(
    files: List[UploadFile] = File(...),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    prune_after_ingest: bool = Form(default=False),
) -> BatchIngestResponse:
    service = IngestService()
    file_payloads = [
        {
            "file_bytes": await upload.read(),
            "file_name": upload.filename or f"upload_{uuid.uuid4().hex}.bin",
            "content_type": upload.content_type,
        }
        for upload in files
    ]
    return service.ingest_uploaded_file_batch(
        tenant_id=tenant_id,
        client_id=client_id,
        files=file_payloads,
        prune_after_ingest=prune_after_ingest,
    )


@router.post("/batch/web", response_model=BatchIngestResponse)
def batch_ingest_web(req: BatchWebRequest) -> BatchIngestResponse:
    service = IngestService()
    return service.ingest_web_batch(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        items=[item.model_dump() for item in req.items],
        prune_after_ingest=req.prune_after_ingest,
    )

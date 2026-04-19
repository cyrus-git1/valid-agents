"""
/valid/ingest router
--------------------
Ingest files into Valid's internal knowledge base.

POST /valid/ingest — Upload a file (PDF, DOCX, XLSX, XLS) to Valid's own KG.

Chunks, extracts entities, and sends to the core API's /ingest/valid endpoint.
No tenant/client scoping — this is Valid's internal documentation.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core_client import CORE_API_URL, _outbound_headers, _TIMEOUT
from app.services.chunking_service import document_bytes_to_chunks
from app.services.ingest.service import IngestService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/valid", tags=["valid-ingest"])

_ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "xls"}


class ValidIngestResponse(BaseModel):
    document_id: str = ""
    file_name: str = ""
    source_type: str = ""
    chunks_upserted: int = 0
    entities_linked: int = 0
    warnings: List[str] = Field(default_factory=list)


@router.post("/ingest", response_model=ValidIngestResponse)
async def ingest_valid_file(
    file: UploadFile = File(..., description="PDF, DOCX, XLSX, or XLS file."),
    title: Optional[str] = Form(default=None),
    extract_entities: bool = Form(default=True, description="Run LLM-based NER"),
) -> ValidIngestResponse:
    """Ingest a file into Valid's internal knowledge base.

    Accepts PDF, DOCX, XLSX, XLS. Chunks locally, extracts entities,
    then stores via the core API's /ingest/valid endpoint.
    """
    file_name = file.filename or "upload.bin"
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    warnings: List[str] = []

    # ── Chunk ──────────────────────────────────────────────────────────
    try:
        raw_chunks = document_bytes_to_chunks(file_bytes, ext)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    # Filter chunks (same as IngestService)
    from app.services.chunking_service import ChunkingService
    svc = ChunkingService()
    filtered_chunks, filter_warnings = svc._process_chunks(raw_chunks)
    warnings.extend(filter_warnings)

    if not filtered_chunks:
        raise HTTPException(status_code=400, detail="No usable content after processing.")

    # ── Extract entities ───────────────────────────────────────────────
    entities: List[Dict[str, Any]] = []
    if extract_entities:
        try:
            ingest_svc = IngestService()
            entities = ingest_svc._extract_entities_llm(filtered_chunks)
        except Exception as e:
            warnings.append(f"Entity extraction failed: {e}")
            logger.warning("Valid ingest NER failed: %s", e)

    # ── Send to core API /ingest/valid ─────────────────────────────────
    source_type = {"xlsx": "xlsx", "xls": "xlsx"}.get(ext, ext)

    payload = {
        "file_name": file_name,
        "file_bytes_b64": base64.b64encode(file_bytes).decode("utf-8"),
        "title": title or file_name,
        "source_type": source_type,
        "metadata": {},
        "chunks": [
            {
                "text": c.get("text", ""),
                "start_page": c.get("page_start"),
                "end_page": c.get("page_end"),
                "token_count": c.get("token_count", 0),
            }
            for c in filtered_chunks
        ],
        "entities": entities,
    }

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{CORE_API_URL}/ingest/valid",
                json=payload,
                headers=_outbound_headers(),
            )
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        logger.exception("Valid ingest core API call failed")
        raise HTTPException(status_code=502, detail=f"Core API ingest failed: {e}")

    return ValidIngestResponse(
        document_id=result.get("document_id", ""),
        file_name=file_name,
        source_type=source_type,
        chunks_upserted=result.get("chunks_upserted", 0),
        entities_linked=result.get("entities_linked", 0),
        warnings=warnings + result.get("warnings", []),
    )

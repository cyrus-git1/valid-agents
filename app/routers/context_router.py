"""
/context router
---------------
POST   /context/build                            -- Synchronous ingest + summary generation
POST   /context/summary/generate                 -- Generate (or regenerate) a context summary
POST   /context/summary/get                      -- Retrieve an existing summary
DELETE /context/summary/{tenant_id}/{client_id} -- Delete a summary
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.models.api.context import (
    ContextBuildRequest,
    ContextBuildResponse,
)
from app.models.api.context_summary import (
    ContextSummaryGenerateRequest,
    ContextSummaryGenerateResponse,
    ContextSummaryGetRequest,
    ContextSummaryResponse,
    ContextSummaryDeleteResponse,
)
from app.agents.context_agent import run_context_agent
from app.models.ingest import IngestInput
from app.services.ingest import IngestService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["context"])


def _ensure_parsed(value: Any, fallback: Any = None) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback
    return value if value is not None else fallback


def _row_to_response(row: Dict[str, Any]) -> ContextSummaryResponse:
    return ContextSummaryResponse(
        id=row["id"],
        tenant_id=row["tenant_id"],
        client_id=row["client_id"],
        summary=row["summary"],
        topics=_ensure_parsed(row.get("topics"), []),
        metadata=_ensure_parsed(row.get("metadata"), {}),
        source_stats=_ensure_parsed(row.get("source_stats"), {}),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@router.post("/build", response_model=ContextBuildResponse)
def build_context(req: ContextBuildRequest) -> ContextBuildResponse:
    """Synchronously ingest sources and regenerate the context summary.

    Stateless by design: no background task state, no in-process job store.
    """
    svc = IngestService()
    warnings: list[str] = []
    documents_ingested = 0
    weblinks_ingested = 0
    transcripts_ingested = 0
    total_chunks = 0

    # File documents
    for doc_path in req.context.docs:
        path = Path(doc_path)
        if not path.exists():
            warnings.append(f"File not found: {doc_path}")
            continue
        if path.suffix.lower() not in (".pdf", ".docx"):
            warnings.append(f"Skipping unsupported file type: {doc_path}")
            continue
        try:
            result = svc.ingest(IngestInput(
                tenant_id=req.tenant_id,
                client_id=req.client_id,
                file_bytes=path.read_bytes(),
                file_name=path.name,
                title=path.stem,
                skip_context_generation=True,
            ))
            documents_ingested += 1
            total_chunks += result.chunks_upserted
            warnings.extend(result.warnings)
        except Exception as e:
            logger.exception("Context build ingest failed for %s", doc_path)
            warnings.append(f"Failed to ingest {doc_path}: {e}")

    # Transcript files
    for transcript_path in req.context.transcripts:
        path = Path(transcript_path)
        if not path.exists():
            warnings.append(f"Transcript not found: {transcript_path}")
            continue
        if path.suffix.lower() != ".vtt":
            warnings.append(f"Skipping non-VTT transcript: {transcript_path}")
            continue
        try:
            result = svc.ingest(IngestInput(
                tenant_id=req.tenant_id,
                client_id=req.client_id,
                file_bytes=path.read_bytes(),
                file_name=path.name,
                title=path.stem,
                skip_context_generation=True,
            ))
            transcripts_ingested += 1
            total_chunks += result.chunks_upserted
            warnings.extend(result.warnings)
        except Exception as e:
            logger.exception("Context build transcript ingest failed for %s", transcript_path)
            warnings.append(f"Failed to ingest transcript {transcript_path}: {e}")

    # Web links
    for url in req.context.weblinks:
        try:
            result = svc.ingest(IngestInput(
                tenant_id=req.tenant_id,
                client_id=req.client_id,
                web_url=url,
                skip_context_generation=True,
            ))
            weblinks_ingested += 1
            total_chunks += result.chunks_upserted
            warnings.extend(result.warnings)
        except Exception as e:
            logger.exception("Context build web ingest failed for %s", url)
            warnings.append(f"Failed to ingest {url}: {e}")

    if documents_ingested == 0 and weblinks_ingested == 0 and transcripts_ingested == 0:
        raise HTTPException(status_code=400, detail="No valid sources were ingested.")

    try:
        summary_result = run_context_agent(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            client_profile=req.client_profile.model_dump(),
            force_regenerate=True,
        )
        if summary_result.get("status") in ("generation_failed", "store_failed"):
            warnings.append(summary_result.get("error", "Context generation failed"))
    except Exception as e:
        logger.exception("Context summary generation failed after build")
        warnings.append(f"Context summary generation failed: {e}")

    return ContextBuildResponse(
        job_id=None,
        status="complete",
        documents_ingested=documents_ingested,
        weblinks_ingested=weblinks_ingested,
        transcripts_ingested=transcripts_ingested,
        total_chunks=total_chunks,
        kg_nodes_upserted=0,
        kg_edges_upserted=0,
        warnings=warnings,
    )


@router.post("/summary/generate", response_model=ContextSummaryGenerateResponse)
def generate_context_summary(req: ContextSummaryGenerateRequest) -> ContextSummaryGenerateResponse:
    from app.agents.context_agent import run_context_agent

    try:
        result = run_context_agent(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            client_profile=req.client_profile,
            force_regenerate=req.force_regenerate,
        )
    except Exception as e:
        logger.exception("Context summary generation failed")
        raise HTTPException(status_code=500, detail=f"Context summary generation failed: {e}")

    if result.get("status") in ("generation_failed", "store_failed"):
        raise HTTPException(status_code=500, detail=result.get("error", "Context generation failed"))

    # Build response from agent result
    summary_data = {
        "id": str(req.tenant_id),  # placeholder — real ID comes from core API
        "tenant_id": str(req.tenant_id),
        "client_id": str(req.client_id),
        "summary": result.get("summary", ""),
        "topics": result.get("topics", []),
        "metadata": {},
        "source_stats": {},
        "created_at": None,
        "updated_at": None,
    }
    return ContextSummaryGenerateResponse(
        summary=_row_to_response(summary_data),
        status=result.get("status", "complete"),
        regenerated=result.get("regenerated", False),
    )


@router.post("/summary/get", response_model=ContextSummaryResponse)
def get_context_summary(req: ContextSummaryGetRequest) -> ContextSummaryResponse:
    from app import core_client
    row = core_client.get_context_summary(tenant_id=str(req.tenant_id), client_id=str(req.client_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"No context summary found for tenant={req.tenant_id}, client={req.client_id}.")
    return _row_to_response(row)


@router.delete("/summary/{tenant_id}/{client_id}", response_model=ContextSummaryDeleteResponse)
def delete_context_summary(tenant_id: UUID, client_id: UUID) -> ContextSummaryDeleteResponse:
    # TODO: add core_client.delete_context_summary() when core API supports it
    return ContextSummaryDeleteResponse(deleted=False, tenant_id=tenant_id, client_id=client_id)

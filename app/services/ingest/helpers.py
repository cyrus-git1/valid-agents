"""Helper functions for stateless ingest routes."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import HTTPException, UploadFile

from app.models.ingest import IngestEntity, IngestInput
from app.services.ingest.constants import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    ALLOWED_INLINE_INGEST_TYPES,
    INLINE_SOURCE_TYPE,
)

def normalize_url(url: str) -> str:
    url = url.strip()
    return url if url.startswith(("http://", "https://")) else f"https://{url}"


def parse_entities_json(entities: str | None) -> List[IngestEntity]:
    if not entities:
        return []
    try:
        raw = json.loads(entities)
        return [IngestEntity(**entity) for entity in raw]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid entities JSON: {exc}")


def validate_file_upload(file_name: str, content_type: str | None) -> str:
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}' (ext='{ext}'). "
                "Send a PDF, DOCX, VTT, or XLSX file, or use webvtt_content / survey_results_json."
            ),
        )
    return ext


def validate_batch_file_name(file_name: str | None) -> None:
    ext = (file_name.rsplit(".", 1)[-1] if file_name and "." in file_name else "").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file '{file_name}'.")
def init_batch_item(index: int, source: str) -> Dict[str, Any]:
    return {
        "index": index,
        "source": source,
        "status": "pending",
        "document_id": None,
        "chunks_upserted": 0,
        "warnings": [],
        "detail": None,
    }


def build_file_ingest_input(
    *,
    tenant_id,
    client_id,
    file_bytes: bytes,
    file_name: str,
    title: str | None = None,
    entities: List[IngestEntity] | None = None,
    extract_entities: bool = True,
    prune_after_ingest: bool = False,
    skip_context_generation: bool = False,
) -> IngestInput:
    return IngestInput(
        tenant_id=tenant_id,
        client_id=client_id,
        file_bytes=file_bytes,
        file_name=file_name,
        title=title,
        entities=entities or [],
        extract_entities=extract_entities,
        prune_after_ingest=prune_after_ingest,
        skip_context_generation=skip_context_generation,
    )


def build_web_ingest_input(
    *,
    tenant_id,
    client_id,
    url: str,
    title: str | None = None,
    metadata: Dict[str, Any] | None = None,
    entities: List[IngestEntity] | None = None,
    extract_entities: bool = True,
    prune_after_ingest: bool = False,
    skip_context_generation: bool = False,
) -> IngestInput:
    return IngestInput(
        tenant_id=tenant_id,
        client_id=client_id,
        web_url=normalize_url(url),
        title=title,
        metadata=metadata or {},
        entities=entities or [],
        extract_entities=extract_entities,
        prune_after_ingest=prune_after_ingest,
        skip_context_generation=skip_context_generation,
    )


def build_serialized_ingest_input(
    *,
    tenant_id,
    client_id,
    chunks: List[Dict[str, Any]],
    source_type: str,
    source_uri: str,
    title: str,
    metadata: Dict[str, Any] | None = None,
    entities: List[IngestEntity] | None = None,
    extract_entities: bool = True,
    prune_after_ingest: bool = False,
    skip_context_generation: bool = False,
) -> IngestInput:
    return IngestInput(
        tenant_id=tenant_id,
        client_id=client_id,
        serialized_chunks=chunks,
        serialized_source_type=source_type,
        serialized_source_uri=source_uri,
        title=title,
        metadata=metadata or {},
        entities=entities or [],
        extract_entities=extract_entities,
        prune_after_ingest=prune_after_ingest,
        skip_context_generation=skip_context_generation,
    )


def choose_ingest_mode(
    *,
    file: UploadFile | None,
    webvtt_content: str | None,
    survey_results_json: str | None,
) -> str:
    provided = [
        ("file", file is not None),
        ("webvtt", bool(webvtt_content and webvtt_content.strip())),
        ("survey_results", bool(survey_results_json and survey_results_json.strip())),
    ]
    selected = [name for name, is_present in provided if is_present]
    if len(selected) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: file, webvtt_content, or survey_results_json.",
        )
    return selected[0]


def inline_source_type(mode: str) -> str:
    if mode not in ALLOWED_INLINE_INGEST_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported inline ingest mode: {mode}")
    return INLINE_SOURCE_TYPE[mode]

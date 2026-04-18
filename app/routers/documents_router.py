"""
/documents router
-----------------
GET    /documents  — List all documents + chunks for a tenant
DELETE /documents  — Bulk-delete documents + cascade chunks by tenant + document IDs

Proxies to the data plane via core_client.
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.core_client import list_documents, delete_documents, patch_document
from app.models.api.documents import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    DocumentFlagRequest,
    DocumentFlagResponse,
    DocumentPatchRequest,
    DocumentPatchResponse,
    DocumentRevisionResponse,
    DocumentWithChunksListResponse,
)
from app.services.revision_service import RevisionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentWithChunksListResponse)
def get_documents(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> DocumentWithChunksListResponse:
    """Return every document for a tenant+client with all associated chunks."""
    try:
        data = list_documents(tenant_id=str(tenant_id), client_id=str(client_id))
        return DocumentWithChunksListResponse(**data)
    except Exception as e:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=502, detail=str(e))


@router.patch("/{document_id}", response_model=DocumentPatchResponse)
def update_document(
    document_id: str,
    body: DocumentPatchRequest,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> DocumentPatchResponse:
    """Update document flags (status, is_pinned, is_canonical)."""
    try:
        data = patch_document(
            tenant_id=str(tenant_id),
            client_id=str(client_id),
            document_id=document_id,
            status=body.status,
            is_pinned=body.is_pinned,
            is_canonical=body.is_canonical,
        )
        return DocumentPatchResponse(**data)
    except Exception as e:
        logger.exception("Failed to update document")
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("", response_model=BulkDeleteResponse)
def bulk_delete_documents(
    body: BulkDeleteRequest,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> BulkDeleteResponse:
    """Delete documents by tenant_id + list of document_ids."""
    try:
        data = delete_documents(
            tenant_id=str(tenant_id),
            client_id=str(client_id),
            document_ids=body.document_ids,
        )
        return BulkDeleteResponse(**data)
    except Exception as e:
        logger.exception("Failed to delete documents")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/{document_id}/flag", response_model=DocumentFlagResponse)
def flag_document(
    document_id: str,
    body: DocumentFlagRequest,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> DocumentFlagResponse:
    """Flag a document for revision.

    Sets the document status to 'flagged'. If ``corrected_text`` is
    provided, immediately triggers a revision (ingest new content,
    delete old, regenerate summaries).
    """
    # Mark as flagged
    try:
        patch_document(
            tenant_id=str(tenant_id),
            client_id=str(client_id),
            document_id=document_id,
            status="flagged",
        )
    except Exception as e:
        logger.exception("Failed to flag document %s", document_id)
        raise HTTPException(status_code=502, detail=str(e))

    # If corrected content provided, run revision inline
    if body.corrected_text:
        try:
            svc = RevisionService()
            result = svc.revise_document(
                tenant_id=tenant_id,
                client_id=client_id,
                old_document_id=document_id,
                corrected_text=body.corrected_text,
            )
            return DocumentFlagResponse(
                document_id=document_id,
                status="flagged",
                revision_triggered=True,
                new_document_id=result.new_document_id,
                warnings=result.warnings,
            )
        except Exception as e:
            logger.exception("Revision failed for flagged document %s", document_id)
            return DocumentFlagResponse(
                document_id=document_id,
                status="flagged",
                revision_triggered=False,
                warnings=[f"Flagged but revision failed: {e}"],
            )

    return DocumentFlagResponse(
        document_id=document_id,
        status="flagged",
        revision_triggered=False,
    )


@router.post("/{document_id}/revise", response_model=DocumentRevisionResponse)
def revise_document(
    document_id: str,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    file: UploadFile = File(...),
    title: str = Form(default=None),
    extract_entities: bool = Form(default=True),
) -> DocumentRevisionResponse:
    """Replace a document with a re-uploaded file.

    Ingests the new file, deletes the old document, and regenerates
    affected summaries (document-level and tenant-level).
    """
    try:
        file_bytes = file.file.read()
        svc = RevisionService()
        return svc.revise_document(
            tenant_id=tenant_id,
            client_id=client_id,
            old_document_id=document_id,
            file_bytes=file_bytes,
            file_name=file.filename or "upload",
            title=title,
            extract_entities=extract_entities,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to revise document %s", document_id)
        raise HTTPException(status_code=502, detail=str(e))

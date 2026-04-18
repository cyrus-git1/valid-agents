"""
Document revision service — replaces a flagged document with corrected content.

Pipeline: ingest new → delete old → regenerate summaries.
Deterministic (no LLM routing), so a plain service class rather than a
LangGraph agent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from app import core_client
from app.models.api.documents import DocumentRevisionResponse
from app.models.ingest import IngestInput, IngestOutput

logger = logging.getLogger(__name__)


class RevisionService:
    """Orchestrates document replacement with summary regeneration."""

    def revise_document(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        old_document_id: str,
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        corrected_text: Optional[str] = None,
        title: Optional[str] = None,
        extract_entities: bool = True,
    ) -> DocumentRevisionResponse:
        """Replace an existing document with corrected content.

        Accepts either ``file_bytes`` + ``file_name`` (re-upload) **or**
        ``corrected_text`` (inline edit).  Exactly one must be provided.

        Steps:
          1. Ingest the new content (skip context generation — we do it ourselves).
          2. Delete the old document (only after successful ingest).
          3. Regenerate the document-level summary for the new document.
          4. Regenerate the tenant-level summary.
        """
        warnings: List[str] = []

        # ── Step 1: Ingest new content ─────────────────────────────────
        ingest_result = self._ingest_new_content(
            tenant_id=tenant_id,
            client_id=client_id,
            old_document_id=old_document_id,
            file_bytes=file_bytes,
            file_name=file_name,
            corrected_text=corrected_text,
            title=title,
            extract_entities=extract_entities,
        )
        new_document_id = str(ingest_result.document_id)
        warnings.extend(ingest_result.warnings)

        # ── Step 2: Delete old document ────────────────────────────────
        old_deleted = self._delete_old_document(
            tenant_id=str(tenant_id),
            client_id=str(client_id),
            old_document_id=old_document_id,
            warnings=warnings,
        )

        # ── Step 3 & 4: Regenerate summaries ───────────────────────────
        regenerated = self._regenerate_summaries(
            tenant_id=str(tenant_id),
            client_id=str(client_id),
            new_document_id=new_document_id,
            warnings=warnings,
        )

        return DocumentRevisionResponse(
            old_document_id=old_document_id,
            new_document_id=new_document_id,
            chunks_upserted=ingest_result.chunks_upserted,
            old_document_deleted=old_deleted,
            summaries_regenerated=regenerated,
            warnings=warnings,
        )

    # ── Private helpers ────────────────────────────────────────────────

    def _ingest_new_content(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        old_document_id: str,
        file_bytes: Optional[bytes],
        file_name: Optional[str],
        corrected_text: Optional[str],
        title: Optional[str],
        extract_entities: bool,
    ) -> IngestOutput:
        from app.services.ingest.service import IngestService

        if file_bytes is not None and file_name is not None:
            inp = IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                file_bytes=file_bytes,
                file_name=file_name,
                title=title or file_name,
                extract_entities=extract_entities,
                skip_context_generation=True,
            )
        elif corrected_text is not None:
            inp = IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                serialized_chunks=[{
                    "text": corrected_text,
                    "chunk_index": 0,
                    "page_start": None,
                    "page_end": None,
                }],
                serialized_source_type="corrected_text",
                serialized_source_uri=f"revision:{old_document_id}",
                title=title or f"Revision of {old_document_id}",
                extract_entities=extract_entities,
                skip_context_generation=True,
            )
        else:
            raise ValueError(
                "Revision requires either file_bytes + file_name or corrected_text."
            )

        return IngestService().ingest(inp)

    @staticmethod
    def _delete_old_document(
        *,
        tenant_id: str,
        client_id: str,
        old_document_id: str,
        warnings: List[str],
    ) -> bool:
        try:
            core_client.delete_documents(
                tenant_id=tenant_id,
                client_id=client_id,
                document_ids=[old_document_id],
            )
            return True
        except Exception as exc:
            msg = f"Failed to delete old document {old_document_id}: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            return False

    @staticmethod
    def _regenerate_summaries(
        *,
        tenant_id: str,
        client_id: str,
        new_document_id: str,
        warnings: List[str],
    ) -> List[str]:
        from app.agents.context_agent import run_context_agent

        regenerated: List[str] = []

        # Document-level summary for the new document
        try:
            run_context_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                force_regenerate=True,
                granularity_level="document",
                scope_ref=new_document_id,
            )
            regenerated.append("document")
        except Exception as exc:
            msg = f"Document summary regeneration failed: {exc}"
            logger.warning(msg)
            warnings.append(msg)

        # Tenant-level summary (may reference old document content)
        try:
            run_context_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                force_regenerate=True,
                granularity_level="tenant",
            )
            regenerated.append("tenant")
        except Exception as exc:
            msg = f"Tenant summary regeneration failed: {exc}"
            logger.warning(msg)
            warnings.append(msg)

        return regenerated

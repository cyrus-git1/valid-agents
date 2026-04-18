"""Presentation helpers for ingest API responses."""
from __future__ import annotations

from app.models.ingest import IngestOutput
from app.models.ingest_api import IngestFileResponse


def to_ingest_response(result: IngestOutput) -> IngestFileResponse:
    return IngestFileResponse(
        document_id=str(result.document_id),
        source_type=result.source_type,
        source_uri=result.source_uri,
        chunks_upserted=result.chunks_upserted,
        entities_linked=result.entities_linked,
        warnings=result.warnings,
        prune_result=result.prune_result,
    )

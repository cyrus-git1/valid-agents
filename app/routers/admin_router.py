"""
/admin router (agent service)
-----------------------------
POST /admin/reindex/{document_id} -- Re-chunk and re-embed a document
POST /admin/rebuild-kg            -- Rebuild the full KG for a client
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.supabase_client import get_supabase
from app.models.api.admin import (
    RebuildKGRequest,
    RebuildKGResponse,
    ReindexRequest,
    ReindexResponse,
)
from app.models.ingest import IngestInput
from app.services.ingest import IngestService

# TODO: KGService has moved to the memory service
KGService = None
KGBuildConfig = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reindex/{document_id}", response_model=ReindexResponse)
def reindex_document(document_id: str, req: ReindexRequest) -> ReindexResponse:
    sb = get_supabase()

    res = (
        sb.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("tenant_id", str(req.tenant_id))
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    doc = res.data[0]
    source_uri = doc.get("source_uri", "")
    meta = doc.get("metadata", {})

    if not source_uri.startswith("bucket:"):
        raise HTTPException(
            status_code=400,
            detail=f"Document source_uri '{source_uri}' is not a bucket URI.",
        )

    try:
        svc = IngestService(sb)
        file_bytes, file_type, bucket, path = svc.download_from_storage(source_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage download failed: {e}")

    try:
        result = svc.ingest(IngestInput(
            tenant_id=req.tenant_id,
            client_id=doc.get("client_id") and UUID(doc["client_id"]),
            file_bytes=file_bytes,
            file_name=meta.get("file_name") or path.split("/")[-1],
            title=doc.get("title"),
            metadata=meta,
            embed_model=req.embed_model,
            embed_batch_size=req.embed_batch_size,
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

    return ReindexResponse(
        document_id=str(result.document_id),
        chunks_upserted=result.chunks_upserted,
        warnings=result.warnings,
    )


@router.post("/rebuild-kg", response_model=RebuildKGResponse)
def rebuild_kg(req: RebuildKGRequest) -> RebuildKGResponse:
    sb = get_supabase()
    kg_svc = KGService(sb)

    cfg = KGBuildConfig(
        similarity_threshold=req.similarity_threshold,
        max_edges_per_chunk=req.max_edges_per_chunk,
        batch_size=req.batch_size,
    )

    try:
        result = kg_svc.build_kg_from_chunk_embeddings(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            document_id=None,
            config=cfg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG rebuild failed: {e}")

    return RebuildKGResponse(
        nodes_upserted=result.get("nodes_upserted", 0),
        edges_upserted=result.get("edges_upserted", 0),
        chunks_processed=result.get("chunks_valid", 0),
    )

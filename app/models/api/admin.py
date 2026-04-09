"""Pydantic models for admin operations (reindex, rebuild-kg)."""
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel

from app.models.base import TenantScoped


class ReindexRequest(TenantScoped):
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64


class ReindexResponse(BaseModel):
    document_id: str
    chunks_upserted: int
    warnings: list


class RebuildKGRequest(TenantScoped):
    similarity_threshold: float = 0.82
    max_edges_per_chunk: int = 10
    batch_size: int = 500


class RebuildKGResponse(BaseModel):
    nodes_upserted: int
    edges_upserted: int
    chunks_processed: int

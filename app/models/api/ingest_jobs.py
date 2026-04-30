"""Pydantic models for the ingest-jobs proxy endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IngestJob(BaseModel):
    """Single ingest job — mirrors the core API ingest_jobs row shape."""
    id: str
    tenant_id: str
    client_id: Optional[str] = None
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    source_app: Optional[str] = None
    request_id: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    status: str = "unknown"
    document_id: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class IngestJobListResponse(BaseModel):
    items: List[IngestJob] = Field(default_factory=list)
    total: int = 0
    warning: Optional[str] = None


class IngestJobListRequest(BaseModel):
    tenant_id: UUID
    client_id: Optional[UUID] = None
    actor_id: Optional[str] = None
    since: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp; only return jobs started at or after this.",
    )
    limit: int = Field(default=50, ge=1, le=500)

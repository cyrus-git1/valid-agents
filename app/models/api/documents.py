"""Pydantic models for the /documents router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content: str
    content_tokens: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    has_embedding: bool


class DocumentWithChunksResponse(BaseModel):
    id: str
    tenant_id: str
    client_id: Optional[str] = None
    source_type: str
    source_uri: Optional[str] = None
    title: Optional[str] = None
    source_timestamp: Optional[datetime] = None
    is_pinned: bool
    is_canonical: bool
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunks: List[ChunkResponse] = Field(default_factory=list)


class DocumentWithChunksListResponse(BaseModel):
    items: List[DocumentWithChunksResponse]
    total: int


class DocumentPatchRequest(BaseModel):
    status: Optional[str] = Field(default=None, description="active | draft | deprecated | archived")
    is_pinned: Optional[bool] = None
    is_canonical: Optional[bool] = None


class DocumentPatchResponse(BaseModel):
    updated: bool
    document_id: str
    status: Optional[str] = None
    is_pinned: Optional[bool] = None
    is_canonical: Optional[bool] = None


class BulkDeleteRequest(BaseModel):
    document_ids: List[str] = Field(
        min_length=1,
        description="List of document IDs to delete (with cascading chunk cleanup).",
    )


class BulkDeleteResponse(BaseModel):
    deleted: int = Field(description="Number of documents successfully deleted.")
    not_found: List[str] = Field(
        default_factory=list,
        description="Document IDs that were not found.",
    )


class DocumentFlagRequest(BaseModel):
    reason: str = Field(description="Why this document needs revision.")
    corrected_text: Optional[str] = Field(
        default=None,
        description="Inline corrected content. Alternative to file re-upload via /revise.",
    )


class DocumentFlagResponse(BaseModel):
    document_id: str
    status: str = Field(description="Document status after flagging (e.g. 'flagged').")
    revision_triggered: bool = Field(
        description="True if corrected content was provided and revision started.",
    )
    new_document_id: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class DocumentRevisionResponse(BaseModel):
    old_document_id: str
    new_document_id: str
    chunks_upserted: int
    old_document_deleted: bool
    summaries_regenerated: List[str] = Field(
        default_factory=list,
        description="Granularity levels regenerated, e.g. ['document', 'tenant'].",
    )
    warnings: List[str] = Field(default_factory=list)

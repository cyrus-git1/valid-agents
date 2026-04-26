"""Core ingest models shared across routers and services."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IngestEntity(BaseModel):
    """A named entity submitted alongside an ingest request."""

    name: str = Field(..., description="Entity name (e.g., 'Acme Corp', 'John Smith')")
    type: str = Field(..., description="Entity type (e.g., 'organization', 'person', 'product', 'topic')")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties (role, url, etc.)")


class IngestInput(BaseModel):
    """Parameters for the ingest pipeline (service layer)."""

    tenant_id: UUID
    client_id: UUID
    study_id: Optional[UUID] = Field(
        default=None,
        description=(
            "Optional study scope. When provided, the document is tagged "
            "with this study_id in its metadata so insight tools can scope "
            "analysis to a single research study within the tenant/client."
        ),
    )
    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None
    web_url: Optional[str] = None
    serialized_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    serialized_source_type: Optional[str] = None
    serialized_source_uri: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entities: List[IngestEntity] = Field(default_factory=list)
    extract_entities: bool = Field(default=True, description="Run LLM-based NER on chunks to auto-extract entities")
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    prune_after_ingest: bool = False
    skip_context_generation: bool = Field(
        default=False,
        description="Skip auto context summary; used by batch ingest to generate once at end",
    )

    model_config = {"arbitrary_types_allowed": True}


class IngestOutput(BaseModel):
    """Result returned by the ingest pipeline (service layer)."""

    document_id: UUID
    source_type: str
    source_uri: str
    chunks_upserted: int
    chunk_ids: List[UUID] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    entities_linked: int = 0
    prune_result: Optional[Dict[str, Any]] = None

    # Internal: chunk text data for entity linking (excluded from API responses)
    _chunks_data: List[Dict[str, Any]] = []

    model_config = {"arbitrary_types_allowed": True}

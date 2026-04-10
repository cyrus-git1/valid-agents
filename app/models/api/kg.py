"""Pydantic models for the /kg router — entity and relationship batch upsert."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.base import TenantScoped


# -- Entity upsert --


class EntityUpsertItem(BaseModel):
    """A single entity to upsert into the KG."""
    name: str = Field(..., description="Entity name (e.g., 'Acme Corp')")
    type: str = Field(..., description="Semantic entity type (e.g., 'organization', 'person', 'product')")
    description: Optional[str] = Field(default=None, description="Optional longer description for richer embedding")
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: List[UUID] = Field(default_factory=list, description="Chunk IDs this entity was extracted from")


class EntityBatchRequest(TenantScoped):
    """Batch upsert entities into the KG. Entities are embedded automatically."""
    entities: List[EntityUpsertItem] = Field(..., min_length=1, max_length=500)


class EntityUpsertResult(BaseModel):
    """Per-item result from an entity upsert."""
    name: str
    type: str
    node_id: Optional[str] = None
    node_key: str
    status: str  # "ok" | "failed"
    error: Optional[str] = None


class EntityBatchResponse(BaseModel):
    """Response from POST /kg/entities."""
    total: int
    succeeded: int
    failed: int
    results: List[EntityUpsertResult]


# -- Relationship upsert --


class RelationshipUpsertItem(BaseModel):
    """A single relationship (edge) between two entities."""
    source_entity_name: str
    source_entity_type: str
    target_entity_name: str
    target_entity_type: str
    rel_type: str = Field(..., description="Relationship label (e.g., 'works_at', 'manufactures')")
    weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    evidence_chunk_ids: List[UUID] = Field(default_factory=list)


class RelationshipBatchRequest(TenantScoped):
    """Batch upsert relationships. Entities must exist first (call /kg/entities first)."""
    relationships: List[RelationshipUpsertItem] = Field(..., min_length=1, max_length=500)


class RelationshipUpsertResult(BaseModel):
    """Per-item result from a relationship upsert."""
    source_entity_name: str
    target_entity_name: str
    rel_type: str
    edge_id: Optional[str] = None
    status: str  # "ok" | "failed"
    error: Optional[str] = None


class RelationshipBatchResponse(BaseModel):
    """Response from POST /kg/relationships."""
    total: int
    succeeded: int
    failed: int
    results: List[RelationshipUpsertResult]

"""Pydantic models for the /enrich router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.base import StatusResponse, TenantScopedRequest


class EnrichmentGap(BaseModel):
    """A knowledge gap identified in the KG."""
    topic: str = Field(description="What is missing or thin")
    reason: str = Field(description="Why this is a gap")
    priority: str = Field(description="high, medium, or low")
    search_queries: List[str] = Field(default_factory=list, description="Queries to fill this gap")


class EnrichmentSource(BaseModel):
    """A web source selected to fill a knowledge gap."""
    url: str
    title: str
    relevance_reason: str = Field(description="Why this source was selected")
    gap_topic: str = Field(description="Which gap this source addresses")
    job_id: Optional[str] = Field(default=None, description="Ingest job ID for tracking")


class EnrichmentRunRequest(TenantScopedRequest):
    """Request to analyze KG gaps and enrich with web content."""
    request: Optional[str] = Field(
        default=None,
        description=(
            "Optional focus for enrichment "
            "(e.g., 'Find competitor pricing data'). "
            "If omitted, performs broad gap analysis."
        ),
    )
    max_sources: int = Field(default=5, ge=1, le=20, description="Max URLs to ingest")
    top_k: int = Field(default=15, ge=1, le=50, description="KG nodes to sample for gap analysis")


class EnrichmentRunResponse(StatusResponse):
    """Response containing identified gaps and ingestion jobs."""
    gaps: List[EnrichmentGap] = Field(default_factory=list)
    sources: List[EnrichmentSource] = Field(default_factory=list)
    job_ids: List[str] = Field(default_factory=list, description="Ingest job IDs to poll via /ingest/status/{id}")
    context_sampled: int = Field(default=0, description="Number of KG excerpts analyzed")

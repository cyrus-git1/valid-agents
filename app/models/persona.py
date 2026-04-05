"""Pydantic models for the /persona router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.base import StatusResponse, TenantScopedRequest


class PersonaDemographics(BaseModel):
    """Demographic attributes of a persona."""
    age_range: Optional[str] = None
    income_level: Optional[str] = None
    location: Optional[str] = None
    occupation: Optional[str] = None
    education: Optional[str] = None


class PersonaItem(BaseModel):
    """A single audience persona extracted from KG context."""
    name: str = Field(description="Concise archetype label")
    description: str = Field(description="2-3 sentence description")
    demographics: PersonaDemographics = Field(default_factory=PersonaDemographics)
    motivations: List[str] = Field(default_factory=list, description="Goals, needs, desires")
    pain_points: List[str] = Field(default_factory=list, description="Frustrations, challenges")
    behaviors: List[str] = Field(default_factory=list, description="Decision patterns, media habits")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How much KG evidence supports this persona (0.0-1.0)",
    )


class PersonaFindRequest(TenantScopedRequest):
    """Request to discover audience personas from KG context."""
    request: Optional[str] = Field(
        default=None,
        description=(
            "Optional prompt to focus persona discovery "
            "(e.g., 'Find personas interested in sustainability'). "
            "If omitted, performs broad audience discovery."
        ),
    )
    max_personas: int = Field(default=5, ge=1, le=10, description="Maximum personas to return")
    top_k: int = Field(default=15, ge=1, le=50, description="KG nodes to retrieve for context")
    hop_limit: int = Field(default=1, ge=0, le=2, description="Graph expansion depth")


class PersonaFindResponse(StatusResponse):
    """Response containing discovered audience personas."""
    personas: List[PersonaItem] = Field(default_factory=list)
    context_used: int = Field(default=0, description="Number of KG excerpts used")

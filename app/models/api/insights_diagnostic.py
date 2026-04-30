"""Pydantic models for the multi-agent insights diagnostic endpoint."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosticRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    survey_ids: Optional[List[UUID]] = None
    study_id: Optional[UUID] = None
    client_profile: Optional[Dict[str, Any]] = None
    focus_query: Optional[str] = Field(
        default=None,
        description="Optional focus to scope the diagnostic (e.g., 'pricing concerns Q3').",
    )
    parallel: bool = True
    critic_enabled: bool = True


class DiagnosticPipelineSpecialistSummary(BaseModel):
    name: str
    status: str
    harness_score: Optional[float] = None
    tool_calls: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    skip_reason: Optional[str] = None


class DiagnosticPipelineSummary(BaseModel):
    plan: Dict[str, Any]
    specialists: List[DiagnosticPipelineSpecialistSummary]
    critic: Optional[Dict[str, Any]] = None
    revision_round_run: bool = False
    data_inventory: Dict[str, Any] = Field(default_factory=dict)


class DiagnosticResponse(BaseModel):
    tenant_id: str
    client_id: str
    survey_ids: List[str] = Field(default_factory=list)
    study_id: Optional[str] = None
    generated_at: str
    from_cache: bool = False

    # v2 report (flattened — same keys as the legacy /insights output)
    executive_summary: str = ""
    current_state_assessment: Dict[str, Any] = Field(default_factory=dict)
    quantitative_findings: List[Dict[str, Any]] = Field(default_factory=list)
    qualitative_findings: List[Dict[str, Any]] = Field(default_factory=list)
    competitive_landscape: Dict[str, Any] = Field(default_factory=dict)
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    objections_and_blockers: List[Dict[str, Any]] = Field(default_factory=list)
    contradictions_and_blind_spots: List[Dict[str, Any]] = Field(default_factory=list)
    hypotheses_to_test: List[Dict[str, Any]] = Field(default_factory=list)
    external_context_via_enrichment: Dict[str, Any] = Field(default_factory=dict)
    key_findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations_future_steps: List[Dict[str, Any]] = Field(default_factory=list)
    data_sources_used: Dict[str, Any] = Field(default_factory=dict)
    personas_referenced: List[str] = Field(default_factory=list)
    data_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    enrichment_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    meta_insights: Dict[str, Any] = Field(default_factory=dict)

    pipeline: DiagnosticPipelineSummary
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "complete"
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        extra = "allow"

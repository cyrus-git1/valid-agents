"""Pydantic models for `/transcripts/individual` and `/transcripts/aggregate`."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


VALID_ANALYSES = {
    "discriminate",
    "sentiment",
    "themes",
    "summary",
    "insights",
    "quotes",
}


# ── Single-session: /transcripts/individual ──────────────────────────────


class OrchestrateRequest(BaseModel):
    vtt_content: str = Field(min_length=1)
    tenant_id: UUID
    client_id: UUID
    survey_id: UUID
    session_id: UUID
    study_id: Optional[UUID] = None
    respondent_id: Optional[UUID] = None
    interviewer_id: Optional[UUID] = None
    recorded_at: Optional[datetime] = None
    language: str = "en"
    analyses: Optional[List[str]] = Field(
        default=None,
        description=(
            "Subset of: discriminate, sentiment, themes, summary, insights, quotes. "
            "Omit to run all."
        ),
    )
    focus: Optional[str] = None
    summary_type: str = Field(
        default="general",
        description="Summary flavour: general | meeting | interview",
    )

    @field_validator("vtt_content")
    @classmethod
    def must_start_with_webvtt(cls, v: str) -> str:
        if not v.strip().startswith("WEBVTT"):
            raise ValueError("vtt_content must start with 'WEBVTT'")
        return v

    @field_validator("analyses")
    @classmethod
    def filter_known_analyses(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        bad = [a for a in v if a not in VALID_ANALYSES]
        if bad:
            raise ValueError(f"Unknown analyses: {bad}. Allowed: {sorted(VALID_ANALYSES)}")
        return v


class OrchestrateResponse(BaseModel):
    tenant_id: str
    client_id: str
    survey_id: str
    session_id: str
    vtt_hash: str
    from_cache: bool
    language: str
    duration_seconds: Optional[float] = None
    discriminate: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    themes: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    quotes: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, str]] = Field(default_factory=list)
    status: str
    elapsed_ms: float


# ── Cross-session: /transcripts/aggregate ────────────────────────────────


class AggregateRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    survey_id: UUID
    session_ids: Optional[List[UUID]] = Field(
        default=None,
        description=(
            "Subset of session_ids to analyse. Omit to analyse ALL sessions "
            "tagged with the given survey_id."
        ),
    )
    compare_to_session_ids: Optional[List[UUID]] = Field(
        default=None,
        description=(
            "If set, runs synthesis twice (primary group vs comparison group) "
            "and includes a 'comparison' block in the response."
        ),
    )
    analyses: Optional[List[str]] = Field(
        default=None,
        description="Per-session analyses scope. Same allowed values as /individual.",
    )
    focus: Optional[str] = None
    summary_type: str = "interview"

    @field_validator("analyses")
    @classmethod
    def filter_known_analyses(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        bad = [a for a in v if a not in VALID_ANALYSES]
        if bad:
            raise ValueError(f"Unknown analyses: {bad}. Allowed: {sorted(VALID_ANALYSES)}")
        return v


class AggregateResponse(BaseModel):
    tenant_id: str
    client_id: str
    survey_id: str
    session_ids: List[str]
    per_session_results: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    comparison: Optional[Dict[str, Any]] = None
    coverage: Dict[str, Any]
    errors: List[Dict[str, str]] = Field(default_factory=list)
    status: str
    elapsed_ms: float

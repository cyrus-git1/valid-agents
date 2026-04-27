"""Pydantic models for `/clusters/analyze`."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ClustersAnalyzeRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    survey_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Filter to sessions whose metadata.survey_id is in this list. Omit to include all.",
    )
    study_id: Optional[UUID] = Field(
        default=None,
        description="Filter to sessions whose metadata.study_id matches.",
    )
    k: Optional[int] = Field(
        default=None,
        ge=2, le=50,
        description="If provided, KMeans with this many clusters. Otherwise HDBSCAN.",
    )
    min_cluster_size: int = Field(
        default=3, ge=2,
        description="HDBSCAN min_cluster_size (ignored when k is provided).",
    )
    min_word_count: int = Field(
        default=30, ge=0,
        description="Skip sessions whose concatenated text has fewer than this many words.",
    )
    include_text_embedding: bool = Field(
        default=True,
        description="If true, include OpenAI text-embedding-3-small features in the cluster vector. Set false for cheaper, deterministic-only clustering.",
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus hint passed into the LLM cluster labeller (e.g., 'pricing concerns').",
    )
    produce_labels: bool = Field(
        default=True,
        description="If true, runs ONE LLM call per cluster to generate human-readable labels.",
    )

    @field_validator("k")
    @classmethod
    def _validate_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 2:
            raise ValueError("k must be >= 2")
        return v


class DefiningTag(BaseModel):
    field: str
    value: str
    in_cluster_count: int
    in_cluster_rate: float
    baseline_count: int
    baseline_rate: float
    p_value: float
    rate_ratio: float


class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    description: str
    size: int
    defining_tags: List[DefiningTag] = Field(default_factory=list)
    top_terms: List[str] = Field(default_factory=list)
    mean_vader: float
    dominant_traits: List[str] = Field(default_factory=list)
    sample_session_ids: List[str] = Field(default_factory=list)


class FeatureSummary(BaseModel):
    numeric_dimensions: int
    embedding_dimensions: int
    tags_normalised: int
    tags_via_llm_fallback: int


class ClustersAnalyzeResponse(BaseModel):
    tenant_id: str
    client_id: str
    session_count: int
    noise_count: int
    n_clusters: int
    algorithm: str
    clusters: List[ClusterSummary] = Field(default_factory=list)
    noise_session_ids: List[str] = Field(default_factory=list)
    feature_summary: FeatureSummary
    errors: List[Dict[str, str]] = Field(default_factory=list)
    status: str
    elapsed_ms: float

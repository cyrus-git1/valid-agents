"""LangGraph workflow state definitions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document


class RouterState(TypedDict, total=False):
    input: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    intent: str
    intent_confidence: float
    classification_attempt: int
    output: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


class RAGState(TypedDict, total=False):
    question: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    documents: List[Document]
    context: str
    answer: str
    confidence: float
    top_similarity: float
    attempt: int
    model: str


class ContextBuildState(TypedDict, total=False):
    tenant_id: str
    client_id: str
    docs: List[str]
    weblinks: List[str]
    transcripts: List[str]
    client_profile: Dict[str, Any]
    ingest_results: List[Dict[str, Any]]
    documents: List[Document]
    status: str
    error: Optional[str]
    warnings: List[str]


class EnrichmentState(TypedDict, total=False):
    request: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    # Step outputs
    kg_context: str
    context_summary: str
    profile_section: str
    gaps: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    ranked_sources: List[Dict[str, Any]]
    ingested_sources: List[Dict[str, Any]]
    job_ids: List[str]
    # Tracking
    context_sampled: int
    max_sources: int
    status: str
    error: Optional[str]


class SurveyState(TypedDict, total=False):
    request: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    question_types: List[str]
    title: str
    description: str
    documents: List[Document]
    context: str
    tenant_profile: str
    context_analysis: str
    profile_section: str
    prior_questions: str
    title_description_section: str
    raw_output: str
    survey: str
    generated_title: str
    generated_description: str
    context_used: int
    confidence: float
    attempt: int
    error: Optional[str]
    status: str

"""Pydantic models for the /ingest router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from app.models.base import TenantScoped


# -- Entity model --


class IngestEntity(BaseModel):
    """A named entity submitted alongside an ingest request."""
    name: str = Field(..., description="Entity name (e.g., 'Acme Corp', 'John Smith')")
    type: str = Field(..., description="Entity type (e.g., 'organization', 'person', 'product', 'topic')")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties (role, url, etc.)")


# -- Service-layer DTOs --


class IngestInput(BaseModel):
    """Parameters for the ingest pipeline (service layer)."""
    tenant_id: UUID
    client_id: UUID

    # File ingest -- provide both
    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None

    # Web ingest -- provide this
    web_url: Optional[str] = None

    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Entities to link to chunks
    entities: List[IngestEntity] = Field(default_factory=list)

    # NER extraction — run LLM entity extraction on chunks
    extract_entities: bool = Field(default=True, description="Run LLM-based NER on chunks to auto-extract entities")

    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    prune_after_ingest: bool = False
    skip_context_generation: bool = Field(default=False, description="Skip auto context summary — used by batch ingest to generate once at end")

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


# -- Router response models --


class IngestFileResponse(BaseModel):
    job_id: str
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    warnings: List[str] = []
    prune_result: Optional[Dict[str, Any]] = None


class IngestWebRequest(TenantScoped):
    url: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entities: List[IngestEntity] = Field(default_factory=list)
    extract_entities: bool = Field(default=True, description="Run LLM-based NER on chunks")
    prune_after_ingest: bool = False


class IngestWebResponse(BaseModel):
    job_id: str
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    warnings: List[str] = []
    prune_result: Optional[Dict[str, Any]] = None


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str        # "complete" | "running" | "failed"
    detail: Optional[str] = None


# -- Batch ingest models --

class BatchWebItem(BaseModel):
    url: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchWebRequest(TenantScoped):
    items: List[BatchWebItem] = Field(..., min_length=1, max_length=50)
    prune_after_ingest: bool = False


class BatchItemStatus(BaseModel):
    index: int
    source: str                # file name or URL
    status: str                # "running" | "complete" | "failed"
    document_id: Optional[str] = None
    chunks_upserted: int = 0
    warnings: List[str] = []
    detail: Optional[str] = None


class BatchIngestResponse(BaseModel):
    batch_id: str
    total: int
    status: str                # "running" | "complete" | "partial_failure"
    items: List[BatchItemStatus] = []


class BatchIngestStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    status: str                # "running" | "complete" | "partial_failure"
    items: List[BatchItemStatus] = []


# -- Survey results ingest --


class SurveyResponseItem(BaseModel):
    """A single question + response pair."""
    question_id: Optional[str] = None
    question_type: str = Field(description="multiple_choice, rating, nps, yes_no, etc.")
    question_label: str = Field(description="The question text")
    response: Any = Field(description="The response value (string, number, list, etc.)")
    options: Optional[List[str]] = Field(default=None, description="Options for choice questions")


class IngestSurveyResultsRequest(TenantScoped):
    """Ingest completed survey responses as KB chunks."""
    survey_id: Optional[str] = Field(default=None, description="ID of the survey these responses belong to")
    survey_title: Optional[str] = Field(default=None, description="Title of the survey")
    responses: List[SurveyResponseItem] = Field(..., min_length=1, description="List of question+response pairs")
    respondent_id: Optional[str] = Field(default=None, description="Anonymous respondent identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchSurveyResultsRequest(TenantScoped):
    """Batch ingest multiple survey response sets."""
    survey_id: Optional[str] = None
    survey_title: Optional[str] = None
    items: List[IngestSurveyResultsRequest] = Field(..., min_length=1, max_length=200)


class IngestSurveyResultsResponse(BaseModel):
    job_id: str
    chunks_upserted: int = 0
    status: str = "pending"


# -- Transcript ingest --


class IngestTranscriptRequest(TenantScoped):
    """Ingest a transcript as KB chunks."""
    title: Optional[str] = Field(default=None, description="Title or label for the transcript")
    content: str = Field(..., min_length=10, description="Raw transcript text")
    source: Optional[str] = Field(default=None, description="Where it came from (e.g., 'Zoom', 'Gong', 'manual')")
    speaker_labels: bool = Field(default=False, description="Whether content includes speaker labels")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchTranscriptRequest(TenantScoped):
    """Batch ingest multiple transcripts."""
    items: List[IngestTranscriptRequest] = Field(..., min_length=1, max_length=50)


class IngestTranscriptResponse(BaseModel):
    job_id: str
    chunks_upserted: int = 0
    status: str = "pending"

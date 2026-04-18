"""HTTP request/response models and examples for stateless ingest routes."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.base import TenantScoped
from app.models.ingest import IngestEntity


SURVEY_RESULTS_JSON_EXAMPLE = json.dumps(
    {
        "q1_multiple_choice": "Option A",
        "q2_checkbox": ["Option 1", "Option 2", "Option 3"],
        "q3_short_text": "Brief user input",
        "q4_long_text": "A longer paragraph of feedback from the user...",
        "q5_rating": 4,
        "q6_yes_no": "yes",
        "q7_nps": 8,
        "q8_ranking": ["Item C", "Item A", "Item B"],
        "q9_card_sort": {
            "item1": "cat_A",
            "item2": "cat_A",
            "item3": "cat_B",
        },
        "q10_interface_test": {
            "https://example.com": True,
            "https://test.com": False,
        },
        "q11_tree_testing": ["root", "branch1", "leaf"],
        "q12_matrix": {
            "row1": "col_A",
            "row2": "col_B",
            "row3": "col_C",
        },
        "q13_sus": {
            "sus_0": "4",
            "sus_1": "2",
            "sus_2": "5",
            "sus_3": "1",
            "sus_4": "4",
            "sus_5": "3",
            "sus_6": "5",
            "sus_7": "2",
            "sus_8": "4",
            "sus_9": "1",
        },
        "_heatmap": {
            "q10_interface_test": {
                "sessions": {
                    "session_1": "heatmap-session-uuid-here",
                },
                "page_url": "https://example.com",
            }
        },
    },
    indent=2,
)


class IngestFileResponse(BaseModel):
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    entities_linked: int = 0
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
    document_id: str
    source_type: str
    source_uri: str
    chunks_upserted: int
    entities_linked: int = 0
    warnings: List[str] = []
    prune_result: Optional[Dict[str, Any]] = None


class BatchWebItem(BaseModel):
    url: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchWebRequest(TenantScoped):
    items: List[BatchWebItem] = Field(..., min_length=1, max_length=50)
    prune_after_ingest: bool = False


class BatchItemStatus(BaseModel):
    index: int
    source: str
    status: str
    document_id: Optional[str] = None
    chunks_upserted: int = 0
    warnings: List[str] = []
    detail: Optional[str] = None


class BatchIngestResponse(BaseModel):
    total: int
    completed: int
    failed: int
    status: str
    items: List[BatchItemStatus] = []

"""
/search router (agent service)
-------------------------------
POST /search/ask — Full RAG: graph retrieval + LLM answer generation
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.models.api.search import (
    AskRequest,
    AskResponse,
    SearchResultItem,
)
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


def _docs_to_result_items(docs) -> List[SearchResultItem]:
    items = []
    for doc in docs:
        m = doc.metadata
        items.append(SearchResultItem(
            node_id=m.get("node_id", ""),
            node_key=m.get("node_key", ""),
            node_type=m.get("node_type", ""),
            content=doc.page_content,
            similarity_score=m.get("similarity_score"),
            document_id=m.get("document_id"),
            chunk_index=m.get("chunk_index"),
            source=m.get("source", "vector"),
            retrieval_reason=m.get("retrieval_reason"),
            evidence_quote=m.get("evidence_quote"),
            evidence_score=m.get("evidence_score"),
            evidence_count=m.get("evidence_count", 0),
        ))
    return items


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Full RAG pipeline: graph retrieval + LLM answer generation."""
    svc = SearchService(
        tenant_id=req.tenant_id,
        client_id=req.client_id,
        llm_model=req.model,
    )

    try:
        answer, docs = svc.ask(
            req.question,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
        )
    except Exception as e:
        logger.exception("RAG pipeline failed in /ask")
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")

    return AskResponse(
        question=req.question,
        answer=answer,
        sources=_docs_to_result_items(docs),
    )

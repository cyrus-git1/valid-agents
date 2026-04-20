"""
/insights router — evidence-backed Q&A over KB.

POST /insights/analyze — Question answering with citations and contradiction checking.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from uuid import UUID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


class InsightAnalyzeRequest(BaseModel):
    tenant_id: UUID
    client_id: Optional[UUID] = None
    question: str = Field(min_length=1)
    contradiction_check: bool = False


@router.post("/analyze")
def analyze_insight(req: InsightAnalyzeRequest):
    """Evidence-backed synthesis over summary + source chunks.

    Retrieves via hop-1 from summary chunks (which mention-edge back to their
    source evidence), then synthesizes an answer whose every factual claim
    cites a SOURCE chunk.
    """
    from app.workflows.insight_workflow import run_insight_analysis

    try:
        result = run_insight_analysis(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id) if req.client_id else None,
            question=req.question,
            contradiction_check=req.contradiction_check,
        )
    except Exception as e:
        logger.exception("Insight analysis failed")
        raise HTTPException(status_code=500, detail=f"Insight analysis failed: {e}")

    return result

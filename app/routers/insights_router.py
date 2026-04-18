"""
/insights router — business insights report generation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from uuid import UUID

from app.models.base import TenantScopedRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


class InsightsGenerateRequest(TenantScopedRequest):
    focus_query: Optional[str] = Field(
        default=None,
        description="Optional focus for the analysis (e.g., 'pricing concerns')",
    )


@router.post("/generate")
def generate_insights(req: InsightsGenerateRequest):
    """Generate a full business insights report.

    The agent checks what data exists (transcripts, surveys, documents, personas),
    runs relevant analyses, and produces a unified report with evidence attribution.
    Never blocks on missing data — reports what's missing and continues.
    """
    from app.agents.insights_agent import run_insights_agent

    try:
        result = run_insights_agent(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            client_profile=req.client_profile,
            focus_query=req.focus_query,
        )
    except Exception as e:
        logger.exception("Insights generation failed")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {e}")

    return result


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
    cites a SOURCE chunk. Summaries are navigation only — they are never used
    as the citation target.
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

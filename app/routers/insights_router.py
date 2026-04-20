"""
/insights router — deep analysis endpoints.

POST /insights/analyze   — Evidence-backed Q&A over KB
POST /insights/generate  — Deep analysis (transcript, competitive, cross-doc synthesis)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from uuid import UUID

from app.models.base import TenantScopedRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


# ── Evidence-backed Q&A ───────────────────────────────────────────────────


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


# ── Deep analysis ─────────────────────────────────────────────────────────


class InsightsGenerateRequest(TenantScopedRequest):
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus for the analysis (e.g., 'pricing concerns', 'competitor X')",
    )
    analyses: List[str] = Field(
        default=["transcript", "competitive", "synthesis"],
        description="Which analyses to run: 'transcript', 'competitive', 'synthesis'. Defaults to all.",
    )


@router.post("/generate")
def generate_insights(req: InsightsGenerateRequest):
    """Run deep analysis across the knowledge base.

    Combines up to three analysis types:
    - transcript: summary + sentiment + insights from transcript/interview data
    - competitive: competitor profiles, positioning gaps, win/loss signals
    - synthesis: cross-document patterns, contradictions, blind spots

    Each analysis searches the KB independently and returns structured results.
    """
    from app.tools.analysis_tools import create_analysis_tools

    tools = create_analysis_tools(str(req.tenant_id), str(req.client_id))
    tool_map = {t.name: t for t in tools}

    valid_analyses = {"transcript", "competitive", "synthesis"}
    requested = [a for a in req.analyses if a in valid_analyses]
    if not requested:
        requested = list(valid_analyses)

    tool_name_map = {
        "transcript": "analyze_transcript",
        "competitive": "competitive_intelligence",
        "synthesis": "cross_document_synthesis",
    }

    results: Dict[str, Any] = {}
    errors: List[str] = []

    for analysis in requested:
        tool_name = tool_name_map[analysis]
        tool_fn = tool_map.get(tool_name)
        if not tool_fn:
            errors.append(f"Tool {tool_name} not found")
            continue

        try:
            result = tool_fn.invoke({"focus": req.focus})
            results[analysis] = result
        except Exception as e:
            logger.warning("Analysis %s failed: %s", analysis, e)
            results[analysis] = {"status": "failed", "error": str(e)}
            errors.append(f"{analysis}: {e}")

    return {
        "tenant_id": str(req.tenant_id),
        "client_id": str(req.client_id),
        "focus": req.focus,
        "analyses_requested": requested,
        "results": results,
        "errors": errors,
        "status": "complete" if not errors else "partial",
    }

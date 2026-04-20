"""
/insights router — insight generation endpoints.

POST /insights/analyze   — Evidence-backed Q&A with citations
POST /insights/generate  — Actionable insights, strengths, advantages (deep analysis)
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


# ── Deep analysis: actionable insights, strengths, advantages ─────────────


class InsightsGenerateRequest(TenantScopedRequest):
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus for the analysis (e.g., 'pricing strategy', 'customer onboarding')",
    )
    analyses: List[str] = Field(
        default=["transcript", "competitive", "synthesis"],
        description="Which analyses to run: 'transcript', 'competitive', 'synthesis'. Defaults to all.",
    )


@router.post("/generate")
def generate_insights(req: InsightsGenerateRequest):
    """Generate actionable insights, strengths, and competitive advantages.

    Runs up to three deep analyses:
    - transcript: executive summary, sentiment, prioritized insights, action items,
      key moments, recommendations
    - competitive: competitor profiles, our positioning (strengths + gaps),
      win/loss signals, opportunities, threats
    - synthesis: recurring themes, contradictions between sources, convergent
      evidence, blind spots, executive summary, cross-document recommendations

    Each analysis returns structured data with evidence quotes and priorities.
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

    # Build a unified view of actionable items across all analyses
    actionable_insights: List[Dict[str, Any]] = []
    strengths: List[Dict[str, Any]] = []
    advantages: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    # From transcript analysis
    t = results.get("transcript", {})
    if isinstance(t, dict):
        for insight in t.get("key_insights", []) or []:
            if isinstance(insight, dict):
                actionable_insights.append({
                    "source": "transcript",
                    **insight,
                })
        for action in t.get("decisions_and_action_items", []) or []:
            if isinstance(action, dict):
                actionable_insights.append({
                    "source": "transcript",
                    "type": "action_item",
                    **action,
                })
        for rec in t.get("recommendations", []) or []:
            recommendations.append(rec if isinstance(rec, str) else str(rec))

    # From competitive intelligence
    c = results.get("competitive", {})
    if isinstance(c, dict):
        pos = c.get("our_positioning", {}) or {}
        for s in pos.get("strengths", []) or []:
            if isinstance(s, dict):
                strengths.append({"source": "competitive", **s})
        for opp in c.get("opportunities", []) or []:
            if isinstance(opp, dict):
                advantages.append({"source": "competitive", "type": "opportunity", **opp})
        for win in c.get("win_signals", []) or []:
            if isinstance(win, dict):
                advantages.append({"source": "competitive", "type": "win_signal", **win})

    # From cross-document synthesis
    s = results.get("synthesis", {})
    if isinstance(s, dict):
        for rec in s.get("recommendations", []) or []:
            recommendations.append(rec if isinstance(rec, str) else str(rec))
        for ev in s.get("convergent_evidence", []) or []:
            if isinstance(ev, dict):
                actionable_insights.append({
                    "source": "synthesis",
                    "type": "convergent_finding",
                    "insight": ev.get("conclusion", ""),
                    "confidence": ev.get("confidence", "medium"),
                    "evidence": ev.get("evidence", []),
                })

    return {
        "tenant_id": str(req.tenant_id),
        "client_id": str(req.client_id),
        "focus": req.focus,
        "analyses_requested": requested,
        "actionable_insights": actionable_insights,
        "strengths": strengths,
        "advantages": advantages,
        "recommendations": recommendations,
        "results": results,
        "errors": errors,
        "status": "complete" if not errors else "partial",
    }

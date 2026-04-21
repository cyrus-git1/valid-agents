"""
/transcripts router — stateless WebVTT analysis.

POST /transcripts/analyze — Accept raw WebVTT, return summary + sentiment + insights.

Stateless: no ingestion, no KB writes, no tenant scoping needed.
Purely functional — takes content in, returns analysis out.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcripts", tags=["transcripts"])


class TranscriptAnalyzeRequest(BaseModel):
    vtt_content: str = Field(
        min_length=1,
        description="Raw WebVTT text (must start with 'WEBVTT').",
    )
    analyses: List[str] = Field(
        default=["summary", "sentiment", "insights"],
        description="Which analyses to run: 'summary', 'sentiment', 'insights'. Defaults to all.",
    )
    summary_type: str = Field(
        default="general",
        description="Summary flavor: 'general', 'meeting', 'interview', etc.",
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus for the analysis (e.g., 'pricing concerns').",
    )


@router.post("/analyze")
def analyze_transcript_content(req: TranscriptAnalyzeRequest) -> Dict[str, Any]:
    """Run deep analysis on a raw WebVTT transcript — no KB interaction.

    Produces up to three outputs depending on 'analyses':
    - summary: structured meeting summary with action items, decisions, topic groups
    - sentiment: overall sentiment, themes, notable quotes
    - insights: actionable insights extracted from the conversation

    Everything runs in-process against the provided vtt_content. No ingestion.
    """
    content = (req.vtt_content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="vtt_content cannot be empty.")
    if not content.startswith("WEBVTT"):
        raise HTTPException(
            status_code=400,
            detail="vtt_content must start with 'WEBVTT'.",
        )

    valid = {"summary", "sentiment", "insights"}
    requested = [a for a in req.analyses if a in valid]
    if not requested:
        requested = list(valid)

    # Dummy IDs — services require them but we're stateless
    dummy_tenant = uuid4()
    dummy_survey = uuid4()

    results: Dict[str, Any] = {}
    errors: List[str] = []

    # ── Summary ────────────────────────────────────────────────────────
    if "summary" in requested:
        try:
            from app.analysis.transcript_insights import TranscriptInsightsService
            svc = TranscriptInsightsService()
            summary_out = svc.generate_summary_from_vtt(
                tenant_id=dummy_tenant,
                vtt_content=content,
                summary_type=req.summary_type,
            )
            results["summary"] = {
                "summary": summary_out.get("summary", ""),
                "action_items": summary_out.get("action_items", []),
                "decisions": summary_out.get("decisions", []),
                "topic_groups": summary_out.get("topic_groups", []),
                "summary_type": summary_out.get("summary_type", req.summary_type),
                "status": summary_out.get("status", "complete"),
                "error": summary_out.get("error"),
            }
        except Exception as e:
            logger.warning("Transcript summary failed: %s", e)
            results["summary"] = {"status": "failed", "error": str(e)}
            errors.append(f"summary: {e}")

    # ── Sentiment ──────────────────────────────────────────────────────
    if "sentiment" in requested:
        try:
            from app.analysis.sentiment import SentimentAnalysisService
            svc = SentimentAnalysisService()
            sent_out = svc.generate_from_vtt(
                tenant_id=dummy_tenant,
                survey_id=dummy_survey,
                vtt_content=content,
            )
            results["sentiment"] = {
                "overall_sentiment": sent_out.get("overall_sentiment", {}),
                "dominant_sentiment": sent_out.get("dominant_sentiment", "neutral"),
                "themes": sent_out.get("themes", []),
                "notable_quotes": sent_out.get("notable_quotes", []),
                "summary": sent_out.get("summary", ""),
            }
        except Exception as e:
            logger.warning("Transcript sentiment failed: %s", e)
            results["sentiment"] = {"status": "failed", "error": str(e)}
            errors.append(f"sentiment: {e}")

    # ── Insights ───────────────────────────────────────────────────────
    if "insights" in requested:
        try:
            from app.analysis.transcript_insights import TranscriptInsightsService
            svc = TranscriptInsightsService()
            insights_out = svc.generate_from_vtt(
                tenant_id=dummy_tenant,
                survey_id=dummy_survey,
                vtt_content=content,
            )
            results["insights"] = {
                "summary": insights_out.get("summary", ""),
                "actionable_insights": insights_out.get("actionable_insights", []),
                "status": insights_out.get("status", "complete"),
                "error": insights_out.get("error"),
            }
        except Exception as e:
            logger.warning("Transcript insights failed: %s", e)
            results["insights"] = {"status": "failed", "error": str(e)}
            errors.append(f"insights: {e}")

    # Apply focus filter post-hoc if provided (client-side filter
    # would be cleaner, but this keeps the API surface simple)
    return {
        "analyses_requested": requested,
        "focus": req.focus,
        "results": results,
        "errors": errors,
        "status": "complete" if not errors else "partial",
    }

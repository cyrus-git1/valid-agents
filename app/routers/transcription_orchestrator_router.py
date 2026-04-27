"""
/transcripts router endpoints for the new orchestrated pipelines.

POST /transcripts/individual — single-session orchestration with discriminate
                              + 5 LLM agents + cache
POST /transcripts/aggregate  — cross-session synthesis within one survey
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agents.transcription.aggregator import run_cross_session_aggregation
from app.agents.transcription.orchestrator import run_orchestration
from app.models.api.transcription_orchestrator import (
    AggregateRequest,
    AggregateResponse,
    OrchestrateRequest,
    OrchestrateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcripts", tags=["transcripts"])


@router.post("/individual", response_model=OrchestrateResponse)
def transcripts_individual(req: OrchestrateRequest) -> OrchestrateResponse:
    """Single-session transcription orchestration.

    Runs the discriminate (deterministic) agent + 5 parallel LLM agents
    over a raw WebVTT. Cached by content hash + scope (24hr TTL).
    """
    try:
        result = run_orchestration(
            vtt_content=req.vtt_content,
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            survey_id=str(req.survey_id),
            session_id=str(req.session_id),
            analyses=req.analyses,
            focus=req.focus,
            summary_type=req.summary_type,
            language=req.language,
        )
    except ValueError as e:
        # Validation errors (e.g. not WEBVTT) → 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("transcripts_individual failed")
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {e}")
    return result  # response_model coerces dict → OrchestrateResponse


@router.post("/aggregate", response_model=AggregateResponse)
def transcripts_aggregate(req: AggregateRequest) -> AggregateResponse:
    """Cross-session synthesis within one survey.

    Loads each session's already-ingested transcript chunks from the KB
    (filtered by survey_id + session_id metadata), runs each through the
    single-session orchestrator (cache-friendly), then runs a final
    cross-session synthesis pass.

    If `compare_to_session_ids` is provided, a `comparison` block is
    included with deltas between the two groups.
    """
    try:
        result = run_cross_session_aggregation(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            survey_id=str(req.survey_id),
            session_ids=[str(s) for s in (req.session_ids or [])] or None,
            compare_to_session_ids=(
                [str(s) for s in (req.compare_to_session_ids or [])]
                if req.compare_to_session_ids else None
            ),
            analyses=req.analyses,
            focus=req.focus,
            summary_type=req.summary_type,
        )
    except Exception as e:
        logger.exception("transcripts_aggregate failed")
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {e}")
    return result

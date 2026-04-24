"""
/intake router — Vera's 8-stage context intake flow.

POST /intake/stream — SSE-streamed intake conversation
POST /intake/reset  — clear an in-progress intake session
"""
from __future__ import annotations

import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.intake_agent import reset_intake_session, stream_intake_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intake", tags=["intake"])


class IntakeRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    input: str = Field(..., description="The user's next message in the intake conversation.")
    session_id: Optional[str] = Field(
        default="default",
        description="Identifier to track a multi-turn intake session.",
    )


class IntakeResetRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    session_id: Optional[str] = Field(default="default")


@router.post("/stream")
async def intake_stream(req: IntakeRequest):
    """Stream one turn of Vera's context intake conversation via SSE.

    Events:
      status    — progress update (stage N/8)
      partial   — assistant response text
      done      — normal turn complete
      completed — intake finished, profile ingested (includes document_id)
      error     — failure

    The conversation advances through 8 stages. When the user confirms
    the final summary, the profile is ingested as a document and a
    'completed' event fires.
    """
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        async for event in stream_intake_agent(
            request=req.input,
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            session_id=req.session_id or "default",
        ):
            yield {"data": json.dumps(event, default=str)}

    return EventSourceResponse(event_generator())


@router.post("/reset")
def intake_reset(req: IntakeResetRequest):
    """Clear an in-progress intake session so the user can start over."""
    try:
        reset_intake_session(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            session_id=req.session_id or "default",
        )
        return {"status": "reset"}
    except Exception as e:
        logger.exception("Intake reset failed")
        raise HTTPException(status_code=500, detail=str(e))

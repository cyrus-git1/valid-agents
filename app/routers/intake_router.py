"""
/intake router — Vera's 8-stage context intake flow.

POST /intake/stream — SSE-streamed intake conversation (no auth required —
                     anonymous founders pre-signup can use this)
POST /intake/reset  — clear an in-progress intake session
POST /intake/claim  — after the user signs up, attach a completed
                     anonymous intake profile to their new tenant/client
                     so it gets ingested into their KB
GET  /intake/profile/{session_id} — peek at a stored profile (for the
                     frontend to render the structured profile after
                     completion before claiming)
"""
from __future__ import annotations

import json
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.intake_agent import (
    claim_intake_profile,
    get_stored_profile,
    reset_intake_session,
    stream_intake_agent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intake", tags=["intake"])


class IntakeRequest(BaseModel):
    input: str = Field(..., description="The user's next message in the intake conversation.")
    session_id: Optional[str] = Field(
        default=None,
        description="Identifier to track a multi-turn intake session. "
                    "Provide one on the first message and reuse it. "
                    "If omitted, the server generates one and returns it in events.",
    )
    tenant_id: Optional[UUID] = Field(
        default=None,
        description="Optional. If provided AND client_id is provided, "
                    "the completed profile will be ingested into the KB. "
                    "Omit for anonymous pre-signup intakes — call /intake/claim later.",
    )
    client_id: Optional[UUID] = Field(default=None)


class IntakeResetRequest(BaseModel):
    session_id: str
    tenant_id: Optional[UUID] = None
    client_id: Optional[UUID] = None


class IntakeClaimRequest(BaseModel):
    session_id: str = Field(..., description="The session_id from the anonymous intake")
    tenant_id: UUID
    client_id: UUID


@router.post("/stream")
async def intake_stream(req: IntakeRequest):
    """Stream one turn of Vera's context intake conversation via SSE.

    Authentication is OPTIONAL — anonymous users (no tenant/client) can
    run the intake before signing up. The server stores the profile
    temporarily (2hr TTL) keyed by session_id.

    Events:
      session  — session_id assigned on first message (save this)
      status   — progress update (stage N/8)
      partial  — assistant response text
      done     — normal turn complete
      completed — intake finished. If tenant/client provided, profile is
                  ingested. Otherwise it's stored temporarily — frontend
                  must call /intake/claim after signup.
      error    — failure
    """
    from sse_starlette.sse import EventSourceResponse

    session_id = req.session_id or f"anon-{uuid4()}"
    tenant_id = str(req.tenant_id) if req.tenant_id else None
    client_id = str(req.client_id) if req.client_id else None

    async def event_generator():
        # Echo the session_id so the frontend can persist it for follow-ups
        yield {"data": json.dumps({"type": "session", "session_id": session_id})}

        async for event in stream_intake_agent(
            request=req.input,
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
        ):
            yield {"data": json.dumps(event, default=str)}

    return EventSourceResponse(event_generator())


@router.post("/reset")
def intake_reset(req: IntakeResetRequest):
    """Clear an in-progress intake session so the user can start over."""
    try:
        reset_intake_session(
            tenant_id=str(req.tenant_id) if req.tenant_id else None,
            client_id=str(req.client_id) if req.client_id else None,
            session_id=req.session_id,
        )
        return {"status": "reset"}
    except Exception as e:
        logger.exception("Intake reset failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/claim")
def intake_claim(req: IntakeClaimRequest):
    """Attach an anonymous intake profile to a newly-created tenant/client.

    Call this after the user signs up. The previously-completed intake
    profile (stored temporarily under session_id) gets ingested into
    the KB under the new tenant/client.
    """
    result = claim_intake_profile(
        session_id=req.session_id,
        tenant_id=str(req.tenant_id),
        client_id=str(req.client_id),
    )
    if result.get("status") == "not_found":
        raise HTTPException(
            status_code=404,
            detail=(
                "No completed intake profile found for that session_id. "
                "It may have expired (2hr TTL) or never completed."
            ),
        )
    if result.get("status") == "incomplete":
        raise HTTPException(
            status_code=400,
            detail="Intake session exists but hasn't been completed yet.",
        )
    return result


@router.get("/profile/{session_id}")
def intake_profile(session_id: str):
    """Read a stored profile (anonymous or claimed) without ingesting."""
    profile = get_stored_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

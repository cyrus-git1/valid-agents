"""
Intake agent — Vera runs an 8-stage context intake conversation for users
with no existing documentation.

Anonymous-friendly: tenant_id and client_id are OPTIONAL. If omitted, the
profile is held in memory under the session_id and can be claimed later
via /intake/claim once the user signs up.

At the end of the conversation:
  - If tenant/client provided: profile ingested directly as a doc
  - If anonymous: profile stored under session_id, awaiting /intake/claim
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time as _time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.intake_prompts import build_intake_prompt

logger = logging.getLogger(__name__)

# ── Session state ──────────────────────────────────────────────────────────

_SESSION_TTL_S = 3600 * 2  # 2 hours
_intake_sessions: Dict[str, Dict[str, Any]] = {}
_intake_lock = threading.Lock()


def _session_key(
    session_id: str,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> str:
    """Anonymous sessions key on session_id only.
    Authenticated sessions use tenant:client:session for isolation.
    """
    if tenant_id and client_id:
        return f"{tenant_id}:{client_id}:{session_id}"
    return f"anon:{session_id}"


def _get_session(key: str) -> Optional[Dict[str, Any]]:
    with _intake_lock:
        s = _intake_sessions.get(key)
        if not s or _time.monotonic() > s["expires_at"]:
            _intake_sessions.pop(key, None)
            return None
        return s


def _update_session(
    key: str,
    user_msg: str,
    assistant_msg: str,
    profile: Optional[Dict[str, Any]] = None,
    completed: bool = False,
) -> None:
    with _intake_lock:
        if key not in _intake_sessions:
            _intake_sessions[key] = {
                "messages": [],
                "profile": None,
                "completed": False,
                "expires_at": _time.monotonic() + _SESSION_TTL_S,
            }
        s = _intake_sessions[key]
        s["messages"].append({"role": "user", "content": user_msg})
        s["messages"].append({"role": "assistant", "content": assistant_msg})
        if profile is not None:
            s["profile"] = profile
        if completed:
            s["completed"] = True
        s["expires_at"] = _time.monotonic() + _SESSION_TTL_S


def _reset_session_internal(key: str) -> None:
    with _intake_lock:
        _intake_sessions.pop(key, None)


# ── Completion marker parsing ──────────────────────────────────────────────

_COMPLETE_RE = re.compile(
    r"<<INTAKE_COMPLETE>>\s*(\{.*?\})\s*<<END_INTAKE>>",
    re.DOTALL,
)


def _extract_profile(output: str) -> Tuple[Optional[Dict[str, Any]], str]:
    m = _COMPLETE_RE.search(output)
    if not m:
        return None, output
    try:
        profile = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse intake profile JSON: %s", e)
        return None, output
    cleaned = _COMPLETE_RE.sub("", output).strip()
    return profile, cleaned


def _estimate_current_stage(messages: List[Dict[str, str]]) -> int:
    assistant_msgs = sum(1 for m in messages if m["role"] == "assistant")
    return min(8, max(1, assistant_msgs // 2 + 1))


# ── Streaming entry point ──────────────────────────────────────────────────


async def stream_intake_agent(
    request: str,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    session_id: str = "default",
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run one turn of the intake conversation. Yields SSE events.

    tenant_id and client_id are optional. If omitted, the profile is
    stored temporarily under session_id and must be claimed via
    /intake/claim after the user signs up.
    """
    is_anonymous = not (tenant_id and client_id)
    key = _session_key(session_id, tenant_id, client_id)
    session = _get_session(key) or {
        "messages": [],
        "profile": None,
        "completed": False,
    }

    if session.get("completed"):
        if is_anonymous:
            out = (
                "We already completed your intake — your profile is saved "
                "and ready to be attached to your account once you sign up."
            )
        else:
            out = (
                "We already completed your intake — I've saved your context "
                "profile. You can now ask me questions about your business, "
                "generate surveys, or explore other tools."
            )
        yield {"type": "done", "output": out, "stage": "post_intake"}
        return

    history = session.get("messages") or []
    stage_num = _estimate_current_stage(history)

    yield {"type": "status", "message": f"Thinking... (stage {stage_num}/8)"}

    prompt = build_intake_prompt()
    llm = get_llm("context_analysis")
    chain = prompt | llm | StrOutputParser()

    lc_messages = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=request))

    try:
        output = await asyncio.to_thread(
            chain.invoke,
            {
                "current_stage": stage_num,
                "collected_so_far": json.dumps(session.get("profile") or {}, indent=2),
                "messages": lc_messages,
            },
        )
    except Exception as e:
        logger.exception("Intake chain failed")
        yield {"type": "error", "message": f"Intake agent failed: {e}"}
        return

    profile, cleaned_output = _extract_profile(output)

    if profile:
        # Save the profile to session state
        _update_session(
            key,
            user_msg=request,
            assistant_msg=cleaned_output,
            profile=profile,
            completed=True,
        )

        # If we have tenant/client, ingest immediately
        document_id = ""
        warnings: List[str] = []
        if not is_anonymous:
            yield {"type": "status", "message": "Saving your profile..."}
            document_id, warnings = await asyncio.to_thread(
                _ingest_profile, profile, tenant_id, client_id,
            )

        yield {"type": "partial", "text": cleaned_output}
        yield {
            "type": "completed",
            "output": cleaned_output,
            "profile": profile,
            "document_id": document_id,
            "session_id": session_id,
            "anonymous": is_anonymous,
            "next_step": (
                "Call POST /intake/claim with this session_id and your "
                "tenant_id/client_id after signup to ingest the profile."
                if is_anonymous else
                "Profile ingested into your KB."
            ),
            "warnings": warnings,
        }
        return

    _update_session(key, user_msg=request, assistant_msg=cleaned_output)
    yield {"type": "partial", "text": cleaned_output}
    yield {"type": "done", "output": cleaned_output, "stage": stage_num}


# ── Ingest the final profile ───────────────────────────────────────────────


def _ingest_profile(
    profile: Dict[str, Any],
    tenant_id: str,
    client_id: str,
) -> Tuple[str, List[str]]:
    """Ingest the structured profile as a document so downstream agents use it."""
    from uuid import UUID
    from app.models.ingest import IngestInput
    from app.services.ingest.service import IngestService

    profile_text = _format_profile_markdown(profile)

    try:
        inp = IngestInput(
            tenant_id=UUID(tenant_id),
            client_id=UUID(client_id),
            serialized_chunks=[{
                "text": profile_text,
                "chunk_index": 0,
                "page_start": None,
                "page_end": None,
            }],
            serialized_source_type="intake_profile",
            serialized_source_uri=f"intake:{tenant_id}:{client_id}",
            title="Context Profile (Intake)",
            metadata={"intake_raw": profile},
            extract_entities=True,
        )
        result = IngestService().ingest(inp)
        return str(result.document_id), list(result.warnings or [])
    except Exception as e:
        logger.exception("Failed to ingest intake profile")
        return "", [f"Ingest failed: {e}"]


def _format_profile_markdown(profile: Dict[str, Any]) -> str:
    """Format the structured profile as markdown for ingestion."""
    sections = ["# Context Profile (Intake)", ""]

    identity = profile.get("identity_and_role", {}) or {}
    if identity:
        sections.append("## Identity and Role")
        for k, v in identity.items():
            if v:
                sections.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        sections.append("")

    if profile.get("business_summary"):
        sections.append("## Business Summary")
        sections.append(str(profile["business_summary"]))
        sections.append("")

    for section_key, title in [
        ("stage_and_resources", "Stage and Resources"),
        ("customer_and_gtm", "Customer and GTM"),
        ("current_bottleneck", "Current Bottleneck"),
        ("daily_operations", "Daily Operations"),
        ("working_style", "Working Style"),
        ("goals_and_horizon", "Goals and Horizon"),
        ("vocabulary", "Vocabulary"),
    ]:
        val = profile.get(section_key)
        if not val:
            continue
        sections.append(f"## {title}")
        if isinstance(val, dict):
            for k, v in val.items():
                if v:
                    if isinstance(v, list):
                        v = ", ".join(str(x) for x in v)
                    sections.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        elif isinstance(val, list):
            for item in val:
                sections.append(f"- {item}")
        else:
            sections.append(str(val))
        sections.append("")

    if profile.get("open_assumptions"):
        sections.append("## Open Assumptions")
        for item in profile["open_assumptions"]:
            sections.append(f"- {item}")
        sections.append("")

    return "\n".join(sections)


# ── Public helpers (called by router) ──────────────────────────────────────


def reset_intake_session(
    session_id: str,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> None:
    """Drop an in-flight intake session (anonymous or authenticated)."""
    _reset_session_internal(_session_key(session_id, tenant_id, client_id))


def get_stored_profile(session_id: str) -> Optional[Dict[str, Any]]:
    """Look up a stored profile by session_id (checks anonymous bucket).

    Returns the full session dict (messages + profile + completed) or None.
    """
    # Try anonymous first — that's the common case for /intake/profile lookups
    s = _get_session(_session_key(session_id))
    if s:
        return {
            "session_id": session_id,
            "completed": s.get("completed", False),
            "profile": s.get("profile"),
            "message_count": len(s.get("messages", [])),
        }
    return None


def claim_intake_profile(
    session_id: str,
    tenant_id: str,
    client_id: str,
) -> Dict[str, Any]:
    """Attach an anonymous completed profile to a real tenant/client.

    Returns one of:
      {"status": "claimed", "document_id": ..., "warnings": [...]}
      {"status": "not_found"}
      {"status": "incomplete"}
    """
    anon_key = _session_key(session_id)
    s = _get_session(anon_key)
    if not s:
        return {"status": "not_found"}
    if not s.get("completed") or not s.get("profile"):
        return {"status": "incomplete"}

    profile = s["profile"]
    document_id, warnings = _ingest_profile(profile, tenant_id, client_id)

    # Move the session under the authenticated key and clear the anon entry
    auth_key = _session_key(session_id, tenant_id, client_id)
    with _intake_lock:
        _intake_sessions[auth_key] = {
            **s,
            "expires_at": _time.monotonic() + _SESSION_TTL_S,
        }
        _intake_sessions.pop(anon_key, None)

    return {
        "status": "claimed",
        "document_id": document_id,
        "warnings": warnings,
        "profile": profile,
    }

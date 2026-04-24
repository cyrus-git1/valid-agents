"""
Intake agent — Vera runs an 8-stage context intake conversation for users
with no existing documentation.

At the end of the conversation, produces a structured profile and ingests
it as a document in the user's KB so downstream agents have context.

Usage
-----
    POST /intake/stream
    {
      "tenant_id": "...",
      "client_id": "...",
      "session_id": "optional",
      "input": "user message"
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time as _time
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.intake_prompts import build_intake_prompt

logger = logging.getLogger(__name__)

# ── Session state ──────────────────────────────────────────────────────────
#
# Per-session state keyed by f"{tenant_id}:{client_id}:{session_id}".
# Holds conversation history + completion flag + collected profile.

_SESSION_TTL_S = 3600 * 2  # 2 hours — intake is a longer-running flow
_intake_sessions: Dict[str, Dict[str, Any]] = {}
_intake_lock = threading.Lock()


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


def _reset_session(key: str) -> None:
    with _intake_lock:
        _intake_sessions.pop(key, None)


# ── Completion marker parsing ──────────────────────────────────────────────

_COMPLETE_RE = re.compile(
    r"<<INTAKE_COMPLETE>>\s*(\{.*?\})\s*<<END_INTAKE>>",
    re.DOTALL,
)


def _extract_profile(output: str) -> tuple[Optional[Dict[str, Any]], str]:
    """If the output contains the completion marker, return (profile, cleaned_text).

    Otherwise returns (None, output).
    """
    m = _COMPLETE_RE.search(output)
    if not m:
        return None, output
    try:
        profile = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse intake profile JSON: %s", e)
        return None, output
    # Strip the marker block from the user-facing output
    cleaned = _COMPLETE_RE.sub("", output).strip()
    return profile, cleaned


def _estimate_current_stage(messages: List[Dict[str, str]]) -> int:
    """Rough stage counter based on assistant message count."""
    # Each stage typically takes 2-4 exchanges (core + follow-up + summary)
    assistant_msgs = sum(1 for m in messages if m["role"] == "assistant")
    stage = min(8, max(1, assistant_msgs // 2 + 1))
    return stage


# ── Streaming entry point ──────────────────────────────────────────────────


async def stream_intake_agent(
    request: str,
    tenant_id: str,
    client_id: str,
    session_id: str = "default",
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run one turn of the intake conversation. Yields SSE events.

    Event types:
      status    — progress update
      partial   — assistant response text
      done      — turn complete, normal flow
      completed — intake finished, profile ingested. Includes profile + document_id
      error     — something went wrong
    """
    session_key = f"{tenant_id}:{client_id}:{session_id}"
    session = _get_session(session_key) or {
        "messages": [],
        "profile": None,
        "completed": False,
    }

    # If intake was already completed in this session, gently redirect
    if session.get("completed"):
        out = (
            "We already completed your intake — I've saved your context profile. "
            "You can now ask me questions about your business, generate surveys, "
            "or explore other tools. Want to start fresh? Let me know."
        )
        yield {"type": "done", "output": out, "stage": "post_intake"}
        return

    history = session.get("messages") or []
    stage_num = _estimate_current_stage(history)

    yield {"type": "status", "message": f"Thinking... (stage {stage_num}/8)"}

    # Build the LLM call
    prompt = build_intake_prompt()
    llm = get_llm("context_analysis")
    chain = prompt | llm | StrOutputParser()

    # Assemble message history as LangChain messages
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

    # Check for completion marker
    profile, cleaned_output = _extract_profile(output)

    if profile:
        # Save + ingest the profile as a document
        yield {"type": "status", "message": "Saving your profile..."}
        document_id, warnings = await asyncio.to_thread(
            _ingest_profile, profile, tenant_id, client_id,
        )

        # Update session
        _update_session(
            session_key,
            user_msg=request,
            assistant_msg=cleaned_output,
            profile=profile,
            completed=True,
        )

        yield {"type": "partial", "text": cleaned_output}
        yield {
            "type": "completed",
            "output": cleaned_output,
            "profile": profile,
            "document_id": document_id,
            "warnings": warnings,
        }
        return

    # Normal turn — save and return
    _update_session(session_key, user_msg=request, assistant_msg=cleaned_output)

    yield {"type": "partial", "text": cleaned_output}
    yield {"type": "done", "output": cleaned_output, "stage": stage_num}


# ── Ingest the final profile ───────────────────────────────────────────────


def _ingest_profile(
    profile: Dict[str, Any],
    tenant_id: str,
    client_id: str,
) -> tuple[str, List[str]]:
    """Ingest the structured profile as a document so downstream agents use it.

    Returns (document_id, warnings).
    """
    from uuid import UUID
    from app.models.ingest import IngestInput
    from app.services.ingest.service import IngestService

    # Format the profile as readable markdown for the chunks
    sections = [
        "# Context Profile (Intake)",
        "",
    ]

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

    profile_text = "\n".join(sections)

    # Ingest as a serialized-chunks document
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


# ── Reset endpoint helper ──────────────────────────────────────────────────


def reset_intake_session(tenant_id: str, client_id: str, session_id: str = "default") -> None:
    """Drop an in-flight intake session (useful for 'start over')."""
    _reset_session(f"{tenant_id}:{client_id}:{session_id}")

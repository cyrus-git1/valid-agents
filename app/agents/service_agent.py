"""
Service agent — plan-execute-reflect chat agent.

Replaces the rigid router agent with a three-phase flow:
  0. Fast path: casual conversation handled by lightweight LLM (no tools)
  1. Plan:      LLM analyzes the request, produces a structured tool execution plan
  2. Execute:   ReAct agent follows the plan, adapting if steps fail
  3. Reflect:   Optional quality check on multi-step results

Usage
-----
    from app.agents.service_agent import run_service_agent

    result = run_service_agent(
        request="What are our top risks?",
        tenant_id="...",
        client_id="...",
    )
    print(result["output"])
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.service_prompts import (
    CONVERSATION_PROMPT,
    EXECUTE_PROMPT,
    PLAN_PROMPT,
    REFLECT_PROMPT,
)
from app.tools.service_tools import create_service_tools

logger = logging.getLogger(__name__)

# ── Episodic memory (session history) ─────────────────────────────────────
#
# In-memory cache of recent conversation turns keyed by "tenant_id:client_id".
# Keeps the last N exchanges so follow-up messages have context.
# TTL-based: entries expire after 30 minutes of inactivity.

import threading
import time as _time

_SESSION_TTL_S = 1800  # 30 minutes
_SESSION_MAX_TURNS = 20  # keep last 20 exchanges (40 messages)
_session_history: Dict[str, Dict[str, Any]] = {}
_session_lock = threading.Lock()


def _get_session_history(key: str) -> Optional[List[Dict[str, str]]]:
    """Get conversation history for a session, or None if expired/missing."""
    with _session_lock:
        session = _session_history.get(key)
        if not session:
            return None
        if _time.monotonic() > session["expires_at"]:
            del _session_history[key]
            return None
        return session["messages"]


def _append_to_history(key: str, user_msg: str, assistant_msg: str) -> None:
    """Append a user/assistant exchange to session history."""
    with _session_lock:
        if key not in _session_history:
            _session_history[key] = {
                "messages": [],
                "expires_at": _time.monotonic() + _SESSION_TTL_S,
            }
        session = _session_history[key]
        session["messages"].append({"role": "user", "content": user_msg})
        session["messages"].append({"role": "assistant", "content": assistant_msg})
        # Trim to max turns
        if len(session["messages"]) > _SESSION_MAX_TURNS * 2:
            session["messages"] = session["messages"][-_SESSION_MAX_TURNS * 2:]
        session["expires_at"] = _time.monotonic() + _SESSION_TTL_S


def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
    """Format conversation history as a string for prompt injection."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-10:]:  # last 5 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content'][:200]}")
    return "\n".join(lines)


# Map tool name → intent string for AgentQueryResponse compatibility
_TOOL_INTENT_MAP: Dict[str, str] = {
    "ask_question": "retrieval",
    "generate_survey": "survey",
    "extend_survey": "survey_extend",
    "find_personas": "persona",
    "enrich_kb": "enrich",
    "ingest_url": "ingest",
    "build_context": "context",
    "list_documents": "documents",
    "flag_document": "revision",
    "get_summary": "summary",
    "search_kb": "search",
    "check_status": "status",
    "analyze_transcript": "transcript_analysis",
    "competitive_intelligence": "competitive_intel",
    "cross_document_synthesis": "synthesis",
}


def run_service_agent(
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Entry point. Returns {intent, output, sources, confidence}.

    Conversation history is managed automatically via episodic memory —
    a per-session cache keyed by (tenant_id, client_id) that stores
    recent exchanges so follow-up messages have context.
    """
    session_key = f"{tenant_id}:{client_id}"
    history = _get_session_history(session_key)

    # ── Phase 0: Quick conversation check ──────────────────────────────
    if _is_likely_conversation(request) and not history:
        result = _handle_conversation(request)
        _append_to_history(session_key, request, result["output"])
        return result

    # ── Phase 1: Plan ──────────────────────────────────────────────────
    plan = _plan(request, history)

    # Plan detected conversation
    if plan.get("is_conversation"):
        return _handle_conversation(request)

    # Plan detected out-of-scope request
    if plan.get("is_out_of_scope"):
        return {
            "intent": "out_of_scope",
            "output": (
                "I'm here to help with your market research on Valid — I can search "
                "your knowledge base, generate surveys, discover personas, and more. "
                "That question falls outside what I can help with."
            ),
            "sources": [],
            "confidence": 1.0,
        }

    if plan.get("needs_clarification"):
        return {
            "intent": "clarification",
            "output": plan.get("clarification_message", "Could you clarify your request?"),
            "sources": [],
            "confidence": plan.get("confidence", 0.0),
        }

    # ── Phase 2: Execute ───────────────────────────────────────────────
    result = _execute(request, plan, tenant_id, client_id, client_profile, history)

    # ── Phase 3: Reflect (only for multi-step plans) ───────────────────
    steps = plan.get("steps", [])
    if len(steps) >= 2 and result.get("output"):
        reflection = _reflect(request, result["output"])
        if reflection.get("quality") == "poor" and reflection.get("gaps"):
            gaps_text = "; ".join(reflection["gaps"])
            result["output"] += f"\n\nNote: some aspects may be incomplete — {gaps_text}"

    _append_to_history(session_key, request, result.get("output", ""))
    return result


# ── Streaming entry point ─────────────────────────────────────────────────


async def stream_service_agent(
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streaming version of run_service_agent. Yields SSE events.

    Event types:
      status  — progress update (e.g., "Planning...", "Searching knowledge base...")
      partial — content chunk from the response
      done    — final result with intent, output, sources, confidence
      error   — something went wrong
    """
    session_key = f"{tenant_id}:{client_id}"
    history = _get_session_history(session_key)

    try:
        # ── Phase 0: Conversation fast path ────────────────────────────
        if _is_likely_conversation(request) and not history:
            result = _handle_conversation(request)
            _append_to_history(session_key, request, result["output"])
            yield {"type": "done", **result}
            return

        # ── Phase 1: Plan ──────────────────────────────────────────────
        yield {"type": "status", "message": "Planning..."}
        plan = await asyncio.to_thread(_plan, request, history)

        if plan.get("is_conversation"):
            result = _handle_conversation(request)
            _append_to_history(session_key, request, result["output"])
            yield {"type": "done", **result}
            return

        if plan.get("is_out_of_scope"):
            out = (
                "I'm here to help with your market research on Valid — I can search "
                "your knowledge base, generate surveys, discover personas, and more. "
                "That question falls outside what I can help with."
            )
            _append_to_history(session_key, request, out)
            yield {"type": "done", "intent": "out_of_scope", "output": out,
                   "sources": [], "confidence": 1.0}
            return

        if plan.get("needs_clarification"):
            out = plan.get("clarification_message", "Could you clarify your request?")
            _append_to_history(session_key, request, out)
            yield {"type": "done", "intent": "clarification", "output": out,
                   "sources": [], "confidence": plan.get("confidence", 0.0)}
            return

        # ── Phase 2: Execute ───────────────────────────────────────────
        steps = plan.get("steps", [])
        tool_names = [s.get("tool", "") for s in steps]
        status_msg = _tool_status_message(tool_names)
        yield {"type": "status", "message": status_msg}

        result = await asyncio.to_thread(
            _execute, request, plan, tenant_id, client_id, client_profile, history,
        )

        # Stream the output
        if result.get("output"):
            yield {"type": "partial", "text": result["output"]}

        # ── Phase 3: Reflect (multi-step only) ─────────────────────────
        if len(steps) >= 2 and result.get("output"):
            yield {"type": "status", "message": "Reviewing response..."}
            reflection = await asyncio.to_thread(
                _reflect, request, result["output"],
            )
            if reflection.get("quality") == "poor" and reflection.get("gaps"):
                gaps_text = "; ".join(reflection["gaps"])
                result["output"] += f"\n\nNote: some aspects may be incomplete — {gaps_text}"

        _append_to_history(session_key, request, result.get("output", ""))
        yield {"type": "done", **result}

    except Exception as e:
        logger.exception("stream_service_agent failed")
        yield {"type": "error", "message": str(e)}


def _tool_status_message(tool_names: List[str]) -> str:
    """Generate a human-friendly status message from planned tool names."""
    messages = {
        "ask_question": "Searching the knowledge base...",
        "generate_survey": "Generating your survey...",
        "find_personas": "Discovering audience personas...",
        "enrich_kb": "Analyzing knowledge base gaps...",
        "ingest_url": "Ingesting URL into the knowledge base...",
        "build_context": "Building context summary...",
        "list_documents": "Fetching your documents...",
        "flag_document": "Updating document...",
        "get_summary": "Fetching summary...",
        "search_kb": "Searching the knowledge base...",
        "check_status": "Checking knowledge base status...",
    }
    if tool_names:
        return messages.get(tool_names[0], "Working on it...")
    return "Working on it..."


# ── Conversation fast path ─────────────────────────────────────────────────

_CONVERSATION_PATTERNS = {
    "hi", "hello", "hey", "hewwo", "yo", "sup", "hiya", "howdy",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see ya", "cya",
    "ok", "okay", "cool", "nice", "great",
    "lol", "haha", "hehe",
    "help", "what can you do", "who are you",
}


_TOOL_KEYWORDS = {
    "survey", "persona", "ingest", "document", "search", "enrich",
    "summary", "context", "question", "analyze", "flag", "revise",
    "upload", "add", "import", "delete", "list", "find", "generate",
    "create", "build", "tell me", "what", "who", "how", "why",
}


def _is_likely_conversation(request: str) -> bool:
    """Fast keyword check to skip the planner for obvious chitchat."""
    cleaned = request.strip().lower().rstrip("!?.,:;")
    # Exact match on short messages
    if cleaned in _CONVERSATION_PATTERNS:
        return True
    # Contains a URL or domain — never conversation
    if "http://" in cleaned or "https://" in cleaned:
        return False
    words = cleaned.split()
    if words and "." in words[-1]:
        return False
    # Very short messages (< 4 words) that aren't tool-related
    if len(words) <= 3 and not any(kw in cleaned for kw in _TOOL_KEYWORDS):
        return True
    return False


def _handle_conversation(request: str) -> Dict[str, Any]:
    """Respond to casual messages with a lightweight LLM call."""
    llm = get_llm("default")
    chain = CONVERSATION_PROMPT | llm | StrOutputParser()

    try:
        output = chain.invoke({"request": request})
    except Exception as e:
        logger.warning("Conversation response failed: %s", e)
        output = "Hey there! I'm Vera, your research assistant. How can I help you today?"

    return {
        "intent": "conversation",
        "output": output,
        "sources": [],
        "confidence": 1.0,
    }


# ── Plan ───────────────────────────────────────────────────────────────────


def _plan(request: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Single LLM call to produce a structured execution plan."""
    llm = get_llm("service_planner")
    chain = PLAN_PROMPT | llm | StrOutputParser()

    # Prepend conversation history so the planner has context
    history_text = _format_history(history)
    plan_input = f"{history_text}\n\nCurrent request: {request}" if history_text else request

    try:
        raw = chain.invoke({"request": plan_input})
        plan = _parse_json(raw)
        if plan and ("steps" in plan or plan.get("is_conversation")):
            return plan
    except Exception as e:
        logger.warning("Plan generation failed: %s", e)

    # Fallback: single-step plan based on request shape
    looks_like_question = any(
        request.strip().lower().startswith(w)
        for w in ("what", "who", "where", "when", "why", "how", "is ", "are ", "do ", "does ", "can ", "tell me")
    )
    fallback_tool = "ask_question" if looks_like_question else "search_kb"
    fallback_args = {"question": request} if looks_like_question else {"query": request}

    return {
        "reasoning": "Fallback: could not parse plan from LLM",
        "is_conversation": False,
        "steps": [{"tool": fallback_tool, "args": fallback_args, "purpose": "fallback"}],
        "confidence": 0.5,
        "needs_clarification": False,
        "clarification_message": None,
    }


# ── Execute ────────────────────────────────────────────────────────────────


def _execute(
    request: str,
    plan: Dict[str, Any],
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run the ReAct agent with the plan injected into the system prompt."""
    tools = create_service_tools(tenant_id, client_id, client_profile)

    # Inject plan into execute prompt
    plan_text = json.dumps(plan.get("steps", []), indent=2)
    system_prompt = EXECUTE_PROMPT.replace("{plan_json}", plan_text)

    # Add conversation history to system prompt
    history_text = _format_history(history)
    if history_text:
        system_prompt += f"\n\n{history_text}"

    try:
        from langgraph.prebuilt import create_react_agent
        llm = get_llm("service_agent")
        agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    except ImportError:
        logger.error("create_react_agent not available")
        return {
            "intent": "unknown",
            "output": "Agent execution is not available (missing langgraph).",
            "sources": [],
            "confidence": 0.0,
        }

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"recursion_limit": 15},
        )
    except Exception as e:
        logger.exception("Service agent execution failed")
        return {
            "intent": _intent_from_plan(plan),
            "output": f"I ran into an issue processing your request — {e}. Could you try again?",
            "sources": [],
            "confidence": plan.get("confidence", 0.0),
        }

    # Extract output, tool trace, tool results, and sources from messages
    messages = result.get("messages", [])
    output = ""
    tool_calls_trace: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for msg in messages:
        # Track tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_trace.append(tc.get("name", "unknown"))

        # Get the final AI message (no tool calls = final synthesis)
        if hasattr(msg, "content") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                output = content

        # Capture tool results for fallback synthesis
        if msg.type == "tool" and hasattr(msg, "content"):
            try:
                tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(tool_result, dict):
                    tool_results.append(tool_result)
                    if tool_result.get("citations"):
                        sources.extend(tool_result["citations"])
                elif isinstance(tool_result, list):
                    tool_results.append({"items": tool_result})
            except (json.JSONDecodeError, TypeError):
                pass

    # If the agent didn't produce a final synthesis (just an affirmation
    # or empty response), build one from the tool results
    if not output or _is_just_affirmation(output):
        fallback = _synthesize_from_results(tool_results, tool_calls_trace)
        if fallback:
            output = fallback

    # Determine intent from the first tool actually called
    intent = "unknown"
    if tool_calls_trace:
        intent = _TOOL_INTENT_MAP.get(tool_calls_trace[0], "unknown")
    else:
        intent = _intent_from_plan(plan)

    return {
        "intent": intent,
        "output": output or "No response generated.",
        "sources": sources,
        "confidence": plan.get("confidence", 0.0),
    }


# ── Reflect ────────────────────────────────────────────────────────────────


def _reflect(request: str, response: str) -> Dict[str, Any]:
    """Optional quality check on the agent's response."""
    llm = get_llm("service_planner")
    chain = REFLECT_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"request": request, "response": response[:2000]})
        result = _parse_json(raw)
        if result:
            return result
    except Exception as e:
        logger.warning("Reflect step failed: %s", e)

    return {"satisfied": True, "gaps": [], "quality": "good"}


# ── Helpers ────────────────────────────────────────────────────────────────


_AFFIRMATION_PHRASES = [
    "let me", "i'll", "just a moment", "pulling up", "searching",
    "looking up", "one moment", "sure thing", "great question",
    "let me check", "let me look", "let me pull", "let me search",
    "i will", "working on", "generating", "fetching",
]


def _is_just_affirmation(text: str) -> bool:
    """Check if the text is just an acknowledgment with no real content.

    An affirmation is a short message that acknowledges the user's request
    but doesn't contain actual data (numbers, lists, specific answers).
    """
    cleaned = text.strip().lower()
    # Must be short and contain affirmation language
    if len(cleaned) > 300:
        return False
    if not any(phrase in cleaned for phrase in _AFFIRMATION_PHRASES):
        return False
    # Check for actual data content — concrete values, not just keywords
    # that might appear in the affirmation itself
    has_concrete_data = (
        # Contains numbers (chunk counts, entity counts, IDs)
        any(c.isdigit() for c in cleaned)
        # Contains bullet points or lists
        or "\n-" in cleaned or "\n*" in cleaned or "1." in cleaned
        # Contains quoted text or citations
        or '"' in cleaned or "[c" in cleaned
        # Contains a UUID-like pattern
        or len(cleaned) > 100 and "-" in cleaned and any(
            len(part) >= 8 and all(c in "0123456789abcdef-" for c in part)
            for part in cleaned.split()
        )
    )
    return not has_concrete_data


def _synthesize_from_results(
    tool_results: List[Dict[str, Any]],
    tool_names: List[str],
) -> str:
    """Build a response from raw tool results when the agent didn't synthesize."""
    if not tool_results:
        return ""

    parts: List[str] = []
    for i, result in enumerate(tool_results):
        tool_name = tool_names[i] if i < len(tool_names) else "unknown"

        # Check for errors
        if result.get("error"):
            parts.append(f"I encountered an issue: {result['error']}")
            continue

        # Check for a human-readable message
        if result.get("message"):
            parts.append(result["message"])
            continue

        # Format based on tool type
        if tool_name == "list_documents":
            docs = result.get("documents", [])
            total = result.get("total", len(docs))
            if not docs:
                parts.append("Your knowledge base is empty — no documents have been ingested yet.")
            else:
                lines = [f"You have {total} document(s) in your knowledge base:\n"]
                for d in docs:
                    title = d.get("title") or "Untitled"
                    source_url = d.get("source_url", "")
                    status = d.get("status", "active")

                    if source_url and source_url.startswith("http"):
                        lines.append(f"- **{title}** — {source_url} ({status})")
                    else:
                        lines.append(f"- **{title}** ({status})")
                parts.append("\n".join(lines))

        elif tool_name == "check_status":
            doc_count = result.get("document_count", 0)
            summary_status = result.get("context_summary", "unknown")
            parts.append(
                f"Knowledge base status: {doc_count} document(s), "
                f"context summary: {summary_status}."
            )
            docs = result.get("documents", [])
            if docs:
                lines = ["\nDocuments:"]
                for d in docs:
                    title = d.get("title") or d.get("id", "Untitled")
                    lines.append(f"- **{title}** ({d.get('source_type', 'unknown')}, {d.get('chunks', 0)} chunks)")
                parts.append("\n".join(lines))

        elif tool_name == "ingest_url":
            doc_id = result.get("document_id", "")
            chunks = result.get("chunks_upserted", 0)
            entities = result.get("entities_linked", 0)
            uri = result.get("source_uri", "")
            parts.append(
                f"Successfully ingested {uri}. "
                f"Stored {chunks} content chunks and linked {entities} entities."
            )

        elif result.get("answer"):
            parts.append(result["answer"])

        elif result.get("summary"):
            parts.append(result["summary"])

    return "\n\n".join(parts) if parts else ""


def _intent_from_plan(plan: Dict[str, Any]) -> str:
    """Derive intent string from the first step in the plan."""
    steps = plan.get("steps", [])
    if steps:
        tool_name = steps[0].get("tool", "")
        return _TOOL_INTENT_MAP.get(tool_name, "unknown")
    return "unknown"


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM output, handling markdown code blocks."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None

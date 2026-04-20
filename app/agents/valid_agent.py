"""
Valid sales chat agent — ReAct agent that can search Valid's KB,
show pricing/sales models, and book demos.

Guardrails: only answers questions about Valid, pricing, or demos.
Anything outside that scope is politely refused.

Usage
-----
    POST /valid/stream
    {"input": "What does Valid do?"}
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.llm_config import get_llm
from app.tools.valid_tools import create_valid_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Vera, Valid's friendly and professional sales assistant. "
    "Valid is a market research platform that helps businesses validate ideas "
    "through real customer feedback and AI-driven insights.\n\n"
    "YOUR CAPABILITIES (use the tools):\n"
    "1. Answer questions about Valid — product, features, use cases, team\n"
    "2. Show pricing, plans, and sales models\n"
    "3. Book demos for interested prospects\n\n"
    "GUARDRAILS — strictly enforced:\n"
    "- You ONLY discuss Valid, its product, pricing, and demo booking.\n"
    "- REFUSE anything unrelated: programming, medical advice, general knowledge, "
    "math, science, other companies, news, weather, etc.\n"
    "- If someone asks about THEIR company (not Valid), explain: 'I'm Vera, "
    "Valid's sales assistant — I can tell you all about Valid, but I don't have "
    "access to your company's data. Once you sign up, our platform's AI assistant "
    "can answer questions about your business using your own documents.'\n"
    "- When refusing other topics, say: 'I'm here to help you learn about Valid "
    "and book a demo. Feel free to ask me anything about Valid!'\n"
    "- NEVER answer from your own training data. ALL answers must come from "
    "search_valid or get_pricing tool results.\n"
    "- If the search returns no results, say so honestly — don't guess.\n\n"
    "DEMO BOOKING FLOW:\n"
    "- When someone wants a demo, collect their name and email (required), "
    "plus company and role (optional).\n"
    "- If they provide info gradually across messages, gather it step by step.\n"
    "- Once you have name + email, call book_demo.\n"
    "- Be conversational — don't ask for all fields at once like a form.\n\n"
    "PERSONALITY:\n"
    "- Warm, concise, professional — like a knowledgeable sales rep.\n"
    "- Proactively suggest a demo when the user seems interested.\n"
    "- When discussing features, relate them to business outcomes.\n"
    "- Call the tools. Do NOT respond until you have tool results.\n"
    "- Your final response MUST include actual data from tool results."
)

_CONVERSATION_PATTERNS = {
    "hi", "hello", "hey", "hewwo", "yo", "sup", "hiya", "howdy",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see ya", "cya",
    "ok", "okay", "cool", "nice", "great",
    "lol", "haha", "hehe",
}

# ── Episodic memory ───────────────────────────────────────────────────────

import threading
import time as _time

_SESSION_TTL_S = 1800
_SESSION_MAX_TURNS = 20
_valid_sessions: Dict[str, Dict[str, Any]] = {}
_valid_session_lock = threading.Lock()


def _get_history(session_id: str) -> Optional[List[Dict[str, str]]]:
    with _valid_session_lock:
        s = _valid_sessions.get(session_id)
        if not s or _time.monotonic() > s["expires_at"]:
            _valid_sessions.pop(session_id, None)
            return None
        return s["messages"]


def _append_history(session_id: str, user_msg: str, assistant_msg: str) -> None:
    with _valid_session_lock:
        if session_id not in _valid_sessions:
            _valid_sessions[session_id] = {
                "messages": [],
                "expires_at": _time.monotonic() + _SESSION_TTL_S,
            }
        s = _valid_sessions[session_id]
        s["messages"].append({"role": "user", "content": user_msg})
        s["messages"].append({"role": "assistant", "content": assistant_msg})
        if len(s["messages"]) > _SESSION_MAX_TURNS * 2:
            s["messages"] = s["messages"][-_SESSION_MAX_TURNS * 2:]
        s["expires_at"] = _time.monotonic() + _SESSION_TTL_S


def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-10:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content'][:200]}")
    return "\n".join(lines)


# ── Agent ──────────────────────────────────────────────────────────────────


def _is_chitchat(request: str) -> bool:
    cleaned = request.strip().lower().rstrip("!?.,:;")
    if cleaned in _CONVERSATION_PATTERNS:
        return True
    return False


async def stream_valid_agent(
    request: str,
    session_id: str = "default",
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream responses from the Valid sales chat agent."""
    history = _get_history(session_id)

    try:
        # Chitchat fast path
        if _is_chitchat(request) and not history:
            out = (
                "Hey! I'm Vera, Valid's assistant. I can tell you about our platform, "
                "show you our pricing and plans, or book a demo with our team. "
                "What would you like to know?"
            )
            _append_history(session_id, request, out)
            yield {"type": "done", "output": out, "sources": []}
            return

        # Run ReAct agent
        yield {"type": "status", "message": "Thinking..."}
        output, sources = await asyncio.to_thread(
            _run_agent, request, history,
        )

        _append_history(session_id, request, output)

        yield {"type": "partial", "text": output}
        yield {"type": "done", "output": output, "sources": sources}

    except Exception as e:
        logger.exception("stream_valid_agent failed")
        yield {"type": "error", "message": str(e)}


def _run_agent(
    request: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple:
    """Run the ReAct agent synchronously."""
    tools = create_valid_tools()

    # Build system prompt with history
    system = _SYSTEM_PROMPT
    history_text = _format_history(history)
    if history_text:
        system += f"\n\n{history_text}"

    try:
        from langgraph.prebuilt import create_react_agent
        llm = get_llm("service_agent")
        agent = create_react_agent(model=llm, tools=tools, prompt=system)
    except ImportError:
        logger.error("create_react_agent not available")
        return "Agent is not available.", []

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"recursion_limit": 10},
        )
    except Exception as e:
        logger.exception("Valid ReAct agent failed")
        return f"I ran into an issue — {e}. Could you try again?", []

    # Extract output and sources
    messages = result.get("messages", [])
    output = ""
    sources: List[Dict[str, Any]] = []

    for msg in messages:
        # Final AI message (no tool calls)
        if hasattr(msg, "content") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                output = content

        # Extract search sources from tool results
        if msg.type == "tool" and hasattr(msg, "content"):
            try:
                tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(tool_result, list):
                    for r in tool_result:
                        if isinstance(r, dict) and r.get("document_id"):
                            sources.append({
                                "document_id": r.get("document_id"),
                                "chunk_index": r.get("chunk_index"),
                                "similarity_score": r.get("similarity_score"),
                            })
                elif isinstance(tool_result, dict) and tool_result.get("message"):
                    # book_demo or get_pricing message — use as output if agent didn't synthesize
                    if not output or len(output.strip()) < 20:
                        output = tool_result["message"]
            except (json.JSONDecodeError, TypeError):
                pass

    if not output:
        output = "I'm here to help you learn about Valid. Could you rephrase your question?"

    return output, sources

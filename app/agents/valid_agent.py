"""
Valid docs chat agent — lightweight agent that only searches Valid's
internal knowledge base via /search/valid.

No tenant/client scoping, no plan-execute-reflect, no tool chaining.
Just: search the KB, synthesize an answer, stream it back.

Usage
-----
    POST /valid/stream
    {"input": "What does Valid do?"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm_config import get_llm
from app.tools.valid_tools import create_valid_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful, professional assistant for Valid — a market research "
    "platform that helps businesses validate ideas through real customer "
    "feedback and AI-driven insights.\n\n"
    "You answer questions about Valid using ONLY the search results provided. "
    "Never make up information. If the search results don't contain the answer, "
    "say so honestly.\n\n"
    "Keep responses concise and friendly. Use the search results to back up "
    "every claim."
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
    words = cleaned.split()
    if len(words) <= 2 and not any(
        kw in cleaned for kw in ("valid", "product", "feature", "price", "survey", "research")
    ):
        return True
    return False


async def stream_valid_agent(
    request: str,
    session_id: str = "default",
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream responses from the Valid docs chat agent."""
    history = _get_history(session_id)

    try:
        # Chitchat fast path
        if _is_chitchat(request) and not history:
            out = "Hey! I'm Valid's assistant. Ask me anything about Valid — our product, features, pricing, or how we help with market research."
            _append_history(session_id, request, out)
            yield {"type": "done", "output": out, "sources": []}
            return

        # Search
        yield {"type": "status", "message": "Searching Valid's knowledge base..."}
        search_results = await asyncio.to_thread(_search, request, history)

        if not search_results or (len(search_results) == 1 and search_results[0].get("error")):
            out = "I couldn't find anything about that in Valid's knowledge base. Could you rephrase your question?"
            _append_history(session_id, request, out)
            yield {"type": "done", "output": out, "sources": []}
            return

        # Synthesize
        yield {"type": "status", "message": "Generating response..."}
        output, sources = await asyncio.to_thread(
            _synthesize, request, search_results, history
        )

        _append_history(session_id, request, output)

        yield {"type": "partial", "text": output}
        yield {"type": "done", "output": output, "sources": sources}

    except Exception as e:
        logger.exception("stream_valid_agent failed")
        yield {"type": "error", "message": str(e)}


def _search(
    request: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Search Valid's KB, using history for context."""
    tools = create_valid_tools()
    search_tool = tools[0]

    # Build search query with history context
    query = request
    if history:
        # Include recent context for better search relevance
        recent = [m["content"] for m in history[-4:] if m["role"] == "user"]
        if recent:
            query = f"{' '.join(recent)} {request}"

    return search_tool.invoke({"query": query, "top_k": 5})


def _synthesize(
    request: str,
    results: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple:
    """Synthesize an answer from search results."""
    context = "\n\n".join(
        f"[{i+1}] {r.get('content', '')}"
        for i, r in enumerate(results)
        if r.get("content")
    )

    history_text = _format_history(history)
    system = _SYSTEM_PROMPT
    if history_text:
        system += f"\n\n{history_text}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Search results:\n{context}\n\nQuestion: {question}\n\nAnswer based on the search results above."),
    ])

    llm = get_llm("context_analysis")
    chain = prompt | llm | StrOutputParser()

    try:
        output = chain.invoke({"context": context, "question": request})
    except Exception as e:
        logger.warning("Valid synthesis failed: %s", e)
        output = "I found some relevant information but had trouble putting it together. Please try again."

    # Build sources
    sources = [
        {
            "document_id": r.get("document_id"),
            "chunk_index": r.get("chunk_index"),
            "similarity_score": r.get("similarity_score"),
        }
        for r in results
        if r.get("document_id")
    ]

    return output, sources

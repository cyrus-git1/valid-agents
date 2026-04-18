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

import json
import logging
import re
from typing import Any, Dict, List, Optional

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

# Map tool name → intent string for AgentQueryResponse compatibility
_TOOL_INTENT_MAP: Dict[str, str] = {
    "ask_question": "retrieval",
    "generate_survey": "survey",
    "find_personas": "persona",
    "enrich_kb": "enrich",
    "ingest_url": "ingest",
    "build_context": "context",
    "list_documents": "documents",
    "flag_document": "revision",
    "get_summary": "summary",
    "search_kb": "search",
}


def run_service_agent(
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Entry point. Returns {intent, output, sources, confidence}.

    Compatible with the existing AgentQueryResponse contract.
    """
    # ── Phase 0: Quick conversation check ──────────────────────────────
    if _is_likely_conversation(request):
        return _handle_conversation(request)

    # ── Phase 1: Plan ──────────────────────────────────────────────────
    plan = _plan(request)

    # Plan detected conversation
    if plan.get("is_conversation"):
        return _handle_conversation(request)

    if plan.get("needs_clarification"):
        return {
            "intent": "clarification",
            "output": plan.get("clarification_message", "Could you clarify your request?"),
            "sources": [],
            "confidence": plan.get("confidence", 0.0),
        }

    # ── Phase 2: Execute ───────────────────────────────────────────────
    result = _execute(request, plan, tenant_id, client_id, client_profile)

    # ── Phase 3: Reflect (only for multi-step plans) ───────────────────
    steps = plan.get("steps", [])
    if len(steps) >= 2 and result.get("output"):
        reflection = _reflect(request, result["output"])
        if reflection.get("quality") == "poor" and reflection.get("gaps"):
            gaps_text = "; ".join(reflection["gaps"])
            result["output"] += f"\n\nNote: some aspects may be incomplete — {gaps_text}"

    return result


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
        output = "Hey there! How can I help you today?"

    return {
        "intent": "conversation",
        "output": output,
        "sources": [],
        "confidence": 1.0,
    }


# ── Plan ───────────────────────────────────────────────────────────────────


def _plan(request: str) -> Dict[str, Any]:
    """Single LLM call to produce a structured execution plan."""
    llm = get_llm("service_planner")
    chain = PLAN_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"request": request})
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
) -> Dict[str, Any]:
    """Run the ReAct agent with the plan injected into the system prompt."""
    tools = create_service_tools(tenant_id, client_id, client_profile)

    # Inject plan into execute prompt
    plan_text = json.dumps(plan.get("steps", []), indent=2)
    system_prompt = EXECUTE_PROMPT.replace("{plan_json}", plan_text)

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

    # Extract output, tool trace, and sources from messages
    messages = result.get("messages", [])
    output = ""
    tool_calls_trace: List[str] = []
    sources: List[Dict[str, Any]] = []

    for msg in messages:
        # Track tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_trace.append(tc.get("name", "unknown"))

        # Get the final AI message (no tool calls = final synthesis)
        if hasattr(msg, "content") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
            output = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Extract citations from tool results
        if msg.type == "tool" and hasattr(msg, "content"):
            try:
                tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(tool_result, dict) and tool_result.get("citations"):
                    sources.extend(tool_result["citations"])
            except (json.JSONDecodeError, TypeError):
                pass

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

"""
Persona discovery agent — ReAct agent that explores the KB to identify
audience personas with evidence tracking.

This is the first true ReAct agent in the system. Unlike deterministic
pipelines, the persona agent decides what to search based on what it
finds — iterating on thin segments, checking transcripts, and optionally
searching the web for demographic data.

The harness wraps the entire agent invocation — if the output personas
are low quality, the whole exploration is retried with feedback.

Usage
-----
    from app.agents.persona_agent import run_persona_agent

    result = run_persona_agent(
        tenant_id="...",
        client_id="...",
        request="Find personas interested in sustainability",
        client_profile={"industry": "SaaS"},
    )
    for p in result["personas"]:
        print(p["name"], p["evidence_sources"])
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.llm_config import get_llm
from app.harness_pkg import run_with_harness, StepOutput
from app.harness_pkg.configs import PERSONA_STEP_CONFIG
from app.prompts.persona_prompts import PERSONA_AGENT_SYSTEM_PROMPT
from app.tools.persona_tools import create_persona_tools

logger = logging.getLogger(__name__)


def run_persona_agent(
    tenant_id: str,
    client_id: str,
    request: Optional[str] = None,
    client_profile: Optional[Dict[str, Any]] = None,
    max_personas: int = 5,
    top_k: int = 15,
    hop_limit: int = 1,
) -> Dict[str, Any]:
    """Discover audience personas using a ReAct agent with harness quality gating.

    The agent explores the KB iteratively, checks transcripts, and optionally
    searches the web. The harness validates the final output.

    Returns dict with keys: personas, context_used, status, error, warnings.
    """
    user_request = request or "Identify the key audience personas for this organization."

    # Build closure-bound tools
    tools = create_persona_tools(tenant_id, client_id)

    def step_fn(inputs: dict, feedback_section: str):
        """Run the ReAct agent and capture execution traces."""

        # Build system prompt with optional feedback
        system_prompt = PERSONA_AGENT_SYSTEM_PROMPT.replace(
            "{feedback_section}",
            f"\n\nREVISION GUIDANCE (address these issues):\n{feedback_section}" if feedback_section else "",
        )

        # Add client profile context
        if client_profile:
            profile_parts = []
            for key in ("industry", "headcount", "revenue", "company_name", "persona"):
                if client_profile.get(key):
                    profile_parts.append(f"{key.replace('_', ' ').title()}: {client_profile[key]}")
            demo = client_profile.get("demographic", {})
            if isinstance(demo, dict):
                for key in ("age_range", "income_bracket", "occupation", "location"):
                    if demo.get(key):
                        profile_parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
            if profile_parts:
                system_prompt += "\n\nClient profile:\n" + "\n".join(profile_parts)

        system_prompt += f"\n\nMaximum personas to produce: {max_personas}"

        # Build the ReAct agent
        try:
            from langgraph.prebuilt import create_react_agent
            llm = get_llm("persona")
            agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
        except ImportError:
            logger.warning("create_react_agent not available, falling back to sequential")
            return _fallback_sequential(
                tenant_id, client_id, user_request, client_profile,
                max_personas, tools, feedback_section,
            )

        # Invoke the agent
        agent_input = {"messages": [{"role": "user", "content": user_request}]}

        try:
            result = agent.invoke(agent_input, config={"recursion_limit": 15})
        except Exception as e:
            logger.exception("Persona ReAct agent failed")
            return StepOutput(
                result=[],
                prompt_sent=system_prompt[:500],
                raw_llm_output=f"Agent error: {e}",
                tool_calls=[],
            )

        # Extract the final output and tool call trace
        messages = result.get("messages", [])
        tool_calls_trace = []
        raw_output = ""
        personas = []

        for msg in messages:
            # Track tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_trace.append({
                        "tool": tc.get("name", "unknown"),
                        "args": {k: str(v)[:100] for k, v in tc.get("args", {}).items()},
                    })

            # Get the final AI message content
            if hasattr(msg, "content") and msg.type == "ai" and not hasattr(msg, "tool_calls"):
                raw_output = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Parse personas from the final output
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                if isinstance(parsed, list):
                    personas = parsed
                elif isinstance(parsed, dict) and "personas" in parsed:
                    personas = parsed["personas"]
            except json.JSONDecodeError:
                # Try extracting JSON from markdown
                import re
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                        if isinstance(parsed, list):
                            personas = parsed
                    except json.JSONDecodeError:
                        pass

        return StepOutput(
            result=personas,
            prompt_sent=system_prompt[:500] + "...",
            raw_llm_output=raw_output[:2000] if raw_output else "(no output)",
            tool_calls=tool_calls_trace,
        )

    # Run through harness
    harness_result = run_with_harness(
        step_fn,
        {
            "request": user_request,
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_profile": client_profile or {},
        },
        PERSONA_STEP_CONFIG,
    )

    from datetime import datetime, timezone

    personas = harness_result.output if isinstance(harness_result.output, list) else []

    # Count tool calls across all attempts
    total_tool_calls = 0
    transcripts_checked = 0
    for attempt in harness_result.attempt_traces:
        trace_data = attempt.to_dict()
        exec_trace = trace_data.get("execution_trace", {})
        calls = exec_trace.get("tool_calls", [])
        total_tool_calls += len(calls)
        transcripts_checked += sum(
            1 for c in calls if c.get("tool") in ("count_transcripts", "get_transcripts")
        )

    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "request": user_request,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "personas": personas,
        "metadata": {
            "context_sampled": sum(
                len(t.to_dict().get("execution_trace", {}).get("tool_calls", []))
                for t in harness_result.attempt_traces
                if any(c.get("tool") == "search_kb" for c in t.to_dict().get("execution_trace", {}).get("tool_calls", []))
            ),
            "transcripts_checked": transcripts_checked,
            "tool_calls_used": total_tool_calls,
            "harness_score": harness_result.final_score,
            "harness_status": harness_result.status.value,
        },
        "status": harness_result.status.value,
        "error": None if harness_result.status.value == "passed" else "Persona extraction did not meet quality threshold.",
    }


def _fallback_sequential(
    tenant_id: str,
    client_id: str,
    user_request: str,
    client_profile: Optional[Dict[str, Any]],
    max_personas: int,
    tools: list,
    feedback_section: str,
) -> StepOutput:
    """Fallback if create_react_agent is not available — call tools sequentially."""
    from langchain_core.output_parsers import StrOutputParser
    from app.prompts.persona_prompts import PERSONA_EXTRACTION_PROMPT
    from app.analysis.base import BaseAnalysisService

    tool_calls_trace = []

    # Search KB
    search_tool = tools[0]  # search_kb
    all_content = []
    queries = [
        "customers audience demographics target market",
        "pain points challenges frustrations needs",
        "buyer persona consumer behavior decision making",
    ]
    for q in queries:
        results = search_tool.invoke({"query": q, "top_k": 15})
        tool_calls_trace.append({"tool": "search_kb", "args": {"query": q}, "result_summary": f"{len(results)} results"})
        all_content.extend(results)

    if user_request != "Identify the key audience personas for this organization.":
        results = search_tool.invoke({"query": user_request, "top_k": 15})
        tool_calls_trace.append({"tool": "search_kb", "args": {"query": user_request[:50]}, "result_summary": f"{len(results)} results"})
        all_content.extend(results)

    # Get summary
    summary_tool = tools[1]  # get_summary
    summary = summary_tool.invoke({})
    tool_calls_trace.append({"tool": "get_summary", "result_summary": "found" if summary else "none"})

    # Build context
    seen = {}
    for r in all_content:
        nid = r.get("node_id")
        if nid and nid not in seen:
            seen[nid] = r
    sorted_results = sorted(seen.values(), key=lambda r: r.get("similarity_score") or 0.0, reverse=True)[:15]
    context = "\n\n---\n\n".join(f"[Source {i + 1}]\n{r['content']}" for i, r in enumerate(sorted_results) if r.get("content", "").strip())

    summary_section = ""
    if summary:
        summary_section = f"\nContext summary:\nSummary: {summary.get('summary', '')}\nTopics: {', '.join(summary.get('topics', []))}\n\n"

    profile_section = BaseAnalysisService._build_profile_section(client_profile)

    # Generate
    llm = get_llm("persona")
    chain = PERSONA_EXTRACTION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "context": context, "profile_section": profile_section,
        "summary_section": summary_section, "user_request": user_request,
        "max_personas": str(max_personas), "feedback_section": feedback_section,
    })

    # Parse
    import re
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        personas = json.loads(cleaned)
        if not isinstance(personas, list):
            personas = []
    except json.JSONDecodeError:
        personas = []

    return StepOutput(
        result=personas,
        prompt_sent=f"[fallback] Sequential persona extraction with {len(sorted_results)} docs",
        raw_llm_output=raw[:2000],
        tool_calls=tool_calls_trace,
    )

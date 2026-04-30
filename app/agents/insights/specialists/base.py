"""Shared types + ReAct-specialist runner for the insights pipeline."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.harness_pkg import run_with_harness, StepOutput
from app.harness_pkg.engine import StepConfig
from app.llm_config import get_llm

logger = logging.getLogger(__name__)


@dataclass
class SpecialistOutput:
    """Uniform return shape for all specialists."""
    name: str
    result: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    harness_score: Optional[float] = None
    status: str = "complete"  # complete | partial | failed | skipped
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "result": self.result,
            "tool_calls": self.tool_calls,
            "harness_score": self.harness_score,
            "status": self.status,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def _parse_json_loose(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m2 = re.search(r"(\{[\s\S]*\})", text)
        if m2:
            try:
                return json.loads(m2.group(1))
            except json.JSONDecodeError:
                return None
    return None


def _build_revision_section(revision_feedback: Optional[str]) -> str:
    if not revision_feedback:
        return ""
    return (
        "\n\nREVISION GUIDANCE (address these issues from the critic):\n"
        f"{revision_feedback}\n"
    )


def run_react_specialist(
    *,
    name: str,
    tools: list,
    system_prompt_template: str,
    plan_focus: str,
    revision_feedback: Optional[str],
    user_message: str,
    step_config: StepConfig,
    recursion_limit: int = 8,
    empty_result: Optional[Dict[str, Any]] = None,
) -> SpecialistOutput:
    """Run a ReAct specialist wrapped in the harness.

    The specialist's system prompt is rendered with `plan_focus` and a `revision_section`,
    invoked via langgraph's create_react_agent with the given tools, and parsed as JSON.
    """
    started = time.monotonic()
    empty = empty_result if empty_result is not None else {}

    def step_fn(inputs: dict, feedback_section: str):
        revision_section = _build_revision_section(revision_feedback)
        if feedback_section:
            revision_section += f"\n\nADDITIONAL FEEDBACK (cheap-check loop):\n{feedback_section}"
        system_prompt = system_prompt_template.format(
            plan_focus=plan_focus or "(no specific focus — use general scope)",
            revision_section=revision_section,
        )

        try:
            from langgraph.prebuilt import create_react_agent
            llm = get_llm("insights_specialist")
            agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
        except ImportError as e:
            logger.warning("create_react_agent not available for %s: %s", name, e)
            return StepOutput(result=dict(empty), prompt_sent=system_prompt[:500],
                              raw_llm_output=f"langgraph unavailable: {e}", tool_calls=[])

        agent_input = {"messages": [{"role": "user", "content": user_message}]}
        try:
            result = agent.invoke(agent_input, config={"recursion_limit": recursion_limit})
        except Exception as e:
            logger.exception("Specialist %s ReAct invocation failed", name)
            return StepOutput(result=dict(empty), prompt_sent=system_prompt[:500],
                              raw_llm_output=f"agent error: {e}", tool_calls=[])

        messages = result.get("messages", [])
        tool_calls_trace: list[dict] = []
        raw_output = ""
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_trace.append({
                        "tool": tc.get("name", "unknown"),
                        "args": {k: str(v)[:100] for k, v in tc.get("args", {}).items()},
                    })
            if hasattr(msg, "content") and getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
                raw_output = msg.content if isinstance(msg.content, str) else str(msg.content)

        parsed = _parse_json_loose(raw_output) or dict(empty)
        if not isinstance(parsed, dict):
            parsed = dict(empty)

        return StepOutput(
            result=parsed,
            prompt_sent=system_prompt[:500] + "...",
            raw_llm_output=raw_output[:3000] if raw_output else "(no output)",
            tool_calls=tool_calls_trace,
        )

    try:
        harness_result = run_with_harness(step_fn, {"request": user_message}, step_config)
        result = harness_result.output if isinstance(harness_result.output, dict) else dict(empty)

        # Aggregate tool calls across attempts.
        all_tool_calls: list[dict] = []
        for attempt in harness_result.attempt_traces:
            calls = attempt.to_dict().get("execution_trace", {}).get("tool_calls", []) or []
            all_tool_calls.extend(calls)

        status = "complete" if harness_result.status.value == "passed" else "partial"
        return SpecialistOutput(
            name=name,
            result=result,
            tool_calls=all_tool_calls,
            harness_score=harness_result.final_score,
            status=status,
            error=None if status == "complete" else "below threshold or cheap-check failed",
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
    except Exception as e:
        logger.exception("Specialist %s run failed", name)
        return SpecialistOutput(
            name=name,
            result=dict(empty),
            tool_calls=[],
            harness_score=None,
            status="failed",
            error=str(e),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )


def run_single_shot_specialist(
    *,
    name: str,
    runner: Callable[[], SpecialistOutput],
) -> SpecialistOutput:
    """Wrap a deterministic specialist runner with timing + error isolation."""
    started = time.monotonic()
    try:
        out = runner()
        out.elapsed_ms = (time.monotonic() - started) * 1000
        return out
    except Exception as e:
        logger.exception("Single-shot specialist %s failed", name)
        return SpecialistOutput(
            name=name,
            result={},
            tool_calls=[],
            harness_score=None,
            status="failed",
            error=str(e),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

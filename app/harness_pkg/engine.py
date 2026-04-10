"""
Agent harness — wraps LLM generation steps with two-tier validation and
feedback-driven retries.

Tier 1: Cheap structural check (no LLM call).
Tier 2: Optional LLM manager evaluation with weighted rubric scoring.

Each step defines a rubric — a list of named dimensions with weights.
The manager scores each dimension individually (0.0-1.0), and the harness
computes a deterministic weighted composite score.

Execution traces capture the full LLM I/O (prompts sent, raw responses,
tool calls) following Meta-Harness methodology — no compression, the
optimizer reads raw traces for counterfactual diagnosis.

Usage
-----
    from app.harness_pkg import run_with_harness, StepConfig, CheapCheckResult, RubricDimension

    config = StepConfig(
        name="survey_generation",
        cheap_check=my_cheap_check,
        manager_prompt=SURVEY_MANAGER_PROMPT,
        manager_context_builder=build_survey_context,
        rubric=[
            RubricDimension("relevance", 0.30, "Does output address the request?"),
            RubricDimension("context_usage", 0.25, "Does it use provided context?"),
        ],
    )
    result = run_with_harness(step_fn, inputs, config)
    output = result.output
    print(result.status)  # "passed" | "exhausted" | "failed"
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, NamedTuple

from pydantic import BaseModel, Field

from app.llm_config import get_llm
from app.harness_pkg import store as harness_store

logger = logging.getLogger(__name__)


# ── Status ──────────────────────────────────────────────────────────────────


class HarnessStatus(str, Enum):
    PASSED = "passed"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


# ── Types ───────────────────────────────────────────────────────────────────


class CheapCheckResult(NamedTuple):
    passed: bool
    feedback: str = ""


@dataclass
class RubricDimension:
    name: str
    weight: float
    description: str


class DimensionScore(BaseModel):
    dimension: str = Field(description="Name of the dimension being scored")
    score: float = Field(description="Score from 0.0 (fails completely) to 1.0 (excellent)")
    reasoning: str = Field(description="Brief explanation for this score")


class ManagerVerdict(BaseModel):
    dimension_scores: list[DimensionScore] = Field(description="One score per rubric dimension")
    overall_feedback: str = Field(
        description="Specific critique if quality is low. Focus on the weakest dimensions. "
        "Empty string if all dimensions pass."
    )


@dataclass
class StepConfig:
    name: str
    cheap_check: Callable[[Any], CheapCheckResult]
    manager_prompt: str
    manager_context_builder: Callable[[Any, dict], str]
    rubric: list[RubricDimension] = field(default_factory=list)
    score_threshold: float = 0.7
    max_retries: int = 2
    use_manager: bool = True


# ── Step output (for capturing raw LLM I/O) ────────────────────────────────


@dataclass
class StepOutput:
    """Return this from step_fn to capture execution traces.

    If step_fn returns a plain value, the harness wraps it automatically.
    If step_fn returns a StepOutput, the harness captures the trace fields.

    Usage in step_fn:
        def step_fn(inputs, feedback_section):
            prompt = build_prompt(...)
            raw = llm.invoke(prompt)
            parsed = parse(raw)
            return StepOutput(
                result=parsed,
                prompt_sent=str(prompt),
                raw_llm_output=raw,
                tool_calls=[{"tool": "search_kb", "args": {...}, "result_summary": "5 docs"}],
            )
    """
    result: Any
    prompt_sent: str = ""
    raw_llm_output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_name: str = ""


# ── Attempt trace ───────────────────────────────────────────────────────────


@dataclass
class AttemptTrace:
    """Record of a single attempt — includes full LLM I/O for Meta-Harness diagnosis."""
    attempt: int
    started_at: str

    # Timing
    step_latency_ms: float | None = None
    manager_latency_ms: float | None = None

    # Cheap check
    cheap_check_passed: bool | None = None
    cheap_check_feedback: str = ""

    # Manager eval
    manager_score: float | None = None
    dimension_breakdown: dict[str, float] = field(default_factory=dict)
    manager_feedback: str = ""

    # Errors
    error: str | None = None
    outcome: str = ""

    # ── Execution trace (Meta-Harness fields) ───────────────────────────
    # These capture the full LLM I/O so the optimizer can do counterfactual
    # diagnosis on raw prompts/responses, not just scores.

    prompt_sent: str = ""                          # assembled prompt → LLM
    raw_llm_output: str = ""                       # raw LLM response before parsing
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # tools invoked during this step
    model_name: str = ""                           # which model was used

    # Manager I/O (separate from the generation step)
    manager_prompt_sent: str = ""                   # system + human messages → manager LLM
    manager_raw_response: str = ""                  # raw manager LLM response

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "attempt": self.attempt,
            "started_at": self.started_at,
            "outcome": self.outcome,
        }
        if self.step_latency_ms is not None:
            d["step_latency_ms"] = round(self.step_latency_ms, 1)
        if self.manager_latency_ms is not None:
            d["manager_latency_ms"] = round(self.manager_latency_ms, 1)
        if self.cheap_check_passed is not None:
            d["cheap_check_passed"] = self.cheap_check_passed
        if self.cheap_check_feedback:
            d["cheap_check_feedback"] = self.cheap_check_feedback
        if self.manager_score is not None:
            d["manager_score"] = round(self.manager_score, 3)
            d["dimension_breakdown"] = {
                k: round(v, 3) for k, v in self.dimension_breakdown.items()
            }
        if self.manager_feedback:
            d["manager_feedback"] = self.manager_feedback
        if self.error:
            d["error"] = self.error

        # Execution trace — always include (even if empty, so optimizer sees the structure)
        d["execution_trace"] = {
            "prompt_sent": self.prompt_sent,
            "raw_llm_output": self.raw_llm_output,
            "tool_calls": self.tool_calls,
            "model_name": self.model_name,
        }

        # Manager I/O
        if self.manager_prompt_sent or self.manager_raw_response:
            d["manager_trace"] = {
                "prompt_sent": self.manager_prompt_sent,
                "raw_response": self.manager_raw_response,
            }

        return d


# ── Harness result (this IS the job record) ────────────────────────────────


@dataclass
class HarnessResult:
    """Full job record for a harness run. Persisted to Redis + Supabase."""
    output: Any

    # Job identity
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str = ""

    # Tenant scope
    tenant_id: str = ""
    client_id: str = ""

    # Status
    status: HarnessStatus = HarnessStatus.FAILED

    # Timing
    started_at: str = ""
    completed_at: str = ""
    total_latency_ms: float = 0.0

    # Summary
    attempts: int = 0
    final_score: float | None = None
    cheap_check_failures: int = 0
    used_manager: bool = False

    # Context for optimizer
    inputs_snapshot: dict[str, Any] = field(default_factory=dict)
    genome_version: int | None = None

    # Per-attempt detail
    attempt_traces: list[AttemptTrace] = field(default_factory=list)

    # Legacy compat
    scores: list[float] = field(default_factory=list)
    dimension_breakdowns: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "step_name": self.step_name,
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "attempts": self.attempts,
            "final_score": round(self.final_score, 3) if self.final_score is not None else None,
            "cheap_check_failures": self.cheap_check_failures,
            "used_manager": self.used_manager,
            "inputs_snapshot": self.inputs_snapshot,
            "genome_version": self.genome_version,
            "attempt_traces": [t.to_dict() for t in self.attempt_traces],
        }


# ── Rubric helpers ──────────────────────────────────────────────────────────


def _build_rubric_instruction(rubric: list[RubricDimension]) -> str:
    if not rubric:
        return ""
    lines = ["\n\nScore each of the following dimensions from 0.0 to 1.0:"]
    for dim in rubric:
        pct = int(dim.weight * 100)
        lines.append(f"- **{dim.name}** ({pct}% weight): {dim.description}")
    lines.append(
        "\nReturn a score for EVERY dimension listed above. "
        "Be calibrated: 0.5 = mediocre, 0.7 = acceptable, 0.9 = strong."
    )
    return "\n".join(lines)


def _compute_weighted_score(
    verdict: ManagerVerdict,
    rubric: list[RubricDimension],
) -> tuple[float, dict[str, float]]:
    score_map: dict[str, float] = {}
    for ds in verdict.dimension_scores:
        score_map[ds.dimension] = max(0.0, min(1.0, ds.score))
    total = 0.0
    weight_sum = 0.0
    breakdown: dict[str, float] = {}
    for dim in rubric:
        dim_score = score_map.get(dim.name, 0.5)
        breakdown[dim.name] = dim_score
        total += dim_score * dim.weight
        weight_sum += dim.weight
    composite = total / weight_sum if weight_sum > 0 else 0.5
    return composite, breakdown


def _format_score_breakdown(
    composite: float,
    breakdown: dict[str, float],
    rubric: list[RubricDimension],
) -> str:
    parts = [f"composite={composite:.2f}"]
    for dim in rubric:
        if dim.name in breakdown:
            parts.append(f"{dim.name}={breakdown[dim.name]:.2f}")
    return " | ".join(parts)


# ── Feedback formatting ─────────────────────────────────────────────────────


def _format_feedback(feedback: str) -> str:
    if not feedback:
        return ""
    return (
        "\n\nREVISION GUIDANCE (address these issues):\n"
        f"{feedback}"
    )


def _format_manager_feedback(
    verdict: ManagerVerdict,
    composite: float,
    breakdown: dict[str, float],
    rubric: list[RubricDimension],
) -> str:
    parts = []
    weak_dims = [dim for dim in rubric if breakdown.get(dim.name, 1.0) < 0.6]
    if weak_dims:
        parts.append("Weak dimensions:")
        for dim in weak_dims:
            score = breakdown.get(dim.name, 0.0)
            reasoning = ""
            for ds in verdict.dimension_scores:
                if ds.dimension == dim.name:
                    reasoning = ds.reasoning
                    break
            parts.append(f"- {dim.name} ({score:.1f}/1.0): {reasoning}")
    if verdict.overall_feedback:
        parts.append(f"\n{verdict.overall_feedback}")
    return "\n".join(parts) if parts else verdict.overall_feedback


# ── Manager evaluation ──────────────────────────────────────────────────────


@dataclass
class _ManagerEvalResult:
    verdict: ManagerVerdict
    composite: float
    breakdown: dict[str, float]
    latency_ms: float
    prompt_sent: str
    raw_response: str


def _run_manager_eval(
    output: Any,
    inputs: dict,
    config: StepConfig,
) -> _ManagerEvalResult:
    llm = get_llm("manager")
    structured_llm = llm.with_structured_output(ManagerVerdict)
    system_prompt = config.manager_prompt + _build_rubric_instruction(config.rubric)
    human_message = config.manager_context_builder(output, inputs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": human_message},
    ]

    # Capture the prompt for the trace
    prompt_text = f"[system]\n{system_prompt}\n\n[human]\n{human_message}"

    t0 = time.monotonic()
    try:
        verdict = structured_llm.invoke(messages)
        raw_response = verdict.model_dump_json()
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        logger.warning("Manager eval failed for %s: %s — passing by default", config.name, e)
        verdict = ManagerVerdict(dimension_scores=[], overall_feedback="")
        return _ManagerEvalResult(
            verdict=verdict, composite=1.0, breakdown={},
            latency_ms=latency, prompt_sent=prompt_text, raw_response=f"ERROR: {e}",
        )

    latency = (time.monotonic() - t0) * 1000
    composite, breakdown = _compute_weighted_score(verdict, config.rubric)
    return _ManagerEvalResult(
        verdict=verdict, composite=composite, breakdown=breakdown,
        latency_ms=latency, prompt_sent=prompt_text, raw_response=raw_response,
    )


# ── Main harness ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_with_harness(
    step_fn: Callable[[dict, str], Any],
    inputs: dict,
    config: StepConfig,
) -> HarnessResult:
    """Run a step function through the harness with validation and retries.

    step_fn can return either:
      - A plain value (the parsed output)
      - A StepOutput instance (parsed output + raw LLM I/O for tracing)

    The HarnessResult IS the job record — persisted to Redis + Supabase JSONL.
    """
    harness_start = time.monotonic()
    trace = HarnessResult(
        output=None,
        step_name=config.name,
        tenant_id=inputs.get("tenant_id", ""),
        client_id=inputs.get("client_id", ""),
        started_at=_now_iso(),
        inputs_snapshot=inputs,
    )

    best_output = None
    best_score: float = -1.0
    feedback_section = ""

    total_attempts = 1 + config.max_retries

    for attempt_num in range(1, total_attempts + 1):
        trace.attempts = attempt_num
        attempt = AttemptTrace(
            attempt=attempt_num,
            started_at=_now_iso(),
        )

        # ── Run the step ────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            raw_result = step_fn(inputs, feedback_section)
        except Exception as e:
            attempt.step_latency_ms = (time.monotonic() - t0) * 1000
            attempt.error = str(e)
            attempt.outcome = "error"
            trace.attempt_traces.append(attempt)

            logger.error("Harness: %s step_fn failed on attempt %d: %s", config.name, attempt_num, e)
            feedback_section = _format_feedback(f"Previous attempt raised an error: {e}")
            continue

        attempt.step_latency_ms = (time.monotonic() - t0) * 1000

        # ── Unwrap StepOutput if returned ───────────────────────────────
        if isinstance(raw_result, StepOutput):
            output = raw_result.result
            attempt.prompt_sent = raw_result.prompt_sent
            attempt.raw_llm_output = raw_result.raw_llm_output
            attempt.tool_calls = raw_result.tool_calls
            attempt.model_name = raw_result.model_name
        else:
            output = raw_result

        # ── Tier 1: Cheap check ─────────────────────────────────────────
        check = config.cheap_check(output)
        attempt.cheap_check_passed = check.passed

        if not check.passed:
            trace.cheap_check_failures += 1
            attempt.cheap_check_feedback = check.feedback
            attempt.outcome = "cheap_check_failed"
            trace.attempt_traces.append(attempt)

            logger.info(
                "Harness: %s cheap check failed (attempt %d/%d): %s",
                config.name, attempt_num, total_attempts, check.feedback,
            )
            feedback_section = _format_feedback(check.feedback)

            if best_output is None:
                best_output = output
            continue

        # ── Tier 2: Manager eval (optional) ─────────────────────────────
        if config.use_manager and config.rubric:
            trace.used_manager = True
            mgr = _run_manager_eval(output, inputs, config)

            attempt.manager_latency_ms = mgr.latency_ms
            attempt.manager_score = mgr.composite
            attempt.dimension_breakdown = mgr.breakdown
            attempt.manager_feedback = mgr.verdict.overall_feedback
            attempt.manager_prompt_sent = mgr.prompt_sent
            attempt.manager_raw_response = mgr.raw_response

            trace.scores.append(mgr.composite)
            trace.dimension_breakdowns.append(mgr.breakdown)

            logger.info(
                "Harness: %s manager eval (attempt %d/%d): %s",
                config.name, attempt_num, total_attempts,
                _format_score_breakdown(mgr.composite, mgr.breakdown, config.rubric),
            )

            if mgr.composite > best_score:
                best_score = mgr.composite
                best_output = output

            if mgr.composite >= config.score_threshold:
                attempt.outcome = "passed"
                trace.attempt_traces.append(attempt)

                trace.output = output
                trace.final_score = mgr.composite
                trace.status = HarnessStatus.PASSED
                trace.completed_at = _now_iso()
                trace.total_latency_ms = (time.monotonic() - harness_start) * 1000

                logger.info(
                    "Harness: %s passed on attempt %d [job=%s, %.0fms]",
                    config.name, attempt_num, trace.job_id, trace.total_latency_ms,
                )
                harness_store.record_trace(trace)
                return trace

            # Below threshold — retry
            attempt.outcome = "below_threshold"
            trace.attempt_traces.append(attempt)

            feedback_section = _format_feedback(
                _format_manager_feedback(mgr.verdict, mgr.composite, mgr.breakdown, config.rubric)
            )
        else:
            # No manager — cheap check passed, we're done
            attempt.outcome = "passed"
            trace.attempt_traces.append(attempt)

            trace.output = output
            trace.status = HarnessStatus.PASSED
            trace.completed_at = _now_iso()
            trace.total_latency_ms = (time.monotonic() - harness_start) * 1000
            harness_store.record_trace(trace)
            return trace

    # ── Exhausted retries ───────────────────────────────────────────────
    trace.completed_at = _now_iso()
    trace.total_latency_ms = (time.monotonic() - harness_start) * 1000

    if best_output is not None:
        trace.output = best_output
        trace.final_score = best_score if best_score >= 0 else None
        trace.status = HarnessStatus.EXHAUSTED
        logger.warning(
            "Harness: %s exhausted %d attempts. Best score: %.2f [job=%s, %.0fms]",
            config.name, total_attempts, best_score, trace.job_id, trace.total_latency_ms,
        )
    else:
        trace.status = HarnessStatus.FAILED
        logger.error(
            "Harness: %s failed on all %d attempts — no usable output [job=%s, %.0fms]",
            config.name, total_attempts, trace.job_id, trace.total_latency_ms,
        )

    harness_store.record_trace(trace)
    return trace

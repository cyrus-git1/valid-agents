"""
Agent harness — wraps LLM generation steps with two-tier validation and
feedback-driven retries.

Tier 1: Cheap structural check (no LLM call).
Tier 2: Optional LLM manager evaluation with weighted rubric scoring.

Each step defines a rubric — a list of named dimensions with weights.
The manager scores each dimension individually (0.0-1.0), and the harness
computes a deterministic weighted composite score. This means the threshold
is grounded and consistent across steps.

Usage
-----
    from app.harness import run_with_harness, StepConfig, CheapCheckResult, RubricDimension

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

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, NamedTuple

from pydantic import BaseModel, Field

from app.llm_config import get_llm
from app import harness_store

logger = logging.getLogger(__name__)


# ── Status ──────────────────────────────────────────────────────────────────


class HarnessStatus(str, Enum):
    """Terminal status of a harness run."""
    PASSED = "passed"             # output met threshold on some attempt
    EXHAUSTED = "exhausted"       # all retries used, returning best output
    FAILED = "failed"             # step_fn errored on every attempt, no usable output


# ── Types ───────────────────────────────────────────────────────────────────


class CheapCheckResult(NamedTuple):
    passed: bool
    feedback: str = ""


@dataclass
class RubricDimension:
    """A single scoring dimension in the evaluation rubric."""
    name: str
    weight: float
    description: str


class DimensionScore(BaseModel):
    """Score for a single rubric dimension."""
    dimension: str = Field(description="Name of the dimension being scored")
    score: float = Field(description="Score from 0.0 (fails completely) to 1.0 (excellent)")
    reasoning: str = Field(description="Brief explanation for this score")


class ManagerVerdict(BaseModel):
    """Structured output from the manager LLM — scores per rubric dimension."""
    dimension_scores: list[DimensionScore] = Field(
        description="One score per rubric dimension"
    )
    overall_feedback: str = Field(
        description="Specific critique if quality is low. Focus on the weakest dimensions. "
        "Empty string if all dimensions pass."
    )


@dataclass
class StepConfig:
    """Configuration for a single harnessed step."""
    name: str
    cheap_check: Callable[[Any], CheapCheckResult]
    manager_prompt: str
    manager_context_builder: Callable[[Any, dict], str]
    rubric: list[RubricDimension] = field(default_factory=list)
    score_threshold: float = 0.7
    max_retries: int = 2
    use_manager: bool = True


# ── Attempt trace ───────────────────────────────────────────────────────────


@dataclass
class AttemptTrace:
    """Record of a single attempt within a harness run."""
    attempt: int
    started_at: str                                # ISO 8601
    step_latency_ms: float | None = None           # step_fn wall time
    manager_latency_ms: float | None = None        # manager eval wall time
    cheap_check_passed: bool | None = None
    cheap_check_feedback: str = ""
    manager_score: float | None = None
    dimension_breakdown: dict[str, float] = field(default_factory=dict)
    manager_feedback: str = ""
    error: str | None = None                       # step_fn exception message
    outcome: str = ""                              # "passed" | "cheap_check_failed" | "below_threshold" | "error"

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
        return d


# ── Harness result ──────────────────────────────────────────────────────────


@dataclass
class HarnessResult:
    """Wraps step output with full job trace information."""
    output: Any

    # Job identity
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str = ""

    # Status
    status: HarnessStatus = HarnessStatus.FAILED

    # Timing
    started_at: str = ""                           # ISO 8601
    completed_at: str = ""                         # ISO 8601
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
        """Serialize to a dict for logging / API responses."""
        return {
            "job_id": self.job_id,
            "step_name": self.step_name,
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
    """Build the rubric instruction block for the manager prompt."""
    if not rubric:
        return ""

    lines = [
        "\n\nScore each of the following dimensions from 0.0 to 1.0:",
    ]
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
    """Compute weighted composite score from dimension scores.

    Returns (composite_score, {dimension_name: score}).
    """
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
    """Format score breakdown for logging."""
    parts = [f"composite={composite:.2f}"]
    for dim in rubric:
        if dim.name in breakdown:
            parts.append(f"{dim.name}={breakdown[dim.name]:.2f}")
    return " | ".join(parts)


# ── Feedback formatting ─────────────────────────────────────────────────────


def _format_feedback(feedback: str) -> str:
    """Format feedback for injection into the prompt's {feedback_section}."""
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
    """Build detailed feedback from the manager verdict for retry injection."""
    parts = []

    weak_dims = [
        dim for dim in rubric
        if breakdown.get(dim.name, 1.0) < 0.6
    ]
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


def _run_manager_eval(
    output: Any,
    inputs: dict,
    config: StepConfig,
) -> tuple[ManagerVerdict, float, dict[str, float], float]:
    """Run the LLM manager evaluation on a step's output.

    Returns (verdict, composite_score, dimension_breakdown, latency_ms).
    """
    llm = get_llm("manager")
    structured_llm = llm.with_structured_output(ManagerVerdict)

    system_prompt = config.manager_prompt + _build_rubric_instruction(config.rubric)
    human_message = config.manager_context_builder(output, inputs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": human_message},
    ]

    t0 = time.monotonic()
    try:
        verdict = structured_llm.invoke(messages)
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        logger.warning("Manager eval failed for %s: %s — passing by default", config.name, e)
        verdict = ManagerVerdict(dimension_scores=[], overall_feedback="")
        return verdict, 1.0, {}, latency

    latency = (time.monotonic() - t0) * 1000
    composite, breakdown = _compute_weighted_score(verdict, config.rubric)
    return verdict, composite, breakdown, latency


# ── Main harness ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_with_harness(
    step_fn: Callable[[dict, str], Any],
    inputs: dict,
    config: StepConfig,
) -> HarnessResult:
    """Run a step function through the harness with validation and retries.

    Parameters
    ----------
    step_fn : callable
        Signature: (inputs: dict, feedback_section: str) -> Any
        Called with feedback_section="" on first attempt, populated on retries.
    inputs : dict
        The inputs to pass to step_fn.
    config : StepConfig
        Step-specific configuration (cheap check, manager prompt, rubric, thresholds).

    Returns
    -------
    HarnessResult
        The best output across all attempts, with full job trace.
    """
    harness_start = time.monotonic()
    trace = HarnessResult(
        output=None,
        step_name=config.name,
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
            output = step_fn(inputs, feedback_section)
        except Exception as e:
            attempt.step_latency_ms = (time.monotonic() - t0) * 1000
            attempt.error = str(e)
            attempt.outcome = "error"
            trace.attempt_traces.append(attempt)

            logger.error("Harness: %s step_fn failed on attempt %d: %s", config.name, attempt_num, e)
            feedback_section = _format_feedback(f"Previous attempt raised an error: {e}")
            continue

        attempt.step_latency_ms = (time.monotonic() - t0) * 1000

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
            verdict, composite, breakdown, mgr_latency = _run_manager_eval(output, inputs, config)

            attempt.manager_latency_ms = mgr_latency
            attempt.manager_score = composite
            attempt.dimension_breakdown = breakdown
            attempt.manager_feedback = verdict.overall_feedback

            trace.scores.append(composite)
            trace.dimension_breakdowns.append(breakdown)

            logger.info(
                "Harness: %s manager eval (attempt %d/%d): %s",
                config.name, attempt_num, total_attempts,
                _format_score_breakdown(composite, breakdown, config.rubric),
            )

            if composite > best_score:
                best_score = composite
                best_output = output

            if composite >= config.score_threshold:
                attempt.outcome = "passed"
                trace.attempt_traces.append(attempt)

                trace.output = output
                trace.final_score = composite
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
                _format_manager_feedback(verdict, composite, breakdown, config.rubric)
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

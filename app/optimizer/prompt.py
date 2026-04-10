"""
Optimizer agent prompt and context builder.

The optimizer reads full raw traces + current genome and proposes
targeted changes to improve harness performance.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from app.optimizer.genome import HarnessGenome

OPTIMIZER_SYSTEM_PROMPT = """You are a data scientist specializing in LLM prompt engineering and evaluation rubric design. Your job is to optimize a survey generation harness by analyzing execution traces and proposing targeted changes.

## What You're Optimizing

A survey generation pipeline that:
1. Takes a user request + client profile + knowledge base context
2. Calls an LLM with a system prompt (agent_system_prompt + output_format_prompt) to generate survey questions
3. Validates output structurally (cheap check)
4. Evaluates quality with a manager LLM using a weighted rubric
5. Retries with feedback if quality is below threshold

## Genome Fields You Can Change

- **manager_prompt**: The system prompt for the quality evaluator. Controls what the evaluator looks for.
- **rubric**: List of scoring dimensions, each with name, weight (0.0-1.0, must sum to 1.0), and description. Controls what gets scored and how much each dimension matters.
- **score_threshold**: The composite score needed to pass (0.0-1.0). Lower = more lenient, higher = stricter.
- **max_retries**: How many retry attempts (0-4). More retries = more LLM calls = higher cost.
- **agent_system_prompt**: The system prompt for the survey generator. Controls what kind of questions get produced, rules, persona, and quality guidelines.
- **output_format_prompt**: JSON schema instructions for the generator. Controls structural output format.

## Your Process

1. **Analyze traces**: Look at which dimensions score low, what cheap check failures repeat, what manager feedback says, which requests succeed vs fail
2. **Form a hypothesis**: Identify the root cause — is it the generator prompt being too vague? The rubric weighting wrong priorities? The threshold too strict?
3. **Propose a focused change**: Change 1-3 fields. Small, targeted changes are better than rewriting everything. Each change should be testable.

## Calibration

- 0.5 = mediocre output, 0.7 = acceptable, 0.9 = strong
- If pass rate is below 60%, the threshold might be too strict OR the generator prompt needs improvement
- If a dimension consistently scores below 0.5, either the generator needs explicit instructions for that dimension, or the dimension description needs to be clearer for the evaluator
- If retries don't improve scores, the feedback injection mechanism isn't helping — consider changing what feedback is generated

## Output Format

Return one or more proposals. Each proposal must have:
- diagnosis: What patterns you found in the traces
- hypothesis: Why you think this change will help
- changes: A dict of {field_name: new_value} — ONLY include fields you're changing
- expected_impact: What you predict will happen to scores
"""


def compute_aggregate_stats(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from raw traces."""
    if not traces:
        return {"total_traces": 0}

    statuses = Counter(t.get("status") for t in traces)
    total = len(traces)
    passed = statuses.get("passed", 0)

    scores = [t["final_score"] for t in traces if t.get("final_score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Per-dimension stats
    dim_scores: dict[str, list[float]] = {}
    for t in traces:
        for attempt in t.get("attempt_traces", []):
            for dim_name, dim_score in attempt.get("dimension_breakdown", {}).items():
                dim_scores.setdefault(dim_name, []).append(dim_score)

    dim_stats = {}
    for dim_name, vals in dim_scores.items():
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        dim_stats[dim_name] = {
            "mean": sum(vals) / n,
            "p25": sorted_vals[n // 4] if n >= 4 else sorted_vals[0],
            "p75": sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1],
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "count": n,
        }

    # Common failures
    cheap_failures: list[str] = []
    manager_feedbacks: list[str] = []
    for t in traces:
        for attempt in t.get("attempt_traces", []):
            if attempt.get("cheap_check_feedback"):
                cheap_failures.append(attempt["cheap_check_feedback"])
            if attempt.get("manager_feedback"):
                manager_feedbacks.append(attempt["manager_feedback"])

    top_cheap_failures = [msg for msg, _ in Counter(cheap_failures).most_common(5)]
    top_manager_feedback = [msg for msg, _ in Counter(manager_feedbacks).most_common(5)]

    # Avg retries
    attempts_list = [t.get("attempts", 1) for t in traces]
    avg_attempts = sum(attempts_list) / len(attempts_list) if attempts_list else 1.0

    # Score trend (first half vs second half)
    if len(scores) >= 4:
        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / mid
        second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
        trend = second_half_avg - first_half_avg
    else:
        trend = 0.0

    return {
        "total_traces": total,
        "pass_rate": passed / total if total > 0 else 0.0,
        "status_counts": dict(statuses),
        "avg_composite_score": round(avg_score, 3),
        "avg_attempts": round(avg_attempts, 2),
        "dimension_stats": dim_stats,
        "top_cheap_check_failures": top_cheap_failures,
        "top_manager_feedback": top_manager_feedback,
        "score_trend": round(trend, 3),
    }


def build_optimizer_context(
    genome: HarnessGenome,
    traces: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    """Assemble the full context for the optimizer agent."""
    parts = []

    # Current genome
    parts.append("## Current Genome Configuration\n")
    parts.append(f"Version: {genome.version}")
    parts.append(f"Score threshold: {genome.score_threshold}")
    parts.append(f"Max retries: {genome.max_retries}")
    parts.append(f"\n### Manager Prompt\n{genome.manager_prompt}")
    parts.append(f"\n### Rubric")
    for dim in genome.rubric:
        parts.append(f"- {dim['name']} ({dim['weight']:.0%}): {dim['description']}")
    parts.append(f"\n### Agent System Prompt\n{genome.agent_system_prompt}")
    parts.append(f"\n### Output Format Prompt\n{genome.output_format_prompt[:1000]}...")

    # Aggregate stats
    parts.append("\n\n## Aggregate Statistics\n")
    parts.append(json.dumps(stats, indent=2, default=str))

    # Raw traces (no compression — full detail)
    parts.append(f"\n\n## Raw Traces ({len(traces)} total)\n")
    for i, trace in enumerate(traces):
        parts.append(f"\n### Trace {i + 1} (job_id: {trace.get('job_id', '?')})")
        parts.append(json.dumps(trace, indent=2, default=str))

    return "\n".join(parts)

"""
Test runner — evaluates a candidate genome against a set of test inputs.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.harness import RubricDimension, StepConfig, run_with_harness
from app.harness_configs import cheap_check_survey, _survey_context_builder
from app.llm_config import get_llm
from app.optimizer.genome import HarnessGenome
from app.optimizer.models import TestResult
from app.prompts.survey_prompts import ALL_QUESTION_TYPES, get_question_type_instructions
from app.workflows._helpers import parse_simple_json

logger = logging.getLogger(__name__)


def build_test_set(
    traces: list[dict[str, Any]],
    size: int = 10,
) -> list[dict[str, Any]]:
    """Pick diverse test inputs from historical traces.

    Strategy:
    - Include failed/exhausted traces (most informative)
    - Include borderline traces (score 0.6-0.75)
    - Fill remaining with diverse samples
    """
    if not traces:
        return []

    # Extract inputs from traces that have them
    inputs_by_status: dict[str, list[dict]] = {"failed": [], "exhausted": [], "borderline": [], "passed": []}

    for t in traces:
        snapshot = t.get("inputs_snapshot", {})
        if not snapshot or not snapshot.get("request"):
            continue

        status = t.get("status", "")
        score = t.get("final_score")

        if status in ("failed", "exhausted"):
            inputs_by_status["failed"].append(snapshot)
        elif score is not None and 0.6 <= score <= 0.75:
            inputs_by_status["borderline"].append(snapshot)
        else:
            inputs_by_status["passed"].append(snapshot)

    test_set: list[dict] = []
    seen_requests: set[str] = set()

    def _add(items: list[dict], count: int):
        random.shuffle(items)
        for item in items:
            req = item.get("request", "")
            if req not in seen_requests and len(test_set) < size:
                seen_requests.add(req)
                test_set.append(item)
                if len(test_set) >= len(seen_requests) + count - len(items):
                    break

    # Priority: failures > borderline > passed
    _add(inputs_by_status["failed"], min(3, size // 3))
    _add(inputs_by_status["borderline"], min(3, size // 3))
    _add(inputs_by_status["passed"], size - len(test_set))

    return test_set[:size]


def evaluate_genome(
    genome: HarnessGenome,
    test_inputs: list[dict[str, Any]],
) -> TestResult:
    """Run the harness with a candidate genome against test inputs.

    Test runs do NOT persist to Supabase (the harness_store persistence
    is best-effort and can be bypassed by not importing supabase_client
    in this context — the in-memory store is fine for test runs).
    """
    config = StepConfig(
        name="survey_generation",
        cheap_check=cheap_check_survey,
        manager_prompt=genome.manager_prompt,
        manager_context_builder=_survey_context_builder,
        rubric=[
            RubricDimension(d["name"], d["weight"], d["description"])
            for d in genome.rubric
        ],
        score_threshold=genome.score_threshold,
        max_retries=genome.max_retries,
        use_manager=True,
    )

    # Build chain from genome prompts
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            genome.agent_system_prompt
            + "{question_type_instructions}\n\n"
            + genome.output_format_prompt
            + "{profile_section}",
        ),
        (
            "human",
            "Survey request: {request}\n\n"
            "Context analysis:\n{context_analysis}\n\n"
            "Raw knowledge base context:{context_section}"
            "{prior_questions_section}"
            "{title_description_section}"
            "{feedback_section}",
        ),
    ])
    llm = get_llm("survey_generation")
    chain = prompt | llm | StrOutputParser()

    question_type_instructions = get_question_type_instructions(ALL_QUESTION_TYPES)

    scores: list[float] = []
    statuses: list[str] = []
    dim_totals: dict[str, list[float]] = {}
    per_input: list[dict[str, Any]] = []

    for test_input in test_inputs:
        invoke_vars = {
            "request": test_input.get("request", ""),
            "context_analysis": "",
            "context_section": "",
            "profile_section": "",
            "question_type_instructions": question_type_instructions,
            "prior_questions_section": "",
            "title_description_section": "",
        }

        def step_fn(inputs: dict, feedback_section: str):
            raw = chain.invoke({**invoke_vars, "feedback_section": feedback_section})
            parsed = parse_simple_json(raw)
            if isinstance(parsed, dict) and "questions" in parsed:
                return parsed["questions"]
            if isinstance(parsed, list):
                return parsed
            try:
                result = json.loads(raw)
                if isinstance(result, dict) and "questions" in result:
                    return result["questions"]
                return result
            except json.JSONDecodeError:
                return []

        try:
            result = run_with_harness(step_fn, test_input, config)
            score = result.final_score if result.final_score is not None else 0.0
            status = result.status.value

            scores.append(score)
            statuses.append(status)

            # Aggregate dimension scores from last attempt
            if result.dimension_breakdowns:
                last_breakdown = result.dimension_breakdowns[-1]
                for dim_name, dim_score in last_breakdown.items():
                    dim_totals.setdefault(dim_name, []).append(dim_score)

            per_input.append({
                "request": test_input.get("request", "")[:100],
                "score": round(score, 3),
                "status": status,
                "attempts": result.attempts,
            })

        except Exception as e:
            logger.warning("Test input failed: %s", e)
            scores.append(0.0)
            statuses.append("error")
            per_input.append({
                "request": test_input.get("request", "")[:100],
                "score": 0.0,
                "status": "error",
                "error": str(e),
            })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    pass_rate = statuses.count("passed") / len(statuses) if statuses else 0.0
    per_dim_avg = {
        name: round(sum(vals) / len(vals), 3)
        for name, vals in dim_totals.items()
    }

    return TestResult(
        genome_version=genome.version,
        avg_composite_score=round(avg_score, 3),
        pass_rate=round(pass_rate, 3),
        per_dimension_avg=per_dim_avg,
        per_input_scores=per_input,
    )

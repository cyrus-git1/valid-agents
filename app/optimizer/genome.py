"""
Harness genome — a versionable snapshot of all tunable harness parameters.

The genome captures everything the optimizer can change:
  - Evaluator side: manager_prompt, rubric, score_threshold, max_retries
  - Generator side: agent_system_prompt, output_format_prompt
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HarnessGenome(BaseModel):
    """Complete tunable configuration for a harness step."""

    # Identity
    version: int
    step_name: str
    parent_version: int | None = None

    # Evaluator side
    manager_prompt: str
    rubric: list[dict[str, Any]] = Field(
        description="List of {name: str, weight: float, description: str}"
    )
    score_threshold: float = 0.7
    max_retries: int = 2

    # Generator side
    agent_system_prompt: str
    output_format_prompt: str

    # Optimization metadata
    optimization_notes: str = ""
    test_score: float | None = None
    test_details: dict[str, Any] = Field(default_factory=dict)


def baseline_genome(step_name: str) -> HarnessGenome:
    """Build version 0 from the current hardcoded constants.

    Only supports 'survey_generation' for now.
    """
    if step_name != "survey_generation":
        raise ValueError(f"No baseline genome defined for step: {step_name}")

    from app.prompts.harness_prompts import SURVEY_MANAGER_PROMPT
    from app.prompts.survey_prompts import (
        SURVEY_AGENT_SYSTEM_PROMPT,
        SURVEY_OUTPUT_FORMAT_PROMPT,
    )
    from app.harness_configs import SURVEY_STEP_CONFIG

    return HarnessGenome(
        version=0,
        step_name=step_name,
        parent_version=None,
        manager_prompt=SURVEY_MANAGER_PROMPT,
        rubric=[
            {"name": dim.name, "weight": dim.weight, "description": dim.description}
            for dim in SURVEY_STEP_CONFIG.rubric
        ],
        score_threshold=SURVEY_STEP_CONFIG.score_threshold,
        max_retries=SURVEY_STEP_CONFIG.max_retries,
        agent_system_prompt=SURVEY_AGENT_SYSTEM_PROMPT,
        output_format_prompt=SURVEY_OUTPUT_FORMAT_PROMPT,
        optimization_notes="Baseline from hardcoded constants.",
    )

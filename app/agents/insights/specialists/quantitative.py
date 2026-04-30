"""Quantitative specialist — runs NPS / T2B / proportions / CIs / crosstabs."""
from __future__ import annotations

from typing import Any, Dict

from app.agents.insights.specialists.base import (
    SpecialistOutput,
    run_react_specialist,
)
from app.harness_pkg.configs import QUANTITATIVE_SPECIALIST_STEP_CONFIG
from app.prompts.insights_specialist_prompts import QUANTITATIVE_SPECIALIST_PROMPT
from app.tools.insights_tools import create_insights_tools

_KEEP = {"compute_quantitative_metrics", "compute_crosstab", "compute_confidence_intervals"}


def run_quantitative_specialist(inputs: Dict[str, Any]) -> Dict[str, Any]:
    all_tools = create_insights_tools(
        inputs["tenant_id"], inputs["client_id"],
        inputs.get("client_profile"),
        survey_ids=inputs.get("survey_ids"),
        study_id=inputs.get("study_id"),
    )
    tools = [t for t in all_tools if getattr(t, "name", "") in _KEEP]

    user_message = (
        "Compute the most informative quantitative findings on the scoped survey responses. "
        "Run compute_quantitative_metrics first; pick 1-3 informative dimension pairs and "
        "call compute_crosstab; then return ONLY the JSON object specified in the system prompt."
    )

    out: SpecialistOutput = run_react_specialist(
        name="quantitative",
        tools=tools,
        system_prompt_template=QUANTITATIVE_SPECIALIST_PROMPT,
        plan_focus=inputs.get("plan_focus", "") or "",
        revision_feedback=inputs.get("revision_feedback"),
        user_message=user_message,
        step_config=QUANTITATIVE_SPECIALIST_STEP_CONFIG,
        recursion_limit=8,
        empty_result={"quantitative_findings": [], "crosstabs_run": [], "tool_call_summary": []},
    )
    return out.to_dict()

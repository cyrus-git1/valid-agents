"""Qualitative specialist — themes / objections / contradictions / hypotheses."""
from __future__ import annotations

from typing import Any, Dict

from app.agents.insights.specialists.base import (
    SpecialistOutput,
    run_react_specialist,
)
from app.harness_pkg.configs import QUALITATIVE_SPECIALIST_STEP_CONFIG
from app.prompts.insights_specialist_prompts import QUALITATIVE_SPECIALIST_PROMPT
from app.tools.insights_tools import create_insights_tools

_KEEP = {
    "cross_document_synthesis", "extract_objections", "analyze_transcript",
    "generate_hypotheses", "analyze_sentiment", "extract_transcript_insights",
    "get_personas",
}


def run_qualitative_specialist(inputs: Dict[str, Any]) -> Dict[str, Any]:
    all_tools = create_insights_tools(
        inputs["tenant_id"], inputs["client_id"],
        inputs.get("client_profile"),
        survey_ids=inputs.get("survey_ids"),
        study_id=inputs.get("study_id"),
    )
    tools = [t for t in all_tools if getattr(t, "name", "") in _KEEP]

    user_message = (
        "Surface the most informative qualitative findings, objections, contradictions, "
        "and testable hypotheses from the scoped transcripts and documents. Start with "
        "get_personas, then cross_document_synthesis + extract_objections, then "
        "generate_hypotheses. Return ONLY the JSON object specified in the system prompt."
    )

    out: SpecialistOutput = run_react_specialist(
        name="qualitative",
        tools=tools,
        system_prompt_template=QUALITATIVE_SPECIALIST_PROMPT,
        plan_focus=inputs.get("plan_focus", "") or "",
        revision_feedback=inputs.get("revision_feedback"),
        user_message=user_message,
        step_config=QUALITATIVE_SPECIALIST_STEP_CONFIG,
        recursion_limit=10,
        empty_result={
            "qualitative_findings": [],
            "objections_and_blockers": [],
            "contradictions_and_blind_spots": [],
            "hypotheses_to_test": [],
            "personas_referenced": [],
            "tool_call_summary": [],
        },
    )
    return out.to_dict()

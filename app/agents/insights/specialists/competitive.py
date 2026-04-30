"""Competitive specialist — extracts named competitors + win/loss signals."""
from __future__ import annotations

from typing import Any, Dict

from app.agents.insights.specialists.base import (
    SpecialistOutput,
    run_react_specialist,
)
from app.harness_pkg.configs import COMPETITIVE_SPECIALIST_STEP_CONFIG
from app.prompts.insights_specialist_prompts import COMPETITIVE_SPECIALIST_PROMPT
from app.tools.insights_tools import create_insights_tools

_KEEP = {"competitive_intelligence"}


def run_competitive_specialist(inputs: Dict[str, Any]) -> Dict[str, Any]:
    all_tools = create_insights_tools(
        inputs["tenant_id"], inputs["client_id"],
        inputs.get("client_profile"),
        survey_ids=inputs.get("survey_ids"),
        study_id=inputs.get("study_id"),
    )
    tools = [t for t in all_tools if getattr(t, "name", "") in _KEEP]

    user_message = (
        "Extract named competitors with strengths/weaknesses, win/loss signals, and "
        "positioning gaps from the scoped sources. Use competitive_intelligence. "
        "Return ONLY the JSON object specified in the system prompt."
    )

    out: SpecialistOutput = run_react_specialist(
        name="competitive",
        tools=tools,
        system_prompt_template=COMPETITIVE_SPECIALIST_PROMPT,
        plan_focus=inputs.get("plan_focus", "") or "",
        revision_feedback=inputs.get("revision_feedback"),
        user_message=user_message,
        step_config=COMPETITIVE_SPECIALIST_STEP_CONFIG,
        recursion_limit=5,
        empty_result={
            "competitive_landscape": {
                "competitors_mentioned": [],
                "win_signals": [],
                "loss_signals": [],
                "positioning_gaps": [],
            },
            "tool_call_summary": [],
        },
    )
    return out.to_dict()

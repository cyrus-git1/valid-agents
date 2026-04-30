"""Segments specialist — calls analyze_clusters then transforms output to v2 schema."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agents.insights.specialists.base import (
    SpecialistOutput,
    run_single_shot_specialist,
)
from app.llm_config import get_llm
from app.prompts.insights_specialist_prompts import SEGMENTS_TRANSFORMER_PROMPT
from app.tools.insights_tools import create_insights_tools

logger = logging.getLogger(__name__)


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
                return None
    return None


def _runner(inputs: Dict[str, Any]) -> SpecialistOutput:
    all_tools = create_insights_tools(
        inputs["tenant_id"], inputs["client_id"],
        inputs.get("client_profile"),
        survey_ids=inputs.get("survey_ids"),
        study_id=inputs.get("study_id"),
    )
    cluster_tool = next((t for t in all_tools if getattr(t, "name", "") == "analyze_clusters"), None)
    if cluster_tool is None:
        return SpecialistOutput(
            name="segments", result={"segments": []},
            status="failed", error="analyze_clusters tool not available",
        )

    try:
        cluster_result = cluster_tool.invoke({})
    except Exception as e:
        logger.warning("analyze_clusters failed: %s", e)
        return SpecialistOutput(
            name="segments", result={"segments": []},
            tool_calls=[{"tool": "analyze_clusters", "error": str(e)}],
            status="failed", error=str(e),
        )

    tool_calls = [{"tool": "analyze_clusters", "result_summary": "ok"}]

    # Single-shot LLM transform into v2 segments schema
    try:
        llm = get_llm("insights_specialist")
        prompt = ChatPromptTemplate.from_messages(
            [("system", SEGMENTS_TRANSFORMER_PROMPT), ("human", "Produce the segments JSON now.")]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "plan_focus": inputs.get("plan_focus", "") or "(no specific focus)",
            "cluster_result": json.dumps(cluster_result, default=str)[:6000],
        })
        parsed = _parse_json_loose(raw) or {"segments": []}
        if not isinstance(parsed, dict):
            parsed = {"segments": []}
        if not isinstance(parsed.get("segments"), list):
            parsed["segments"] = []
        return SpecialistOutput(
            name="segments", result=parsed, tool_calls=tool_calls, status="complete",
        )
    except Exception as e:
        logger.warning("segments transform failed: %s", e)
        return SpecialistOutput(
            name="segments", result={"segments": []}, tool_calls=tool_calls,
            status="partial", error=str(e),
        )


def run_segments_specialist(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = run_single_shot_specialist(name="segments", runner=lambda: _runner(inputs))
    return out.to_dict()

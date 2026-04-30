"""External specialist — runs the enrichment agent (web search) and transforms output."""
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
from app.prompts.insights_specialist_prompts import EXTERNAL_TRANSFORMER_PROMPT

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
    request = inputs.get("plan_focus") or inputs.get("focus_query")
    try:
        from app.agents.enrichment_agent import run_enrichment_agent
        enrichment_result = run_enrichment_agent(
            tenant_id=inputs["tenant_id"],
            client_id=inputs["client_id"],
            request=request,
            client_profile=inputs.get("client_profile"),
            max_sources=3,
        )
    except Exception as e:
        logger.warning("enrichment agent failed: %s", e)
        return SpecialistOutput(
            name="external",
            result={
                "external_context_via_enrichment": {
                    "summary": "", "key_external_facts": [], "trends_to_watch": [],
                },
                "enrichment_recommendations": [],
            },
            tool_calls=[{"tool": "recommend_enrichment", "error": str(e)}],
            status="failed", error=str(e),
        )

    tool_calls = [{"tool": "recommend_enrichment", "result_summary": "ok"}]

    try:
        llm = get_llm("insights_specialist")
        prompt = ChatPromptTemplate.from_messages(
            [("system", EXTERNAL_TRANSFORMER_PROMPT),
             ("human", "Produce the external context JSON now.")]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "plan_focus": inputs.get("plan_focus", "") or "(no specific focus)",
            "enrichment_result": json.dumps(enrichment_result, default=str)[:6000],
        })
        parsed = _parse_json_loose(raw) or {}
        if not isinstance(parsed, dict):
            parsed = {}
        parsed.setdefault("external_context_via_enrichment", {
            "summary": "", "key_external_facts": [], "trends_to_watch": [],
        })
        parsed.setdefault("enrichment_recommendations", [])
        return SpecialistOutput(
            name="external", result=parsed, tool_calls=tool_calls, status="complete",
        )
    except Exception as e:
        logger.warning("external transform failed: %s", e)
        return SpecialistOutput(
            name="external",
            result={
                "external_context_via_enrichment": {
                    "summary": "", "key_external_facts": [], "trends_to_watch": [],
                },
                "enrichment_recommendations": [],
            },
            tool_calls=tool_calls,
            status="partial", error=str(e),
        )


def run_external_specialist(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = run_single_shot_specialist(name="external", runner=lambda: _runner(inputs))
    return out.to_dict()

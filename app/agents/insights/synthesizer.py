"""Synthesizer — single LLM call merging specialist outputs into v2 schema."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm_config import get_llm
from app.prompts.insights_synthesizer_prompts import INSIGHTS_SYNTHESIZER_PROMPT

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
                pass
        m2 = re.search(r"(\{[\s\S]*\})", text)
        if m2:
            try:
                return json.loads(m2.group(1))
            except json.JSONDecodeError:
                return None
    return None


def _empty_report() -> Dict[str, Any]:
    return {
        "executive_summary": "",
        "current_state_assessment": {
            "overall_diagnostic": "",
            "what_is_working": [],
            "what_is_not_working": [],
            "confidence_level": "low",
        },
        "quantitative_findings": [],
        "qualitative_findings": [],
        "competitive_landscape": {
            "competitors_mentioned": [], "win_signals": [],
            "loss_signals": [], "positioning_gaps": [],
        },
        "segments": [],
        "objections_and_blockers": [],
        "contradictions_and_blind_spots": [],
        "hypotheses_to_test": [],
        "external_context_via_enrichment": {
            "summary": "", "key_external_facts": [], "trends_to_watch": [],
        },
        "key_findings": [],
        "recommendations_future_steps": [],
        "data_sources_used": {},
        "personas_referenced": [],
        "data_gaps": [],
        "enrichment_recommendations": [],
        "meta_insights": {
            "data_coverage": "sparse", "freshness": "unknown",
            "sample_bias_flags": [], "confidence_calibration": "",
        },
    }


def _compact_specialist(name: str, out: Dict[str, Any]) -> Dict[str, Any]:
    """Trim each specialist output to its result + status (drop tool_calls noise)."""
    return {
        "name": name,
        "status": out.get("status", "unknown"),
        "harness_score": out.get("harness_score"),
        "result": out.get("result", {}),
        "error": out.get("error"),
    }


def run_synthesizer(
    *,
    plan: Dict[str, Any],
    specialist_outputs: List[Dict[str, Any]],
    tenant_id: str,
    client_id: str,
    focus_query: Optional[str],
    survey_ids: Optional[List[str]],
    study_id: Optional[str],
) -> Dict[str, Any]:
    """Merge specialist outputs into the v2 unified schema. Returns the report dict."""
    compact = [_compact_specialist(o.get("name", "?"), o) for o in specialist_outputs]
    try:
        llm = get_llm("insights_synthesizer")
        prompt = ChatPromptTemplate.from_messages(
            [("system", INSIGHTS_SYNTHESIZER_PROMPT),
             ("human", "Produce the unified v2 report JSON now.")]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "plan": json.dumps(plan, default=str)[:3000],
            "tenant_id": tenant_id,
            "client_id": client_id,
            "survey_ids": survey_ids or [],
            "study_id": study_id or "(none)",
            "focus_query": focus_query or "(broad current-state diagnostic)",
            "specialist_outputs": json.dumps(compact, default=str)[:18000],
        })
        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, dict):
            logger.warning("Synthesizer returned unparseable output; falling back to empty report.")
            return _empty_report()

        # Backfill any missing top-level keys with empty defaults so callers can rely on shape.
        empty = _empty_report()
        for k, v in empty.items():
            parsed.setdefault(k, v)
        return parsed
    except Exception as e:
        logger.exception("Synthesizer failed: %s", e)
        return _empty_report()

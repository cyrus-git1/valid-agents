"""Planner — single LLM call producing the structured plan."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm_config import get_llm
from app.prompts.insights_planner_prompts import INSIGHTS_PLANNER_PROMPT

logger = logging.getLogger(__name__)

ALL_SPECIALISTS = ["quantitative", "qualitative", "competitive", "segments", "external"]


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
        m2 = re.search(r"(\{[\s\S]*\})", text)
        if m2:
            try:
                return json.loads(m2.group(1))
            except json.JSONDecodeError:
                return None
    return None


def _heuristic_plan(data_inventory: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback plan based on data inventory only."""
    transcript_count = int(data_inventory.get("transcript_count", 0) or 0)
    survey_count = int(data_inventory.get("survey_count", 0) or 0)
    has_documents = bool(data_inventory.get("has_documents"))

    run: List[str] = []
    skip: Dict[str, Optional[str]] = {s: None for s in ALL_SPECIALISTS}

    if survey_count > 0:
        run.append("quantitative")
    else:
        skip["quantitative"] = "no surveys in scope"

    if transcript_count > 0 or has_documents:
        run.append("qualitative")
        run.append("competitive")
    else:
        skip["qualitative"] = "no transcripts or documents"
        skip["competitive"] = "no transcripts or documents"

    if transcript_count >= 3:
        run.append("segments")
    else:
        skip["segments"] = f"only {transcript_count} transcripts; clustering uninformative below 3"

    run.append("external")

    return {
        "plan_summary": "Heuristic fallback plan based on data inventory.",
        "specialists_to_run": run,
        "per_specialist_focus": {s: "" for s in run},
        "skip_reasons": {s: skip[s] for s in ALL_SPECIALISTS if s not in run},
        "confidence": 0.5,
    }


def _validate_and_normalize(plan: Dict[str, Any], data_inventory: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce LLM output into the expected shape and enforce data prereqs."""
    base = _heuristic_plan(data_inventory)
    if not isinstance(plan, dict):
        return base

    run = plan.get("specialists_to_run") or []
    if not isinstance(run, list):
        run = []
    run = [s for s in run if s in ALL_SPECIALISTS]

    # Enforce hard prerequisites — never let the planner schedule a specialist
    # whose data inventory prereqs are not met.
    transcript_count = int(data_inventory.get("transcript_count", 0) or 0)
    survey_count = int(data_inventory.get("survey_count", 0) or 0)
    has_documents = bool(data_inventory.get("has_documents"))

    forced_skips: Dict[str, str] = {}
    if "quantitative" in run and survey_count == 0:
        run.remove("quantitative")
        forced_skips["quantitative"] = "no surveys in scope"
    if "qualitative" in run and transcript_count == 0 and not has_documents:
        run.remove("qualitative")
        forced_skips["qualitative"] = "no transcripts or documents"
    if "competitive" in run and transcript_count == 0 and not has_documents:
        run.remove("competitive")
        forced_skips["competitive"] = "no transcripts or documents"
    if "segments" in run and transcript_count < 3:
        run.remove("segments")
        forced_skips["segments"] = f"only {transcript_count} transcripts; clustering uninformative below 3"

    focus_in = plan.get("per_specialist_focus") or {}
    if not isinstance(focus_in, dict):
        focus_in = {}
    per_focus = {s: str(focus_in.get(s, "") or "") for s in run}

    skip_in = plan.get("skip_reasons") or {}
    if not isinstance(skip_in, dict):
        skip_in = {}
    skip_reasons: Dict[str, Optional[str]] = {}
    for s in ALL_SPECIALISTS:
        if s in run:
            continue
        reason = forced_skips.get(s) or skip_in.get(s) or "skipped by planner"
        skip_reasons[s] = str(reason)

    confidence = plan.get("confidence")
    try:
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            confidence = 0.5
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "plan_summary": str(plan.get("plan_summary") or "")[:1500] or base["plan_summary"],
        "specialists_to_run": run,
        "per_specialist_focus": per_focus,
        "skip_reasons": skip_reasons,
        "confidence": confidence,
    }


def run_planner(
    *,
    tenant_id: str,
    client_id: str,
    focus_query: Optional[str],
    survey_ids: Optional[List[str]],
    study_id: Optional[str],
    data_inventory: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the insights planner. Returns the validated plan dict."""
    try:
        llm = get_llm("insights_planner")
        prompt = ChatPromptTemplate.from_messages(
            [("system", INSIGHTS_PLANNER_PROMPT), ("human", "Produce the plan now.")]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "focus_query": focus_query or "(no focus — produce a broad current-state diagnostic + forward steps)",
            "tenant_id": tenant_id,
            "client_id": client_id,
            "survey_ids": survey_ids or [],
            "study_id": study_id or "(none)",
            "data_inventory": json.dumps(data_inventory, default=str),
        })
        parsed = _parse_json_loose(raw)
        if parsed is None:
            logger.warning("Planner returned unparseable output; falling back to heuristic plan")
            return _heuristic_plan(data_inventory)
        return _validate_and_normalize(parsed, data_inventory)
    except Exception as e:
        logger.exception("Planner failed: %s", e)
        return _heuristic_plan(data_inventory)

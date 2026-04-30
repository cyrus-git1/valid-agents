"""Critic — single LLM call producing weak-section flags + targeted revisions."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm_config import get_llm
from app.prompts.insights_critic_prompts import INSIGHTS_CRITIC_PROMPT

logger = logging.getLogger(__name__)


# Map critic section names → specialist names that should re-run on weak flag.
SECTION_TO_SPECIALIST = {
    "quantitative_findings": "quantitative",
    "qualitative_findings": "qualitative",
    "competitive_landscape": "competitive",
    "segments": "segments",
    "external_context": "external",
}


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


def _passing_default() -> Dict[str, Any]:
    return {
        "passes": True,
        "section_scores": {},
        "weak_sections": [],
        "targeted_revisions": {},
        "global_issues": [],
    }


def run_critic(
    *,
    plan: Dict[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """Score the synthesized report. Returns critic dict (always shaped)."""
    try:
        llm = get_llm("insights_critic")
        prompt = ChatPromptTemplate.from_messages(
            [("system", INSIGHTS_CRITIC_PROMPT),
             ("human", "Score the report now.")]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "plan": json.dumps(plan, default=str)[:2000],
            "report": json.dumps(report, default=str)[:14000],
        })
        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, dict):
            logger.warning("Critic returned unparseable output; defaulting to passing.")
            return _passing_default()

        # Normalise shape
        out = _passing_default()
        out["passes"] = bool(parsed.get("passes", True))
        scores = parsed.get("section_scores") or {}
        if isinstance(scores, dict):
            cleaned: Dict[str, float] = {}
            for k, v in scores.items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            out["section_scores"] = cleaned
        weak = parsed.get("weak_sections") or []
        if isinstance(weak, list):
            out["weak_sections"] = [str(x) for x in weak if str(x) in SECTION_TO_SPECIALIST]
        revisions = parsed.get("targeted_revisions") or {}
        if isinstance(revisions, dict):
            out["targeted_revisions"] = {
                str(k): str(v) for k, v in revisions.items()
                if str(k) in SECTION_TO_SPECIALIST and v
            }
        gi = parsed.get("global_issues") or []
        if isinstance(gi, list):
            out["global_issues"] = [str(x) for x in gi]
        return out
    except Exception as e:
        logger.exception("Critic failed: %s", e)
        return _passing_default()


def specialists_to_revise(critic_output: Dict[str, Any]) -> List[tuple[str, str]]:
    """Return [(specialist_name, revision_feedback), ...] for the orchestrator to re-run.

    Caps at 2 specialists (so the critic loop doesn't re-run everything).
    """
    weak = critic_output.get("weak_sections") or []
    revisions = critic_output.get("targeted_revisions") or {}
    pairs: List[tuple[str, str]] = []
    for section in weak:
        spec = SECTION_TO_SPECIALIST.get(section)
        if not spec:
            continue
        feedback = revisions.get(section, f"Section '{section}' was flagged as weak; deepen evidence.")
        pairs.append((spec, feedback))
        if len(pairs) >= 2:
            break
    return pairs

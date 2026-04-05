"""
src/workflows/follow_up_workflow.py
-------------------------------------
Follow-up survey generation: given a completed survey (with optional response
summaries), generate deeper-dive questions that probe the findings.

Used by POST /survey/generate-follow-up.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.survey_prompts import (
    ALL_QUESTION_TYPES,
    FOLLOW_UP_SURVEY_PROMPT,
    get_question_type_instructions,
)
from app.workflows._helpers import (
    build_profile_section,
    format_completed_survey,
    parse_recommendation_output,
    retrieve_kg_context,
    run_context_analysis,
)

logger = logging.getLogger(__name__)


def generate_follow_up_survey(
    *,
    original_request: str,
    completed_questions: List[Dict[str, Any]],
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    question_types: List[str] | None = None,
    count: int = 5,
) -> Dict[str, Any]:
    """Generate follow-up survey questions from a completed survey.

    Takes the original survey questions (with optional response summaries) and
    produces a new set of questions that probe deeper into the findings.

    Returns dict with keys: questions (list), reasoning (str), status, error.
    """
    question_types = question_types or ALL_QUESTION_TYPES
    client_profile = client_profile or {}

    # ── retrieve context ──
    context = retrieve_kg_context(original_request, tenant_id, client_id)

    # ── build profile ──
    profile_section = build_profile_section(client_profile)

    # ── analyse context ──
    context_analysis = run_context_analysis(
        request=original_request,
        context=context,
        client_profile=client_profile,
    )

    # ── format completed survey for prompt ──
    completed_text = format_completed_survey(completed_questions)

    # ── generate follow-up ──
    llm = get_llm("follow_up")
    chain = FOLLOW_UP_SURVEY_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": original_request,
            "count": str(count),
            "completed_survey_text": completed_text,
            "context_analysis": context_analysis,
            "context_section": f"\n\n{context}" if context else "",
            "profile_section": profile_section,
            "question_type_instructions": get_question_type_instructions(question_types),
        })
    except Exception as e:
        logger.exception("Follow-up survey generation failed")
        return {"questions": [], "reasoning": "", "status": "failed", "error": str(e)}

    return parse_recommendation_output(raw, key="questions")

"""
src/workflows/question_workflow.py
------------------------------------
Question recommendation: given existing survey questions, recommend new
ones that fill coverage gaps and strengthen the survey.

Used by POST /survey/generate-question.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.survey_prompts import (
    ALL_QUESTION_TYPES,
    QUESTION_RECOMMENDATION_PROMPT,
    get_question_type_instructions,
)
from app.workflows._helpers import (
    build_profile_section,
    parse_recommendation_output,
    retrieve_kg_context,
    run_context_analysis,
)

logger = logging.getLogger(__name__)


def generate_question(
    *,
    request: str,
    existing_questions: List[Dict[str, Any]],
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    question_types: List[str] | None = None,
    count: int = 3,
) -> Dict[str, Any]:
    """Generate question recommendations based on already-created survey questions.

    Retrieves KG context, analyses it, then asks the LLM to recommend new
    questions that complement the ones already in the survey.

    Returns dict with keys: recommendations (list), reasoning (str), status, error.
    """
    question_types = question_types or ALL_QUESTION_TYPES
    client_profile = client_profile or {}

    # ── retrieve context ──
    context = retrieve_kg_context(request, tenant_id, client_id)

    # ── build profile ──
    profile_section = build_profile_section(client_profile)

    # ── analyse context ──
    context_analysis = run_context_analysis(
        request=request,
        context=context,
        client_profile=client_profile,
    )

    # ── format existing questions for prompt ──
    existing_text = json.dumps(existing_questions, indent=2) if existing_questions else "[]"

    # ── generate recommendations ──
    llm = get_llm("question_rec")
    chain = QUESTION_RECOMMENDATION_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "count": str(count),
            "existing_questions_text": existing_text,
            "context_analysis": context_analysis,
            "context_section": f"\n\n{context}" if context else "",
            "profile_section": profile_section,
            "question_type_instructions": get_question_type_instructions(question_types),
        })
    except Exception as e:
        logger.exception("Question recommendation failed")
        return {"recommendations": [], "reasoning": "", "status": "failed", "error": str(e)}

    return parse_recommendation_output(raw)

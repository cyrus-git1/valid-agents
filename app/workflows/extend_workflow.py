"""
src/workflows/extend_workflow.py
---------------------------------
Extend an existing survey: takes the ORIGINAL prompt that was submitted to
generate the survey, plus the existing questions, and produces N more
questions grounded in the SAME KB context as the original generation.

Used by POST /survey/extend and the service agent's extend_survey tool.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm_config import get_llm
from app.prompts.survey_prompts import (
    ALL_QUESTION_TYPES,
    SURVEY_OUTPUT_FORMAT_PROMPT,
    get_question_type_instructions,
)
from app.workflows._helpers import (
    build_profile_section,
    parse_recommendation_output,
    retrieve_kg_context,
)

logger = logging.getLogger(__name__)


_EXTEND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert survey designer. You've already created a survey "
        "for this request. Now you're being asked to ADD MORE questions to "
        "it — using the SAME original prompt and the SAME KB context that "
        "informed the original survey.\n\n"
        "Your new questions should:\n"
        "- Stay within the scope of the ORIGINAL request — do NOT drift to "
        "unrelated topics\n"
        "- Complement the existing questions — fill coverage gaps, go deeper "
        "on angles the existing ones opened, or explore untested dimensions\n"
        "- NOT duplicate anything already in the existing questions (check "
        "labels carefully — semantic duplicates count as duplicates)\n"
        "- Use the KB context to ground questions in the company's actual "
        "products, terminology, and audience\n"
        "- Diversify question types — if existing questions lack interactive "
        "types (card_sort, ranking, tree_testing, matrix), prefer those\n\n"
        "Return a JSON object with:\n"
        '  "reasoning": short paragraph on why these questions\n'
        '  "questions": JSON array of new question objects\n\n'
        "{question_type_instructions}\n\n"
        + SURVEY_OUTPUT_FORMAT_PROMPT
        + "{profile_section}",
    ),
    (
        "human",
        "ORIGINAL prompt: {request}\n\n"
        "Number of NEW questions to add: {count}\n\n"
        "Survey title: {title}\n"
        "Survey description: {description}\n\n"
        "Existing questions in the survey (do NOT duplicate):\n"
        "{existing_questions_text}\n\n"
        "KB context (same as used for the original generation):\n"
        "{context_section}",
    ),
])


def extend_survey(
    *,
    request: str,
    existing_questions: List[Dict[str, Any]],
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    question_types: List[str] | None = None,
    count: int = 3,
    title: str | None = None,
    description: str | None = None,
) -> Dict[str, Any]:
    """Generate N new questions for an existing survey.

    Re-runs the same KB retrieval that the original survey generation
    would have done (using `request` as the search query), so the new
    questions sit in the same context as the original ones.

    Returns dict with: questions (list), reasoning, status, error, original_request.
    """
    question_types = question_types or ALL_QUESTION_TYPES
    client_profile = client_profile or {}

    # Same retrieval as original generation — KB context is scoped by the
    # original request text
    context = retrieve_kg_context(request, tenant_id, client_id)

    profile_section = build_profile_section(client_profile)

    existing_text = "[]"
    if existing_questions:
        # Trim to labels + types for the prompt — full objects are too verbose
        compact = [
            {
                "type": q.get("type", "unknown"),
                "label": q.get("label", ""),
            }
            for q in existing_questions
        ]
        existing_text = json.dumps(compact, indent=2)

    llm = get_llm("question_rec")
    chain = _EXTEND_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "count": str(count),
            "title": title or "(none)",
            "description": description or "(none)",
            "existing_questions_text": existing_text,
            "context_section": f"\n{context}" if context else "(no KB context retrieved)",
            "profile_section": profile_section,
            "question_type_instructions": get_question_type_instructions(question_types),
        })
    except Exception as e:
        logger.exception("extend_survey generation failed")
        return {
            "questions": [],
            "reasoning": "",
            "status": "failed",
            "error": str(e),
            "original_request": request,
        }

    parsed = parse_recommendation_output(raw, key="questions")
    parsed["original_request"] = request
    return parsed

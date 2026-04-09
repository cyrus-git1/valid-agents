"""
src/workflows/title_description_workflow.py
---------------------------------------------
Survey title, description, and whole-survey generation.

Contains standalone functions (no LangGraph) for generating survey metadata:
  - generate_title()       — KG-aware title generation
  - generate_description() — KG-aware description generation
  - generate_whole_survey() — title + description + questions from a prompt (no KG)

Used by POST /survey/generate-title, /generate-description, /generate-whole.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.survey_prompts import (
    ALL_QUESTION_TYPES,
    SURVEY_DESCRIPTION_PROMPT,
    SURVEY_GENERATION_PROMPT,
    SURVEY_TITLE_PROMPT,
    get_question_type_instructions,
)
from app.workflows._helpers import (
    build_profile_section,
    build_questions_section,
    normalize_question,
    parse_simple_json,
    retrieve_kg_context,
    run_context_analysis,
)

logger = logging.getLogger(__name__)


def generate_title(
    *,
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    existing_questions: List[Dict[str, Any]] | None = None,
    description: str | None = None,
) -> Dict[str, Any]:
    """Generate a survey title based on business context.

    Retrieves KG context, analyses it, then asks the LLM to produce a concise
    survey title informed by the organization's profile and knowledge base.

    Returns dict with keys: title (str), status (str), error (str | None).
    """
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

    # ── build questions section ──
    questions_section = build_questions_section(existing_questions)

    # ── build description section ──
    description_section = ""
    if description and description.strip():
        description_section = f"\n\nSurvey description: {description.strip()}"

    # ── generate title ──
    llm = get_llm("survey_title")
    chain = SURVEY_TITLE_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "context_analysis": context_analysis,
            "profile_section": profile_section,
            "questions_section": questions_section,
            "description_section": description_section,
        })
    except Exception as e:
        logger.exception("Survey title generation failed")
        return {"title": "", "status": "failed", "error": str(e)}

    data = parse_simple_json(raw)
    title = data.get("title", "").strip() if isinstance(data, dict) else ""

    if not title:
        return {"title": "", "status": "parse_error", "error": "Could not extract title from LLM output"}

    return {"title": title, "status": "complete", "error": None}


def generate_description(
    *,
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    title: str | None = None,
    existing_questions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Generate a survey description based on title + context, or context alone.

    Returns dict with keys: description (str), status (str), error (str | None).
    """
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

    # ── build sections ──
    title_section = f"Survey title: {title.strip()}\n\n" if title and title.strip() else ""
    questions_section = build_questions_section(existing_questions)

    # ── generate description ──
    llm = get_llm("survey_description")
    chain = SURVEY_DESCRIPTION_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "context_analysis": context_analysis,
            "profile_section": profile_section,
            "title_section": title_section,
            "questions_section": questions_section,
        })
    except Exception as e:
        logger.exception("Survey description generation failed")
        return {"description": "", "status": "failed", "error": str(e)}

    data = parse_simple_json(raw)
    description = data.get("description", "").strip() if isinstance(data, dict) else ""

    if not description:
        return {"description": "", "status": "parse_error", "error": "Could not extract description from LLM output"}

    return {"description": description, "status": "complete", "error": None}


def generate_whole_survey(
    *,
    prompt: str,
    question_types: List[str] | None = None,
) -> Dict[str, Any]:
    """Generate a complete survey (title, description, questions) from a prompt alone.

    Skips KG retrieval and context analysis — the LLM generates everything
    directly from the user's free-text prompt.

    Returns dict with keys: title, description, questions (list), status, error.
    """
    question_types = question_types or ALL_QUESTION_TYPES
    question_type_instructions = get_question_type_instructions(question_types)

    # ── generate questions ──
    llm_gen = get_llm("survey_generation")
    chain = SURVEY_GENERATION_PROMPT | llm_gen | StrOutputParser()
    try:
        raw_output = chain.invoke({
            "request": prompt,
            "context_analysis": "",
            "context_section": "",
            "profile_section": "",
            "question_type_instructions": question_type_instructions,
            "prior_questions_section": "",
            "title_description_section": "",
            "feedback_section": "",
        })
    except Exception as e:
        logger.exception("Whole survey generation failed")
        return {"title": "", "description": "", "questions": [], "status": "failed", "error": str(e)}

    # ── parse and normalize questions ──
    parsed = parse_simple_json(raw_output)
    questions_raw = parsed if isinstance(parsed, list) else parsed.get("questions", []) if isinstance(parsed, dict) else []

    if not questions_raw:
        try:
            questions_raw = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
            if match:
                try:
                    questions_raw = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

    if isinstance(questions_raw, dict) and "questions" in questions_raw:
        questions_raw = questions_raw["questions"]

    if not isinstance(questions_raw, list):
        questions_raw = []

    normalized = [normalize_question(q) for q in questions_raw]

    # ── build questions section for title/description ──
    questions_section = build_questions_section(normalized)

    # ── generate title ──
    title = ""
    llm_title = get_llm("survey_title")
    chain_title = SURVEY_TITLE_PROMPT | llm_title | StrOutputParser()
    try:
        raw_title = chain_title.invoke({
            "request": prompt,
            "context_analysis": "",
            "profile_section": "",
            "questions_section": questions_section,
            "description_section": "",
        })
        data = parse_simple_json(raw_title)
        title = data.get("title", "").strip() if isinstance(data, dict) else ""
    except Exception as e:
        logger.warning("Title generation in generate_whole failed: %s", e)

    # ── generate description ──
    description = ""
    llm_desc = get_llm("survey_description")
    chain_desc = SURVEY_DESCRIPTION_PROMPT | llm_desc | StrOutputParser()
    try:
        raw_desc = chain_desc.invoke({
            "request": prompt,
            "context_analysis": "",
            "profile_section": "",
            "title_section": f"Survey title: {title}\n\n" if title else "",
            "questions_section": questions_section,
        })
        data = parse_simple_json(raw_desc)
        description = data.get("description", "").strip() if isinstance(data, dict) else ""
    except Exception as e:
        logger.warning("Description generation in generate_whole failed: %s", e)

    return {
        "title": title,
        "description": description,
        "questions": normalized,
        "status": "complete",
        "error": None,
    }

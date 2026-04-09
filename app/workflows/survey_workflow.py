"""
src/workflows/survey_workflow.py
----------------------------------
LangGraph survey generation workflow:
  request → retrieve → grade → build → analyze → generate → validate → title/desc

Implements confidence-gated context retrieval with automatic retry, then
generates a survey matching the flat-array output schema.

Usage
-----
    from app.workflows.survey_workflow import build_survey_graph

    app = build_survey_graph()
    result = app.invoke({
        "request": "Create a customer satisfaction survey",
        "tenant_id": "...",
        "client_id": "...",
        "client_profile": {...},
        "question_types": ["multiple_choice"],
    })
    print(result["survey"])  # JSON array string
"""
from __future__ import annotations

import json
import logging
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from app.llm_config import get_llm
from app.models.states import SurveyState
from app.prompts.survey_prompts import (
    ALL_QUESTION_TYPES,
    CONTEXT_ANALYSIS_PROMPT,
    SURVEY_DESCRIPTION_PROMPT,
    SURVEY_GENERATION_PROMPT,
    SURVEY_TITLE_PROMPT,
    get_question_type_instructions,
)
from app import core_client  # replaces SearchService
# get_supabase removed — use core_client
from app.harness import run_with_harness
from app.harness_configs import SURVEY_STEP_CONFIG, get_active_survey_config
from app.workflows._helpers import (
    build_profile_section,
    build_questions_section,
    normalize_question,
    parse_simple_json,
)

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.60

# Re-export standalone functions so existing imports don't break.
# These now live in their own workflow files.
from app.workflows.question_workflow import generate_question  # noqa: F401
from app.workflows.follow_up_workflow import generate_follow_up_survey  # noqa: F401
from app.workflows.title_description_workflow import (  # noqa: F401
    generate_title,
    generate_description,
    generate_whole_survey,
)


# ── Nodes ────────────────────────────────────────────────────────────────────


def retrieve_context(state: SurveyState) -> SurveyState:
    """Retrieve context from KG via graph-expanded search."""
    attempt = state.get("attempt", 0) + 1
    top_k = 10 if attempt == 1 else 15
    hop_limit = 1 if attempt == 1 else 2

    docs = core_client.search_graph(
        tenant_id=state["tenant_id"],
        client_id=state["client_id"],
        query=state["request"],
        top_k=top_k,
        hop_limit=hop_limit,
    )

    top_sim = 0.0
    if docs:
        top_sim = docs[0].metadata.get("similarity_score", 0.0)

    return {
        **state,
        "documents": docs,
        "confidence": top_sim,
        "context_used": len(docs),
        "attempt": attempt,
        "status": "retrieving",
    }


def grade_context(state: SurveyState) -> SurveyState:
    """Grade retrieval quality for routing."""
    return state


def build_prompt(state: SurveyState) -> SurveyState:
    """Build context string and tenant profile for the analysis step."""
    docs = state.get("documents", [])
    context = ""
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

    context_section = f"\n\n{context}" if context else ""

    # Build profile section
    client_profile = state.get("client_profile", {})
    profile_section = build_profile_section(client_profile)

    # Build tenant profile string for context analysis
    profile_parts = []
    for key in ("industry", "headcount", "revenue", "company_name", "persona"):
        if client_profile.get(key):
            profile_parts.append(f"{key.replace('_', ' ').title()}: {client_profile[key]}")
    demo = client_profile.get("demographic", {})
    for key in ("age_range", "income_bracket", "occupation", "location"):
        if demo.get(key):
            profile_parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
    if demo.get("language") and demo["language"] != "en":
        profile_parts.append(f"Survey Language: {demo['language']}")
    tenant_profile = "\n".join(profile_parts) if profile_parts else "No profile provided."

    # Fetch prior questions via core API
    prior_questions_section = ""
    try:
        prior_outputs = core_client.get_survey_outputs(
            tenant_id=state["tenant_id"],
            client_id=state["client_id"],
            output_type="survey",
            limit=5,
        )
        if prior_outputs:
            seen_labels: set[str] = set()
            prior_list: list[dict] = []
            for row in prior_outputs:
                questions = row.get("questions") or []
                if isinstance(questions, str):
                    try:
                        questions = json.loads(questions)
                    except json.JSONDecodeError:
                        continue
                for q in questions:
                    label = (q.get("label") or "").strip().lower()
                    if label and label not in seen_labels:
                        seen_labels.add(label)
                        prior_list.append(q)
                    if len(prior_list) >= 30:
                        break
                if len(prior_list) >= 30:
                    break

            if prior_list:
                formatted = "\n".join(
                    f"- [{q.get('type', 'unknown')}] {q.get('label', '')}"
                    for q in prior_list
                )
                prior_questions_section = (
                    f"\n\nPreviously generated questions for this client:\n{formatted}"
                )
    except Exception as e:
        logger.warning("Failed to fetch prior survey questions: %s", e)

    # Build title/description section if either was provided
    td_parts = []
    title = state.get("title", "")
    description = state.get("description", "")
    if title and title.strip():
        td_parts.append(f"Survey title: {title.strip()}")
    if description and description.strip():
        td_parts.append(f"Survey description: {description.strip()}")
    title_description_section = ""
    if td_parts:
        title_description_section = (
            "\n\nThe following title and/or description have been provided for this survey. "
            "Use them to guide the tone, scope, and focus of the questions you generate:\n"
            + "\n".join(td_parts)
        )

    return {
        **state,
        "context": context_section,
        "tenant_profile": tenant_profile,
        "profile_section": profile_section,
        "prior_questions": prior_questions_section,
        "title_description_section": title_description_section,
        "status": "analyzing",
    }


def analyze_context(state: SurveyState) -> SurveyState:
    """Use LLM to extract survey-relevant insights from KG context + tenant profile."""
    context = state.get("context", "")
    tenant_profile = state.get("tenant_profile", "No profile provided.")

    if not context.strip() and tenant_profile == "No profile provided.":
        return {
            **state,
            "context_analysis": "No context or profile available. Generate general-purpose survey questions.",
            "status": "generating",
        }

    llm = get_llm("context_analysis")
    chain = CONTEXT_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        analysis = chain.invoke({
            "tenant_profile": tenant_profile,
            "request": state["request"],
            "context": context if context.strip() else "No knowledge base context available.",
        })
    except Exception as e:
        logger.exception("Context analysis failed")
        analysis = f"Analysis unavailable: {e}. Proceed with general survey design."

    logger.info("Context analysis completed (%d chars) for request: %r", len(analysis), state["request"][:80])

    return {
        **state,
        "context_analysis": analysis,
        "status": "generating",
    }


def _build_survey_chain(genome=None):
    """Build the survey LLM chain, optionally from a genome's prompts."""
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm("survey_generation")

    if genome is not None:
        # Build prompt template from genome's dynamic prompts
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                genome.agent_system_prompt
                + "{question_type_instructions}\n\n"
                + genome.output_format_prompt
                + "{profile_section}",
            ),
            (
                "human",
                "Survey request: {request}\n\n"
                "Context analysis:\n{context_analysis}\n\n"
                "Raw knowledge base context:{context_section}"
                "{prior_questions_section}"
                "{title_description_section}"
                "{feedback_section}",
            ),
        ])
    else:
        prompt = SURVEY_GENERATION_PROMPT

    return prompt | llm | StrOutputParser()


def generate_survey(state: SurveyState) -> SurveyState:
    """Generate survey questions via LLM, with harness validation and retries."""
    question_types = state.get("question_types", ALL_QUESTION_TYPES)
    question_type_instructions = get_question_type_instructions(question_types)

    # Load active genome config (falls back to hardcoded default)
    active_config, genome_version = get_active_survey_config()

    # Build chain — from genome prompts if a genome is active, else static
    genome = None
    if genome_version is not None:
        try:
            from app.supabase_client import get_supabase
            from app.optimizer.genome_store import load_active_genome
            genome = load_active_genome(get_supabase(), "survey_generation")
        except Exception:
            pass

    chain = _build_survey_chain(genome)

    invoke_vars = {
        "request": state["request"],
        "context_analysis": state.get("context_analysis", ""),
        "context_section": state.get("context", ""),
        "profile_section": state.get("profile_section", ""),
        "question_type_instructions": question_type_instructions,
        "prior_questions_section": state.get("prior_questions", ""),
        "title_description_section": state.get("title_description_section", ""),
    }

    def step_fn(inputs: dict, feedback_section: str):
        raw = chain.invoke({**invoke_vars, "feedback_section": feedback_section})
        parsed = parse_simple_json(raw)
        # Unwrap {"questions": [...]} if needed
        if isinstance(parsed, dict) and "questions" in parsed:
            return parsed["questions"]
        if isinstance(parsed, list):
            return parsed
        # Fallback: try direct parse
        try:
            result = json.loads(raw)
            if isinstance(result, dict) and "questions" in result:
                return result["questions"]
            return result
        except json.JSONDecodeError:
            return []

    result = run_with_harness(
        step_fn,
        {"request": state["request"], "client_profile": state.get("client_profile", {})},
        active_config,
    )
    result.genome_version = genome_version

    # Serialize back to raw JSON for the validate_output node
    output = result.output if result.output is not None else []
    return {**state, "raw_output": json.dumps(output, indent=2)}


def validate_output(state: SurveyState) -> SurveyState:
    """Parse and validate the LLM output into the required flat-array schema."""
    raw = state.get("raw_output", "")

    survey_data = parse_simple_json(raw)
    if not survey_data:
        try:
            survey_data = json.loads(raw)
        except json.JSONDecodeError:
            pass

    if survey_data is None or survey_data == {}:
        return {**state, "error": "Could not parse JSON from LLM output", "status": "parse_error"}

    # Unwrap if LLM returned {"questions": [...]} instead of flat array
    if isinstance(survey_data, dict) and "questions" in survey_data:
        survey_data = survey_data["questions"]

    if not isinstance(survey_data, list):
        return {**state, "error": "Survey output is not a JSON array", "status": "parse_error"}

    normalized = [normalize_question(q) for q in survey_data]

    return {
        **state,
        "survey": json.dumps(normalized, indent=2),
        "status": "complete",
    }


def fallback_output(state: SurveyState) -> SurveyState:
    """Handle unparseable LLM output."""
    logger.error("Survey output parse failed: %s", state.get("error"))
    return {
        **state,
        "survey": json.dumps([]),
        "status": "failed",
    }


def generate_title_description_node(state: SurveyState) -> SurveyState:
    """Generate title and description from the completed survey questions.

    Runs after validate_output — uses the generated questions, context analysis,
    and profile to produce a title and description for the survey.
    """
    survey_json = state.get("survey", "[]")
    try:
        questions = json.loads(survey_json)
    except json.JSONDecodeError:
        questions = []

    request = state["request"]
    context_analysis = state.get("context_analysis", "")
    profile_section = state.get("profile_section", "")
    questions_section = build_questions_section(questions)

    # ── generate title ──
    user_title = state.get("title", "")
    generated_title = user_title if user_title and user_title.strip() else ""

    if not generated_title:
        description_for_title = state.get("description", "")
        description_section = ""
        if description_for_title and description_for_title.strip():
            description_section = f"\n\nSurvey description: {description_for_title.strip()}"

        llm_title = get_llm("survey_title")
        chain_title = SURVEY_TITLE_PROMPT | llm_title | StrOutputParser()
        try:
            raw_title = chain_title.invoke({
                "request": request,
                "context_analysis": context_analysis,
                "profile_section": profile_section,
                "questions_section": questions_section,
                "description_section": description_section,
            })
            data = parse_simple_json(raw_title)
            generated_title = data.get("title", "").strip() if isinstance(data, dict) else ""
        except Exception as e:
            logger.warning("Title generation in workflow failed: %s", e)

    # ── generate description ──
    user_description = state.get("description", "")
    generated_description = user_description if user_description and user_description.strip() else ""

    if not generated_description:
        title_section = f"Survey title: {generated_title}\n\n" if generated_title else ""

        llm_desc = get_llm("survey_description")
        chain_desc = SURVEY_DESCRIPTION_PROMPT | llm_desc | StrOutputParser()
        try:
            raw_desc = chain_desc.invoke({
                "request": request,
                "context_analysis": context_analysis,
                "profile_section": profile_section,
                "title_section": title_section,
                "questions_section": questions_section,
            })
            data = parse_simple_json(raw_desc)
            generated_description = data.get("description", "").strip() if isinstance(data, dict) else ""
        except Exception as e:
            logger.warning("Description generation in workflow failed: %s", e)

    return {
        **state,
        "generated_title": generated_title,
        "generated_description": generated_description,
    }


# ── Routing ──────────────────────────────────────────────────────────────────


def route_on_context_confidence(state: SurveyState) -> str:
    """Route based on retrieval confidence. Proceeds to generation after one retry."""
    confidence = state.get("confidence", 0.0)
    attempt = state.get("attempt", 1)

    if confidence < CONFIDENCE_THRESHOLD and attempt < 2:
        return "retrieve_context"  # retry with broader search
    return "build_prompt"          # proceed regardless after retry


def route_on_validation_to_title(state: SurveyState) -> str:
    """Route based on output validation — go to title/description gen or fallback."""
    if state.get("status") == "parse_error":
        return "fallback_output"
    return "generate_title_description"


# ── Graph ────────────────────────────────────────────────────────────────────


def build_survey_graph():
    """Build and compile the survey generation LangGraph."""
    graph = StateGraph(SurveyState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("grade_context", grade_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("analyze_context", analyze_context)
    graph.add_node("generate_survey", generate_survey)
    graph.add_node("validate_output", validate_output)
    graph.add_node("generate_title_description", generate_title_description_node)
    graph.add_node("fallback_output", fallback_output)

    graph.set_entry_point("retrieve_context")

    graph.add_edge("retrieve_context", "grade_context")
    graph.add_conditional_edges("grade_context", route_on_context_confidence)
    graph.add_edge("build_prompt", "analyze_context")
    graph.add_edge("analyze_context", "generate_survey")
    graph.add_edge("generate_survey", "validate_output")
    graph.add_conditional_edges("validate_output", route_on_validation_to_title)
    graph.add_edge("generate_title_description", END)
    graph.add_edge("fallback_output", END)

    return graph.compile()

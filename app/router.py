"""
Agent service router — all endpoints.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agents.router_agent import build_router_agent
from app.agents.persona_agent import run_persona_agent
from app.agents.enrichment_agent import run_enrichment_agent
from app.models.base import StatusResponse, TenantScopedRequest
from app.models.persona import PersonaDemographics, PersonaFindRequest, PersonaFindResponse, PersonaItem
from app.models.enrichment import EnrichmentGap, EnrichmentRunRequest, EnrichmentRunResponse, EnrichmentSource
from app.models.survey import (
    GenerateDescriptionRequest, GenerateDescriptionResponse,
    GenerateFollowUpRequest, GenerateFollowUpResponse,
    GenerateQuestionRequest, GenerateQuestionResponse,
    GenerateScopedRequest, GenerateScopedResponse,
    GenerateTitleRequest, GenerateTitleResponse,
    GenerateWholeRequest, GenerateWholeResponse,
    SurveyGenerateRequest, SurveyGenerateResponse,
    SurveyQuestionItem,
)
from app.workflows.survey_workflow import (
    build_survey_graph,
    generate_question,
    generate_follow_up_survey,
    generate_title,
    generate_description,
    generate_whole_survey,
)
from app import core_client

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])
persona_router = APIRouter(prefix="/persona", tags=["persona"])
enrich_router = APIRouter(prefix="/enrich", tags=["enrich"])
survey_router = APIRouter(prefix="/survey", tags=["survey"])


# ── Agent ────────────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    input: str = Field(..., description="User query or request")
    client_profile: Optional[Dict[str, Any]] = None


class AgentQueryResponse(BaseModel):
    intent: str
    output: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None


@agent_router.post("/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    try:
        agent = build_router_agent()
        result = agent.invoke({
            "input": req.input,
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
            "client_profile": req.client_profile,
        })
    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=f"Agent query failed: {e}")
    return AgentQueryResponse(
        intent=result.get("intent", "unknown"),
        output=result.get("output", ""),
        sources=result.get("sources", []),
        confidence=result.get("intent_confidence"),
    )


# ── Persona ──────────────────────────────────────────────────────────────────

@persona_router.post("/find", response_model=PersonaFindResponse)
def find_personas(req: PersonaFindRequest) -> PersonaFindResponse:
    try:
        result = run_persona_agent(
            tenant_id=str(req.tenant_id), client_id=str(req.client_id),
            request=req.request, client_profile=req.client_profile,
            max_personas=req.max_personas, top_k=req.top_k, hop_limit=req.hop_limit,
        )
    except Exception as e:
        logger.exception("Persona discovery failed")
        raise HTTPException(status_code=500, detail=f"Persona discovery failed: {e}")
    personas = [
        PersonaItem(name=p["name"], description=p["description"],
                    demographics=PersonaDemographics(**p.get("demographics", {})),
                    motivations=p.get("motivations", []), pain_points=p.get("pain_points", []),
                    behaviors=p.get("behaviors", []), confidence=p.get("confidence", 0.5),
                    evidence_sources=p.get("evidence_sources", []))
        for p in result.get("personas", [])
    ]
    metadata = result.get("metadata", {})
    return PersonaFindResponse(
        personas=personas,
        context_used=metadata.get("context_sampled", 0),
        status=result.get("status", "complete"),
        error=result.get("error"),
    )


# ── Enrichment ───────────────────────────────────────────────────────────────

@enrich_router.post("/run", response_model=EnrichmentRunResponse)
def enrich_run(req: EnrichmentRunRequest) -> EnrichmentRunResponse:
    try:
        result = run_enrichment_agent(
            tenant_id=str(req.tenant_id), client_id=str(req.client_id),
            request=req.request, client_profile=req.client_profile,
            max_sources=req.max_sources, top_k=req.top_k,
        )
    except Exception as e:
        logger.exception("Enrichment failed")
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {e}")
    gaps = [EnrichmentGap(topic=g.get("topic", ""), reason=g.get("reason", ""), priority=g.get("priority", "medium"), search_queries=g.get("search_queries", [])) for g in result.get("gaps", [])]
    sources = [EnrichmentSource(url=s.get("url", ""), title=s.get("title", ""), relevance_reason=s.get("relevance_reason", ""), gap_topic=s.get("gap_topic", ""), job_id=s.get("job_id")) for s in result.get("sources", [])]
    return EnrichmentRunResponse(gaps=gaps, sources=sources, job_ids=result.get("job_ids", []),
                                  context_sampled=result.get("context_sampled", 0),
                                  status=result.get("status", "complete"), error=result.get("error"))


# ── Survey ───────────────────────────────────────────────────────────────────

def _parse_questions(questions_raw: List[Dict[str, Any]]) -> List[SurveyQuestionItem]:
    return [SurveyQuestionItem(
        id=q.get("id", ""), type=q.get("type", "multiple_choice"),
        label=q.get("label", ""), required=q.get("required", False),
        options=q.get("options"), min=q.get("min"), max=q.get("max"),
        lowLabel=q.get("lowLabel"), highLabel=q.get("highLabel"),
        items=q.get("items"), categories=q.get("categories"),
    ) for q in questions_raw]


@survey_router.post("/generate", response_model=SurveyGenerateResponse)
def survey_generate(req: SurveyGenerateRequest) -> SurveyGenerateResponse:
    try:
        graph = build_survey_graph()
        result = graph.invoke({
            "request": req.request, "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id), "client_profile": req.client_profile or {},
            "question_types": req.question_types, "title": req.title or "", "description": req.description or "",
        })
    except Exception as e:
        logger.exception("Survey generation failed")
        raise HTTPException(status_code=500, detail=f"Survey generation failed: {e}")

    survey_json = result.get("survey", "[]")
    try:
        questions_raw = json.loads(survey_json)
    except json.JSONDecodeError:
        questions_raw = []

    # Save output via core API
    try:
        core_client.save_survey_output(
            tenant_id=str(req.tenant_id), client_id=str(req.client_id),
            output_type="survey", request=req.request, questions=questions_raw,
        )
    except Exception:
        logger.warning("Failed to persist survey output", exc_info=True)

    return SurveyGenerateResponse(
        questions=_parse_questions(questions_raw),
        context_used=result.get("context_used", 0),
        title=result.get("generated_title", ""),
        description=result.get("generated_description", ""),
        status=result.get("status", "complete"),
        error=result.get("error"),
    )


@survey_router.post("/generate-whole", response_model=GenerateWholeResponse)
def survey_generate_whole(req: GenerateWholeRequest) -> GenerateWholeResponse:
    try:
        result = generate_whole_survey(prompt=req.prompt, question_types=req.question_types)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whole survey generation failed: {e}")
    return GenerateWholeResponse(title=result.get("title", ""), description=result.get("description", ""),
                                  questions=_parse_questions(result.get("questions", [])),
                                  status=result.get("status", "complete"), error=result.get("error"))


@survey_router.post("/generate-question", response_model=GenerateQuestionResponse)
def survey_generate_question(req: GenerateQuestionRequest) -> GenerateQuestionResponse:
    existing_dicts = [q.model_dump(exclude_none=True) for q in req.existing_questions]
    try:
        result = generate_question(request=req.request, existing_questions=existing_dicts,
                                    tenant_id=str(req.tenant_id), client_id=str(req.client_id),
                                    client_profile=req.client_profile, question_types=req.question_types, count=req.count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question recommendation failed: {e}")
    try:
        core_client.save_survey_output(tenant_id=str(req.tenant_id), client_id=str(req.client_id),
                                        output_type="recommendation", request=req.request,
                                        questions=result.get("recommendations", []), reasoning=result.get("reasoning"))
    except Exception:
        logger.warning("Failed to persist recommendation output", exc_info=True)
    return GenerateQuestionResponse(recommendations=_parse_questions(result.get("recommendations", [])),
                                     reasoning=result.get("reasoning", ""), status=result.get("status", "complete"), error=result.get("error"))


@survey_router.post("/generate-follow-up", response_model=GenerateFollowUpResponse)
def survey_generate_follow_up(req: GenerateFollowUpRequest) -> GenerateFollowUpResponse:
    completed_dicts = [q.model_dump(exclude_none=True) for q in req.completed_questions]
    try:
        result = generate_follow_up_survey(original_request=req.original_request, completed_questions=completed_dicts,
                                            tenant_id=str(req.tenant_id), client_id=str(req.client_id),
                                            client_profile=req.client_profile, question_types=req.question_types, count=req.count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up generation failed: {e}")
    try:
        core_client.save_survey_output(tenant_id=str(req.tenant_id), client_id=str(req.client_id),
                                        output_type="follow_up", request=req.original_request,
                                        questions=result.get("questions", []), reasoning=result.get("reasoning"))
    except Exception:
        logger.warning("Failed to persist follow-up output", exc_info=True)
    return GenerateFollowUpResponse(questions=_parse_questions(result.get("questions", [])),
                                     reasoning=result.get("reasoning", ""), status=result.get("status", "complete"), error=result.get("error"))


@survey_router.post("/generate-title", response_model=GenerateTitleResponse)
def survey_generate_title(req: GenerateTitleRequest) -> GenerateTitleResponse:
    existing_dicts = [q.model_dump(exclude_none=True) for q in req.existing_questions] if req.existing_questions else None
    try:
        result = generate_title(request=req.request, tenant_id=str(req.tenant_id), client_id=str(req.client_id),
                                 client_profile=req.client_profile, existing_questions=existing_dicts, description=req.description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation failed: {e}")
    return GenerateTitleResponse(title=result.get("title", ""), status=result.get("status", "complete"), error=result.get("error"))


@survey_router.post("/generate-description", response_model=GenerateDescriptionResponse)
def survey_generate_description(req: GenerateDescriptionRequest) -> GenerateDescriptionResponse:
    existing_dicts = [q.model_dump(exclude_none=True) for q in req.existing_questions] if req.existing_questions else None
    try:
        result = generate_description(request=req.request, tenant_id=str(req.tenant_id), client_id=str(req.client_id),
                                       client_profile=req.client_profile, title=req.title, existing_questions=existing_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description generation failed: {e}")
    return GenerateDescriptionResponse(description=result.get("description", ""), status=result.get("status", "complete"), error=result.get("error"))


@survey_router.post("/generate-scoped", response_model=GenerateScopedResponse)
def survey_generate_scoped(req: GenerateScopedRequest) -> GenerateScopedResponse:
    """Generate more questions within the scope of an existing survey.

    Fast path: one KB search + one LLM call. No harness manager eval.
    Uses title + description as scope boundaries. Questions must be
    relevant to the seed question's topic and the company's KB content.
    """
    import json as _json
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from app.llm_config import get_llm
    from app.prompts.survey_prompts import SURVEY_OUTPUT_FORMAT_PROMPT, get_question_type_instructions
    from app.workflows._helpers import normalize_question

    # KB search based on the seed question + title
    search_query = f"{req.seed_question.label} {req.title}"
    kb_context = ""
    try:
        docs = core_client.search_graph(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            query=search_query,
            top_k=5,
            hop_limit=1,
            node_types=["Chunk"],
        )
        if docs:
            kb_context = "\n\n".join(d.page_content[:300] for d in docs[:5])
    except Exception:
        pass

    # Build existing questions text
    existing_text = ""
    all_existing = [req.seed_question] + (req.existing_questions or [])
    if all_existing:
        existing_text = "\n".join(
            f"- [{q.type}] {q.label}" for q in all_existing
        )

    question_type_instructions = get_question_type_instructions(req.question_types)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert survey designer. Generate additional questions that "
            "stay STRICTLY within the scope defined by the survey title and description.\n\n"
            "Rules:\n"
            "- Every question must be relevant to BOTH the title/description scope AND "
            "the seed question's topic area\n"
            "- Do NOT drift to other topics — if the title says 'Pricing Experience' "
            "only generate pricing-related questions\n"
            "- Do NOT duplicate existing questions — complement them\n"
            "- Use the KB context to make questions specific to the company\n"
            "- Use diverse question types from the allowed list\n\n"
            + question_type_instructions + "\n\n"
            + SURVEY_OUTPUT_FORMAT_PROMPT
        ),
        (
            "human",
            "Survey Title: {title}\n"
            "Survey Description: {description}\n\n"
            "Seed Question: [{seed_type}] {seed_label}\n\n"
            "Existing Questions (do not duplicate):\n{existing}\n\n"
            "KB Context:\n{kb_context}\n\n"
            "Generate {count} more questions within this scope."
        ),
    ])

    llm = get_llm("survey_generation")
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "title": req.title,
            "description": req.description,
            "seed_type": req.seed_question.type,
            "seed_label": req.seed_question.label,
            "existing": existing_text or "(none)",
            "kb_context": kb_context or "(no KB context available)",
            "count": str(req.count),
        })

        # Parse
        import re
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        parsed = _json.loads(cleaned)
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
        if not isinstance(parsed, list):
            parsed = []

        normalized = [normalize_question(q) for q in parsed]

        return GenerateScopedResponse(
            questions=_parse_questions(normalized),
            status="complete",
        )
    except Exception as e:
        logger.exception("Scoped generation failed")
        raise HTTPException(status_code=500, detail=f"Scoped generation failed: {e}")

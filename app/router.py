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
                    behaviors=p.get("behaviors", []), confidence=p.get("confidence", 0.5))
        for p in result.get("personas", [])
    ]
    return PersonaFindResponse(personas=personas, context_used=result.get("context_used", 0),
                               status=result.get("status", "complete"), error=result.get("error"))


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

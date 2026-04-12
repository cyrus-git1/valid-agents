"""
/form router — survey targeting and distribution.

POST /form/target — Given a survey and company context, recommends
demographic targeting parameters for respondent recruitment.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from uuid import UUID

from app import core_client
from app.llm_config import get_llm
from app.models.base import TenantScopedRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/form", tags=["form"])


# ── Models ──────────────────────────────────────────────────────────────────


class DemographicTarget(BaseModel):
    age_range: Optional[str] = Field(None, description="e.g., '25-34', '35-54'")
    gender: Optional[str] = Field(None, description="e.g., 'all', 'female', 'male', 'non-binary'")
    income_level: Optional[str] = Field(None, description="e.g., 'low', 'middle', 'upper-middle', 'high'")
    location: Optional[str] = Field(None, description="e.g., 'urban', 'suburban', 'US', 'North America'")
    occupation: Optional[str] = Field(None, description="e.g., 'product managers', 'IT professionals', 'small business owners'")
    education: Optional[str] = Field(None, description="e.g., 'bachelor+', 'any', 'graduate degree'")
    industry: Optional[str] = Field(None, description="e.g., 'technology', 'healthcare', 'retail'")


class TargetingRecommendation(BaseModel):
    primary_target: DemographicTarget = Field(description="The main audience to recruit")
    secondary_target: Optional[DemographicTarget] = Field(None, description="Optional secondary audience")
    sample_size_recommendation: int = Field(description="Recommended number of respondents")
    reasoning: str = Field(description="Why these demographics were chosen based on the context and survey")
    exclusion_criteria: List[str] = Field(default_factory=list, description="Who to exclude from the panel")


class FormTargetRequest(TenantScopedRequest):
    survey_id: Optional[str] = Field(None, description="ID of the survey to target for")
    survey_questions: Optional[List[Dict[str, Any]]] = Field(None, description="Survey questions (if not using survey_id)")
    survey_title: Optional[str] = None
    focus: Optional[str] = Field(None, description="Optional focus for targeting (e.g., 'enterprise buyers', 'mobile users')")


class FormTargetResponse(BaseModel):
    targeting: TargetingRecommendation
    context_used: bool
    survey_used: bool


# ── Prompt ──────────────────────────────────────────────────────────────────


TARGETING_SYSTEM_PROMPT = (
    "You are an expert market research panel recruiter. Given a company's "
    "business context and a specific survey, recommend the ideal demographic "
    "targeting parameters for respondent recruitment.\n\n"
    "Your recommendations should:\n"
    "- Match the survey's topic and questions to the right audience\n"
    "- Use the company's context (industry, products, target market) to inform targeting\n"
    "- Be specific enough for a panel recruitment platform (not generic)\n"
    "- Include a realistic sample size based on the survey complexity\n"
    "- Note any exclusion criteria (e.g., exclude competitors, exclude minors)\n\n"
    "Return a JSON object with this structure:\n"
    "{\n"
    '  "primary_target": {\n'
    '    "age_range": "25-54",\n'
    '    "gender": "all",\n'
    '    "income_level": "middle to upper-middle",\n'
    '    "location": "urban, US",\n'
    '    "occupation": "product managers, UX designers",\n'
    '    "education": "bachelor+",\n'
    '    "industry": "technology, SaaS"\n'
    "  },\n"
    '  "secondary_target": null,\n'
    '  "sample_size_recommendation": 150,\n'
    '  "reasoning": "Based on the company\'s focus on enterprise SaaS...",\n'
    '  "exclusion_criteria": ["direct competitors", "respondents under 18"]\n'
    "}\n\n"
    "If a secondary audience would strengthen the research, include it. "
    "Otherwise set secondary_target to null."
)


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post("/target", response_model=FormTargetResponse)
def generate_targeting(req: FormTargetRequest) -> FormTargetResponse:
    """Generate demographic targeting recommendations for a survey.

    Pulls the company's context summary and the specified survey,
    then uses an LLM to recommend ideal respondent demographics.
    """
    # Fetch context
    context = core_client.get_context_summary(
        tenant_id=str(req.tenant_id),
        client_id=str(req.client_id),
    )
    context_text = ""
    if context:
        context_text = (
            f"Company Context:\n{context.get('summary', '')}\n"
            f"Topics: {', '.join(context.get('topics', []))}\n\n"
        )

    # Get survey questions
    survey_questions = req.survey_questions
    survey_title = req.survey_title

    if not survey_questions and req.survey_id:
        # Try to fetch from survey outputs
        try:
            outputs = core_client.get_survey_outputs(
                tenant_id=str(req.tenant_id),
                client_id=str(req.client_id),
                limit=20,
            )
            for out in outputs:
                qs = out.get("questions", [])
                if isinstance(qs, str):
                    try:
                        qs = json.loads(qs)
                    except Exception:
                        continue
                if qs:
                    survey_questions = qs
                    survey_title = survey_title or out.get("request", "")
                    break
        except Exception as e:
            logger.warning("Failed to fetch survey for targeting: %s", e)

    survey_text = ""
    if survey_questions:
        lines = []
        for i, q in enumerate(survey_questions):
            q_type = q.get("type", "unknown")
            q_label = q.get("label", "")
            lines.append(f"{i+1}. [{q_type}] {q_label}")
            if q.get("options"):
                lines.append(f"   Options: {', '.join(q['options'][:6])}")
        survey_text = f"Survey: {survey_title or 'Untitled'}\n" + "\n".join(lines) + "\n\n"

    # Build profile section
    profile_text = ""
    if req.client_profile:
        parts = []
        for key in ("industry", "headcount", "revenue", "company_name", "persona"):
            val = req.client_profile.get(key)
            if val:
                parts.append(f"{key.replace('_', ' ').title()}: {val}")
        if parts:
            profile_text = "Client Profile:\n" + "\n".join(parts) + "\n\n"

    focus_text = ""
    if req.focus:
        focus_text = f"Targeting Focus: {req.focus}\n\n"

    if not context_text and not survey_text:
        raise HTTPException(
            status_code=400,
            detail="Need either a context summary or survey questions to generate targeting. Ingest content or provide survey_questions.",
        )

    # Generate targeting
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", TARGETING_SYSTEM_PROMPT),
        ("human", "{context}{profile}{survey}{focus}Generate the demographic targeting recommendation."),
    ])

    llm = get_llm("context_analysis")
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "context": context_text,
            "profile": profile_text,
            "survey": survey_text,
            "focus": focus_text,
        })

        # Parse JSON
        import re
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
            if match:
                cleaned = match.group(1).strip()
        parsed = json.loads(cleaned)

        # Build response
        primary = DemographicTarget(**(parsed.get("primary_target", {})))
        secondary = None
        if parsed.get("secondary_target"):
            secondary = DemographicTarget(**parsed["secondary_target"])

        targeting = TargetingRecommendation(
            primary_target=primary,
            secondary_target=secondary,
            sample_size_recommendation=parsed.get("sample_size_recommendation", 100),
            reasoning=parsed.get("reasoning", ""),
            exclusion_criteria=parsed.get("exclusion_criteria", []),
        )

        return FormTargetResponse(
            targeting=targeting,
            context_used=bool(context_text),
            survey_used=bool(survey_text),
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse targeting response: {e}")
    except Exception as e:
        logger.exception("Targeting generation failed")
        raise HTTPException(status_code=500, detail=f"Targeting generation failed: {e}")

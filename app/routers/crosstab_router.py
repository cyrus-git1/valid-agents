"""
/surveys/crosstab — segmentation analysis for survey responses.

Take any two dimensions (a question's responses OR a respondent tag), pair
them per-respondent, and produce a contingency table + chi-square + Cramér's V
+ per-cell standardised residuals.

Two ways to provide data:
  1. Inline: caller passes the survey questions JSON (with response arrays)
     plus an optional respondent_tags list.
  2. Lookup: caller passes a survey output id and the endpoint fetches
     prior survey output via core_client.get_survey_outputs.

Pure-computation; zero LLM cost.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.analysis.crosstab import (
    compute_crosstab,
    extract_paired_dimensions_from_survey,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/surveys", tags=["survey"])


class CrosstabRequest(BaseModel):
    """Inline-data variant. Caller supplies the survey questions + optional tags."""
    tenant_id: UUID
    client_id: UUID
    survey_questions: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "Array of question objects. Each must have 'id' (or 'question_id') "
            "and 'responses' (one entry per respondent, in the SAME order as "
            "respondent_tags if provided)."
        ),
    )
    respondent_tags: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional per-respondent tags, e.g. [{industry: 'saas', seniority: 'senior'}, ...]",
    )
    row_question_id: Optional[str] = None
    col_question_id: Optional[str] = None
    row_tag_field: Optional[str] = None
    col_tag_field: Optional[str] = None
    row_categories: Optional[List[str]] = Field(
        default=None,
        description="Optional ordering for row categories (e.g. ['1','2','3','4','5']).",
    )
    col_categories: Optional[List[str]] = Field(
        default=None,
        description="Optional ordering for column categories.",
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "CrosstabRequest":
        if not (self.row_question_id or self.row_tag_field):
            raise ValueError("Must provide row_question_id OR row_tag_field")
        if not (self.col_question_id or self.col_tag_field):
            raise ValueError("Must provide col_question_id OR col_tag_field")
        return self


class CrosstabResponse(BaseModel):
    row_label: str
    col_label: str
    row_categories: List[str]
    col_categories: List[str]
    table: List[List[int]]
    row_totals: List[int]
    col_totals: List[int]
    n: int
    row_percentages: List[List[float]]
    col_percentages: List[List[float]]
    chi_square: Dict[str, Any]
    cramers_v: Optional[float]
    standardised_residuals: List[List[float]]
    warning: Optional[str] = None


@router.post("/crosstab", response_model=CrosstabResponse)
def survey_crosstab(req: CrosstabRequest) -> CrosstabResponse:
    """Compute a contingency table + chi-square + Cramér's V for two dimensions.

    Either dimension can be a survey question's responses or a respondent
    demographic tag. The pairing is by index — `survey_questions[*].responses[i]`
    must correspond to `respondent_tags[i]`.
    """
    try:
        rows, cols, row_label, col_label = extract_paired_dimensions_from_survey(
            survey_questions=req.survey_questions,
            row_question_id=req.row_question_id,
            col_question_id=req.col_question_id,
            row_tag_field=req.row_tag_field,
            col_tag_field=req.col_tag_field,
            respondent_tags=req.respondent_tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = compute_crosstab(
            rows=rows,
            cols=cols,
            row_label=row_label,
            col_label=col_label,
            row_categories=req.row_categories,
            col_categories=req.col_categories,
        )
    except Exception as e:
        logger.exception("crosstab computation failed")
        raise HTTPException(status_code=500, detail=f"Crosstab failed: {e}")
    return result

"""
Pydantic models for confidence-interval / box-score outputs on survey responses.

Shared by `app.analysis.confidence_interval.ConfidenceIntervalService`.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# Question types we support quantitative analysis for. Other types (short_text,
# long_text, etc.) are analysed via the qualitative pipelines.
QUANTITATIVE_QUESTION_TYPES = {
    "rating",
    "nps",
    "yes_no",
    "multiple_choice",
    "checkbox",
    "ranking",
}

# Below this many responses we surface a strong warning. Below half this we
# refuse to compute CIs at all.
MIN_SAMPLE_SIZE = 5


# ── Mean CI (rating, nps) ────────────────────────────────────────────────


class MeanCI(BaseModel):
    mean: float
    ci_lower: float
    ci_upper: float
    std_dev: float
    n: int


# ── Top-2/Bottom-2 box (rating + nps) ────────────────────────────────────


class BoxScore(BaseModel):
    """A box score is the proportion of respondents in the top-N or bottom-N
    points of a Likert / rating scale. T2B and B2B are the standard market-
    research summary stats — far more interpretable than mean alone.

    For NPS specifically: promoters = top-2-box (9-10 on 0-10 scale),
    detractors = bottom-7-box (0-6). We expose generic top_2_box / bottom_2_box
    here and add nps-specific scoring fields on the parent QuestionCI.
    """
    label: str = Field(description="Human-readable label, e.g. 'Top 2 Box'")
    threshold_lower: float = Field(description="Lower bound of the box (inclusive)")
    threshold_upper: float = Field(description="Upper bound of the box (inclusive)")
    count: int
    proportion: float
    ci_lower: float
    ci_upper: float
    n: int


class NPSScore(BaseModel):
    """NPS-specific computed metrics. Only present for question_type='nps'."""
    promoters: int
    passives: int
    detractors: int
    promoter_pct: float
    passive_pct: float
    detractor_pct: float
    nps: float = Field(description="Net Promoter Score (promoter_pct - detractor_pct), range -100..100")
    nps_ci_lower: float = Field(description="95% CI lower bound on the NPS itself")
    nps_ci_upper: float = Field(description="95% CI upper bound on the NPS itself")
    n: int


# ── Proportion CI (yes_no, multiple_choice, checkbox) ────────────────────


class ProportionCI(BaseModel):
    option: str
    count: int
    proportion: float
    ci_lower: float
    ci_upper: float
    n: int


# ── Rank CI (ranking) ────────────────────────────────────────────────────


class RankCI(BaseModel):
    item: str
    mean_rank: float
    ci_lower: float
    ci_upper: float
    n: int


# ── Per-question result ──────────────────────────────────────────────────


class QuestionCI(BaseModel):
    question_id: str
    question_type: str
    label: str
    n: int

    # rating / nps
    mean_ci: Optional[MeanCI] = None
    top_2_box: Optional[BoxScore] = None
    bottom_2_box: Optional[BoxScore] = None
    nps_score: Optional[NPSScore] = None  # only for question_type='nps'

    # yes_no / multiple_choice / checkbox
    proportion_cis: Optional[List[ProportionCI]] = None

    # ranking
    rank_cis: Optional[List[RankCI]] = None

    warning: Optional[str] = None

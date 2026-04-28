"""
src/services/confidence_interval_service.py
----------------------------------------------
Compute confidence intervals for quantitative survey question types.

Supported types (from src/prompts/survey_prompts.QUESTION_TYPE_PROMPTS):
  rating      — CI on the mean score
  nps         — CI on the mean 0-10 score
  yes_no      — CI on the proportion of 'yes'
  multiple_choice — CI on the proportion for each option
  checkbox    — CI on the selection proportion for each option
  ranking     — CI on the mean rank for each item
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from app.models.confidence_interval import (
    MIN_SAMPLE_SIZE,
    QUANTITATIVE_QUESTION_TYPES,
    BoxScore,
    MeanCI,
    NPSScore,
    ProportionCI,
    QuestionCI,
    RankCI,
)

logger = logging.getLogger(__name__)


class ConfidenceIntervalService:
    """Pure-computation service — no DB access, no LLM calls."""

    def compute_all(
        self,
        questions: List[Dict[str, Any]],
        confidence_level: float = 0.95,
    ) -> List[QuestionCI]:
        """Compute CIs for a list of question response sets."""
        results: List[QuestionCI] = []
        for q in questions:
            qtype = q.get("question_type", "")
            if qtype not in QUANTITATIVE_QUESTION_TYPES:
                logger.debug("Skipping non-quantitative type: %s", qtype)
                continue

            result = self._compute_single(q, confidence_level)
            results.append(result)
        return results

    # ── Dispatcher ────────────────────────────────────────────────────────

    def _compute_single(
        self,
        q: Dict[str, Any],
        confidence_level: float,
    ) -> QuestionCI:
        qtype = q["question_type"]
        responses = q.get("responses", [])
        label = q.get("label", "")
        qid = q.get("question_id", "")

        if qtype in ("rating", "nps"):
            return self._ci_mean(qid, qtype, label, responses, confidence_level)
        elif qtype == "yes_no":
            return self._ci_yes_no(qid, label, responses, confidence_level)
        elif qtype == "multiple_choice":
            return self._ci_proportions(
                qid, qtype, label, responses, q.get("options"), confidence_level,
            )
        elif qtype == "checkbox":
            return self._ci_checkbox(
                qid, label, responses, q.get("options"), confidence_level,
            )
        elif qtype == "ranking":
            return self._ci_ranking(
                qid, label, responses, q.get("options"), confidence_level,
            )
        else:
            return QuestionCI(
                question_id=qid, question_type=qtype, label=label, n=0,
                warning=f"Unsupported type: {qtype}",
            )

    # ── rating / nps  →  CI on mean + Top-2/Bottom-2 box + (NPS only) NPS ──

    def _ci_mean(
        self,
        qid: str,
        qtype: str,
        label: str,
        responses: List[Any],
        confidence_level: float,
    ) -> QuestionCI:
        values = _to_floats(responses)
        n = len(values)
        warning = _sample_warning(n)

        if n < MIN_SAMPLE_SIZE:
            return QuestionCI(
                question_id=qid, question_type=qtype, label=label, n=n,
                warning=warning or f"Need at least {MIN_SAMPLE_SIZE} responses.",
            )

        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        sem = stats.sem(arr)
        ci_low, ci_high = stats.t.interval(
            confidence_level, df=n - 1, loc=mean, scale=sem,
        )

        # Determine the scale extent so we can pick T2B/B2B thresholds.
        # NPS is fixed at 0-10. For rating, infer from the data extent
        # (capped at 10 — anything bigger is treated as continuous and we
        # skip box scores).
        scale_max = 10.0 if qtype == "nps" else float(arr.max())
        scale_min = 0.0 if qtype == "nps" else float(arr.min())

        top_2_box: Optional[BoxScore] = None
        bottom_2_box: Optional[BoxScore] = None
        nps_score: Optional[NPSScore] = None

        if qtype == "nps":
            # Standard NPS thresholds: promoters=9-10, passives=7-8, detractors=0-6
            promoters = int(np.sum(arr >= 9))
            passives = int(np.sum((arr >= 7) & (arr <= 8)))
            detractors = int(np.sum(arr <= 6))
            promoter_pct = 100.0 * promoters / n
            passive_pct = 100.0 * passives / n
            detractor_pct = 100.0 * detractors / n
            nps_value = promoter_pct - detractor_pct

            # CI on the NPS itself: NPS = p - d, both binomial proportions on
            # the same n. Treat as a paired/related-sample contrast — exact
            # variance is Var(p) + Var(d) - 2 Cov(p, d) where Cov(p,d) = -p*d/n
            # since promoters and detractors are mutually exclusive bins.
            p_hat = promoters / n
            d_hat = detractors / n
            var_nps = (p_hat * (1 - p_hat) + d_hat * (1 - d_hat) + 2 * p_hat * d_hat) / n
            se_nps = math.sqrt(max(var_nps, 0.0))
            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            margin = z * se_nps * 100.0  # convert proportion-scale to NPS-scale
            nps_score = NPSScore(
                promoters=promoters,
                passives=passives,
                detractors=detractors,
                promoter_pct=round(promoter_pct, 2),
                passive_pct=round(passive_pct, 2),
                detractor_pct=round(detractor_pct, 2),
                nps=round(nps_value, 2),
                nps_ci_lower=round(nps_value - margin, 2),
                nps_ci_upper=round(nps_value + margin, 2),
                n=n,
            )
            # T2B / B2B for NPS using its standard 2-point top/bottom buckets
            top_2_box = _box_score(
                arr, n, scale_min=9.0, scale_max=10.0,
                label="Top 2 Box (Promoters: 9-10)",
                confidence_level=confidence_level,
            )
            bottom_2_box = _box_score(
                arr, n, scale_min=0.0, scale_max=1.0,
                label="Bottom 2 Box (0-1)",
                confidence_level=confidence_level,
            )
        elif scale_max <= 10.0:
            # Likert-style rating: T2B = top 2 points, B2B = bottom 2 points
            top_2_box = _box_score(
                arr, n,
                scale_min=scale_max - 1.0, scale_max=scale_max,
                label="Top 2 Box",
                confidence_level=confidence_level,
            )
            bottom_2_box = _box_score(
                arr, n,
                scale_min=scale_min, scale_max=scale_min + 1.0,
                label="Bottom 2 Box",
                confidence_level=confidence_level,
            )

        return QuestionCI(
            question_id=qid, question_type=qtype, label=label, n=n,
            mean_ci=MeanCI(
                mean=round(mean, 4),
                ci_lower=round(ci_low, 4),
                ci_upper=round(ci_high, 4),
                std_dev=round(std, 4),
                n=n,
            ),
            top_2_box=top_2_box,
            bottom_2_box=bottom_2_box,
            nps_score=nps_score,
            warning=warning,
        )

    # ── yes_no  →  CI on proportion of 'yes' ─────────────────────────────

    def _ci_yes_no(
        self,
        qid: str,
        label: str,
        responses: List[Any],
        confidence_level: float,
    ) -> QuestionCI:
        bools = _to_bools(responses)
        n = len(bools)
        warning = _sample_warning(n)

        if n < MIN_SAMPLE_SIZE:
            return QuestionCI(
                question_id=qid, question_type="yes_no", label=label, n=n,
                warning=warning or f"Need at least {MIN_SAMPLE_SIZE} responses.",
            )

        yes_count = sum(bools)
        no_count = n - yes_count

        props = []
        for option, count in [("Yes", yes_count), ("No", no_count)]:
            lo, hi = _wilson_ci(count, n, confidence_level)
            props.append(ProportionCI(
                option=option,
                count=count,
                proportion=round(count / n, 4),
                ci_lower=round(lo, 4),
                ci_upper=round(hi, 4),
                n=n,
            ))

        return QuestionCI(
            question_id=qid, question_type="yes_no", label=label, n=n,
            proportion_cis=props,
            warning=warning,
        )

    # ── multiple_choice  →  CI on proportion per option ──────────────────

    def _ci_proportions(
        self,
        qid: str,
        qtype: str,
        label: str,
        responses: List[Any],
        options: Optional[List[str]],
        confidence_level: float,
    ) -> QuestionCI:
        # Each response is a single selected option string
        valid = [r for r in responses if isinstance(r, str)]
        n = len(valid)
        warning = _sample_warning(n)

        if n < MIN_SAMPLE_SIZE:
            return QuestionCI(
                question_id=qid, question_type=qtype, label=label, n=n,
                warning=warning or f"Need at least {MIN_SAMPLE_SIZE} responses.",
            )

        counts = Counter(valid)
        all_options = options or sorted(counts.keys())

        props = []
        for opt in all_options:
            count = counts.get(opt, 0)
            lo, hi = _wilson_ci(count, n, confidence_level)
            props.append(ProportionCI(
                option=opt,
                count=count,
                proportion=round(count / n, 4),
                ci_lower=round(lo, 4),
                ci_upper=round(hi, 4),
                n=n,
            ))

        return QuestionCI(
            question_id=qid, question_type=qtype, label=label, n=n,
            proportion_cis=props,
            warning=warning,
        )

    # ── checkbox  →  CI on selection proportion per option ────────────────

    def _ci_checkbox(
        self,
        qid: str,
        label: str,
        responses: List[Any],
        options: Optional[List[str]],
        confidence_level: float,
    ) -> QuestionCI:
        # Each response is a list of selected options
        valid = [r for r in responses if isinstance(r, list)]
        n = len(valid)
        warning = _sample_warning(n)

        if n < MIN_SAMPLE_SIZE:
            return QuestionCI(
                question_id=qid, question_type="checkbox", label=label, n=n,
                warning=warning or f"Need at least {MIN_SAMPLE_SIZE} responses.",
            )

        # Count how many respondents selected each option
        counts: Counter = Counter()
        for selections in valid:
            for opt in selections:
                counts[opt] += 1

        all_options = options or sorted(counts.keys())

        props = []
        for opt in all_options:
            count = counts.get(opt, 0)
            lo, hi = _wilson_ci(count, n, confidence_level)
            props.append(ProportionCI(
                option=opt,
                count=count,
                proportion=round(count / n, 4),
                ci_lower=round(lo, 4),
                ci_upper=round(hi, 4),
                n=n,
            ))

        return QuestionCI(
            question_id=qid, question_type="checkbox", label=label, n=n,
            proportion_cis=props,
            warning=warning,
        )

    # ── ranking  →  CI on mean rank per item ─────────────────────────────

    def _ci_ranking(
        self,
        qid: str,
        label: str,
        responses: List[Any],
        items: Optional[List[str]],
        confidence_level: float,
    ) -> QuestionCI:
        # Each response is an ordered list e.g. ["A","B","C"] meaning A=rank1, B=rank2, C=rank3
        valid = [r for r in responses if isinstance(r, list) and len(r) > 0]
        n = len(valid)
        warning = _sample_warning(n)

        if n < MIN_SAMPLE_SIZE:
            return QuestionCI(
                question_id=qid, question_type="ranking", label=label, n=n,
                warning=warning or f"Need at least {MIN_SAMPLE_SIZE} responses.",
            )

        # Collect ranks per item (1-indexed)
        ranks_per_item: Dict[str, List[float]] = {}
        for ranking in valid:
            for rank_idx, item in enumerate(ranking, start=1):
                ranks_per_item.setdefault(item, []).append(float(rank_idx))

        all_items = items or sorted(ranks_per_item.keys())

        rank_cis = []
        for item in all_items:
            item_ranks = ranks_per_item.get(item, [])
            if len(item_ranks) < MIN_SAMPLE_SIZE:
                rank_cis.append(RankCI(
                    item=item, mean_rank=0.0, ci_lower=0.0, ci_upper=0.0,
                    n=len(item_ranks),
                ))
                continue

            arr = np.array(item_ranks)
            mean_rank = float(np.mean(arr))
            sem = stats.sem(arr)
            ci_low, ci_high = stats.t.interval(
                confidence_level, df=len(arr) - 1, loc=mean_rank, scale=sem,
            )
            rank_cis.append(RankCI(
                item=item,
                mean_rank=round(mean_rank, 4),
                ci_lower=round(ci_low, 4),
                ci_upper=round(ci_high, 4),
                n=len(item_ranks),
            ))

        return QuestionCI(
            question_id=qid, question_type="ranking", label=label, n=n,
            rank_cis=rank_cis,
            warning=warning,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_floats(responses: List[Any]) -> List[float]:
    """Coerce responses to floats, dropping non-numeric values."""
    out: List[float] = []
    for r in responses:
        try:
            out.append(float(r))
        except (TypeError, ValueError):
            continue
    return out


def _to_bools(responses: List[Any]) -> List[bool]:
    """Coerce responses to booleans."""
    out: List[bool] = []
    for r in responses:
        if isinstance(r, bool):
            out.append(r)
        elif isinstance(r, str):
            if r.lower() in ("yes", "true", "1"):
                out.append(True)
            elif r.lower() in ("no", "false", "0"):
                out.append(False)
        elif isinstance(r, (int, float)):
            out.append(bool(r))
    return out


def _wilson_ci(
    count: int, n: int, confidence_level: float,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal approximation for small n or extreme
    proportions (near 0 or 1).
    """
    if n == 0:
        return 0.0, 0.0

    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p_hat = count / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    return max(0.0, centre - margin), min(1.0, centre + margin)


def _sample_warning(n: int) -> Optional[str]:
    """Return a warning string if sample size is problematic."""
    if n < MIN_SAMPLE_SIZE:
        return f"Insufficient data: {n} responses (minimum {MIN_SAMPLE_SIZE} required)."
    if n < 10:
        return f"Small sample size ({n}): confidence interval may be very wide."
    return None


def _box_score(
    arr: np.ndarray,
    n: int,
    *,
    scale_min: float,
    scale_max: float,
    label: str,
    confidence_level: float,
) -> BoxScore:
    """Compute a top/bottom-N-box score with a Wilson CI on the proportion.

    Counts responses with `scale_min <= value <= scale_max` and computes
    a Wilson CI on count/n.
    """
    count = int(np.sum((arr >= scale_min) & (arr <= scale_max)))
    proportion = count / n if n else 0.0
    lo, hi = _wilson_ci(count, n, confidence_level)
    return BoxScore(
        label=label,
        threshold_lower=round(scale_min, 4),
        threshold_upper=round(scale_max, 4),
        count=count,
        proportion=round(proportion, 4),
        ci_lower=round(lo, 4),
        ci_upper=round(hi, 4),
        n=n,
    )

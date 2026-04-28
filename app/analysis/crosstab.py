"""
Cross-tabulation / segmentation analysis for survey data.

Given two dimensions (a question's responses, OR a respondent tag) and a
list of respondents, produces:
  - Contingency table: 2D counts of (row_value, col_value) pairs
  - Row + column totals + percentages
  - Chi-square test (statistic, p-value, dof)
  - Cramér's V (effect size, 0..1) — interpretable independent of sample size
  - Standardised residuals per cell — pinpoints which cells deviate most
    from independence ("|residual| > 2" is significant at p<0.05)

This is the single most-asked-for analyst tool: "How does NPS differ
between SaaS founders vs E-commerce founders?"

Pure-computation service — no DB access, no LLM calls.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_crosstab(
    *,
    rows: List[Any],
    cols: List[Any],
    row_label: str = "row",
    col_label: str = "col",
    row_categories: Optional[List[str]] = None,
    col_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a contingency table from paired (row_value, col_value) lists.

    Args:
        rows: One value per respondent for the row dimension. Same length as cols.
        cols: One value per respondent for the column dimension.
        row_label / col_label: Display names for the dimensions.
        row_categories / col_categories: Optional ordering. If omitted, the
            categories are inferred from the data and sorted alphabetically.

    Returns a dict with `table`, `row_totals`, `col_totals`, `n`,
    `row_percentages`, `col_percentages`, `chi_square`, `cramers_v`,
    `standardised_residuals`, `warning`.
    """
    if len(rows) != len(cols):
        raise ValueError("rows and cols must have the same length")

    # Filter out (row, col) pairs where either side is missing
    pairs: List[Tuple[Any, Any]] = []
    for r, c in zip(rows, cols):
        if r is None or c is None:
            continue
        # Coerce to comparable strings — keeps numeric scale levels usable too
        pairs.append((_to_str(r), _to_str(c)))

    n = len(pairs)
    if n == 0:
        return _empty_crosstab(row_label, col_label, "No usable (row, col) pairs.")

    # Discover categories
    row_set = list(dict.fromkeys(p[0] for p in pairs))
    col_set = list(dict.fromkeys(p[1] for p in pairs))
    if row_categories:
        # Preserve caller's order, append any newly seen
        ordered_rows = [r for r in row_categories if r in row_set] + [r for r in row_set if r not in row_categories]
    else:
        ordered_rows = sorted(row_set)
    if col_categories:
        ordered_cols = [c for c in col_categories if c in col_set] + [c for c in col_set if c not in col_categories]
    else:
        ordered_cols = sorted(col_set)

    # Build the count matrix (row-major)
    matrix: List[List[int]] = [
        [0 for _ in ordered_cols] for _ in ordered_rows
    ]
    row_idx = {r: i for i, r in enumerate(ordered_rows)}
    col_idx = {c: j for j, c in enumerate(ordered_cols)}
    for r, c in pairs:
        i = row_idx[r]
        j = col_idx[c]
        matrix[i][j] += 1

    # Row + column totals
    arr = np.array(matrix, dtype=float)
    row_totals = arr.sum(axis=1).astype(int).tolist()
    col_totals = arr.sum(axis=0).astype(int).tolist()

    # Row-percentages: each cell as % of its row total. Reads as "of all
    # SaaS founders, X% gave NPS 9-10".
    row_percentages: List[List[float]] = []
    for i in range(len(ordered_rows)):
        rt = row_totals[i]
        row_percentages.append(
            [round(100.0 * matrix[i][j] / rt, 2) if rt > 0 else 0.0
             for j in range(len(ordered_cols))]
        )
    # Col-percentages: each cell as % of its column total
    col_percentages: List[List[float]] = []
    for i in range(len(ordered_rows)):
        col_percentages.append(
            [round(100.0 * matrix[i][j] / col_totals[j], 2) if col_totals[j] > 0 else 0.0
             for j in range(len(ordered_cols))]
        )

    # Chi-square test
    chi_square_result: Dict[str, Any] = {}
    cramers_v: Optional[float] = None
    std_residuals: List[List[float]] = []
    warning: Optional[str] = None

    if len(ordered_rows) >= 2 and len(ordered_cols) >= 2 and n >= 5:
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(arr)
        except Exception as e:
            logger.warning("chi2_contingency failed: %s", e)
            chi2, p_value, dof, expected = None, None, None, None

        if chi2 is not None:
            chi_square_result = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "dof": int(dof),
                "significant_at_0_05": bool(p_value < 0.05),
            }
            # Cramér's V = sqrt(chi2 / (n * (min(rows, cols) - 1)))
            min_dim = min(len(ordered_rows), len(ordered_cols)) - 1
            if min_dim > 0:
                cramers_v = round(float(math.sqrt(chi2 / (n * min_dim))), 4)
            # Standardised residuals: (observed - expected) / sqrt(expected)
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = (arr - expected) / np.sqrt(np.where(expected > 0, expected, 1))
                resid = np.where(expected > 0, resid, 0.0)
            std_residuals = [[round(float(v), 3) for v in row] for row in resid.tolist()]

            # Warn if any expected cell count < 5 (chi-square assumption)
            if (expected < 5).any():
                warning = (
                    "Some expected cell counts are < 5; chi-square test "
                    "may be unreliable. Consider Fisher's exact test for 2x2."
                )
    else:
        warning = "Need at least 2 rows, 2 cols, and 5 observations for chi-square."

    return {
        "row_label": row_label,
        "col_label": col_label,
        "row_categories": ordered_rows,
        "col_categories": ordered_cols,
        "table": matrix,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "n": n,
        "row_percentages": row_percentages,
        "col_percentages": col_percentages,
        "chi_square": chi_square_result,
        "cramers_v": cramers_v,
        "standardised_residuals": std_residuals,
        "warning": warning,
    }


# ── Convenience: build paired lists from a survey output ────────────────


def extract_paired_dimensions_from_survey(
    survey_questions: List[Dict[str, Any]],
    *,
    row_question_id: Optional[str] = None,
    col_question_id: Optional[str] = None,
    row_tag: Optional[List[Any]] = None,
    col_tag: Optional[List[Any]] = None,
    respondent_tags: Optional[List[Dict[str, Any]]] = None,
    row_tag_field: Optional[str] = None,
    col_tag_field: Optional[str] = None,
) -> Tuple[List[Any], List[Any], str, str]:
    """Extract paired (row, col) lists from a survey output + respondent tags.

    Two source types per dimension:
      (a) question_id → use that question's response array
      (b) tag_field   → look up `respondent_tags[i][field]` per respondent

    Caller provides one of (row_question_id | row_tag_field) and one of
    (col_question_id | col_tag_field). Returns (rows, cols, row_label, col_label).
    """
    def _resolve(question_id: Optional[str], tag_field: Optional[str]) -> Tuple[List[Any], str]:
        if question_id:
            for q in survey_questions:
                if q.get("id") == question_id or q.get("question_id") == question_id:
                    return list(q.get("responses", []) or []), q.get("label", question_id)
            raise ValueError(f"Question {question_id} not found in survey")
        if tag_field and respondent_tags is not None:
            return [t.get(tag_field) for t in respondent_tags], tag_field
        raise ValueError("Must provide either a question_id or a tag_field")

    rows, row_label = _resolve(row_question_id, row_tag_field)
    cols, col_label = _resolve(col_question_id, col_tag_field)

    if len(rows) != len(cols):
        raise ValueError(
            f"Row and column dimensions have different lengths "
            f"({len(rows)} vs {len(cols)}); cannot pair."
        )
    return rows, cols, row_label, col_label


# ── Helpers ─────────────────────────────────────────────────────────────


def _to_str(value: Any) -> str:
    """Coerce a value to a stable string for grouping. Lists → sorted joined."""
    if isinstance(value, list):
        return ",".join(sorted(_to_str(v) for v in value))
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).strip()


def _empty_crosstab(row_label: str, col_label: str, warning: str) -> Dict[str, Any]:
    return {
        "row_label": row_label,
        "col_label": col_label,
        "row_categories": [],
        "col_categories": [],
        "table": [],
        "row_totals": [],
        "col_totals": [],
        "n": 0,
        "row_percentages": [],
        "col_percentages": [],
        "chi_square": {},
        "cramers_v": None,
        "standardised_residuals": [],
        "warning": warning,
    }

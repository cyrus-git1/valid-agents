"""
Step-specific harness configurations — cheap checks, context builders,
and StepConfig instances for survey generation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.harness import CheapCheckResult, RubricDimension, StepConfig
from app.prompts.harness_prompts import SURVEY_MANAGER_PROMPT
from app.prompts.survey_prompts import ALL_QUESTION_TYPES


# ── Type-specific format validation ────────────────────────────────────────


def _check_question_format(i: int, q: dict) -> CheapCheckResult | None:
    """Validate type-specific required fields for a single question.

    Returns a failed CheapCheckResult if invalid, or None if the question passes.
    """
    qtype = q.get("type", "")
    label = q.get("label", "(unlabeled)")

    # ── multiple_choice / checkbox: need options array with 2+ items ──
    if qtype in ("multiple_choice", "checkbox"):
        options = q.get("options")
        if not isinstance(options, list) or len(options) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} ({qtype}) '{label}': needs an 'options' array with at least 2 items. "
                f"Got: {options!r}",
            )
        for j, opt in enumerate(options):
            if not isinstance(opt, str) or not opt.strip():
                return CheapCheckResult(
                    False,
                    f"Question {i} ({qtype}) '{label}': option {j} is empty or not a string.",
                )

    # ── rating: need min, max, lowLabel, highLabel ──
    elif qtype == "rating":
        for field in ("min", "max"):
            val = q.get(field)
            if val is None or not isinstance(val, (int, float)):
                return CheapCheckResult(
                    False,
                    f"Question {i} (rating) '{label}': missing or non-numeric '{field}' field.",
                )
        if q.get("min", 0) >= q.get("max", 0):
            return CheapCheckResult(
                False,
                f"Question {i} (rating) '{label}': 'min' ({q.get('min')}) must be less than 'max' ({q.get('max')}).",
            )
        for field in ("lowLabel", "highLabel"):
            val = q.get(field)
            if not val or not isinstance(val, str):
                return CheapCheckResult(
                    False,
                    f"Question {i} (rating) '{label}': missing or empty '{field}'.",
                )

    # ── ranking: need items array with 2+ strings ──
    elif qtype == "ranking":
        items = q.get("items")
        if not isinstance(items, list) or len(items) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} (ranking) '{label}': needs an 'items' array with at least 2 items.",
            )
        for j, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                return CheapCheckResult(
                    False,
                    f"Question {i} (ranking) '{label}': item {j} is empty or not a string.",
                )

    # ── card_sort: need items (objects with id+label) and categories (objects with id+label) ──
    elif qtype == "card_sort":
        items = q.get("items")
        if not isinstance(items, list) or len(items) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} (card_sort) '{label}': needs an 'items' array with at least 2 card objects.",
            )
        for j, item in enumerate(items):
            if not isinstance(item, dict) or not item.get("label"):
                return CheapCheckResult(
                    False,
                    f"Question {i} (card_sort) '{label}': item {j} must be an object with a 'label' field.",
                )

        categories = q.get("categories")
        if not isinstance(categories, list) or len(categories) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} (card_sort) '{label}': needs a 'categories' array with at least 2 category objects.",
            )
        for j, cat in enumerate(categories):
            if not isinstance(cat, dict) or not cat.get("label"):
                return CheapCheckResult(
                    False,
                    f"Question {i} (card_sort) '{label}': category {j} must be an object with a 'label' field.",
                )

    # ── tree_testing: need task string and tree array with nested nodes ──
    elif qtype == "tree_testing":
        task = q.get("task")
        if not task or not isinstance(task, str):
            return CheapCheckResult(
                False,
                f"Question {i} (tree_testing) '{label}': missing or empty 'task' string.",
            )
        tree = q.get("tree")
        if not isinstance(tree, list) or len(tree) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} (tree_testing) '{label}': needs a 'tree' array with at least 2 top-level nodes.",
            )
        for j, node in enumerate(tree):
            if not isinstance(node, dict) or not node.get("label"):
                return CheapCheckResult(
                    False,
                    f"Question {i} (tree_testing) '{label}': tree node {j} must be an object with a 'label' field.",
                )
            children = node.get("children")
            if children is not None and not isinstance(children, list):
                return CheapCheckResult(
                    False,
                    f"Question {i} (tree_testing) '{label}': tree node {j} 'children' must be an array.",
                )

    # ── matrix: need rows (2+ strings) and columns (2+ strings) ──
    elif qtype == "matrix":
        rows = q.get("rows")
        if not isinstance(rows, list) or len(rows) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} (matrix) '{label}': needs a 'rows' array with at least 2 items.",
            )
        for j, row in enumerate(rows):
            if not isinstance(row, str) or not row.strip():
                return CheapCheckResult(
                    False,
                    f"Question {i} (matrix) '{label}': row {j} is empty or not a string.",
                )
        columns = q.get("columns")
        if not isinstance(columns, list) or len(columns) < 2:
            return CheapCheckResult(
                False,
                f"Question {i} (matrix) '{label}': needs a 'columns' array with at least 2 items.",
            )
        for j, col in enumerate(columns):
            if not isinstance(col, str) or not col.strip():
                return CheapCheckResult(
                    False,
                    f"Question {i} (matrix) '{label}': column {j} is empty or not a string.",
                )

    # ── short_text, long_text, yes_no, nps, sus: no extra fields required ──

    return None  # passed


# ── Survey generation ──────────────────────────────────────────────────────


def cheap_check_survey(output: Any) -> CheapCheckResult:
    """Structural validation for survey question output, including type-specific format checks."""
    if not isinstance(output, list):
        return CheapCheckResult(False, "Output must be a JSON array of questions.")

    if len(output) < 3:
        return CheapCheckResult(False, f"Only {len(output)} questions generated. Minimum is 3.")

    for i, q in enumerate(output):
        if not isinstance(q, dict):
            return CheapCheckResult(False, f"Question {i} is not an object.")
        if not q.get("type"):
            return CheapCheckResult(False, f"Question {i} is missing a 'type' field.")
        if not q.get("label"):
            return CheapCheckResult(False, f"Question {i} is missing a 'label' field.")
        if q["type"] not in ALL_QUESTION_TYPES:
            return CheapCheckResult(
                False,
                f"Question {i} has invalid type '{q['type']}'. "
                f"Must be one of: {', '.join(ALL_QUESTION_TYPES)}.",
            )

        # Type-specific format validation
        format_error = _check_question_format(i, q)
        if format_error is not None:
            return format_error

    return CheapCheckResult(True)


# ── Manager context builders ──────────────────────────────────────────────


def _format_question_detail(q: dict) -> str:
    """Format a single question with type-specific details for the manager."""
    qtype = q.get("type", "unknown")
    label = q.get("label", "(no label)")
    parts = [f"[{qtype}] {label}"]

    if qtype in ("multiple_choice", "checkbox"):
        opts = q.get("options", [])
        parts.append(f"  options ({len(opts)}): {', '.join(str(o) for o in opts[:6])}")
    elif qtype == "rating":
        parts.append(f"  scale: {q.get('min', '?')}-{q.get('max', '?')} ({q.get('lowLabel', '?')} to {q.get('highLabel', '?')})")
    elif qtype == "ranking":
        items = q.get("items", [])
        parts.append(f"  items ({len(items)}): {', '.join(str(i) for i in items[:6])}")
    elif qtype == "card_sort":
        items = q.get("items", [])
        cats = q.get("categories", [])
        parts.append(f"  cards ({len(items)}): {', '.join(i.get('label', '?') for i in items[:4])}")
        parts.append(f"  categories ({len(cats)}): {', '.join(c.get('label', '?') for c in cats[:4])}")
    elif qtype == "tree_testing":
        tree = q.get("tree", [])
        parts.append(f"  task: {q.get('task', '?')}")
        parts.append(f"  top-level nodes ({len(tree)}): {', '.join(n.get('label', '?') for n in tree[:4])}")
    elif qtype == "matrix":
        rows = q.get("rows", [])
        cols = q.get("columns", [])
        parts.append(f"  rows ({len(rows)}): {', '.join(str(r) for r in rows[:4])}")
        parts.append(f"  columns ({len(cols)}): {', '.join(str(c) for c in cols[:5])}")

    return "\n".join(parts)


def _format_client_profile(inputs: dict) -> str:
    """Format client profile info for manager context."""
    profile = inputs.get("client_profile") or {}
    if not profile:
        return "Client profile: Not provided."

    parts = []
    if profile.get("industry"):
        parts.append(f"Industry: {profile['industry']}")
    if profile.get("headcount"):
        parts.append(f"Company size: {profile['headcount']} employees")
    if profile.get("revenue"):
        parts.append(f"Revenue: {profile['revenue']}")
    if profile.get("company_name"):
        parts.append(f"Company: {profile['company_name']}")
    if profile.get("persona"):
        parts.append(f"Target persona: {profile['persona']}")

    demo = profile.get("demographic", {})
    if demo.get("age_range"):
        parts.append(f"Target age: {demo['age_range']}")
    if demo.get("occupation"):
        parts.append(f"Target occupation: {demo['occupation']}")
    if demo.get("location"):
        parts.append(f"Location: {demo['location']}")
    if demo.get("income_bracket"):
        parts.append(f"Income bracket: {demo['income_bracket']}")
    if demo.get("language") and demo["language"] != "en":
        parts.append(f"Language: {demo['language']}")

    return "Client profile:\n" + "\n".join(parts) if parts else "Client profile: Not provided."


def _survey_context_builder(output: Any, inputs: dict) -> str:
    """Build the human message for the survey manager evaluation with full question details."""
    questions = output if isinstance(output, list) else []
    type_counts: Dict[str, int] = {}
    for q in questions:
        qt = q.get("type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    type_summary = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

    # Show all questions with type-specific details
    question_details = []
    for i, q in enumerate(questions):
        question_details.append(f"{i + 1}. {_format_question_detail(q)}")

    questions_text = "\n".join(question_details)
    profile_text = _format_client_profile(inputs)

    return (
        f"User request: {inputs.get('request', '(none)')}\n\n"
        f"{profile_text}\n\n"
        f"Generated {len(questions)} questions.\n"
        f"Type distribution: {type_summary}\n\n"
        f"Questions:\n{questions_text}"
    )


# ── Config ─────────────────────────────────────────────────────────────────


SURVEY_STEP_CONFIG = StepConfig(
    name="survey_generation",
    cheap_check=cheap_check_survey,
    manager_prompt=SURVEY_MANAGER_PROMPT,
    manager_context_builder=_survey_context_builder,
    rubric=[
        RubricDimension("relevance", 0.25, "Do the questions address what was actually requested?"),
        RubricDimension("profile_alignment", 0.20, "Do questions reflect the client's industry, company size, demographic, and study scope provided in the profile?"),
        RubricDimension("context_usage", 0.20, "Do questions reference specific details from the provided knowledge base context rather than being generic?"),
        RubricDimension("type_diversity", 0.15, "Is there a meaningful mix of question types, not all the same format?"),
        RubricDimension("format_quality", 0.10, "Are type-specific fields well-formed? Options non-redundant, scales labeled correctly, card sorts logical?"),
        RubricDimension("actionability", 0.10, "Would responses yield useful, actionable insights for the stated research goal?"),
    ],
    score_threshold=0.7,
    max_retries=2,
    use_manager=True,
)


def get_active_survey_config() -> tuple[StepConfig, int | None]:
    """Load the active genome-based config if one exists, else return the hardcoded default.

    Returns (StepConfig, genome_version). genome_version is None if using hardcoded default.
    """
    try:
        from app.supabase_client import get_supabase
        from app.optimizer.genome_store import load_active_genome
        sb = get_supabase()
        genome = load_active_genome(sb, "survey_generation")
    except Exception:
        return SURVEY_STEP_CONFIG, None

    if genome is None:
        return SURVEY_STEP_CONFIG, None

    config = StepConfig(
        name="survey_generation",
        cheap_check=cheap_check_survey,
        manager_prompt=genome.manager_prompt,
        manager_context_builder=_survey_context_builder,
        rubric=[
            RubricDimension(d["name"], d["weight"], d["description"])
            for d in genome.rubric
        ],
        score_threshold=genome.score_threshold,
        max_retries=genome.max_retries,
        use_manager=True,
    )
    return config, genome.version

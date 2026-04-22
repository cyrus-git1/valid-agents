"""
Shared helpers used across survey-related workflows.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.survey_prompts import (
    CONTEXT_ANALYSIS_PROMPT,
    get_question_type_instructions,
)

logger = logging.getLogger(__name__)


# ── Profile formatting ───────────────────────────────────────────────────────


def build_profile_section(client_profile: Dict[str, Any]) -> str:
    parts = []
    if client_profile.get("industry"):
        parts.append(f"Industry: {client_profile['industry']}")
    if client_profile.get("headcount"):
        parts.append(f"Headcount: {client_profile['headcount']} employees")
    if client_profile.get("revenue"):
        parts.append(f"Revenue: {client_profile['revenue']}")
    if client_profile.get("company_name"):
        parts.append(f"Company: {client_profile['company_name']}")
    if client_profile.get("persona"):
        parts.append(f"Target persona: {client_profile['persona']}")
    demo = client_profile.get("demographic", {})
    if demo.get("age_range"):
        parts.append(f"Respondent age range: {demo['age_range']}")
    if demo.get("income_bracket"):
        parts.append(f"Income bracket: {demo['income_bracket']}")
    if demo.get("occupation"):
        parts.append(f"Respondent occupation: {demo['occupation']}")
    if demo.get("location"):
        parts.append(f"Location: {demo['location']}")
    if demo.get("language") and demo["language"] != "en":
        parts.append(f"Survey language: {demo['language']}")
    if parts:
        return f"\n\nOrganization profile:\n" + "\n".join(parts)
    return ""


def build_questions_section(questions: List[Dict[str, Any]] | None) -> str:
    if not questions:
        return ""
    formatted = "\n".join(
        f"- [{q.get('type', 'unknown')}] {q.get('label', '')}"
        for q in questions
    )
    return f"\n\nExisting survey questions:\n{formatted}"


# ── Context analysis ─────────────────────────────────────────────────────────


def run_context_analysis(request: str, context: str, client_profile: Dict[str, Any]) -> str:
    parts = []
    for k, v in client_profile.items():
        if k != "demographic" and v:
            parts.append(f"{k}: {v}")
    tenant_profile = "\n".join(parts) if parts else "No profile provided."

    if not context.strip() and tenant_profile == "No profile provided.":
        return "No context or profile available. Generate general-purpose survey questions."

    llm = get_llm("context_analysis")
    chain = CONTEXT_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        return chain.invoke({
            "tenant_profile": tenant_profile,
            "request": request,
            "context": context if context.strip() else "No knowledge base context available.",
        })
    except Exception as e:
        logger.exception("Context analysis failed")
        return f"Analysis unavailable: {e}. Proceed with general survey design."


# ── KG context retrieval (via core API) ──────────────────────────────────────


def retrieve_kg_context(
    request: str,
    tenant_id: str,
    client_id: str,
    top_k: int = 10,
    hop_limit: int = 1,
) -> str:
    from app.core_client import search_graph

    docs = search_graph(
        tenant_id=tenant_id,
        client_id=client_id,
        query=request,
        top_k=top_k,
        hop_limit=hop_limit,
    )
    if not docs:
        return ""
    return "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
        if doc.page_content.strip()
    )


# ── Question normalization ───────────────────────────────────────────────────


def is_valid_uuid(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    try:
        uuid.UUID(val, version=4)
        return True
    except ValueError:
        return False


def normalize_tree_nodes(nodes: list, drop_placeholders: bool = True) -> list:
    """Recursively normalize tree nodes: ensure UUID4 ids, preserve structure.

    Drops empty-label nodes and (when drop_placeholders is True) generic
    placeholder labels like 'Sub option 1' or 'Category 2'.
    """
    normalized = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = (node.get("label") or "").strip()
        if not label:
            continue
        if drop_placeholders and _is_placeholder_label(label):
            # recurse into children in case they have real labels,
            # but drop this placeholder parent and lift its children up
            normalized.extend(normalize_tree_nodes(node.get("children", []), drop_placeholders))
            continue
        normalized.append({
            "id": node.get("id") if is_valid_uuid(node.get("id")) else str(uuid.uuid4()),
            "label": label,
            "children": normalize_tree_nodes(node.get("children", []), drop_placeholders),
        })
    return normalized


def _count_tree_nodes(nodes: list) -> int:
    """Total node count across all depths of a tree."""
    total = 0
    for n in nodes or []:
        if isinstance(n, dict):
            total += 1
            total += _count_tree_nodes(n.get("children", []))
    return total


def _tree_has_nesting(nodes: list) -> bool:
    """True if any node has children (i.e., tree isn't flat)."""
    for n in nodes or []:
        if isinstance(n, dict) and n.get("children"):
            return True
    return False


# Placeholder patterns that indicate the LLM produced generic junk
# instead of company-specific labels
_PLACEHOLDER_PATTERNS = [
    re.compile(r"^sub[\s\-]?option(\s+\d+)?$", re.I),
    re.compile(r"^sub[\s\-]?item(\s+\d+)?$", re.I),
    re.compile(r"^sub[\s\-]?category(\s+\d+)?$", re.I),
    re.compile(r"^option\s+\d+$", re.I),
    re.compile(r"^item\s+\d+$", re.I),
    re.compile(r"^category\s+\d+$", re.I),
    re.compile(r"^section\s+[a-z0-9]+$", re.I),
    re.compile(r"^child\s+\d+$", re.I),
    re.compile(r"^placeholder", re.I),
    re.compile(r"^example\s+\d+$", re.I),
    re.compile(r"^feature\s+[a-z]$", re.I),
    re.compile(r"^product\s+\d+$", re.I),
    re.compile(r"^service\s+[a-z]$", re.I),
    re.compile(r"^node\s+\d+$", re.I),
]


def _is_placeholder_label(label: str) -> bool:
    """True if the label looks like a generic placeholder (e.g. 'Sub option 1')."""
    if not label:
        return True
    s = label.strip()
    return any(p.match(s) for p in _PLACEHOLDER_PATTERNS)


def _count_placeholder_labels(nodes: list) -> int:
    """Count placeholder-label nodes anywhere in the tree."""
    total = 0
    for n in nodes or []:
        if isinstance(n, dict):
            if _is_placeholder_label(n.get("label", "")):
                total += 1
            total += _count_placeholder_labels(n.get("children", []))
    return total


def normalize_question(q: dict) -> dict:
    qtype = q.get("type", "multiple_choice")
    base: Dict[str, Any] = {
        "id": q.get("id") if is_valid_uuid(q.get("id")) else str(uuid.uuid4()),
        "type": qtype,
        "label": q.get("label") or q.get("text", ""),
        "required": bool(q.get("required", False)),
    }
    if qtype in ("multiple_choice", "checkbox"):
        base["options"] = q.get("options", [])
    elif qtype == "rating":
        base["min"] = q.get("min", 1)
        base["max"] = q.get("max", 5)
        base["lowLabel"] = q.get("lowLabel", "Poor")
        base["highLabel"] = q.get("highLabel", "Excellent")
    elif qtype == "ranking":
        base["items"] = q.get("items", [])
    elif qtype == "card_sort":
        base["items"] = [{"id": it.get("id") if is_valid_uuid(it.get("id")) else str(uuid.uuid4()), "label": it.get("label", "")} for it in q.get("items", [])]
        base["categories"] = [{"id": cat.get("id") if is_valid_uuid(cat.get("id")) else str(uuid.uuid4()), "label": cat.get("label", "")} for cat in q.get("categories", [])]
    elif qtype == "tree_testing":
        tree = normalize_tree_nodes(q.get("tree", []))
        base["task"] = (q.get("task") or "").strip()
        base["tree"] = tree
        base["correctPath"] = q.get("correctPath", [])

        # Quality warnings — surfaced via base["_warnings"] so the caller
        # can log or retry if the LLM produced a shallow/empty tree.
        warnings: List[str] = []
        node_count = _count_tree_nodes(tree)
        top_level = len(tree)
        # Count placeholders on the RAW tree (before dropping) to surface
        # that the LLM produced junk
        raw_tree = q.get("tree", [])
        placeholder_count = _count_placeholder_labels(raw_tree)
        if placeholder_count:
            warnings.append(
                f"tree had {placeholder_count} placeholder label(s) (e.g. 'Sub option') "
                "— dropped during normalization"
            )
        if not tree:
            warnings.append("tree is empty")
        elif top_level < 3:
            warnings.append(f"tree has only {top_level} top-level categories (min 3)")
        elif node_count < 8:
            warnings.append(f"tree has only {node_count} total nodes (min 8 for a useful test)")
        elif not _tree_has_nesting(tree):
            warnings.append("tree is flat (no nested children) — not a real IA test")
        if not base["task"]:
            warnings.append("task is empty — tree test needs a user scenario")
        if not base["correctPath"]:
            warnings.append("correctPath is empty — test cannot be scored")
        if warnings:
            base["_warnings"] = warnings
    elif qtype == "matrix":
        base["rows"] = q.get("rows", [])
        base["columns"] = q.get("columns", [])
    return base


# ── JSON parsing ─────────────────────────────────────────────────────────────


def parse_simple_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def parse_recommendation_output(raw: str, key: str = "recommendations") -> Dict[str, Any]:
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    if data is None:
        return {key: [], "reasoning": "", "status": "parse_error", "error": "Could not parse JSON"}
    reasoning = ""
    questions_raw: list = []
    if isinstance(data, dict):
        reasoning = data.get("reasoning", "")
        questions_raw = data.get(key) or data.get("questions") or data.get("recommendations") or []
    elif isinstance(data, list):
        questions_raw = data
    normalized = [normalize_question(q) for q in questions_raw]
    return {key: normalized, "reasoning": reasoning, "status": "complete", "error": None}


def format_completed_survey(questions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, q in enumerate(questions, 1):
        parts = [f"Q{i}. [{q.get('type', 'unknown')}] {q.get('label', '')}"]
        if q.get("options"):
            parts.append(f"   Options: {', '.join(q['options'])}")
        if q.get("response_summary"):
            parts.append(f"   Response summary: {q['response_summary']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines) if lines else "No questions provided."

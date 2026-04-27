"""
One LLM call per cluster to produce a human-readable label + description.

Cluster boundaries are NOT changed by this step — labelling is purely
cosmetic. Failures fall back to a deterministic auto-label like
"Cluster 3 (n=12)".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.cluster_label_prompts import CLUSTER_LABEL_PROMPT

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def label_cluster(
    *,
    cluster_id: int,
    size: int,
    defining_tags: List[Dict[str, Any]],
    top_terms: List[str],
    dominant_traits: List[str],
    mean_vader: float,
    focus: Optional[str] = None,
) -> Dict[str, str]:
    """Generate a label + description for one cluster.

    Returns {"label": str, "description": str}. On failure, returns a
    deterministic fallback so the cluster always has SOME label.
    """
    fallback = {
        "label": _auto_label(cluster_id, defining_tags, top_terms, size),
        "description": _auto_description(defining_tags, dominant_traits, top_terms, size),
    }

    try:
        llm = get_llm("context_analysis")
        chain = CLUSTER_LABEL_PROMPT | llm | StrOutputParser()
        focus_section = f"Focus area: {focus}\n" if focus else ""
        raw = chain.invoke({
            "defining_tags": _format_tags(defining_tags) or "(none)",
            "top_terms": ", ".join(top_terms[:10]) or "(none)",
            "dominant_traits": ", ".join(dominant_traits) or "(none)",
            "mean_vader": f"{mean_vader:.3f}",
            "size": str(size),
            "focus_section": focus_section,
        })
    except Exception as e:
        logger.warning("LLM cluster labeling failed for cluster %d: %s", cluster_id, e)
        return fallback

    parsed = _parse_json_loose(raw)
    label = (parsed.get("label") or "").strip()
    description = (parsed.get("description") or "").strip()
    if not label:
        return fallback
    return {
        "label": label,
        "description": description or fallback["description"],
    }


def label_clusters_bulk(
    cluster_inputs: List[Dict[str, Any]],
    *,
    focus: Optional[str] = None,
) -> Dict[int, Dict[str, str]]:
    """Label many clusters. Sequential — these are cheap and few in number.

    cluster_inputs: each is {cluster_id, size, defining_tags, top_terms,
                              dominant_traits, mean_vader}.
    """
    out: Dict[int, Dict[str, str]] = {}
    for ci in cluster_inputs:
        out[ci["cluster_id"]] = label_cluster(focus=focus, **ci)
    return out


# ── Fallbacks ────────────────────────────────────────────────────────────


def _format_tags(defining_tags: List[Dict[str, Any]]) -> str:
    if not defining_tags:
        return ""
    parts = []
    for t in defining_tags[:5]:
        f = t.get("field", "")
        v = t.get("value", "")
        rate = t.get("in_cluster_rate", 0.0)
        parts.append(f"{f}={v} ({int(rate * 100)}% of cluster)")
    return "; ".join(parts)


def _auto_label(
    cluster_id: int,
    defining_tags: List[Dict[str, Any]],
    top_terms: List[str],
    size: int,
) -> str:
    """Deterministic fallback label using strongest signal we have."""
    if defining_tags:
        first = defining_tags[0]
        return f"{first.get('value', '').replace('_', ' ').title()} (n={size})"
    if top_terms:
        return f"{top_terms[0].title()} cluster (n={size})"
    return f"Cluster {cluster_id} (n={size})"


def _auto_description(
    defining_tags: List[Dict[str, Any]],
    dominant_traits: List[str],
    top_terms: List[str],
    size: int,
) -> str:
    parts = [f"Cluster of {size} respondents"]
    if defining_tags:
        tag_str = ", ".join(f"{t.get('field')}={t.get('value')}" for t in defining_tags[:3])
        parts.append(f"defined by {tag_str}")
    if dominant_traits:
        parts.append(f"with traits {', '.join(dominant_traits[:3])}")
    if top_terms:
        parts.append(f"discussing {', '.join(top_terms[:5])}")
    return ". ".join(parts) + "."


def _parse_json_loose(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        m = _JSON_BLOCK_RE.search(s)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}

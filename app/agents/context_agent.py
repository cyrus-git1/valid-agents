"""
Context agent — generates quality-gated context summaries from KG content.

Delegates to the context workflow which:
  1. Checks for existing summary (returns cached if available)
  2. Retrieves KG content with diverse queries
  3. Generates a summary through the harness (cheap check + rubric eval + retries)
  4. Stores the validated summary back to the memory layer

Usage
-----
    from app.agents.context_agent import run_context_agent

    result = run_context_agent(
        tenant_id="...",
        client_id="...",
        client_profile={"industry": "SaaS"},
        force_regenerate=True,
    )
    print(result["summary"])
    print(result["topics"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.workflows.context_workflow import build_context_graph

logger = logging.getLogger(__name__)


def run_context_agent(
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
    force_regenerate: bool = False,
    new_chunks: Optional[list] = None,
    granularity_level: str = "tenant",
    scope_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate or retrieve a context summary for a tenant+client.

    Args:
        new_chunks: Optional list of chunk dicts from a just-completed ingest.
            These are included alongside KB search results so the summary
            accounts for newly ingested content that may not be indexed yet.
        granularity_level: Summary granularity — 'tenant' (default),
            'document', or 'topic'.
        scope_ref: Required when granularity_level is 'document' (the
            document_id) or 'topic' (the topic label).

    Returns dict with keys: summary, topics, regenerated, context_sampled, status, error.
    """
    graph = build_context_graph()

    result = graph.invoke({
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_profile": client_profile or {},
        "force_regenerate": force_regenerate,
        "new_chunks": new_chunks or [],
        "granularity_level": granularity_level,
        "scope_ref": scope_ref,
    })

    summary_data = result.get("generated_summary", {})
    status = result.get("status", "unknown")

    return {
        "summary": summary_data.get("summary", "") if isinstance(summary_data, dict) else "",
        "topics": summary_data.get("topics", []) if isinstance(summary_data, dict) else [],
        "regenerated": status not in ("cached",),
        "context_sampled": result.get("context_sampled", 0),
        "has_summary": bool(summary_data.get("summary")) if isinstance(summary_data, dict) else False,
        "status": status,
        "error": result.get("error"),
    }

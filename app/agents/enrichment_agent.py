"""
Enrichment agent — thin wrapper around the enrichment workflow.

Delegates all logic to app/workflows/enrichment_workflow.py which uses
tools from app/tools/enrichment_tools.py.

Usage
-----
    from app.agents.enrichment_agent import run_enrichment_agent

    result = run_enrichment_agent(
        tenant_id="...",
        client_id="...",
        request="Find competitor pricing data",
    )
    for src in result["sources"]:
        print(src["url"], src["job_id"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.workflows.enrichment_workflow import build_enrichment_graph

logger = logging.getLogger(__name__)


def run_enrichment_agent(
    tenant_id: str,
    client_id: str,
    request: Optional[str] = None,
    client_profile: Optional[Dict[str, Any]] = None,
    max_sources: int = 5,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Identify KG knowledge gaps and fill them with web content.

    Delegates to the enrichment workflow LangGraph which calls
    enrichment tools for each step.
    """
    graph = build_enrichment_graph()

    result = graph.invoke({
        "request": request or "",
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_profile": client_profile or {},
        "max_sources": max_sources,
    })

    return {
        "gaps": result.get("gaps", []),
        "sources": result.get("ingested_sources", []),
        "job_ids": result.get("job_ids", []),
        "context_sampled": result.get("context_sampled", 0),
        "queries_used": result.get("queries_used", 0),
        "urls_blocked": result.get("urls_blocked", []),
        "warnings": result.get("warnings", []),
        "status": result.get("status", "complete"),
        "error": result.get("error"),
    }

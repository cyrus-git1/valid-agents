"""
Enrichment workflow — LangGraph StateGraph that uses enrichment tools
to identify knowledge gaps and fill them with web content.

Pipeline:
  sample_coverage → fetch_summary → analyze_gaps → search_for_gaps → rank_sources → ingest_sources

Each node calls tools from app.tools.enrichment_tools rather than
making direct service/client calls.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from app.models.states import EnrichmentState
from app.analysis.base import BaseAnalysisService
from app.tools.enrichment_tools import (
    analyze_knowledge_gaps,
    get_context_summary,
    ingest_web_url,
    rank_web_sources,
    search_knowledge_base,
    web_search,
)

logger = logging.getLogger(__name__)

_COVERAGE_QUERIES = [
    "products services offerings solutions",
    "customers market industry trends competitors",
    "strategy goals challenges opportunities",
]


# ── Nodes ───────────────────────────────────────────────────────────────────


def sample_coverage(state: EnrichmentState) -> EnrichmentState:
    """Sample the knowledge base coverage using multiple queries."""
    all_results: Dict[str, Dict[str, Any]] = {}

    for query in _COVERAGE_QUERIES:
        results = search_knowledge_base.invoke({
            "tenant_id": state["tenant_id"],
            "client_id": state["client_id"],
            "query": query,
            "top_k": 15,
            "hop_limit": 1,
        })
        for r in results:
            nid = r.get("node_id")
            if nid and nid not in all_results:
                all_results[nid] = r

    if not all_results:
        return {
            **state,
            "kg_context": "",
            "context_sampled": 0,
            "status": "no_content",
        }

    # Sort by similarity and take top results
    sorted_results = sorted(
        all_results.values(),
        key=lambda r: r.get("similarity_score", 0.0),
        reverse=True,
    )[:15]

    context = "\n\n---\n\n".join(
        f"[Source {i + 1}] {r['content']}"
        for i, r in enumerate(sorted_results)
        if r.get("content", "").strip()
    ) or "(Knowledge base is sparse.)"

    return {
        **state,
        "kg_context": context,
        "context_sampled": len(sorted_results),
        "status": "sampling_complete",
    }


def fetch_summary(state: EnrichmentState) -> EnrichmentState:
    """Fetch the existing context summary."""
    row = get_context_summary.invoke({
        "tenant_id": state["tenant_id"],
        "client_id": state["client_id"],
    })

    summary_section = ""
    if row:
        summary_section = (
            f"\nContext summary:\nSummary: {row.get('summary', '')}\n"
            f"Topics: {', '.join(row.get('topics', []))}\n\n"
        )

    profile_section = BaseAnalysisService._build_profile_section(
        state.get("client_profile")
    )

    return {
        **state,
        "context_summary": summary_section,
        "profile_section": profile_section,
    }


def analyze_gaps(state: EnrichmentState) -> EnrichmentState:
    """Use LLM to identify knowledge gaps."""
    user_request = (
        state.get("request")
        or "Identify gaps in this knowledge base and suggest areas to enrich with external data."
    )

    result = analyze_knowledge_gaps.invoke({
        "context": state.get("kg_context", "(Knowledge base is sparse.)"),
        "profile_section": state.get("profile_section", ""),
        "summary_section": state.get("context_summary", ""),
        "user_request": user_request,
    })

    gaps = result.get("gaps", [])

    if not gaps:
        return {
            **state,
            "gaps": [],
            "status": "no_gaps",
        }

    logger.info("Enrichment: identified %d gaps", len(gaps))

    return {
        **state,
        "gaps": gaps[:5],  # cap at 5 gaps
        "status": "gaps_identified",
    }


def search_for_gaps(state: EnrichmentState) -> EnrichmentState:
    """Search the web for content to fill each gap."""
    all_search_results: List[Dict[str, Any]] = []

    for gap in state.get("gaps", []):
        queries = gap.get("search_queries", [])[:3]
        gap_results: List[Dict[str, Any]] = []
        seen_urls: set = set()

        for query in queries:
            results = web_search.invoke({
                "query": query,
                "num_results": 5,
            })
            for r in results:
                url = r.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    gap_results.append(r)

        if gap_results:
            all_search_results.append({
                "gap_topic": gap.get("topic", ""),
                "gap_reason": gap.get("reason", ""),
                "results": gap_results,
            })

    if not all_search_results:
        return {
            **state,
            "search_results": [],
            "status": "no_search_results",
        }

    return {
        **state,
        "search_results": all_search_results,
        "status": "search_complete",
    }


def rank_sources(state: EnrichmentState) -> EnrichmentState:
    """Rank search results by relevance and quality per gap."""
    max_sources = state.get("max_sources", 5)
    ranked: List[Dict[str, Any]] = []

    for gap_search in state.get("search_results", []):
        search_results_text = "\n".join(
            f"- Title: {r['title']}\n  URL: {r['link']}\n  Snippet: {r['snippet']}"
            for r in gap_search["results"]
        )

        result = rank_web_sources.invoke({
            "gap_topic": gap_search["gap_topic"],
            "gap_reason": gap_search["gap_reason"],
            "search_results_text": search_results_text,
            "max_urls": max(1, max_sources // len(state.get("search_results", [1]))),
        })

        for url_item in result.get("urls", []):
            url_item["gap_topic"] = gap_search["gap_topic"]
            ranked.append(url_item)

    # Deduplicate by URL, keep highest priority
    seen: Dict[str, Dict] = {}
    for src in ranked:
        url = src.get("url", "")
        if url and (url not in seen or src.get("priority", 99) < seen[url].get("priority", 99)):
            seen[url] = src

    final = sorted(seen.values(), key=lambda s: s.get("priority", 99))[:max_sources]

    if not final:
        return {
            **state,
            "ranked_sources": [],
            "status": "no_ranked_sources",
        }

    return {
        **state,
        "ranked_sources": final,
        "status": "ranking_complete",
    }


def ingest_sources(state: EnrichmentState) -> EnrichmentState:
    """Ingest the top-ranked web sources into the knowledge base."""
    ingested: List[Dict[str, Any]] = []
    job_ids: List[str] = []

    for src in state.get("ranked_sources", []):
        url = src.get("url", "")
        result = ingest_web_url.invoke({
            "tenant_id": state["tenant_id"],
            "client_id": state["client_id"],
            "url": url,
            "title": src.get("title"),
            "metadata": {
                "enrichment_source": True,
                "gap_topic": src.get("gap_topic", ""),
                "relevance_reason": src.get("relevance_reason", ""),
            },
        })

        job_id = result.get("job_id")
        if job_id and "error" not in result:
            job_ids.append(job_id)

        ingested.append({
            "url": url,
            "title": src.get("title", ""),
            "relevance_reason": src.get("relevance_reason", ""),
            "gap_topic": src.get("gap_topic", ""),
            "job_id": job_id,
            "error": result.get("error"),
        })

    logger.info("Enrichment: ingested %d sources, %d jobs", len(ingested), len(job_ids))

    return {
        **state,
        "ingested_sources": ingested,
        "job_ids": job_ids,
        "status": "complete",
    }


# ── Routing ─────────────────────────────────────────────────────────────────


def route_after_sample(state: EnrichmentState) -> str:
    if state.get("status") == "no_content":
        return END
    return "fetch_summary"


def route_after_gaps(state: EnrichmentState) -> str:
    if state.get("status") == "no_gaps":
        return END
    return "search_for_gaps"


def route_after_search(state: EnrichmentState) -> str:
    if state.get("status") == "no_search_results":
        return END
    return "rank_sources"


def route_after_rank(state: EnrichmentState) -> str:
    if state.get("status") == "no_ranked_sources":
        return END
    return "ingest_sources"


# ── Graph ───────────────────────────────────────────────────────────────────


def build_enrichment_graph():
    """Build and compile the enrichment LangGraph."""
    graph = StateGraph(EnrichmentState)

    graph.add_node("sample_coverage", sample_coverage)
    graph.add_node("fetch_summary", fetch_summary)
    graph.add_node("analyze_gaps", analyze_gaps)
    graph.add_node("search_for_gaps", search_for_gaps)
    graph.add_node("rank_sources", rank_sources)
    graph.add_node("ingest_sources", ingest_sources)

    graph.set_entry_point("sample_coverage")

    graph.add_conditional_edges("sample_coverage", route_after_sample)
    graph.add_edge("fetch_summary", "analyze_gaps")
    graph.add_conditional_edges("analyze_gaps", route_after_gaps)
    graph.add_conditional_edges("search_for_gaps", route_after_search)
    graph.add_conditional_edges("rank_sources", route_after_rank)
    graph.add_edge("ingest_sources", END)

    return graph.compile()

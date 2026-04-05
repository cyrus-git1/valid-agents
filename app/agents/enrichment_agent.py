"""
KG Enrichment agent — identifies knowledge gaps in a tenant's KG and fills
them by searching the web via Serper, ranking results with an LLM, and
triggering ingestion on the core API.

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

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.enrichment_prompts import GAP_ANALYSIS_PROMPT, URL_RANKING_PROMPT
from app.analysis.base import BaseAnalysisService
from app import core_client
from app.serper_service import SerperService

logger = logging.getLogger(__name__)

_COVERAGE_QUERIES = [
    "products services offerings solutions",
    "customers market industry trends competitors",
    "strategy goals challenges opportunities",
]


def run_enrichment_agent(
    tenant_id: str,
    client_id: str,
    request: Optional[str] = None,
    client_profile: Optional[Dict[str, Any]] = None,
    max_sources: int = 5,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Identify KG knowledge gaps and fill them with web content."""
    serper = SerperService()

    if not serper.is_configured:
        return {
            "gaps": [], "sources": [], "job_ids": [], "context_sampled": 0,
            "status": "failed",
            "error": "SERPER_API_KEY is not configured. Web search is required for enrichment.",
        }

    # Step 1: Sample KG coverage via core API
    all_docs = {}
    for query in _COVERAGE_QUERIES:
        try:
            docs = core_client.search_graph(
                tenant_id=tenant_id, client_id=client_id,
                query=query, top_k=top_k, hop_limit=1,
            )
            for doc in docs:
                nid = doc.metadata.get("node_id")
                if nid and nid not in all_docs:
                    all_docs[nid] = doc
        except Exception as e:
            logger.warning("Coverage query failed (%s): %s", query[:40], e)

    sorted_docs = sorted(
        all_docs.values(),
        key=lambda d: d.metadata.get("similarity_score", 0.0),
        reverse=True,
    )[:top_k]

    # Step 2: Fetch context summary via core API
    summary_section = ""
    row = core_client.get_context_summary(tenant_id=tenant_id, client_id=client_id)
    if row:
        summary_section = (
            f"\nContext summary:\nSummary: {row.get('summary', '')}\n"
            f"Topics: {', '.join(row.get('topics', []))}\n\n"
        )

    if not sorted_docs and not summary_section:
        return {
            "gaps": [], "sources": [], "job_ids": [], "context_sampled": 0,
            "status": "complete",
            "error": "No existing KG content found. Ingest some documents first before enriching.",
        }

    # Step 3: LLM gap analysis
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}] {doc.page_content}"
        for i, doc in enumerate(sorted_docs)
        if doc.page_content.strip()
    ) or "(Knowledge base is sparse.)"

    profile_section = BaseAnalysisService._build_profile_section(client_profile)
    user_request = request or "Identify gaps in this knowledge base and suggest areas to enrich with external data."

    llm = get_llm("enrichment")

    try:
        chain = GAP_ANALYSIS_PROMPT | llm | StrOutputParser()
        raw_gaps = chain.invoke({
            "context": context, "profile_section": profile_section,
            "summary_section": summary_section, "user_request": user_request,
        })
        gap_data = _parse_json(raw_gaps)
        gaps = gap_data.get("gaps", [])
    except Exception as e:
        logger.exception("Gap analysis LLM call failed")
        return {"gaps": [], "sources": [], "job_ids": [], "context_sampled": len(sorted_docs), "status": "failed", "error": f"Gap analysis failed: {e}"}

    if not gaps:
        return {"gaps": [], "sources": [], "job_ids": [], "context_sampled": len(sorted_docs), "status": "complete", "error": None}

    logger.info("Enrichment: identified %d gaps for tenant=%s", len(gaps), tenant_id)

    # Step 4: Search for each gap
    all_search_results: List[Dict[str, Any]] = []
    for gap in gaps[:5]:
        queries = gap.get("search_queries", [])[:3]
        gap_results = []
        seen_urls = set()
        for query in queries:
            try:
                results = serper.search(query, num_results=5)
                for r in results:
                    url = r.get("link", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        gap_results.append(r)
            except Exception as e:
                logger.warning("Serper search failed for query %r: %s", query[:40], e)
        if gap_results:
            all_search_results.append({"gap_topic": gap.get("topic", ""), "gap_reason": gap.get("reason", ""), "results": gap_results})

    if not all_search_results:
        return {"gaps": gaps, "sources": [], "job_ids": [], "context_sampled": len(sorted_docs), "status": "complete", "error": "Web search returned no results for identified gaps."}

    # Step 5: LLM ranks URLs
    ranked_sources: List[Dict[str, Any]] = []
    for gap_search in all_search_results:
        search_results_text = "\n".join(
            f"- Title: {r['title']}\n  URL: {r['link']}\n  Snippet: {r['snippet']}"
            for r in gap_search["results"]
        )
        try:
            chain = URL_RANKING_PROMPT | llm | StrOutputParser()
            raw_ranking = chain.invoke({
                "gap_topic": gap_search["gap_topic"], "gap_reason": gap_search["gap_reason"],
                "search_results": search_results_text,
                "max_urls": str(max(1, max_sources // len(all_search_results))),
            })
            ranking_data = _parse_json(raw_ranking)
            for url_item in ranking_data.get("urls", []):
                url_item["gap_topic"] = gap_search["gap_topic"]
                ranked_sources.append(url_item)
        except Exception as e:
            logger.warning("URL ranking failed for gap %r: %s", gap_search["gap_topic"][:40], e)

    seen = {}
    for src in ranked_sources:
        url = src.get("url", "")
        if url and (url not in seen or src.get("priority", 99) < seen[url].get("priority", 99)):
            seen[url] = src
    final_sources = sorted(seen.values(), key=lambda s: s.get("priority", 99))[:max_sources]

    if not final_sources:
        return {"gaps": gaps, "sources": [], "job_ids": [], "context_sampled": len(sorted_docs), "status": "complete", "error": "No suitable web sources found after filtering."}

    # Step 6: Ingest top URLs via core API
    job_ids = []
    sources_with_jobs = []
    for src in final_sources:
        url = src.get("url", "")
        try:
            resp = core_client.ingest_web(
                tenant_id=tenant_id, client_id=client_id,
                url=url, title=src.get("title"),
                metadata={"enrichment_source": True, "gap_topic": src.get("gap_topic", ""), "relevance_reason": src.get("relevance_reason", "")},
            )
            job_id = resp.get("job_id", str(uuid.uuid4()))
            job_ids.append(job_id)
            sources_with_jobs.append({"url": url, "title": src.get("title", ""), "relevance_reason": src.get("relevance_reason", ""), "gap_topic": src.get("gap_topic", ""), "job_id": job_id})
            logger.info("Enrichment triggered ingest for %s (gap: %s)", url, src.get("gap_topic", ""))
        except Exception as e:
            logger.warning("Failed to trigger ingest for %s: %s", url, e)
            sources_with_jobs.append({"url": url, "title": src.get("title", ""), "relevance_reason": src.get("relevance_reason", ""), "gap_topic": src.get("gap_topic", ""), "job_id": None, "error": str(e)})

    return {"gaps": gaps, "sources": sources_with_jobs, "job_ids": job_ids, "context_sampled": len(sorted_docs), "status": "complete", "error": None}


def _parse_json(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Enrichment LLM returned invalid JSON: %s", cleaned[:500])
        return {}

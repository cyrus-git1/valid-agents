"""
Enrichment workflow — LangGraph StateGraph that uses enrichment tools
to identify knowledge gaps and fill them with web content.

Pipeline:
  check_context → sample_coverage → analyze_gaps → validate_gaps →
  search_for_gaps → validate_results → rank_sources → validate_urls → ingest_sources

Guards:
  1. Context gate — requires existing context summary, detects stale summaries
  2. Gap quality — deduplicates gaps, checks relevance to request
  3. Result sufficiency — minimum URLs, snippet presence, language check
  4. URL safety — blocked domains, file types, IP URLs, shorteners
  5. Content quality — minimum snippet length, duplicate URL detection
  6. Budget caps — max web search queries per run
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List
from urllib.parse import urlparse

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

MAX_SEARCH_QUERIES_PER_RUN = 15  # 5 gaps × 3 queries
STALE_CONTEXT_DAYS = 7  # flag context summaries older than this


# ── URL safety ──────────────────────────────────────────────────────────────

_BLOCKED_DOMAINS = frozenset({
    "reddit.com", "www.reddit.com",
    "twitter.com", "x.com",
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "linkedin.com", "www.linkedin.com",
    "quora.com", "www.quora.com",
    "drive.google.com", "docs.google.com",
    "dropbox.com", "www.dropbox.com",
    "onedrive.live.com",
    "bit.ly", "tinyurl.com", "t.co",
    "youtube.com", "www.youtube.com",
    "pinterest.com", "www.pinterest.com",
})

_BLOCKED_EXTENSIONS = frozenset({
    ".pdf", ".zip", ".exe", ".dmg", ".msi", ".tar", ".gz",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
})


def _is_url_safe(url: str) -> tuple[bool, str]:
    """Check if a URL is safe and scrapeable."""
    if not url:
        return False, "Empty URL"
    try:
        parsed = urlparse(url)
    except Exception:
        return False, f"Unparseable URL"
    if parsed.scheme not in ("http", "https"):
        return False, f"Non-HTTP scheme: {parsed.scheme}"
    domain = parsed.netloc.lower()
    if domain in _BLOCKED_DOMAINS:
        return False, f"Blocked domain: {domain}"
    path_lower = parsed.path.lower()
    for ext in _BLOCKED_EXTENSIONS:
        if path_lower.endswith(ext):
            return False, f"Blocked file type: {ext}"
    if len(url) > 2000:
        return False, f"URL too long ({len(url)} chars)"
    ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    if ip_pattern.match(domain.split(":")[0]):
        return False, f"IP-based URL: {domain}"
    return True, ""


# ── Helpers ─────────────────────────────────────────────────────────────────


def _string_similarity(a: str, b: str) -> float:
    """Ratio similarity between two strings (0.0-1.0)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _detect_snippet_language(snippet: str) -> bool:
    """Quick check if a snippet is predominantly English (Latin script)."""
    if not snippet:
        return False
    latin_chars = sum(1 for c in snippet if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in snippet if c.isalpha())
    if total_alpha == 0:
        return False
    return (latin_chars / total_alpha) > 0.7


# ── Nodes ───────────────────────────────────────────────────────────────────


def check_context(state: EnrichmentState) -> EnrichmentState:
    """Gate: check that a context summary exists and is fresh.

    - No summary → exit with error
    - Stale summary (> 7 days) → continue with warning
    """
    warnings: List[str] = list(state.get("warnings", []))

    row = get_context_summary.invoke({
        "tenant_id": state["tenant_id"],
        "client_id": state["client_id"],
    })

    if not row or not row.get("summary"):
        return {
            **state,
            "warnings": warnings,
            "status": "no_context",
            "error": (
                "No context summary found. Generate a context summary first "
                "(POST /context/summary/generate) before running enrichment."
            ),
        }

    # Check staleness
    age_days = 0.0
    updated_at = row.get("updated_at") or row.get("created_at")
    if updated_at:
        try:
            if isinstance(updated_at, str):
                ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            else:
                ts = updated_at
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days > STALE_CONTEXT_DAYS:
                warnings.append(
                    f"Context summary is {age_days:.0f} days old (threshold: {STALE_CONTEXT_DAYS}). "
                    "Consider regenerating with POST /context/summary/generate?force_regenerate=true"
                )
                logger.warning("Enrichment: stale context summary (%.0f days old)", age_days)
        except Exception:
            pass

    return {
        **state,
        "context_summary": (
            f"\nContext summary:\nSummary: {row.get('summary', '')}\n"
            f"Topics: {', '.join(row.get('topics', []))}\n\n"
        ),
        "context_summary_age_days": age_days,
        "warnings": warnings,
        "status": "context_available",
    }


def sample_coverage(state: EnrichmentState) -> EnrichmentState:
    """Sample the knowledge base coverage using multiple queries."""
    warnings = list(state.get("warnings", []))
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
            "warnings": warnings,
            "status": "no_content",
            "error": "No content found in knowledge base.",
        }

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

    profile_section = BaseAnalysisService._build_profile_section(
        state.get("client_profile")
    )

    return {
        **state,
        "kg_context": context,
        "profile_section": profile_section,
        "context_sampled": len(sorted_results),
        "warnings": warnings,
        "status": "sampling_complete",
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
        return {**state, "gaps": [], "status": "no_gaps"}

    logger.info("Enrichment: identified %d gaps", len(gaps))

    return {**state, "gaps": gaps[:5], "status": "gaps_identified"}


def validate_gaps(state: EnrichmentState) -> EnrichmentState:
    """Quality check on identified gaps.

    - Deduplicates near-identical gaps (>0.8 string similarity on topic)
    - Checks gap relevance to the user's request
    - Ensures each gap has usable search queries
    """
    warnings = list(state.get("warnings", []))
    gaps = list(state.get("gaps", []))
    request = state.get("request", "").lower().strip()

    if not gaps:
        return {**state, "status": "no_gaps"}

    # ── Deduplicate near-identical gaps ─────────────────────────────────
    deduped: List[Dict[str, Any]] = []
    for gap in gaps:
        topic = gap.get("topic", "")
        is_dup = False
        for existing in deduped:
            if _string_similarity(topic, existing.get("topic", "")) > 0.8:
                is_dup = True
                logger.info("Enrichment: deduped gap '%s' (similar to '%s')", topic, existing.get("topic", ""))
                break
        if not is_dup:
            deduped.append(gap)

    if len(deduped) < len(gaps):
        warnings.append(f"Removed {len(gaps) - len(deduped)} duplicate gap(s).")
    gaps = deduped

    # ── Check gap relevance to request ──────────────────────────────────
    if request and request != "identify gaps in this knowledge base and suggest areas to enrich with external data.":
        relevant_gaps = []
        for gap in gaps:
            topic = gap.get("topic", "").lower()
            reason = gap.get("reason", "").lower()
            # Check if any request keywords appear in the gap
            request_words = set(request.split()) - {"the", "a", "an", "and", "or", "for", "to", "in", "of", "with"}
            overlap = sum(1 for w in request_words if w in topic or w in reason)
            if overlap > 0 or len(request_words) < 3:
                relevant_gaps.append(gap)
            else:
                logger.info("Enrichment: gap '%s' seems unrelated to request '%s'", gap.get("topic", ""), request[:50])

        if relevant_gaps and len(relevant_gaps) < len(gaps):
            warnings.append(f"Filtered {len(gaps) - len(relevant_gaps)} gap(s) unrelated to the request.")
            gaps = relevant_gaps
        # If ALL gaps are irrelevant, keep them anyway — the LLM might know better

    # ── Ensure gaps have search queries ─────────────────────────────────
    valid_gaps = []
    for gap in gaps:
        queries = gap.get("search_queries", [])
        if isinstance(queries, list) and len(queries) > 0:
            valid_gaps.append(gap)
        else:
            warnings.append(f"Gap '{gap.get('topic', '?')}' has no search queries — skipped.")

    if not valid_gaps:
        return {
            **state,
            "gaps": [],
            "warnings": warnings,
            "status": "no_gaps",
            "error": "All identified gaps lack actionable search queries.",
        }

    return {**state, "gaps": valid_gaps, "warnings": warnings, "status": "gaps_validated"}


def search_for_gaps(state: EnrichmentState) -> EnrichmentState:
    """Search the web for content to fill each gap. Tracks query budget."""
    warnings = list(state.get("warnings", []))
    all_search_results: List[Dict[str, Any]] = []
    queries_used = 0

    for gap in state.get("gaps", []):
        queries = gap.get("search_queries", [])[:3]
        gap_results: List[Dict[str, Any]] = []
        seen_urls: set = set()

        for query in queries:
            if queries_used >= MAX_SEARCH_QUERIES_PER_RUN:
                warnings.append(f"Search query budget exhausted ({MAX_SEARCH_QUERIES_PER_RUN} max).")
                break

            results = web_search.invoke({"query": query, "num_results": 5})
            queries_used += 1

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
            "queries_used": queries_used,
            "warnings": warnings,
            "status": "no_search_results",
        }

    return {
        **state,
        "search_results": all_search_results,
        "queries_used": queries_used,
        "warnings": warnings,
        "status": "search_complete",
    }


def validate_results(state: EnrichmentState) -> EnrichmentState:
    """Check that search results are sufficient and relevant.

    - Minimum 3 unique URLs
    - Snippets present (not garbage results)
    - Snippets are predominantly English
    - Duplicate URL detection across gaps
    - Minimum snippet length (> 20 chars)
    """
    warnings = list(state.get("warnings", []))
    search_results = state.get("search_results", [])

    all_urls: List[str] = []
    all_domains: set = set()
    has_snippets = False
    non_english_count = 0
    short_snippet_count = 0
    total_results = 0

    # Deduplicate URLs across gaps
    seen_urls_global: set = set()
    cleaned_results: List[Dict[str, Any]] = []

    for gap_search in search_results:
        cleaned_gap_results = []
        for r in gap_search.get("results", []):
            total_results += 1
            url = r.get("link", "")
            snippet = r.get("snippet", "").strip()

            # Skip duplicate URLs across gaps
            if url in seen_urls_global:
                continue
            seen_urls_global.add(url)

            # Check snippet quality
            if snippet:
                has_snippets = True
                if len(snippet) < 20:
                    short_snippet_count += 1
                    continue  # skip garbage snippets
                if not _detect_snippet_language(snippet):
                    non_english_count += 1
                    continue  # skip non-English

            if url:
                all_urls.append(url)
                try:
                    all_domains.add(urlparse(url).netloc.lower())
                except Exception:
                    pass

            cleaned_gap_results.append(r)

        if cleaned_gap_results:
            cleaned_results.append({
                **gap_search,
                "results": cleaned_gap_results,
            })

    if non_english_count > 0:
        warnings.append(f"Filtered {non_english_count} non-English result(s).")
    if short_snippet_count > 0:
        warnings.append(f"Filtered {short_snippet_count} result(s) with snippets too short (<20 chars).")

    if len(all_urls) < 3:
        return {
            **state,
            "search_results": cleaned_results,
            "warnings": warnings,
            "status": "insufficient_results",
            "error": f"Only {len(all_urls)} usable URLs found (need at least 3). "
                     f"Total raw results: {total_results}, after filtering: {len(all_urls)}.",
        }

    if not has_snippets:
        return {
            **state,
            "search_results": cleaned_results,
            "warnings": warnings,
            "status": "insufficient_results",
            "error": "Search results have no usable snippets.",
        }

    if len(all_domains) == 1:
        warnings.append(f"All {len(all_urls)} results from single domain: {next(iter(all_domains))}")

    return {
        **state,
        "search_results": cleaned_results,
        "warnings": warnings,
        "status": "results_validated",
    }


def rank_sources(state: EnrichmentState) -> EnrichmentState:
    """Rank search results by relevance and quality per gap."""
    max_sources = state.get("max_sources", 5)
    ranked: List[Dict[str, Any]] = []

    for gap_search in state.get("search_results", []):
        search_results_text = "\n".join(
            f"- Title: {r.get('title', '')}\n  URL: {r.get('link', '')}\n  Snippet: {r.get('snippet', '')}"
            for r in gap_search["results"]
        )

        result = rank_web_sources.invoke({
            "gap_topic": gap_search["gap_topic"],
            "gap_reason": gap_search["gap_reason"],
            "search_results_text": search_results_text,
            "max_urls": max(1, max_sources // max(len(state.get("search_results", [])), 1)),
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
        return {**state, "ranked_sources": [], "status": "no_ranked_sources"}

    return {**state, "ranked_sources": final, "status": "ranking_complete"}


def validate_urls(state: EnrichmentState) -> EnrichmentState:
    """Safety check on ranked URLs before ingestion."""
    warnings = list(state.get("warnings", []))
    ranked = state.get("ranked_sources", [])
    safe_sources: List[Dict[str, Any]] = []
    blocked: List[Dict[str, str]] = []

    for src in ranked:
        url = src.get("url", "")
        is_safe, reason = _is_url_safe(url)

        if is_safe:
            safe_sources.append(src)
        else:
            blocked.append({"url": url, "reason": reason})
            logger.info("Enrichment: blocked URL %s — %s", url, reason)

    if blocked:
        warnings.append(f"Blocked {len(blocked)} unsafe URL(s): " +
                        "; ".join(f"{b['url']} ({b['reason']})" for b in blocked[:3]))

    if not safe_sources:
        return {
            **state,
            "ranked_sources": [],
            "urls_blocked": blocked,
            "warnings": warnings,
            "status": "all_urls_blocked",
            "error": f"All {len(ranked)} ranked URLs were blocked as unsafe.",
        }

    return {
        **state,
        "ranked_sources": safe_sources,
        "urls_blocked": blocked,
        "warnings": warnings,
        "status": "urls_validated",
    }


def ingest_sources(state: EnrichmentState) -> EnrichmentState:
    """Ingest the validated web sources into the knowledge base."""
    warnings = list(state.get("warnings", []))
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
        elif result.get("error"):
            warnings.append(f"Ingest failed for {url}: {result['error']}")

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
        "warnings": warnings,
        "status": "complete",
    }


# ── Routing ─────────────────────────────────────────────────────────────────


def route_after_context_check(state: EnrichmentState) -> str:
    if state.get("status") == "no_context":
        return END
    return "sample_coverage"


def route_after_sample(state: EnrichmentState) -> str:
    if state.get("status") == "no_content":
        return END
    return "analyze_gaps"


def route_after_gaps(state: EnrichmentState) -> str:
    if state.get("status") in ("no_gaps",):
        return END
    return "validate_gaps"


def route_after_validate_gaps(state: EnrichmentState) -> str:
    if state.get("status") == "no_gaps":
        return END
    return "search_for_gaps"


def route_after_search(state: EnrichmentState) -> str:
    if state.get("status") == "no_search_results":
        return END
    return "validate_results"


def route_after_validate_results(state: EnrichmentState) -> str:
    if state.get("status") == "insufficient_results":
        return END
    return "rank_sources"


def route_after_rank(state: EnrichmentState) -> str:
    if state.get("status") == "no_ranked_sources":
        return END
    return "validate_urls"


def route_after_validate_urls(state: EnrichmentState) -> str:
    if state.get("status") == "all_urls_blocked":
        return END
    return "ingest_sources"


# ── Graph ───────────────────────────────────────────────────────────────────


def build_enrichment_graph():
    """Build and compile the enrichment LangGraph."""
    graph = StateGraph(EnrichmentState)

    graph.add_node("check_context", check_context)
    graph.add_node("sample_coverage", sample_coverage)
    graph.add_node("analyze_gaps", analyze_gaps)
    graph.add_node("validate_gaps", validate_gaps)
    graph.add_node("search_for_gaps", search_for_gaps)
    graph.add_node("validate_results", validate_results)
    graph.add_node("rank_sources", rank_sources)
    graph.add_node("validate_urls", validate_urls)
    graph.add_node("ingest_sources", ingest_sources)

    graph.set_entry_point("check_context")

    graph.add_conditional_edges("check_context", route_after_context_check)
    graph.add_conditional_edges("sample_coverage", route_after_sample)
    graph.add_conditional_edges("analyze_gaps", route_after_gaps)
    graph.add_conditional_edges("validate_gaps", route_after_validate_gaps)
    graph.add_conditional_edges("search_for_gaps", route_after_search)
    graph.add_conditional_edges("validate_results", route_after_validate_results)
    graph.add_conditional_edges("rank_sources", route_after_rank)
    graph.add_conditional_edges("validate_urls", route_after_validate_urls)
    graph.add_edge("ingest_sources", END)

    return graph.compile()

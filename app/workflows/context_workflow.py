"""
Context generation workflow — retrieves KG content, generates a quality-gated
summary, and stores it back to the memory layer.

Pipeline:
  check_existing → retrieve_kg_content → generate_summary (harnessed) → store_summary

The generate_summary step runs through the harness with cheap checks +
rubric-scored manager evaluation + retries.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from app.llm_config import get_llm
from app.models.states import ContextState
from app.prompts.context_prompts import CONTEXT_GENERATION_PROMPT
from app.harness_pkg import run_with_harness, StepOutput
from app.harness_pkg.configs import CONTEXT_STEP_CONFIG
from app import core_client
from app.tools.context_tools import (
    get_document_summary,
    get_existing_summary,
    get_topic_summary,
    search_knowledge_base,
    store_context_summary,
)

logger = logging.getLogger(__name__)

_COVERAGE_QUERIES = [
    "overview of all topics themes and content",
    "products services offerings customers",
    "industry trends challenges goals strategy",
]


# ── Nodes ───────────────────────────────────────────────────────────────────


def _granularity(state: ContextState) -> str:
    return (state.get("granularity_level") or "tenant").lower()


def check_existing(state: ContextState) -> ContextState:
    """Check if a summary at the requested granularity already exists."""
    if state.get("force_regenerate"):
        return {**state, "existing_summary": None, "status": "regenerating"}

    g = _granularity(state)
    scope_ref = state.get("scope_ref")
    result: dict | None = None
    if g == "document" and scope_ref:
        result = get_document_summary.invoke({
            "tenant_id": state["tenant_id"],
            "client_id": state["client_id"],
            "document_id": scope_ref,
        })
    elif g == "topic" and scope_ref:
        result = get_topic_summary.invoke({
            "tenant_id": state["tenant_id"],
            "client_id": state["client_id"],
            "topic": scope_ref,
        })
    else:
        result = get_existing_summary.invoke({
            "tenant_id": state["tenant_id"],
            "client_id": state["client_id"],
        })

    if result and result.get("summary"):
        # If the stored summary is stale, treat it like missing so we regenerate
        if result.get("is_stale"):
            logger.info("check_existing: stale summary (granularity=%s)", g)
            return {**state, "existing_summary": result, "status": "regenerating"}
        return {
            **state,
            "existing_summary": result,
            "generated_summary": result,
            "status": "cached",
        }

    return {**state, "existing_summary": None, "status": "generating"}


def retrieve_kg_content(state: ContextState) -> ContextState:
    """Retrieve KG content scoped to the requested granularity."""
    all_results: Dict[str, Dict[str, Any]] = {}
    g = _granularity(state)
    scope_ref = state.get("scope_ref")

    if g == "document" and scope_ref:
        # Pull every chunk belonging to this document; no semantic expansion needed.
        try:
            docs = core_client.search_graph(
                tenant_id=state["tenant_id"],
                client_id=state["client_id"],
                query=f"document {scope_ref}",
                top_k=50,
                hop_limit=0,
                node_types=["Chunk"],
                exclude_status=["archived", "deprecated"],
            )
            for doc in docs:
                nid = doc.metadata.get("node_id")
                if nid and nid not in all_results:
                    all_results[nid] = {
                        "content": doc.page_content,
                        "similarity_score": doc.metadata.get("similarity_score", 0.0),
                        "node_id": nid,
                        "document_id": doc.metadata.get("document_id"),
                    }
        except Exception as e:
            logger.warning("document-scoped retrieval failed: %s", e)
    elif g == "topic" and scope_ref:
        # Vector search on the topic string + 1-hop expansion for evidence
        results = search_knowledge_base.invoke({
            "tenant_id": state["tenant_id"],
            "client_id": state["client_id"],
            "query": scope_ref,
            "top_k": 20,
            "hop_limit": 1,
        })
        for r in results:
            nid = r.get("node_id")
            if nid and nid not in all_results:
                all_results[nid] = r
    else:
        # Tenant-wide: existing coverage queries
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

    # Merge new chunks that may not be indexed yet
    new_chunks = state.get("new_chunks", [])
    for i, chunk in enumerate(new_chunks):
        text = chunk.get("text", "").strip()
        if text:
            fake_id = f"new_chunk_{i}"
            if fake_id not in all_results:
                all_results[fake_id] = {
                    "content": text,
                    "similarity_score": 1.0,  # prioritize new content
                    "node_id": fake_id,
                    "source": "new_ingest",
                }

    total_sources = len(all_results)

    if not all_results:
        return {
            **state,
            "kg_results": [],
            "kg_context": "",
            "context_sampled": 0,
            "status": "no_content",
            "error": "No content found in knowledge base. Ingest documents first.",
        }

    if total_sources < 3 and not new_chunks:
        return {
            **state,
            "kg_results": list(all_results.values()),
            "kg_context": "",
            "context_sampled": total_sources,
            "status": "insufficient_content",
            "error": f"Only {total_sources} sources found. Need at least 3 for a meaningful summary.",
        }

    sorted_results = sorted(
        all_results.values(),
        key=lambda r: r.get("similarity_score") or 0.0,
        reverse=True,
    )[:20]  # slightly more to include both old and new

    context = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{r['content']}"
        for i, r in enumerate(sorted_results)
        if r.get("content", "").strip()
    )

    # Build profile section
    profile_section = ""
    client_profile = state.get("client_profile", {})
    if client_profile:
        parts = []
        for key in ("industry", "headcount", "revenue", "company_name", "persona"):
            if client_profile.get(key):
                parts.append(f"{key.replace('_', ' ').title()}: {client_profile[key]}")
        demo = client_profile.get("demographic", {})
        if isinstance(demo, dict):
            for key in ("age_range", "income_bracket", "occupation", "location"):
                if demo.get(key):
                    parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
        if parts:
            profile_section = "Company / Client Profile:\n" + "\n".join(parts) + "\n\n"

    return {
        **state,
        "kg_results": sorted_results,
        "kg_context": context,
        "profile_section": profile_section,
        "context_sampled": len(sorted_results),
        "source_stats": {
            "documents_retrieved": len(sorted_results),
            "context_length": len(context),
        },
        "status": "retrieved",
    }


def generate_summary(state: ContextState) -> ContextState:
    """Generate context summary through the harness."""
    llm = get_llm("context_analysis")
    chain = CONTEXT_GENERATION_PROMPT | llm | StrOutputParser()

    context = state.get("kg_context", "")
    profile_section = state.get("profile_section", "")

    invoke_vars = {
        "context": context if context.strip() else "(No knowledge base content available yet.)",
        "profile_section": profile_section,
    }

    def step_fn(inputs: dict, feedback_section: str):
        prompt_text = (
            f"[system] Business analyst generating context summary\n"
            f"[human] {len(context)} chars of KB excerpts, "
            f"profile: {profile_section[:100] if profile_section else '(none)'}\n"
            f"feedback: {feedback_section[:200] if feedback_section else '(none)'}"
        )

        raw = chain.invoke({**invoke_vars, "feedback_section": feedback_section})

        # Parse JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    parsed = {"summary": raw, "topics": []}
            else:
                parsed = {"summary": raw, "topics": []}

        return StepOutput(
            result=parsed,
            prompt_sent=prompt_text,
            raw_llm_output=raw if isinstance(raw, str) else str(raw),
            tool_calls=[
                {
                    "tool": "search_knowledge_base",
                    "result_summary": f"{state.get('context_sampled', 0)} docs retrieved",
                },
            ],
        )

    result = run_with_harness(
        step_fn,
        {
            "tenant_id": state.get("tenant_id", ""),
            "client_id": state.get("client_id", ""),
            "client_profile": state.get("client_profile", {}),
            "kg_context": context,
            "profile_section": profile_section,
        },
        CONTEXT_STEP_CONFIG,
    )

    if result.output and isinstance(result.output, dict):
        return {
            **state,
            "generated_summary": result.output,
            "status": "generated",
        }
    else:
        return {
            **state,
            "generated_summary": {"summary": "", "topics": []},
            "status": "generation_failed",
            "error": "Harness could not produce a valid context summary.",
        }


def store_summary(state: ContextState) -> ContextState:
    """Store the validated summary back to the memory layer at the requested granularity."""
    summary_data = state.get("generated_summary", {})
    summary = summary_data.get("summary", "")
    topics = summary_data.get("topics", [])

    if not summary:
        return {**state, "status": "store_skipped", "error": "No summary to store."}

    # Track which source chunks contributed — enables evidence traceback + staleness diff
    kg_results = state.get("kg_results", [])
    source_chunk_ids = [
        r.get("document_id") for r in kg_results
        if isinstance(r, dict) and r.get("document_id")
    ]

    result = store_context_summary.invoke({
        "tenant_id": state["tenant_id"],
        "client_id": state["client_id"],
        "summary": summary,
        "topics": topics,
        "source_stats": state.get("source_stats", {}),
        "client_profile": state.get("client_profile", {}),
        "granularity_level": _granularity(state),
        "scope_ref": state.get("scope_ref"),
        "source_chunk_ids": source_chunk_ids,
    })

    if result.get("status") == "ok":
        return {**state, "status": "complete"}
    else:
        return {
            **state,
            "status": "store_failed",
            "error": result.get("error", "Unknown storage error"),
        }


# ── Routing ─────────────────────────────────────────────────────────────────


def route_after_check(state: ContextState) -> str:
    if state.get("status") == "cached":
        return END
    return "retrieve_kg_content"


def route_after_retrieve(state: ContextState) -> str:
    if state.get("status") in ("no_content", "insufficient_content"):
        return END
    return "generate_summary"


def route_after_generate(state: ContextState) -> str:
    if state.get("status") == "generation_failed":
        return END
    return "store_summary"


# ── Graph ───────────────────────────────────────────────────────────────────


def build_context_graph():
    """Build and compile the context generation LangGraph."""
    graph = StateGraph(ContextState)

    graph.add_node("check_existing", check_existing)
    graph.add_node("retrieve_kg_content", retrieve_kg_content)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("store_summary", store_summary)

    graph.set_entry_point("check_existing")

    graph.add_conditional_edges("check_existing", route_after_check)
    graph.add_conditional_edges("retrieve_kg_content", route_after_retrieve)
    graph.add_conditional_edges("generate_summary", route_after_generate)
    graph.add_edge("store_summary", END)

    return graph.compile()

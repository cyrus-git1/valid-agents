"""
Evidence-backed insight workflow.

Given a user question, retrieves summary chunks + their source evidence in a
single hop-1 search, then synthesizes an answer whose every factual claim
cites a source chunk (NOT a summary chunk). Summaries act as navigation hints
that route retrieval to the right evidence neighborhood; they are never
treated as ground truth on their own.

Pipeline:
  classify_scope → retrieve → synthesize → contradiction_check (optional)

Returns:
  {
    answer: str,
    citations: [{chunk_id, document_id, quote}, ...],
    confidence: {
      source_chunks_cited, summaries_used, summary_age_days, coverage
    },
    status: 'complete' | 'insufficient_evidence' | 'failed',
    error: str | None,
  }
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from app import core_client
from app.llm_config import get_llm

logger = logging.getLogger(__name__)

SUMMARY_SOURCE_TYPES = ["ContextSummary", "DocumentSummary", "TopicSummary"]


class InsightState(TypedDict, total=False):
    tenant_id: str
    client_id: str
    question: str
    contradiction_check: bool
    # Classification output
    scope: str                         # 'tenant' | 'document' | 'topic' | 'cross-client'
    scope_ref: Optional[str]           # document_id or topic label, when applicable
    # Retrieval output
    summary_docs: List[Dict[str, Any]]
    source_docs: List[Dict[str, Any]]
    # Synthesis output
    answer: str
    citations: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    status: str
    error: Optional[str]


# ── Prompts ─────────────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You classify insight questions into retrieval scopes.\n"
        "Choose ONE scope:\n"
        "  - 'tenant'       -> broad/org-level question (e.g., 'what are our top risks?')\n"
        "  - 'document'     -> about a specific document (only if question names one)\n"
        "  - 'topic'        -> about a specific subject/entity/theme\n"
        "  - 'cross-client' -> explicit comparison across clients\n\n"
        "Return compact JSON: {{\"scope\": \"...\", \"scope_ref\": null}}.\n"
        "Only set scope_ref when the question contains an explicit topic string or document id.",
    ),
    ("human", "{question}"),
])

_SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a grounded analyst. You have been given SUMMARY chunks and SOURCE chunks.\n\n"
        "STRICT RULES:\n"
        "1. Summary chunks are navigation hints — NEVER cite them as evidence.\n"
        "2. Every factual claim in your answer MUST be supported by a SOURCE chunk; cite it inline as [cX] where X matches the [cX] tag on the SOURCE.\n"
        "3. Preserve uncertainty. If sources disagree or hedge, say so explicitly.\n"
        "4. If the retrieved evidence does not support a confident answer, say 'insufficient evidence' and explain what is missing.\n"
        "5. Do not invent numbers, dates, or attributions that are not in the SOURCE chunks.\n\n"
        "Respond in compact JSON:\n"
        "{{\"answer\": \"...\", \"citations\": [{{\"tag\": \"c1\", \"quote\": \"...\"}}, ...]}}",
    ),
    (
        "human",
        "Question: {question}\n\n"
        "SUMMARIES (navigation only, do NOT cite):\n{summaries}\n\n"
        "SOURCES (cite as [c1], [c2], ...):\n{sources}",
    ),
])

_CONTRADICTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You audit answers for unsupported claims. Given a proposed answer and the source chunks that were available, "
        "list every specific factual claim in the answer that is NOT directly supported by at least one source chunk. "
        "Include numbers, attributions, and date claims. If everything is supported, respond with an empty list.\n\n"
        "Respond in compact JSON: {{\"unsupported_claims\": [\"...\", ...]}}.",
    ),
    (
        "human",
        "Proposed answer:\n{answer}\n\nSources:\n{sources}",
    ),
])


# ── Nodes ───────────────────────────────────────────────────────────────────


def classify_scope(state: InsightState) -> InsightState:
    try:
        llm = get_llm("context_analysis")
    except Exception:
        llm = None
    if llm is None:
        # Fallback: treat as tenant-wide
        return {**state, "scope": "tenant", "scope_ref": None}

    chain = _CLASSIFY_PROMPT | llm | StrOutputParser()
    try:
        raw = chain.invoke({"question": state["question"]})
        parsed = _parse_json(raw) or {}
        scope = parsed.get("scope") or "tenant"
        if scope not in {"tenant", "document", "topic", "cross-client"}:
            scope = "tenant"
        return {**state, "scope": scope, "scope_ref": parsed.get("scope_ref")}
    except Exception as e:
        logger.warning("insight.classify_scope failed: %s", e)
        return {**state, "scope": "tenant", "scope_ref": None}


def retrieve(state: InsightState) -> InsightState:
    """Fetch summaries + their source evidence via a single hop-1 search.

    Because the KG is already wired with mention edges from summaries (weight=0.5)
    to the source chunks' entities, hop_limit=1 from a summary pulls in the
    underlying source chunks automatically alongside the summary.
    """
    try:
        docs = core_client.search_graph(
            tenant_id=state["tenant_id"],
            client_id=state.get("client_id"),
            query=state["question"],
            top_k=30,
            hop_limit=1,
            source_types=SUMMARY_SOURCE_TYPES,  # seeds only from summaries
        )
    except Exception as e:
        logger.warning("insight.retrieve(summaries) failed: %s", e)
        docs = []

    # Second pass: pull source-chunk matches directly too, so we always have
    # raw evidence even if summaries are missing/stale.
    try:
        source_docs = core_client.search_graph(
            tenant_id=state["tenant_id"],
            client_id=state.get("client_id"),
            query=state["question"],
            top_k=20,
            hop_limit=0,
            node_types=["Chunk"],
            # exclude summary parents so we only get original-source chunks
            exclude_status=["archived", "deprecated"],
        )
    except Exception as e:
        logger.warning("insight.retrieve(sources) failed: %s", e)
        source_docs = []

    summary_bucket: List[Dict[str, Any]] = []
    source_bucket: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for d in docs + source_docs:
        nid = d.metadata.get("node_id")
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)
        item = {
            "node_id": nid,
            "document_id": d.metadata.get("document_id"),
            "chunk_id": d.metadata.get("chunk_id"),
            "content": d.page_content,
            "similarity_score": d.metadata.get("similarity_score"),
            "source_type": d.metadata.get("source_type"),
            "updated_at": d.metadata.get("updated_at"),
        }
        if (item["source_type"] or "") in SUMMARY_SOURCE_TYPES:
            summary_bucket.append(item)
        else:
            source_bucket.append(item)

    if not source_bucket:
        return {
            **state,
            "summary_docs": summary_bucket,
            "source_docs": [],
            "status": "insufficient_evidence",
            "error": "No source chunks retrieved — refusing to synthesize from summaries alone.",
        }

    return {
        **state,
        "summary_docs": summary_bucket[:10],
        "source_docs": source_bucket[:20],
        "status": "retrieved",
    }


def synthesize(state: InsightState) -> InsightState:
    llm = get_llm("context_analysis")
    chain = _SYNTHESIZE_PROMPT | llm | StrOutputParser()

    summaries_text = "\n\n".join(
        f"[{s.get('source_type','Summary')}] {s['content'][:600]}"
        for s in (state.get("summary_docs") or [])
    ) or "(none retrieved)"
    sources_text = "\n\n".join(
        f"[c{i+1}] (doc={s.get('document_id')}, chunk={s.get('chunk_id')}) {s['content'][:900]}"
        for i, s in enumerate(state.get("source_docs") or [])
    )

    try:
        raw = chain.invoke({
            "question": state["question"],
            "summaries": summaries_text,
            "sources": sources_text,
        })
        parsed = _parse_json(raw) or {}
    except Exception as e:
        logger.exception("insight.synthesize failed")
        return {
            **state,
            "answer": "",
            "citations": [],
            "confidence": {},
            "status": "failed",
            "error": str(e),
        }

    answer = (parsed.get("answer") or "").strip()
    citation_list = parsed.get("citations") or []

    # Resolve citation tags back to concrete source chunk refs
    source_docs = state.get("source_docs") or []
    resolved_citations: List[Dict[str, Any]] = []
    for c in citation_list:
        tag = (c or {}).get("tag", "").lower()
        m = re.match(r"c(\d+)", tag)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(source_docs):
            src = source_docs[idx]
            resolved_citations.append({
                "tag": tag,
                "chunk_id": src.get("chunk_id"),
                "document_id": src.get("document_id"),
                "quote": (c.get("quote") or "")[:300],
            })

    summary_ages = []
    now = datetime.now(timezone.utc)
    for s in state.get("summary_docs") or []:
        ts = s.get("updated_at")
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                summary_ages.append(round((now - dt).total_seconds() / 86400.0, 1))
            except Exception:
                pass

    confidence = {
        "source_chunks_cited": len(resolved_citations),
        "source_chunks_retrieved": len(source_docs),
        "summaries_used": len(state.get("summary_docs") or []),
        "summary_age_days": summary_ages,
    }

    return {
        **state,
        "answer": answer,
        "citations": resolved_citations,
        "confidence": confidence,
        "status": "synthesized",
    }


def contradiction_check(state: InsightState) -> InsightState:
    if not state.get("contradiction_check"):
        return {**state, "status": "complete"}

    llm = get_llm("context_analysis")
    chain = _CONTRADICTION_PROMPT | llm | StrOutputParser()
    sources_text = "\n\n".join(
        f"[c{i+1}] {s['content'][:900]}"
        for i, s in enumerate(state.get("source_docs") or [])
    )
    try:
        raw = chain.invoke({"answer": state.get("answer", ""), "sources": sources_text})
        parsed = _parse_json(raw) or {}
        unsupported = parsed.get("unsupported_claims") or []
    except Exception as e:
        logger.warning("insight.contradiction_check failed: %s", e)
        unsupported = []

    conf = dict(state.get("confidence") or {})
    conf["unsupported_claims"] = unsupported
    return {**state, "confidence": conf, "status": "complete"}


# ── Routing ─────────────────────────────────────────────────────────────────


def route_after_retrieve(state: InsightState) -> str:
    if state.get("status") == "insufficient_evidence":
        return END
    return "synthesize"


def route_after_synthesize(state: InsightState) -> str:
    if state.get("status") == "failed":
        return END
    return "contradiction_check"


# ── Graph ───────────────────────────────────────────────────────────────────


def build_insight_graph():
    graph = StateGraph(InsightState)
    graph.add_node("classify_scope", classify_scope)
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_node("contradiction_check", contradiction_check)

    graph.set_entry_point("classify_scope")
    graph.add_edge("classify_scope", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_conditional_edges("synthesize", route_after_synthesize)
    graph.add_edge("contradiction_check", END)
    return graph.compile()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
    return None


def run_insight_analysis(
    *,
    tenant_id: str,
    client_id: Optional[str],
    question: str,
    contradiction_check: bool = False,
) -> Dict[str, Any]:
    graph = build_insight_graph()
    initial: InsightState = {
        "tenant_id": tenant_id,
        "client_id": client_id or "",
        "question": question,
        "contradiction_check": contradiction_check,
    }
    final = graph.invoke(initial)
    return {
        "question": question,
        "scope": final.get("scope"),
        "scope_ref": final.get("scope_ref"),
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        "confidence": final.get("confidence", {}),
        "status": final.get("status", "unknown"),
        "error": final.get("error"),
    }

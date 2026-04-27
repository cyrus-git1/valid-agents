"""
Cross-session transcription aggregator.

Pulls all (or a subset of) sessions tagged to a single survey, runs each
through the single-session orchestrator (cache-friendly), then a single
cross-session synthesis LLM call surfaces shared themes, contradictions,
persona patterns, etc.

If `compare_to_session_ids` is provided, runs synthesis twice and includes
a `comparison` block in the response (e.g., "3 of 5 vs other 2").
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from app.agents.transcription.orchestrator import run_orchestration, DEFAULT_LLM_ANALYSES
from app.agents.transcription.session_loader import (
    list_sessions_for_survey,
    load_session_transcript,
)
from app.llm_config import get_llm, LLMConfig
from app.prompts.transcript_aggregate_prompts import CROSS_SESSION_AGGREGATE_PROMPT

logger = logging.getLogger(__name__)


_PER_SESSION_CONCURRENCY = 4


def _parse_json_loose(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _compact_session_payload(session_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a per-session result to just the fields the aggregator prompt needs."""
    if not isinstance(result, dict):
        return {"session_id": session_id, "result": "(unavailable)"}
    summary = result.get("summary") or {}
    sentiment = result.get("sentiment") or {}
    themes = result.get("themes") or {}
    quotes = result.get("quotes") or {}
    discriminate = result.get("discriminate") or {}
    return {
        "session_id": session_id,
        "summary_text": (summary.get("summary") or "")[:1500],
        "action_items": (summary.get("action_items") or [])[:6],
        "decisions": (summary.get("decisions") or [])[:6],
        "themes": (themes.get("themes") or [])[:8],
        "dominant_sentiment": sentiment.get("dominant_sentiment"),
        "overall_sentiment": sentiment.get("overall_sentiment"),
        "notable_quotes": (quotes.get("notable_quotes") or [])[:8],
        "vader_compound": (discriminate.get("vader_sentiment") or {}).get("overall", {}).get("compound"),
    }


def _run_one_session(
    *,
    tenant_id: str,
    client_id: str,
    survey_id: str,
    session_id: str,
    analyses: Optional[List[str]],
    focus: Optional[str],
    summary_type: str,
) -> Dict[str, Any]:
    """Load and orchestrate a single session. Never raises."""
    out = {
        "session_id": session_id,
        "from_cache": False,
        "result": None,
        "warnings": [],
        "error": None,
    }
    try:
        vtt, warnings = load_session_transcript(
            tenant_id=tenant_id, client_id=client_id,
            survey_id=survey_id, session_id=session_id,
        )
        out["warnings"] = warnings
        if vtt is None:
            out["error"] = "session has no transcript content in the KB"
            return out

        result = run_orchestration(
            vtt_content=vtt,
            tenant_id=tenant_id,
            client_id=client_id,
            survey_id=survey_id,
            session_id=session_id,
            analyses=analyses,
            focus=focus,
            summary_type=summary_type,
        )
        out["result"] = result
        out["from_cache"] = bool(result.get("from_cache"))
        return out
    except Exception as e:
        logger.exception("session %s orchestration failed", session_id)
        out["error"] = str(e)
        return out


def _synthesize_cross_session(
    *,
    survey_id: str,
    sessions_payload: List[Dict[str, Any]],
    focus: Optional[str],
) -> Dict[str, Any]:
    """Run the single cross-session synthesis LLM call."""
    if not sessions_payload:
        return {
            "shared_themes": [], "contradictions": [], "persona_patterns": [],
            "aggregate_sentiment": {"overall_compound": 0.0, "per_session": []},
            "notable_quotes_top_n": [], "per_session_summary": [],
        }

    llm = get_llm("context_analysis")
    chain = CROSS_SESSION_AGGREGATE_PROMPT | llm | StrOutputParser()
    sessions_json = json.dumps(sessions_payload, default=str)

    try:
        raw = chain.invoke({
            "focus": focus or "(none)",
            "survey_id": survey_id,
            "session_count": str(len(sessions_payload)),
            "sessions_json": sessions_json,
        })
        parsed = _parse_json_loose(raw)
        return parsed if parsed else {
            "shared_themes": [], "contradictions": [], "persona_patterns": [],
            "aggregate_sentiment": {"overall_compound": 0.0, "per_session": []},
            "notable_quotes_top_n": [], "per_session_summary": [],
            "_raw_unparseable": raw[:1000],
        }
    except Exception as e:
        logger.exception("cross-session synthesis failed")
        return {
            "shared_themes": [], "contradictions": [], "persona_patterns": [],
            "aggregate_sentiment": {"overall_compound": 0.0, "per_session": []},
            "notable_quotes_top_n": [], "per_session_summary": [],
            "error": str(e),
        }


def run_cross_session_aggregation(
    *,
    tenant_id: str,
    client_id: str,
    survey_id: str,
    session_ids: Optional[List[str]] = None,
    compare_to_session_ids: Optional[List[str]] = None,
    analyses: Optional[List[str]] = None,
    focus: Optional[str] = None,
    summary_type: str = "interview",
) -> Dict[str, Any]:
    """Run cross-session aggregation. Returns the unified response dict."""
    t0 = time.monotonic()

    # Default per-session scope
    per_session_analyses = analyses if analyses is not None else DEFAULT_LLM_ANALYSES

    # ── Resolve which sessions to analyse ───────────────────────────────
    all_sessions = list_sessions_for_survey(tenant_id, client_id, survey_id)
    total_sessions = len(all_sessions)

    if session_ids:
        primary_ids = [s for s in session_ids if s in all_sessions]
        missing = [s for s in session_ids if s not in all_sessions]
    else:
        primary_ids = list(all_sessions)
        missing = []

    # Comparison group (if any)
    compare_ids = []
    if compare_to_session_ids:
        compare_ids = [s for s in compare_to_session_ids if s in all_sessions]

    union_ids = list({*primary_ids, *compare_ids})

    # ── Run each unique session through the orchestrator (parallel) ─────
    per_session_results: Dict[str, Dict[str, Any]] = {}
    sessions_failed: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    if union_ids:
        max_workers = min(_PER_SESSION_CONCURRENCY, len(union_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_id = {
                ex.submit(
                    _run_one_session,
                    tenant_id=tenant_id, client_id=client_id,
                    survey_id=survey_id, session_id=sid,
                    analyses=per_session_analyses,
                    focus=focus, summary_type=summary_type,
                ): sid
                for sid in union_ids
            }
            for fut in as_completed(future_to_id):
                sid = future_to_id[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    logger.exception("session %s outer failure", sid)
                    out = {"session_id": sid, "result": None, "error": str(e), "from_cache": False, "warnings": []}
                per_session_results[sid] = out
                if out.get("error"):
                    sessions_failed.append({"session_id": sid, "error": out["error"]})
                    errors.append({"agent": f"session:{sid}", "error": out["error"]})

    # ── Build aggregator inputs (compact payloads) ──────────────────────
    def _payloads_for(ids: List[str]) -> List[Dict[str, Any]]:
        out = []
        for sid in ids:
            entry = per_session_results.get(sid)
            if entry and entry.get("result"):
                out.append(_compact_session_payload(sid, entry["result"]))
        return out

    primary_payloads = _payloads_for(primary_ids)
    aggregate = _synthesize_cross_session(
        survey_id=survey_id,
        sessions_payload=primary_payloads,
        focus=focus,
    )

    comparison: Optional[Dict[str, Any]] = None
    if compare_ids:
        compare_payloads = _payloads_for(compare_ids)
        compare_synthesis = _synthesize_cross_session(
            survey_id=survey_id,
            sessions_payload=compare_payloads,
            focus=focus,
        )
        comparison = {
            "primary_session_ids": primary_ids,
            "compare_session_ids": compare_ids,
            "primary_synthesis_summary": {
                "shared_themes": aggregate.get("shared_themes", []),
                "aggregate_sentiment": aggregate.get("aggregate_sentiment"),
            },
            "compare_synthesis_summary": {
                "shared_themes": compare_synthesis.get("shared_themes", []),
                "aggregate_sentiment": compare_synthesis.get("aggregate_sentiment"),
            },
            "delta_notes": "Compare shared_themes and aggregate_sentiment between the two groups to surface differences.",
        }

    # ── Coverage metadata ───────────────────────────────────────────────
    sessions_analyzed = sum(1 for sid in primary_ids if per_session_results.get(sid, {}).get("result"))
    coverage = {
        "total_sessions": total_sessions,
        "sessions_analyzed": sessions_analyzed,
        "sessions_failed": sessions_failed,
        "missing_session_ids": missing,
    }

    # ── Final assembled response ────────────────────────────────────────
    response = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "survey_id": survey_id,
        "session_ids": primary_ids,
        "per_session_results": [
            {
                "session_id": sid,
                "from_cache": per_session_results.get(sid, {}).get("from_cache", False),
                "result": per_session_results.get(sid, {}).get("result"),
                "error": per_session_results.get(sid, {}).get("error"),
            }
            for sid in primary_ids
        ],
        "aggregate": aggregate,
        "comparison": comparison,
        "coverage": coverage,
        "errors": errors,
        "status": "complete" if not errors else "partial",
        "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
    }
    return response

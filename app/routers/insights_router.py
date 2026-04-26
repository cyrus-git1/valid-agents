"""
/insights router — insight generation endpoints.

POST /insights/analyze   — Evidence-backed Q&A with citations
POST /insights/generate  — Actionable insights, customer understanding,
                            decision support, and meta-insights
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from uuid import UUID

from app import core_client
from app.models.base import TenantScopedRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


# ── Evidence-backed Q&A ───────────────────────────────────────────────────


class InsightAnalyzeRequest(BaseModel):
    tenant_id: UUID
    client_id: Optional[UUID] = None
    study_id: Optional[UUID] = Field(
        default=None,
        description="Optional study scope — only documents tagged with this study_id are searched.",
    )
    question: str = Field(min_length=1)
    contradiction_check: bool = False


@router.post("/analyze")
def analyze_insight(req: InsightAnalyzeRequest):
    """Evidence-backed synthesis over summary + source chunks.

    Retrieves via hop-1 from summary chunks (which mention-edge back to their
    source evidence), then synthesizes an answer whose every factual claim
    cites a SOURCE chunk. When study_id is provided, retrieval is scoped
    to documents tagged with that study.
    """
    from app.workflows.insight_workflow import run_insight_analysis

    try:
        result = run_insight_analysis(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id) if req.client_id else None,
            study_id=str(req.study_id) if req.study_id else None,
            question=req.question,
            contradiction_check=req.contradiction_check,
        )
    except Exception as e:
        logger.exception("Insight analysis failed")
        raise HTTPException(status_code=500, detail=f"Insight analysis failed: {e}")

    return result


# ── Deep analysis: actionable insights, strengths, advantages, + more ─────


class InsightsGenerateRequest(TenantScopedRequest):
    study_id: Optional[UUID] = Field(
        default=None,
        description=(
            "Optional study scope. When provided, all 5 analyses are filtered "
            "to documents tagged with this study_id. Otherwise the entire "
            "tenant/client KB is in scope."
        ),
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus (e.g., 'pricing strategy', 'customer onboarding')",
    )
    analyses: List[str] = Field(
        default=["transcript", "competitive", "synthesis", "objections", "hypotheses"],
        description=(
            "Which deep analyses to run: 'transcript', 'competitive', 'synthesis', "
            "'objections', 'hypotheses'. Defaults to all."
        ),
    )


@router.post("/generate")
def generate_insights(req: InsightsGenerateRequest):
    """Generate actionable insights, strengths, advantages, and research outputs.

    Runs deep analyses and returns a unified report organized into:

    - actionable_insights: prioritized findings + action items + convergent evidence
    - strengths: positioning strengths from competitive analysis
    - advantages: opportunities and win signals
    - recommendations: dedup'd recommendations from all analyses
    - customer_understanding: pain points, objections, personas-affected, sentiment
    - decision_support: contradictions, convergent evidence, hypotheses to test
    - meta_insights: data gaps, document coverage, freshness, sample bias signals
    - results: raw structured output from each analysis tool
    """
    from app.tools.analysis_tools import create_analysis_tools

    tenant_str = str(req.tenant_id)
    client_str = str(req.client_id)

    study_str = str(req.study_id) if req.study_id else None
    tools = create_analysis_tools(tenant_str, client_str, study_id=study_str)
    tool_map = {t.name: t for t in tools}

    valid_analyses = {"transcript", "competitive", "synthesis", "objections", "hypotheses"}
    requested = [a for a in req.analyses if a in valid_analyses]
    if not requested:
        requested = list(valid_analyses)

    tool_name_map = {
        "transcript": "analyze_transcript",
        "competitive": "competitive_intelligence",
        "synthesis": "cross_document_synthesis",
        "objections": "extract_objections",
        "hypotheses": "generate_hypotheses",
    }

    results: Dict[str, Any] = {}
    errors: List[str] = []

    for analysis in requested:
        tool_name = tool_name_map[analysis]
        tool_fn = tool_map.get(tool_name)
        if not tool_fn:
            errors.append(f"Tool {tool_name} not found")
            continue
        try:
            result = tool_fn.invoke({"focus": req.focus})
            results[analysis] = result
        except Exception as e:
            logger.warning("Analysis %s failed: %s", analysis, e)
            results[analysis] = {"status": "failed", "error": str(e)}
            errors.append(f"{analysis}: {e}")

    # ── Unified rollups ────────────────────────────────────────────────
    actionable_insights: List[Dict[str, Any]] = []
    strengths: List[Dict[str, Any]] = []
    advantages: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    t = results.get("transcript") or {}
    if isinstance(t, dict):
        for insight in (t.get("key_insights") or []):
            if isinstance(insight, dict):
                actionable_insights.append({"source": "transcript", **insight})
        for action in (t.get("decisions_and_action_items") or []):
            if isinstance(action, dict):
                actionable_insights.append({"source": "transcript", "type": "action_item", **action})
        for rec in (t.get("recommendations") or []):
            recommendations.append(rec if isinstance(rec, str) else str(rec))

    c = results.get("competitive") or {}
    if isinstance(c, dict):
        pos = c.get("our_positioning") or {}
        for s in (pos.get("strengths") or []):
            if isinstance(s, dict):
                strengths.append({"source": "competitive", **s})
        for opp in (c.get("opportunities") or []):
            if isinstance(opp, dict):
                advantages.append({"source": "competitive", "type": "opportunity", **opp})
        for win in (c.get("win_signals") or []):
            if isinstance(win, dict):
                advantages.append({"source": "competitive", "type": "win_signal", **win})

    s = results.get("synthesis") or {}
    if isinstance(s, dict):
        for rec in (s.get("recommendations") or []):
            recommendations.append(rec if isinstance(rec, str) else str(rec))
        for ev in (s.get("convergent_evidence") or []):
            if isinstance(ev, dict):
                actionable_insights.append({
                    "source": "synthesis",
                    "type": "convergent_finding",
                    "insight": ev.get("conclusion", ""),
                    "confidence": ev.get("confidence", "medium"),
                    "evidence": ev.get("evidence", []),
                })

    # ── Customer understanding ─────────────────────────────────────────
    customer_understanding: Dict[str, Any] = {
        "pain_points": [],
        "objections": [],
        "sentiment": {},
        "personas_affected": [],
    }

    # Pain points from transcript key_insights (category = pain_point)
    if isinstance(t, dict):
        for insight in (t.get("key_insights") or []):
            if isinstance(insight, dict) and insight.get("category") == "pain_point":
                customer_understanding["pain_points"].append(insight)
        customer_understanding["sentiment"] = t.get("sentiment", {}) or {}

    # Objections from dedicated analysis
    o = results.get("objections") or {}
    if isinstance(o, dict):
        customer_understanding["objections"] = o.get("objections", []) or []
        if o.get("top_blocker"):
            customer_understanding["top_blocker"] = o["top_blocker"]

    # Personas affected (collected from transcript findings)
    personas_set: set = set()
    if isinstance(t, dict):
        for insight in (t.get("key_insights") or []):
            if isinstance(insight, dict):
                for p in insight.get("personas_affected", []) or []:
                    if isinstance(p, str):
                        personas_set.add(p)
    customer_understanding["personas_affected"] = sorted(personas_set)

    # ── Decision support ───────────────────────────────────────────────
    decision_support: Dict[str, Any] = {
        "contradictions": [],
        "convergent_evidence": [],
        "hypotheses_to_test": [],
        "top_hypothesis": None,
    }

    if isinstance(s, dict):
        decision_support["contradictions"] = s.get("contradictions", []) or []
        decision_support["convergent_evidence"] = s.get("convergent_evidence", []) or []

    h = results.get("hypotheses") or {}
    if isinstance(h, dict):
        decision_support["hypotheses_to_test"] = h.get("hypotheses", []) or []
        decision_support["top_hypothesis"] = h.get("top_hypothesis")

    # ── Meta insights (KB health signals) ──────────────────────────────
    meta_insights = _compute_meta_insights(tenant_str, client_str, results)

    # Dedup recommendations while preserving order
    seen: set = set()
    dedup_recs: List[str] = []
    for r in recommendations:
        key = r.strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup_recs.append(r)

    return {
        "tenant_id": tenant_str,
        "client_id": client_str,
        "study_id": study_str,
        "focus": req.focus,
        "analyses_requested": requested,
        "actionable_insights": actionable_insights,
        "strengths": strengths,
        "advantages": advantages,
        "recommendations": dedup_recs,
        "customer_understanding": customer_understanding,
        "decision_support": decision_support,
        "meta_insights": meta_insights,
        "results": results,
        "errors": errors,
        "status": "complete" if not errors else "partial",
    }


# ── Meta insights computation (fast — no LLM) ─────────────────────────────


def _compute_meta_insights(
    tenant_id: str,
    client_id: str,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute KB health signals from existing metadata. No LLM calls."""
    out: Dict[str, Any] = {
        "document_coverage": {},
        "data_gaps": [],
        "freshness": {},
        "sample_bias": {},
        "kb_totals": {},
    }

    # Document counts by type
    try:
        data = core_client.list_documents(tenant_id=tenant_id, client_id=client_id)
        items = data.get("items", [])
        _INTERNAL = {"ContextSummary", "DocumentSummary", "TopicSummary"}
        user_docs = [d for d in items if d.get("source_type") not in _INTERNAL]

        by_type: Dict[str, int] = {}
        statuses: Dict[str, int] = {}
        oldest_ts: Optional[str] = None
        newest_ts: Optional[str] = None

        for d in user_docs:
            st = d.get("source_type", "unknown")
            by_type[st] = by_type.get(st, 0) + 1
            status = d.get("status", "active")
            statuses[status] = statuses.get(status, 0) + 1
            updated = d.get("updated_at")
            if updated:
                if oldest_ts is None or updated < oldest_ts:
                    oldest_ts = updated
                if newest_ts is None or updated > newest_ts:
                    newest_ts = updated

        out["document_coverage"] = {
            "by_type": by_type,
            "by_status": statuses,
            "total_user_documents": len(user_docs),
        }
        out["kb_totals"]["documents"] = len(user_docs)

        # Freshness — days since newest/oldest
        now = datetime.now(timezone.utc)
        if newest_ts:
            try:
                dt = datetime.fromisoformat(newest_ts.replace("Z", "+00:00"))
                out["freshness"]["newest_doc_days_ago"] = round((now - dt).total_seconds() / 86400.0, 1)
            except Exception:
                pass
        if oldest_ts:
            try:
                dt = datetime.fromisoformat(oldest_ts.replace("Z", "+00:00"))
                out["freshness"]["oldest_doc_days_ago"] = round((now - dt).total_seconds() / 86400.0, 1)
            except Exception:
                pass

        # Data gap detection — what source types are MISSING?
        _KEY_TYPES = {"web", "pdf", "docx", "vtt", "webvtt", "survey_results"}
        present = set(by_type.keys())
        missing = _KEY_TYPES - present
        gap_impact = {
            "vtt": "No transcripts — cannot analyze voice-of-customer directly",
            "webvtt": "No inline transcripts",
            "pdf": "No document-based content (decks, reports, whitepapers)",
            "docx": "No written reports or internal docs",
            "web": "No web-scraped content (public positioning, competitor sites)",
            "survey_results": "No survey response data — cannot compute statistical confidence",
        }
        for m in missing:
            out["data_gaps"].append({
                "missing_source_type": m,
                "impact": gap_impact.get(m, "Source type not ingested"),
            })

        # Sample bias — if > 80% of docs are one type, flag it
        if user_docs:
            dominant = max(by_type.items(), key=lambda kv: kv[1])
            dom_pct = dominant[1] / len(user_docs)
            if dom_pct > 0.8:
                out["sample_bias"]["dominant_type"] = {
                    "source_type": dominant[0],
                    "percentage": round(dom_pct * 100, 1),
                    "risk": (
                        f"{dominant[1]}/{len(user_docs)} documents are {dominant[0]}. "
                        "Findings may be biased to this source type."
                    ),
                }
    except Exception as e:
        logger.warning("meta_insights doc fetch failed: %s", e)
        out["errors"] = [str(e)]

    # Summary counts and staleness
    try:
        all_summaries = core_client.list_summaries(
            tenant_id=tenant_id, client_id=client_id,
        )
        sums = all_summaries.get("summaries", [])
        by_type: Dict[str, int] = {}
        stale_count = 0
        for s in sums:
            st = s.get("source_type", "unknown")
            by_type[st] = by_type.get(st, 0) + 1
            if s.get("is_stale"):
                stale_count += 1
        out["kb_totals"]["summaries"] = len(sums)
        out["kb_totals"]["summaries_by_type"] = by_type
        if stale_count:
            out["freshness"]["stale_summaries"] = stale_count
    except Exception as e:
        logger.debug("meta_insights summary fetch failed: %s", e)

    # Blind spots from cross-document synthesis
    syn = results.get("synthesis") or {}
    if isinstance(syn, dict):
        blind = syn.get("blind_spots", []) or []
        if blind:
            out["blind_spots"] = blind

    return out

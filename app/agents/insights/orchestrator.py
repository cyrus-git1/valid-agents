"""Multi-agent insights diagnostic orchestrator.

Pipeline: discovery → planner → specialists (parallel) → synthesizer → critic → optional re-run.
Cached by (tenant, client, sorted(survey_ids), study_id, sha256(focus_query), critic_enabled).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app import core_client
from app.agents.insights import cache as insights_cache
from app.agents.insights.critic import (
    SECTION_TO_SPECIALIST,
    run_critic,
    specialists_to_revise,
)
from app.agents.insights.planner import ALL_SPECIALISTS, run_planner
from app.agents.insights.specialists.competitive import run_competitive_specialist
from app.agents.insights.specialists.external import run_external_specialist
from app.agents.insights.specialists.qualitative import run_qualitative_specialist
from app.agents.insights.specialists.quantitative import run_quantitative_specialist
from app.agents.insights.specialists.segments import run_segments_specialist
from app.agents.insights.synthesizer import run_synthesizer

logger = logging.getLogger(__name__)


SPECIALIST_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "quantitative": run_quantitative_specialist,
    "qualitative": run_qualitative_specialist,
    "competitive": run_competitive_specialist,
    "segments": run_segments_specialist,
    "external": run_external_specialist,
}


def _gather_data_inventory(tenant_id: str, client_id: str) -> Dict[str, Any]:
    """Cheap synchronous discovery — used by the planner to decide what to run."""
    transcript_count = 0
    survey_count = 0
    has_documents = False
    summary_counts: Dict[str, int] = {}

    try:
        transcript_count = core_client.count_transcripts(tenant_id=tenant_id, client_id=client_id) or 0
    except Exception:
        pass
    try:
        surveys = core_client.get_survey_outputs(tenant_id=tenant_id, client_id=client_id, limit=50)
        survey_count = len(surveys or [])
    except Exception:
        pass
    try:
        docs = core_client.search_graph(
            tenant_id=tenant_id, client_id=client_id,
            query="all content", top_k=1, hop_limit=0,
            exclude_status=["archived", "deprecated"],
        )
        has_documents = bool(docs)
    except Exception:
        pass
    try:
        all_summaries = core_client.list_summaries(tenant_id=tenant_id, client_id=client_id)
        for s in all_summaries.get("summaries", []):
            st = s.get("source_type", "unknown")
            summary_counts[st] = summary_counts.get(st, 0) + 1
    except Exception:
        pass

    return {
        "transcript_count": transcript_count,
        "survey_count": survey_count,
        "has_documents": has_documents,
        "summary_counts": summary_counts,
    }


def _run_specialists_parallel(
    *,
    specialists_to_run: List[str],
    base_inputs: Dict[str, Any],
    plan: Dict[str, Any],
    parallel: bool,
    revision_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Run the requested specialists with per-specialist focus + revision feedback."""
    revision_overrides = revision_overrides or {}
    per_focus = plan.get("per_specialist_focus") or {}

    def _build_inputs(name: str) -> Dict[str, Any]:
        out = dict(base_inputs)
        out["plan_focus"] = per_focus.get(name, "")
        if name in revision_overrides:
            out["revision_feedback"] = revision_overrides[name]
        return out

    runnable = [(n, SPECIALIST_REGISTRY[n], _build_inputs(n))
                for n in specialists_to_run if n in SPECIALIST_REGISTRY]
    results: List[Dict[str, Any]] = []

    if not runnable:
        return results

    if parallel and len(runnable) > 1:
        max_workers = min(5, len(runnable))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_name = {ex.submit(fn, inp): n for n, fn, inp in runnable}
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.exception("Specialist %s crashed", name)
                    results.append({
                        "name": name, "result": {}, "tool_calls": [],
                        "harness_score": None, "status": "failed", "error": str(e),
                        "elapsed_ms": 0.0,
                    })
    else:
        for n, fn, inp in runnable:
            try:
                results.append(fn(inp))
            except Exception as e:
                logger.exception("Specialist %s crashed", n)
                results.append({
                    "name": n, "result": {}, "tool_calls": [],
                    "harness_score": None, "status": "failed", "error": str(e),
                    "elapsed_ms": 0.0,
                })
    return results


def _summarise_specialists(specialist_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": o.get("name"),
            "status": o.get("status"),
            "harness_score": o.get("harness_score"),
            "tool_calls": len(o.get("tool_calls") or []),
            "elapsed_ms": o.get("elapsed_ms", 0.0),
            "error": o.get("error"),
        }
        for o in specialist_outputs
    ]


def _add_skipped_to_summary(
    summary: List[Dict[str, Any]], plan: Dict[str, Any]
) -> List[Dict[str, Any]]:
    skip_reasons = plan.get("skip_reasons") or {}
    seen = {s["name"] for s in summary}
    for spec in ALL_SPECIALISTS:
        if spec in seen:
            continue
        if spec in skip_reasons:
            summary.append({
                "name": spec, "status": "skipped",
                "harness_score": None, "tool_calls": 0,
                "elapsed_ms": 0.0, "error": None,
                "skip_reason": skip_reasons[spec],
            })
    return summary


def run_diagnostic_pipeline(
    *,
    tenant_id: str,
    client_id: str,
    focus_query: Optional[str] = None,
    survey_ids: Optional[List[str]] = None,
    study_id: Optional[str] = None,
    client_profile: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    critic_enabled: bool = True,
) -> Dict[str, Any]:
    """Root entrypoint for the multi-agent insights diagnostic.

    Returns the v2 unified report wrapped with a `pipeline` audit section.
    """
    started = time.monotonic()

    # Cache lookup
    cache_key = insights_cache.make_key(
        tenant_id, client_id, survey_ids, study_id, focus_query, critic_enabled,
    )
    cached = insights_cache.get(cache_key)
    if cached is not None:
        cached = dict(cached)
        cached["from_cache"] = True
        cached.setdefault("metadata", {})["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
        return cached

    # 1. Discovery
    data_inventory = _gather_data_inventory(tenant_id, client_id)

    # 2. Planner
    plan = run_planner(
        tenant_id=tenant_id, client_id=client_id, focus_query=focus_query,
        survey_ids=survey_ids, study_id=study_id, data_inventory=data_inventory,
    )

    base_inputs: Dict[str, Any] = {
        "tenant_id": tenant_id, "client_id": client_id,
        "client_profile": client_profile,
        "focus_query": focus_query,
        "survey_ids": survey_ids, "study_id": study_id,
        "data_inventory": data_inventory,
    }

    # 3. Specialists in parallel
    specialists_to_run = list(plan.get("specialists_to_run") or [])
    specialist_outputs = _run_specialists_parallel(
        specialists_to_run=specialists_to_run,
        base_inputs=base_inputs, plan=plan, parallel=parallel,
    )

    # 4. Synthesizer
    report = run_synthesizer(
        plan=plan, specialist_outputs=specialist_outputs,
        tenant_id=tenant_id, client_id=client_id,
        focus_query=focus_query, survey_ids=survey_ids, study_id=study_id,
    )

    # 5. Critic (optional, single round)
    critic_result: Optional[Dict[str, Any]] = None
    revision_round_run = False
    if critic_enabled:
        critic_result = run_critic(plan=plan, report=report)
        if not critic_result.get("passes") and critic_result.get("weak_sections"):
            pairs = specialists_to_revise(critic_result)
            if pairs and len(pairs) <= 2:
                revision_round_run = True
                revision_specs = [p[0] for p in pairs]
                revision_overrides = {p[0]: p[1] for p in pairs}
                logger.info("Critic flagged sections; re-running specialists: %s", revision_specs)
                revised_outputs = _run_specialists_parallel(
                    specialists_to_run=revision_specs,
                    base_inputs=base_inputs, plan=plan, parallel=parallel,
                    revision_overrides=revision_overrides,
                )
                # Replace the prior specialist outputs for the revised specialists
                by_name = {o.get("name"): o for o in specialist_outputs}
                for o in revised_outputs:
                    by_name[o.get("name")] = o
                specialist_outputs = list(by_name.values())
                # Re-synthesize
                report = run_synthesizer(
                    plan=plan, specialist_outputs=specialist_outputs,
                    tenant_id=tenant_id, client_id=client_id,
                    focus_query=focus_query, survey_ids=survey_ids, study_id=study_id,
                )

    # 6. Assemble final response
    pipeline_summary = _add_skipped_to_summary(
        _summarise_specialists(specialist_outputs), plan,
    )
    total_tool_calls = sum(len(o.get("tool_calls") or []) for o in specialist_outputs)
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)

    failed = [s for s in pipeline_summary if s["status"] == "failed"]
    status = "complete" if not failed else "partial"

    response: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "survey_ids": survey_ids or [],
        "study_id": study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "from_cache": False,
        # v2 report (flattened at top level so it matches existing /insights schema)
        **report,
        # Pipeline audit
        "pipeline": {
            "plan": plan,
            "specialists": pipeline_summary,
            "critic": critic_result,
            "revision_round_run": revision_round_run,
            "data_inventory": data_inventory,
        },
        "metadata": {
            "total_tool_calls": total_tool_calls,
            "elapsed_ms": elapsed_ms,
        },
        "status": status,
        "errors": [{"specialist": s["name"], "error": s.get("error")}
                   for s in pipeline_summary if s["status"] == "failed"],
    }

    insights_cache.put(cache_key, response)
    return response

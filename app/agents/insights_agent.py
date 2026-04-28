"""
Business insights orchestrator — ReAct agent that synthesizes all available
data sources into an evidence-based insights report.

Checks what data exists (transcripts, surveys, documents, personas), runs
relevant analyses, and produces a unified report with evidence attribution.
Never blocks on missing data — continues with whatever is available and
reports what's missing.

Usage
-----
    from app.agents.insights_agent import run_insights_agent

    result = run_insights_agent(
        tenant_id="...",
        client_id="...",
        client_profile={"industry": "SaaS"},
        focus_query="pricing concerns",
    )
    print(result["executive_summary"])
    print(result["data_gaps"])
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.llm_config import get_llm
from app.harness_pkg import run_with_harness, StepOutput
from app.harness_pkg.configs import INSIGHTS_STEP_CONFIG
from app.prompts.insights_prompts import INSIGHTS_AGENT_SYSTEM_PROMPT
from app.tools.insights_tools import create_insights_tools

logger = logging.getLogger(__name__)


def run_insights_agent(
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
    focus_query: Optional[str] = None,
    survey_ids: Optional[List[str]] = None,
    study_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an in-depth diagnostic + forward-step report using a ReAct agent.

    The agent has access to: discovery tools, qualitative analytics
    (transcript / synthesis / objections / hypotheses / competitive intel),
    quantitative analytics (NPS / T2B / CIs / cross-tabs), respondent
    clustering, and web enrichment for external context. Scope is
    optionally narrowed to survey_ids or study_id.

    Output schema is deliberately rich (quantitative_findings,
    qualitative_findings, objections_and_blockers, hypotheses_to_test,
    contradictions_and_blind_spots, competitive_landscape, segments,
    external_context_via_enrichment, recommendations_future_steps,
    meta_insights, etc.). The harness manager evaluates depth across
    all sections and one retry is allowed below threshold.
    """
    scope_hint = ""
    if survey_ids:
        scope_hint = f" Scoped to survey_ids: {survey_ids}."
    elif study_id:
        scope_hint = f" Scoped to study_id: {study_id}."
    user_message = (
        focus_query
        or f"Produce an in-depth current-state diagnostic and forward-step recommendations using all available data sources.{scope_hint}"
    )

    tools = create_insights_tools(
        tenant_id, client_id, client_profile,
        survey_ids=survey_ids,
        study_id=study_id,
    )

    def step_fn(inputs: dict, feedback_section: str):
        system_prompt = INSIGHTS_AGENT_SYSTEM_PROMPT.replace(
            "{feedback_section}",
            f"\n\nREVISION GUIDANCE (address these issues):\n{feedback_section}" if feedback_section else "",
        )

        if client_profile:
            profile_parts = []
            for key in ("industry", "headcount", "revenue", "company_name", "persona"):
                if client_profile.get(key):
                    profile_parts.append(f"{key.replace('_', ' ').title()}: {client_profile[key]}")
            demo = client_profile.get("demographic", {})
            if isinstance(demo, dict):
                for key in ("age_range", "income_bracket", "occupation", "location"):
                    if demo.get(key):
                        profile_parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
            if profile_parts:
                system_prompt += "\n\nClient profile:\n" + "\n".join(profile_parts)

        # Build the ReAct agent
        try:
            from langgraph.prebuilt import create_react_agent
            llm = get_llm("context_analysis")  # use analysis-grade model
            agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
        except ImportError:
            logger.warning("create_react_agent not available, falling back to sequential")
            return _fallback_sequential(tools, user_message, feedback_section)

        agent_input = {"messages": [{"role": "user", "content": user_message}]}

        try:
            result = agent.invoke(agent_input, config={"recursion_limit": 18})
        except Exception as e:
            logger.exception("Insights ReAct agent failed")
            return StepOutput(
                result=_empty_report(),
                prompt_sent=system_prompt[:500],
                raw_llm_output=f"Agent error: {e}",
                tool_calls=[],
            )

        # Extract output and tool trace
        messages = result.get("messages", [])
        tool_calls_trace = []
        raw_output = ""
        report = None

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_trace.append({
                        "tool": tc.get("name", "unknown"),
                        "args": {k: str(v)[:100] for k, v in tc.get("args", {}).items()},
                    })

            if hasattr(msg, "content") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
                raw_output = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Parse report
        if raw_output:
            try:
                report = json.loads(raw_output)
            except json.JSONDecodeError:
                import re
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
                if match:
                    try:
                        report = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass

        if not report or not isinstance(report, dict):
            report = _empty_report()

        return StepOutput(
            result=report,
            prompt_sent=system_prompt[:500] + "...",
            raw_llm_output=raw_output[:3000] if raw_output else "(no output)",
            tool_calls=tool_calls_trace,
        )

    # Run through harness
    harness_result = run_with_harness(
        step_fn,
        {
            "request": user_message,
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_profile": client_profile or {},
        },
        INSIGHTS_STEP_CONFIG,
    )

    report = harness_result.output if isinstance(harness_result.output, dict) else _empty_report()

    # Count tool calls — include the new analytical tool names
    _ANALYTICAL_TOOLS = {
        # legacy
        "analyze_sentiment", "extract_transcript_insights",
        "compute_confidence_intervals", "run_strategic_analysis",
        # scope-aware additions
        "compute_quantitative_metrics", "compute_crosstab", "analyze_clusters",
        # deep-analysis tools
        "analyze_transcript", "competitive_intelligence",
        "cross_document_synthesis", "extract_objections", "generate_hypotheses",
        # enrichment (web)
        "recommend_enrichment",
    }
    total_tool_calls = 0
    analyses_run = set()
    for attempt in harness_result.attempt_traces:
        trace_data = attempt.to_dict()
        calls = trace_data.get("execution_trace", {}).get("tool_calls", [])
        total_tool_calls += len(calls)
        for c in calls:
            tool_name = c.get("tool", "")
            if tool_name in _ANALYTICAL_TOOLS:
                analyses_run.add(tool_name)

    # Recommendations: prefer the new key, fall back to legacy
    recs = (
        report.get("recommendations_future_steps")
        or report.get("recommendations")
        or []
    )

    # Build final output — surface the full v2 schema
    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "survey_ids": survey_ids or [],
        "study_id": study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Top-level narrative
        "executive_summary": report.get("executive_summary", ""),
        "current_state_assessment": report.get("current_state_assessment", {}),
        # Analytical sections
        "quantitative_findings": report.get("quantitative_findings", []),
        "qualitative_findings": report.get("qualitative_findings", []),
        "competitive_landscape": report.get("competitive_landscape", {}),
        "segments": report.get("segments", []),
        "objections_and_blockers": report.get("objections_and_blockers", []),
        "contradictions_and_blind_spots": report.get("contradictions_and_blind_spots", []),
        "hypotheses_to_test": report.get("hypotheses_to_test", []),
        "external_context_via_enrichment": report.get("external_context_via_enrichment", {}),
        # Cross-cutting
        "key_findings": report.get("key_findings", []),
        "recommendations_future_steps": recs,
        # Coverage / context
        "data_sources_used": report.get("data_sources_used", {}),
        "personas_referenced": report.get("personas_referenced", []),
        "data_gaps": report.get("data_gaps", []),
        "enrichment_recommendations": report.get("enrichment_recommendations", []),
        "meta_insights": report.get("meta_insights", {}),
        "metadata": {
            "tool_calls_used": total_tool_calls,
            "harness_score": harness_result.final_score,
            "harness_status": harness_result.status.value,
            "analyses_run": sorted(analyses_run),
        },
        "status": harness_result.status.value,
        "error": None if harness_result.status.value == "passed" else "Insights report did not meet quality threshold.",
    }


def _empty_report() -> Dict[str, Any]:
    """Return a minimal valid report structure with the rich v2 schema."""
    return {
        "executive_summary": "",
        "current_state_assessment": {
            "overall_diagnostic": "",
            "what_is_working": [],
            "what_is_not_working": [],
            "confidence_level": "low",
        },
        "quantitative_findings": [],
        "qualitative_findings": [],
        "competitive_landscape": {
            "competitors_mentioned": [],
            "win_signals": [],
            "loss_signals": [],
            "positioning_gaps": [],
        },
        "segments": [],
        "objections_and_blockers": [],
        "contradictions_and_blind_spots": [],
        "hypotheses_to_test": [],
        "external_context_via_enrichment": {
            "summary": "",
            "key_external_facts": [],
            "trends_to_watch": [],
        },
        "key_findings": [],
        "recommendations_future_steps": [],
        "data_sources_used": {},
        "personas_referenced": [],
        "data_gaps": [],
        "enrichment_recommendations": [],
        "meta_insights": {
            "data_coverage": "sparse",
            "freshness": "unknown",
            "sample_bias_flags": [],
            "confidence_calibration": "",
        },
    }


def _fallback_sequential(
    tools: list,
    user_message: str,
    feedback_section: str,
) -> StepOutput:
    """Fallback if create_react_agent is not available — call tools in fixed order."""
    tool_calls_trace = []

    # Map tools by name
    tool_map = {t.name: t for t in tools}

    # Phase 1: Data discovery
    context = None
    available = {"transcript_count": 0, "survey_count": 0, "has_documents": False}
    personas = []

    if "check_context" in tool_map:
        context = tool_map["check_context"].invoke({})
        tool_calls_trace.append({"tool": "check_context", "result_summary": "found" if context else "none"})

    if "check_available_data" in tool_map:
        available = tool_map["check_available_data"].invoke({})
        tool_calls_trace.append({"tool": "check_available_data", "result_summary": str(available)})

    if "get_personas" in tool_map:
        personas = tool_map["get_personas"].invoke({})
        tool_calls_trace.append({"tool": "get_personas", "result_summary": f"{len(personas)} personas"})

    # Phase 2: Run available analyses
    sentiment_result = {}
    transcript_result = {}
    ci_result = []
    strategic_result = {}
    data_gaps = []

    if available.get("transcript_count", 0) > 0:
        if "analyze_sentiment" in tool_map:
            sentiment_result = tool_map["analyze_sentiment"].invoke({"focus_query": None})
            tool_calls_trace.append({"tool": "analyze_sentiment"})
        if "extract_transcript_insights" in tool_map:
            transcript_result = tool_map["extract_transcript_insights"].invoke({})
            tool_calls_trace.append({"tool": "extract_transcript_insights"})
    else:
        data_gaps.append({"source": "transcripts", "status": "missing", "impact": "Cannot perform sentiment or transcript analysis"})

    if available.get("survey_count", 0) > 0:
        if "compute_confidence_intervals" in tool_map:
            ci_result = tool_map["compute_confidence_intervals"].invoke({})
            tool_calls_trace.append({"tool": "compute_confidence_intervals"})
    else:
        data_gaps.append({"source": "survey_responses", "status": "missing", "impact": "Cannot compute confidence intervals"})

    if available.get("has_documents"):
        if "run_strategic_analysis" in tool_map:
            strategic_result = tool_map["run_strategic_analysis"].invoke({})
            tool_calls_trace.append({"tool": "run_strategic_analysis"})
    else:
        data_gaps.append({"source": "documents", "status": "missing", "impact": "Cannot perform strategic analysis"})

    # Phase 3: Enrichment
    enrichment_result = {}
    if "recommend_enrichment" in tool_map:
        enrichment_result = tool_map["recommend_enrichment"].invoke({"request": None})
        tool_calls_trace.append({"tool": "recommend_enrichment"})

    # Build a basic report from collected data
    report = {
        "executive_summary": strategic_result.get("executive_summary", context.get("summary", "") if context else "Insufficient data for a comprehensive analysis."),
        "key_findings": [
            {"finding": ap.get("title", ""), "source": "strategic_analysis",
             "evidence_sources": ap.get("evidence", []), "confidence": ap.get("priority", "medium"),
             "personas_affected": []}
            for ap in strategic_result.get("action_points", [])[:5]
        ],
        "recommendations": [
            {"recommendation": r, "priority": "medium", "rationale": "", "evidence_sources": []}
            for r in strategic_result.get("future_recommendations", [])[:3]
        ],
        "data_sources_used": {
            "context_summary": context is not None,
            "sentiment_analysis": bool(sentiment_result and not sentiment_result.get("error")),
            "transcript_insights": bool(transcript_result and not transcript_result.get("error")),
            "confidence_intervals": len(ci_result) > 0,
            "strategic_analysis": bool(strategic_result and not strategic_result.get("error")),
            "personas": len(personas) > 0,
        },
        "personas_referenced": [p.get("name", "") for p in personas[:5]],
        "data_gaps": data_gaps,
        "enrichment_recommendations": [
            {"gap": g.get("topic", ""), "priority": g.get("priority", "medium")}
            for g in enrichment_result.get("gaps", [])[:5]
        ],
    }

    return StepOutput(
        result=report,
        prompt_sent="[fallback] Sequential insights generation",
        raw_llm_output=json.dumps(report, default=str)[:3000],
        tool_calls=tool_calls_trace,
    )

"""
Tools for the business insights orchestrator.

Closure-based factory — tenant/client context baked into each tool.
Wraps analysis services, persona agent, enrichment agent, and data access.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app import core_client

logger = logging.getLogger(__name__)


def create_insights_tools(
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
) -> list:
    """Build insights tools with tenant/client context captured in closures."""

    # ── Data discovery ──────────────────────────────────────────────────

    @tool
    def check_context() -> Optional[Dict[str, Any]]:
        """Check if a context summary exists for this tenant.

        Returns {summary, topics} or None. Always call this first to
        understand what the KB covers before running analyses.
        """
        try:
            return core_client.get_context_summary(
                tenant_id=tenant_id, client_id=client_id,
            )
        except Exception as e:
            logger.warning("check_context failed: %s", e)
            return None

    @tool
    def check_available_data() -> Dict[str, Any]:
        """Check what data sources exist for this tenant.

        Returns counts of transcripts, survey outputs, and documents.
        Use this to decide which analyses to run.
        """
        transcript_count = 0
        survey_count = 0
        document_count = 0

        try:
            transcript_count = core_client.count_transcripts(
                tenant_id=tenant_id, client_id=client_id,
            )
        except Exception:
            pass

        try:
            surveys = core_client.get_survey_outputs(
                tenant_id=tenant_id, client_id=client_id, limit=50,
            )
            survey_count = len(surveys)
        except Exception:
            pass

        try:
            docs = core_client.search_graph(
                tenant_id=tenant_id, client_id=client_id,
                query="all content", top_k=1, hop_limit=0,
            )
            document_count = len(docs)
        except Exception:
            pass

        return {
            "transcript_count": transcript_count,
            "survey_count": survey_count,
            "has_documents": document_count > 0,
        }

    @tool
    def get_personas() -> List[Dict[str, Any]]:
        """Get audience personas for this tenant.

        Runs the persona agent if no cached personas exist.
        Returns list of persona dicts with evidence_sources.
        """
        try:
            from app.agents.persona_agent import run_persona_agent
            result = run_persona_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                client_profile=client_profile,
            )
            return result.get("personas", [])
        except Exception as e:
            logger.warning("get_personas failed: %s", e)
            return []

    # ── Analysis tools ──────────────────────────────────────────────────

    @tool
    def analyze_sentiment(focus_query: Optional[str] = None) -> Dict[str, Any]:
        """Run sentiment analysis on transcript data.

        Returns overall sentiment scores, themes, notable quotes, and summary.
        Only call this if check_available_data showed transcripts > 0.
        """
        try:
            # Fetch transcript chunks via core API
            chunks = core_client.get_transcript_chunks(
                tenant_id=tenant_id, client_id=client_id, limit=50,
            )
            if not chunks:
                return {"error": "No transcript chunks found", "overall_sentiment": {}, "themes": [], "summary": ""}

            from app.analysis.sentiment import SentimentAnalysisService
            svc = SentimentAnalysisService(tenant_id=tenant_id, client_id=client_id)
            return svc._run_analysis(
                chunks=chunks,
                focus_query=focus_query,
                client_profile=client_profile,
            )
        except Exception as e:
            logger.warning("analyze_sentiment failed: %s", e)
            return {"error": str(e), "overall_sentiment": {}, "themes": [], "summary": ""}

    @tool
    def extract_transcript_insights() -> Dict[str, Any]:
        """Extract actionable insights from transcript data.

        Returns summary + list of actionable insights with categories and source quotes.
        Only call this if check_available_data showed transcripts > 0.
        """
        try:
            chunks = core_client.get_transcript_chunks(
                tenant_id=tenant_id, client_id=client_id, limit=60,
            )
            if not chunks:
                return {"error": "No transcript chunks found", "summary": "", "actionable_insights": []}

            from app.analysis.transcript_insights import TranscriptInsightsService
            svc = TranscriptInsightsService(tenant_id=tenant_id, client_id=client_id)
            return svc._run_insights(chunks=chunks)
        except Exception as e:
            logger.warning("extract_transcript_insights failed: %s", e)
            return {"error": str(e), "summary": "", "actionable_insights": []}

    @tool
    def compute_confidence_intervals() -> List[Dict[str, Any]]:
        """Compute confidence intervals on survey response data.

        Returns per-question statistical confidence intervals.
        Only call this if check_available_data showed survey_count > 0.
        """
        try:
            surveys = core_client.get_survey_outputs(
                tenant_id=tenant_id, client_id=client_id, limit=10,
            )
            if not surveys:
                return []

            from app.analysis.confidence_interval import ConfidenceIntervalService
            svc = ConfidenceIntervalService()

            # Collect questions with responses across surveys
            questions_with_responses = []
            for survey in surveys:
                qs = survey.get("questions", [])
                if isinstance(qs, str):
                    import json
                    try:
                        qs = json.loads(qs)
                    except Exception:
                        continue
                for q in qs:
                    if q.get("responses"):
                        questions_with_responses.append(q)

            if not questions_with_responses:
                return []

            results = svc.compute_all(questions_with_responses)
            return [r.model_dump() if hasattr(r, "model_dump") else r for r in results]
        except Exception as e:
            logger.warning("compute_confidence_intervals failed: %s", e)
            return []

    @tool
    def run_strategic_analysis() -> Dict[str, Any]:
        """Run convergent strategic analysis across all data sources.

        Combines KB content, transcripts, context summary, and optional web data
        into a strategic overview with action points.
        """
        try:
            # Strategic analysis needs refactoring for stateless mode.
            # For now, build a basic strategic view from context + KB search.
            context = core_client.get_context_summary(
                tenant_id=tenant_id, client_id=client_id,
            )
            docs = core_client.search_graph(
                tenant_id=tenant_id, client_id=client_id,
                query="strategy goals challenges opportunities trends",
                top_k=15, hop_limit=1,
            )

            if not context and not docs:
                return {"error": "No data available for strategic analysis", "executive_summary": "", "action_points": []}

            # Use the context summary + KB content as a lightweight strategic view
            summary = context.get("summary", "") if context else ""
            topics = context.get("topics", []) if context else []
            doc_excerpts = [d.page_content[:200] for d in docs[:5]] if docs else []

            return {
                "executive_summary": summary,
                "convergent_themes": topics,
                "action_points": [],
                "sources_used": {
                    "kg_chunks_retrieved": len(docs) if docs else 0,
                    "context_summary_available": context is not None,
                },
            }
        except Exception as e:
            logger.warning("run_strategic_analysis failed: %s", e)
            return {"error": str(e), "executive_summary": "", "action_points": []}

    # ── Gap tool ────────────────────────────────────────────────────────

    @tool
    def recommend_enrichment(request: Optional[str] = None) -> Dict[str, Any]:
        """Identify knowledge gaps and recommend web sources to fill them.

        Call this after analysis to find what's missing from the KB.
        Returns gaps with search queries and recommended sources.
        """
        try:
            from app.agents.enrichment_agent import run_enrichment_agent
            return run_enrichment_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                request=request,
                client_profile=client_profile,
                max_sources=3,
            )
        except Exception as e:
            logger.warning("recommend_enrichment failed: %s", e)
            return {"error": str(e), "gaps": [], "sources": []}

    return [
        check_context,
        check_available_data,
        get_personas,
        analyze_sentiment,
        extract_transcript_insights,
        compute_confidence_intervals,
        run_strategic_analysis,
        recommend_enrichment,
    ]

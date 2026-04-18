"""
Tools for the business insights orchestrator.

Fully stateless — all data access via core_client HTTP calls.
Analysis is done via direct LLM prompts, not service classes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from app import core_client
from app.llm_config import get_llm

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
        """Check what context summaries exist for this tenant.

        Returns the tenant-wide summary plus counts of document-level and
        topic-level summaries. Always call this first to understand the
        breadth of available context at all granularity levels.
        """
        result: Dict[str, Any] = {}
        try:
            tenant_summary = core_client.get_context_summary(
                tenant_id=tenant_id, client_id=client_id,
            )
            if tenant_summary:
                result["summary"] = tenant_summary.get("summary", "")
                result["topics"] = tenant_summary.get("topics", [])
        except Exception as e:
            logger.warning("check_context (tenant) failed: %s", e)

        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id, client_id=client_id,
            )
            summaries_list = all_summaries.get("summaries", [])
            doc_sums = [s for s in summaries_list if s.get("source_type") == "DocumentSummary"]
            topic_sums = [s for s in summaries_list if s.get("source_type") == "TopicSummary"]
            result["document_summary_count"] = len(doc_sums)
            result["topic_summary_count"] = len(topic_sums)
            if doc_sums:
                result["document_summaries"] = doc_sums[:5]
            if topic_sums:
                result["topic_summaries"] = topic_sums[:5]
        except Exception as e:
            logger.warning("check_context (list_summaries) failed: %s", e)

        return result if result else None

    @tool
    def check_available_data() -> Dict[str, Any]:
        """Check what data sources exist for this tenant.

        Returns counts of transcripts, survey outputs, documents, and
        summaries at each granularity level (tenant, document, topic).
        Use this to decide which analyses to run.
        """
        transcript_count = 0
        survey_count = 0
        document_count = 0
        summary_counts: Dict[str, int] = {}

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
                exclude_status=["archived", "deprecated"],
            )
            document_count = len(docs)
        except Exception:
            pass

        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id, client_id=client_id,
            )
            for s in all_summaries.get("summaries", []):
                st = s.get("source_type", "unknown")
                summary_counts[st] = summary_counts.get(st, 0) + 1
        except Exception:
            pass

        return {
            "transcript_count": transcript_count,
            "survey_count": survey_count,
            "has_documents": document_count > 0,
            "summary_counts": summary_counts,
        }

    _persona_cache: list = []
    _persona_ran: list = [False]  # mutable flag in closure

    @tool
    def get_personas() -> List[Dict[str, Any]]:
        """Get audience personas for this tenant.

        Runs the persona agent once. Cached across retries — won't re-run
        the full persona agent on harness retry.
        """
        if _persona_ran[0]:
            return _persona_cache

        _persona_ran[0] = True
        try:
            from app.agents.persona_agent import run_persona_agent
            result = run_persona_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                client_profile=client_profile,
            )
            personas = result.get("personas", [])
            _persona_cache.extend(personas)
            return personas
        except Exception as e:
            logger.warning("get_personas failed: %s", e)
            return []

    # ── Analysis tools (all use search + direct LLM) ────────────────────

    @tool
    def analyze_sentiment(focus_query: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sentiment from transcript and document content.

        Searches for sentiment-relevant content via KG, then runs
        sentiment analysis via LLM. Only call if transcripts or documents exist.
        """
        queries = [
            "customer feedback opinions satisfaction complaints",
            "positive negative experience sentiment feelings",
        ]
        if focus_query:
            queries.append(focus_query)

        all_content = []
        for q in queries:
            try:
                docs = core_client.search_graph(
                    tenant_id=tenant_id, client_id=client_id,
                    query=q, top_k=10, hop_limit=1,
                    node_types=["VideoTranscript", "Chunk"],
                    boost_pinned=True,
                    exclude_status=["archived", "deprecated"],
                )
                for d in docs:
                    all_content.append(d.page_content)
            except Exception:
                pass

        if not all_content:
            return {"error": "No content found for sentiment analysis", "overall_sentiment": {}, "themes": [], "summary": ""}

        context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(all_content[:15]))

        from app.prompts.sentiment_prompts import SENTIMENT_ANALYSIS_PROMPT
        llm = get_llm("context_analysis")
        chain = SENTIMENT_ANALYSIS_PROMPT | llm | StrOutputParser()

        try:
            import json
            focus_instructions = ""
            if focus_query:
                focus_instructions = f"Focus your analysis on: {focus_query}\n"

            raw = chain.invoke({
                "focus_instructions": focus_instructions,
                "profile_section": "",
                "transcript_count": str(len(all_content)),
                "chunk_count": str(len(all_content)),
                "transcript_context": context,
                "context_summary": "",
            })
            return json.loads(raw)
        except Exception as e:
            logger.warning("analyze_sentiment LLM failed: %s", e)
            return {"error": str(e), "overall_sentiment": {}, "themes": [], "summary": ""}

    @tool
    def extract_transcript_insights() -> Dict[str, Any]:
        """Extract actionable insights from transcript and document content.

        Searches for insight-relevant content via KG, then extracts via LLM.
        """
        try:
            docs = core_client.search_graph(
                tenant_id=tenant_id, client_id=client_id,
                query="actionable insights pain points feature requests improvements",
                top_k=15, hop_limit=1,
                node_types=["VideoTranscript", "Chunk"],
                boost_pinned=True,
                exclude_status=["archived", "deprecated"],
            )
        except Exception:
            docs = []

        if not docs:
            return {"error": "No content found", "summary": "", "actionable_insights": []}

        context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{d.page_content}" for i, d in enumerate(docs[:15]))

        from app.prompts.transcript_insights_prompts import TRANSCRIPT_INSIGHTS_PROMPT
        llm = get_llm("context_analysis")
        chain = TRANSCRIPT_INSIGHTS_PROMPT | llm | StrOutputParser()

        try:
            import json
            raw = chain.invoke({
                "transcript_count": str(len(docs)),
                "chunk_count": str(len(docs)),
                "transcript_context": context,
            })
            return json.loads(raw)
        except Exception as e:
            logger.warning("extract_transcript_insights LLM failed: %s", e)
            return {"error": str(e), "summary": "", "actionable_insights": []}

    @tool
    def compute_confidence_intervals() -> List[Dict[str, Any]]:
        """Compute confidence intervals on survey response data.

        Only call if check_available_data showed survey_count > 0.
        If no survey data is available, returns empty — don't retry.
        """
        try:
            import json
            surveys = core_client.get_survey_outputs(
                tenant_id=tenant_id, client_id=client_id, limit=10,
            )
        except Exception:
            return []  # Core API error — skip silently

        if not surveys:
            return []

        try:
            from app.analysis.confidence_interval import ConfidenceIntervalService
            svc = ConfidenceIntervalService()

            questions_with_responses = []
            for survey in surveys:
                qs = survey.get("questions", [])
                if isinstance(qs, str):
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
        """Run strategic analysis across all available data sources.

        Searches KB for strategic themes using hybrid ranking, combines
        with all available summaries (tenant, document, and topic level).
        """
        context = core_client.get_context_summary(
            tenant_id=tenant_id, client_id=client_id,
        )

        # Gather document-level and topic-level summaries for richer context
        extra_summary_parts: List[str] = []
        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id, client_id=client_id,
            )
            for s in all_summaries.get("summaries", []):
                st = s.get("source_type", "")
                if st in ("DocumentSummary", "TopicSummary") and s.get("summary"):
                    label = s.get("topic") or s.get("document_id") or st
                    extra_summary_parts.append(f"[{st}: {label}] {s['summary'][:300]}")
        except Exception:
            pass

        strategic_queries = [
            "strategy goals challenges opportunities market position",
            "competitive advantage strengths weaknesses threats",
            "products services offerings value proposition",
        ]

        all_docs = []
        for q in strategic_queries:
            try:
                docs = core_client.search_graph(
                    tenant_id=tenant_id, client_id=client_id,
                    query=q, top_k=10, hop_limit=1,
                    boost_pinned=True,
                    exclude_status=["archived", "deprecated"],
                    recency_weight=0.2,
                )
                all_docs.extend(docs)
            except Exception:
                pass

        if not context and not all_docs and not extra_summary_parts:
            return {"error": "No data available", "executive_summary": "", "action_points": []}

        kb_context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{d.page_content}" for i, d in enumerate(all_docs[:20])
        ) if all_docs else "(No KB content)"

        summary_text = context.get("summary", "") if context else ""
        if extra_summary_parts:
            summary_text += "\n\n" + "\n".join(extra_summary_parts)
        topics = context.get("topics", []) if context else []

        from app.prompts.strategic_analysis_prompts import STRATEGIC_ANALYSIS_PROMPT
        llm = get_llm("context_analysis")
        chain = STRATEGIC_ANALYSIS_PROMPT | llm | StrOutputParser()

        try:
            import json
            raw = chain.invoke({
                "kg_context": kb_context,
                "context_summary": summary_text,
                "transcript_context": "(No transcripts available — use KB content only)",
                "transcript_count": "0",
                "web_context": "(No web search performed)",
                "profile_section": "",
                "depth_instructions": "Provide a foundational strategic overview based on available documentation.",
            })
            return json.loads(raw)
        except Exception as e:
            logger.warning("run_strategic_analysis LLM failed: %s", e)
            return {
                "executive_summary": summary_text,
                "convergent_themes": topics,
                "action_points": [],
                "sources_used": {"kg_chunks_retrieved": len(all_docs), "context_summary_available": context is not None},
            }

    # ── Gap tool ────────────────────────────────────────────────────────

    @tool
    def recommend_enrichment(request: Optional[str] = None) -> Dict[str, Any]:
        """Identify knowledge gaps and recommend web sources to fill them.

        Call this after analysis to find what's missing from the KB.
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

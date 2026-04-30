"""
Tools for the business insights orchestrator.

Fully stateless — all data access via core_client HTTP calls.
Analysis is done via direct LLM prompts, not service classes.
"""
from __future__ import annotations

import json
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
    survey_ids: Optional[List[str]] = None,
    study_id: Optional[str] = None,
) -> list:
    """Build insights tools with tenant/client + optional survey/study scope.

    When survey_ids or study_id are provided, deep-analysis tools and
    clustering scope their KB queries to those documents. The agent uses
    this scope to focus on a specific research project's data.
    """

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

    # ── Gap tool ────────────────────────────────────────────────────────

    @tool
    def recommend_enrichment(request: Optional[str] = None) -> Dict[str, Any]:
        """Identify knowledge gaps and ENRICH them via web search.

        Calls the enrichment agent which searches the web (via Serper) for
        external context, competitor info, market data, etc. Returns
        identified gaps + web-discovered sources. Use this to add forward-
        looking external context that isn't in the user's KB yet.
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

    # ── Deep-analysis tools (scope-aware via study_id) ──────────────────

    def _make_deep_analysis_tools():
        """Build scope-aware wrappers around the 5 deep-analysis tools.

        These pass through the active study_id so the deep tools filter
        their KB queries to documents in scope.
        """
        from app.tools.analysis_tools import create_analysis_tools
        analysis_tools_list = create_analysis_tools(
            tenant_id, client_id, study_id=study_id,
        )
        # The deep-analysis tools have rich docstrings already; we just
        # surface them under simpler names to make the system prompt
        # easier to write.
        return analysis_tools_list  # 5 tools: analyze_transcript,
        # competitive_intelligence, cross_document_synthesis,
        # extract_objections, generate_hypotheses

    deep_tools = _make_deep_analysis_tools()

    # ── Quantitative metrics on scoped survey responses ─────────────────

    @tool
    def compute_quantitative_metrics() -> Dict[str, Any]:
        """Compute statistical metrics on scoped survey responses.

        Uses the survey_ids scope (or all surveys for this tenant if no
        scope) and runs ConfidenceIntervalService. Returns:
          - mean_ci, top_2_box, bottom_2_box for rating questions
          - nps_score with promoter/passive/detractor split + NPS CI
          - proportion CIs for yes_no, multiple_choice, checkbox
          - rank CIs for ranking
          - significance flags
        Skip if no surveys exist; never fabricate metrics.
        """
        try:
            outputs = core_client.get_survey_outputs(
                tenant_id=tenant_id, client_id=client_id, limit=50,
            )
        except Exception as e:
            return {"status": "failed", "error": str(e), "questions": []}
        if not outputs:
            return {
                "status": "no_data",
                "message": "No survey outputs in this tenant/client.",
                "questions": [],
            }

        # Scope filter: keep only outputs whose metadata.survey_id matches
        # one of the scoped survey_ids, when scoping is active.
        scoped_outputs = outputs
        if survey_ids:
            scoped_outputs = []
            for o in outputs:
                md = o.get("metadata") or {}
                if md.get("survey_id") in survey_ids or o.get("id") in survey_ids:
                    scoped_outputs.append(o)
            if not scoped_outputs:
                return {
                    "status": "no_data_in_scope",
                    "message": "No survey outputs match the active survey_ids scope.",
                    "questions": [],
                }

        # Flatten all questions with responses across the surveys in scope
        questions_payload: List[Dict[str, Any]] = []
        for o in scoped_outputs:
            qs = o.get("questions") or []
            if isinstance(qs, str):
                try:
                    qs = json.loads(qs)
                except Exception:
                    continue
            for q in qs:
                if not q.get("responses"):
                    continue
                questions_payload.append({
                    "question_id": q.get("id") or q.get("question_id") or "",
                    "question_type": q.get("type") or q.get("question_type") or "",
                    "label": q.get("label", ""),
                    "responses": q.get("responses", []),
                    "options": q.get("options"),
                })

        if not questions_payload:
            return {
                "status": "no_responses",
                "message": "Surveys exist but no questions have responses yet.",
                "questions": [],
            }

        try:
            from app.analysis.confidence_interval import ConfidenceIntervalService
            results = ConfidenceIntervalService().compute_all(questions_payload)
            return {
                "status": "complete",
                "n_questions_analyzed": len(results),
                "questions": [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
            }
        except Exception as e:
            logger.exception("compute_quantitative_metrics failed")
            return {"status": "failed", "error": str(e), "questions": []}

    # ── Cross-tab on scoped survey data ─────────────────────────────────

    @tool
    def compute_crosstab(
        row_question_id: Optional[str] = None,
        col_question_id: Optional[str] = None,
        row_tag_field: Optional[str] = None,
        col_tag_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a cross-tabulation on the scoped survey data.

        Pair any two dimensions (question vs question, question vs tag,
        tag vs tag) — same length per row. Returns contingency table +
        chi-square + Cramér's V + standardised residuals.

        Pass at least one row_* and one col_*. Use this to surface
        statistically significant segmentation patterns: "How does NPS
        differ between Senior and Mid-level respondents?"
        """
        if not (row_question_id or row_tag_field):
            return {"status": "failed", "error": "Need a row dimension."}
        if not (col_question_id or col_tag_field):
            return {"status": "failed", "error": "Need a column dimension."}

        try:
            outputs = core_client.get_survey_outputs(
                tenant_id=tenant_id, client_id=client_id, limit=50,
            )
        except Exception as e:
            return {"status": "failed", "error": str(e)}
        if not outputs:
            return {"status": "no_data", "message": "No survey outputs."}

        scoped_outputs = outputs
        if survey_ids:
            scoped_outputs = [
                o for o in outputs
                if (o.get("metadata") or {}).get("survey_id") in survey_ids
                or o.get("id") in survey_ids
            ]
        if not scoped_outputs:
            return {"status": "no_data_in_scope"}

        # Flatten questions across scoped outputs
        flat_questions: List[Dict[str, Any]] = []
        for o in scoped_outputs:
            qs = o.get("questions") or []
            if isinstance(qs, str):
                try:
                    qs = json.loads(qs)
                except Exception:
                    continue
            flat_questions.extend(qs)

        # Demographic tags pulled from each survey output's metadata
        respondent_tags = [(o.get("metadata") or {}) for o in scoped_outputs]

        try:
            from app.analysis.crosstab import (
                compute_crosstab as _ct,
                extract_paired_dimensions_from_survey,
            )
            rows, cols, row_label, col_label = extract_paired_dimensions_from_survey(
                survey_questions=flat_questions,
                row_question_id=row_question_id,
                col_question_id=col_question_id,
                row_tag_field=row_tag_field,
                col_tag_field=col_tag_field,
                respondent_tags=respondent_tags,
            )
            return _ct(
                rows=rows, cols=cols,
                row_label=row_label, col_label=col_label,
            )
        except ValueError as e:
            return {"status": "failed", "error": str(e)}
        except Exception as e:
            logger.exception("compute_crosstab failed")
            return {"status": "failed", "error": str(e)}

    # ── Cluster analysis scoped by survey_ids ───────────────────────────

    @tool
    def analyze_clusters(k: Optional[int] = None) -> Dict[str, Any]:
        """Run respondent clustering on sessions in the active survey scope.

        Uses HDBSCAN by default (no k needed). Returns clusters with
        defining_tags, top_terms, dominant_traits, and human-readable
        labels. Use this to surface respondent segments: "Three distinct
        groups emerged — pricing-sensitive founders, risk-averse
        enterprise buyers, growth-focused product leads."
        """
        try:
            from app.agents.clustering.respondent_clustering_agent import (
                run_cluster_analysis,
            )
            return run_cluster_analysis(
                tenant_id=tenant_id,
                client_id=client_id,
                survey_ids=survey_ids,
                study_id=study_id,
                k=k,
                produce_labels=True,
                include_text_embedding=True,
            )
        except Exception as e:
            logger.exception("analyze_clusters failed")
            return {"status": "failed", "error": str(e)}

    return [
        check_context,
        check_available_data,
        get_personas,
        analyze_sentiment,
        extract_transcript_insights,
        compute_confidence_intervals,
        recommend_enrichment,
        # Scope-aware additions ↓
        compute_quantitative_metrics,
        compute_crosstab,
        analyze_clusters,
        # Deep-analysis tools (5) — already scoped by study_id when provided
        *deep_tools,
    ]

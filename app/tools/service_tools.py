"""
Tools for the service agent.

Wraps every agent and service as a closure-bound tool so the ReAct
agent's tool signatures are clean (no UUIDs). Each tool handles its
own errors and returns a dict — never raises.

Usage
-----
    tools = create_service_tools(tenant_id="...", client_id="...")
    # Each tool is a @tool-decorated callable with clean signatures
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.tools import tool

from app import core_client

logger = logging.getLogger(__name__)


def create_service_tools(
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
) -> list:
    """Build all service tools with tenant/client context captured in closures."""

    @tool
    def ask_question(question: str) -> Dict[str, Any]:
        """Answer a question using evidence from the knowledge base.

        Uses the insight workflow to retrieve source chunks, synthesize
        an answer, and provide citations. Best for factual questions
        about KB content. Returns answer, citations, confidence, and status.
        """
        try:
            from app.workflows.insight_workflow import run_insight_analysis
            result = run_insight_analysis(
                tenant_id=tenant_id,
                client_id=client_id,
                question=question,
                contradiction_check=False,
            )
            return {
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", {}),
                "status": result.get("status", "unknown"),
                "error": result.get("error"),
            }
        except Exception as e:
            logger.warning("ask_question failed: %s", e)
            return {"error": str(e), "status": "failed", "answer": ""}

    @tool
    def generate_survey(request: str) -> Dict[str, Any]:
        """Generate a survey questionnaire grounded in KB content.

        Creates survey questions based on the request, using KB context
        for industry-specific relevance. Saves the output to the memory
        layer. This is a slow operation (10-30s).
        """
        try:
            from app.agents.survey_agent import run_survey_agent
            result = run_survey_agent(
                request=request,
                tenant_id=tenant_id,
                client_id=client_id,
                client_profile=client_profile,
            )
            # Persist the survey output
            survey_json = result.get("survey", "[]")
            try:
                questions = json.loads(survey_json)
            except json.JSONDecodeError:
                questions = []
            if questions:
                try:
                    core_client.save_survey_output(
                        tenant_id=tenant_id,
                        client_id=client_id,
                        output_type="survey",
                        request=request,
                        questions=questions,
                    )
                except Exception:
                    logger.warning("Failed to persist survey output", exc_info=True)
            return result
        except Exception as e:
            logger.warning("generate_survey failed: %s", e)
            return {"error": str(e), "status": "failed", "survey": "[]"}

    @tool
    def find_personas(request: Optional[str] = None) -> Dict[str, Any]:
        """Discover audience personas from KB data.

        Explores the knowledge base iteratively to identify distinct
        audience segments with evidence. This is a slow operation (15-60s).
        """
        try:
            from app.agents.persona_agent import run_persona_agent
            result = run_persona_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                request=request,
                client_profile=client_profile,
            )
            return {
                "personas": result.get("personas", []),
                "metadata": result.get("metadata", {}),
                "status": result.get("status", "unknown"),
                "error": result.get("error"),
            }
        except Exception as e:
            logger.warning("find_personas failed: %s", e)
            return {"error": str(e), "status": "failed", "personas": []}

    @tool
    def enrich_kb(request: Optional[str] = None) -> Dict[str, Any]:
        """Identify KB knowledge gaps and recommend web sources to fill them.

        Analyzes what the KB covers, finds gaps, searches the web for
        relevant sources. This is a slow operation (15-60s).
        """
        try:
            from app.agents.enrichment_agent import run_enrichment_agent
            result = run_enrichment_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                request=request,
                client_profile=client_profile,
            )
            return {
                "gaps": result.get("gaps", []),
                "sources": result.get("sources", []),
                "status": result.get("status", "unknown"),
                "error": result.get("error"),
            }
        except Exception as e:
            logger.warning("enrich_kb failed: %s", e)
            return {"error": str(e), "status": "failed", "gaps": []}

    @tool
    def ingest_url(web_url: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a web URL into the knowledge base.

        Scrapes the URL, chunks the content, extracts entities, and
        stores everything in the KB. Also regenerates the context summary.
        Accepts URLs with or without https:// prefix.

        This is a synchronous operation that may take 30-60 seconds.
        After completion, report the results to the user including
        document_id, chunks stored, and entities found.
        """
        try:
            from app.models.ingest import IngestInput
            from app.services.ingest.service import IngestService
            # Normalize URL — add https:// if missing
            url = web_url.strip()
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            inp = IngestInput(
                tenant_id=UUID(tenant_id),
                client_id=UUID(client_id),
                web_url=url,
                title=title,
            )
            result = IngestService().ingest(inp)
            return {
                "document_id": str(result.document_id),
                "source_uri": url,
                "source_type": result.source_type,
                "chunks_upserted": result.chunks_upserted,
                "entities_linked": result.entities_linked,
                "warnings": result.warnings,
                "status": "complete",
                "message": (
                    f"Successfully ingested {url}. "
                    f"Stored {result.chunks_upserted} content chunks and "
                    f"linked {result.entities_linked} entities."
                ),
            }
        except Exception as e:
            logger.warning("ingest_url failed: %s", e)
            return {"error": str(e), "status": "failed", "source_uri": web_url}

    @tool
    def check_status() -> Dict[str, Any]:
        """Check the current state of the knowledge base.

        Returns document count, context summary status, and any
        recent activity. Use this when the user asks about the state
        of their data, what's been ingested, or whether things are working.
        """
        result: Dict[str, Any] = {}

        # Document count
        try:
            data = core_client.list_documents(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            items = data.get("items", [])
            result["document_count"] = len(items)
            result["documents"] = [
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "source_type": d.get("source_type"),
                    "status": d.get("status"),
                    "chunks": len(d.get("chunks", [])),
                }
                for d in items[:10]
            ]
        except Exception as e:
            result["document_count"] = 0
            result["document_error"] = str(e)

        # Context summary status
        try:
            summary = core_client.get_context_summary(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            if summary and summary.get("summary"):
                result["context_summary"] = "available"
                result["topics"] = summary.get("topics", [])
            else:
                result["context_summary"] = "not_built"
        except Exception:
            result["context_summary"] = "not_built"

        # Summary counts
        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            summaries_list = all_summaries.get("summaries", [])
            result["document_summaries"] = sum(
                1 for s in summaries_list if s.get("source_type") == "DocumentSummary"
            )
            result["topic_summaries"] = sum(
                1 for s in summaries_list if s.get("source_type") == "TopicSummary"
            )
        except Exception:
            pass

        result["status"] = "ok"
        return result

    @tool
    def build_context(force_regenerate: bool = False) -> Dict[str, Any]:
        """Generate or regenerate the context summary from KB content.

        Scans the KB and produces a tenant-level summary with topics.
        Use force_regenerate=True to rebuild even if a cached summary exists.
        """
        try:
            from app.agents.context_agent import run_context_agent
            return run_context_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                client_profile=client_profile,
                force_regenerate=force_regenerate,
            )
        except Exception as e:
            logger.warning("build_context failed: %s", e)
            return {"error": str(e), "status": "failed", "summary": ""}

    @tool
    def list_documents() -> Dict[str, Any]:
        """List all documents in the knowledge base with chunk previews.

        Returns document metadata (title, source URL, type, status),
        chunk count, and a preview of the first chunk's content for each
        document. Use this when the user asks what's been uploaded,
        what's available, or what can be deleted.
        """
        try:
            data = core_client.list_documents(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            items = data.get("items", [])
            if not items:
                return {
                    "total": 0,
                    "documents": [],
                    "message": "Your knowledge base is empty — no documents have been ingested yet.",
                }
            # Filter out internal documents (summaries are system-generated, not user content)
            _INTERNAL_TYPES = {"ContextSummary", "DocumentSummary", "TopicSummary"}
            user_items = [d for d in items if d.get("source_type") not in _INTERNAL_TYPES]

            if not user_items:
                return {
                    "total": 0,
                    "documents": [],
                    "message": "Your knowledge base is empty — no documents have been ingested yet.",
                }

            documents = []
            for d in user_items[:20]:
                documents.append({
                    "id": d.get("id"),
                    "title": d.get("title") or "Untitled",
                    "source_url": d.get("source_uri") or d.get("source_type", "unknown"),
                    "source_type": d.get("source_type"),
                    "status": d.get("status", "active"),
                })
            return {
                "total": len(user_items),
                "documents": documents,
            }
        except Exception as e:
            logger.warning("list_documents failed: %s", e)
            return {"error": str(e), "status": "failed", "documents": []}

    @tool
    def flag_document(
        old_document_id: str,
        corrected_text: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replace a document with corrected content.

        Ingests the corrected text as a new document, deletes the old one,
        and regenerates affected summaries. Use when a user has edited
        document content and wants to update the KB.
        """
        try:
            from app.services.revision_service import RevisionService
            result = RevisionService().revise_document(
                tenant_id=UUID(tenant_id),
                client_id=UUID(client_id),
                old_document_id=old_document_id,
                corrected_text=corrected_text,
                title=title,
            )
            return {
                "old_document_id": result.old_document_id,
                "new_document_id": result.new_document_id,
                "chunks_upserted": result.chunks_upserted,
                "old_document_deleted": result.old_document_deleted,
                "summaries_regenerated": result.summaries_regenerated,
                "warnings": result.warnings,
                "status": "complete",
            }
        except Exception as e:
            logger.warning("flag_document failed: %s", e)
            return {"error": str(e), "status": "failed"}

    @tool
    def get_summary() -> Optional[Dict[str, Any]]:
        """Fetch the current context summary for this tenant.

        Returns the tenant-wide summary plus counts of document-level
        and topic-level summaries. Fast read-only operation.
        """
        result: Dict[str, Any] = {}
        try:
            tenant_summary = core_client.get_context_summary(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            if tenant_summary:
                result["summary"] = tenant_summary.get("summary", "")
                result["topics"] = tenant_summary.get("topics", [])
        except Exception as e:
            logger.warning("get_summary (tenant) failed: %s", e)

        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            summaries_list = all_summaries.get("summaries", [])
            result["document_summary_count"] = sum(
                1 for s in summaries_list if s.get("source_type") == "DocumentSummary"
            )
            result["topic_summary_count"] = sum(
                1 for s in summaries_list if s.get("source_type") == "TopicSummary"
            )
        except Exception as e:
            logger.warning("get_summary (list) failed: %s", e)

        return result if result else None

    @tool
    def search_kb(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge base for raw content chunks.

        Returns content chunks with similarity scores. Use ask_question
        instead if you need a synthesized answer with citations.
        This tool is for raw search results only.
        """
        try:
            docs = core_client.search_graph(
                tenant_id=tenant_id,
                client_id=client_id,
                query=query,
                top_k=top_k,
                hop_limit=1,
                boost_pinned=True,
                exclude_status=["archived", "deprecated"],
            )
            return [
                {
                    "content": doc.page_content[:500],
                    "similarity_score": doc.metadata.get("similarity_score", 0.0),
                    "document_id": doc.metadata.get("document_id"),
                    "node_id": doc.metadata.get("node_id"),
                }
                for doc in docs
            ]
        except Exception as e:
            logger.warning("search_kb failed: %s", e)
            return []

    return [
        ask_question,
        generate_survey,
        find_personas,
        enrich_kb,
        ingest_url,
        build_context,
        list_documents,
        flag_document,
        get_summary,
        search_kb,
        check_status,
    ]

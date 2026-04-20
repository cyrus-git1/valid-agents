"""
Deep analysis tools for the service agent.

  - analyze_transcript: summary + sentiment + insights from transcript data
  - competitive_intelligence: entity-focused competitor extraction
  - cross_document_synthesis: multi-source pattern detection across all docs
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from app import core_client
from app.llm_config import get_llm

logger = logging.getLogger(__name__)


def create_analysis_tools(
    tenant_id: str,
    client_id: str,
) -> list:
    """Build deep analysis tools with tenant/client context in closures."""

    # ── Transcript analysis ────────────────────────────────────────────

    @tool
    def analyze_transcript(focus: Optional[str] = None) -> Dict[str, Any]:
        """Run deep analysis on transcript content in the knowledge base.

        Searches for transcript/interview data, then produces:
        - Executive summary of key discussions
        - Sentiment analysis (positive/negative themes with quotes)
        - Actionable insights and recommendations
        - Key moments and decisions

        Use 'focus' to narrow the analysis (e.g., 'pricing concerns',
        'product feedback', 'onboarding experience').
        """
        # Fetch transcript content via KG search
        queries = [
            "transcript interview conversation discussion call",
            focus or "customer feedback pain points feature requests",
        ]
        all_content: List[str] = []
        seen: set = set()

        for q in queries:
            try:
                docs = core_client.search_graph(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    query=q,
                    top_k=15,
                    hop_limit=1,
                    node_types=["VideoTranscript", "Chunk"],
                    boost_pinned=True,
                    exclude_status=["archived", "deprecated"],
                )
                for d in docs:
                    nid = d.metadata.get("node_id")
                    if nid and nid not in seen:
                        seen.add(nid)
                        all_content.append(d.page_content)
            except Exception:
                pass

        if not all_content:
            return {
                "status": "no_data",
                "message": "No transcript or interview content found in the knowledge base. Ingest .vtt transcripts first.",
            }

        context = "\n\n---\n\n".join(
            f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(all_content[:20])
        )

        focus_instruction = f"Focus your analysis on: {focus}\n" if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior research analyst. Analyze the transcript excerpts "
                "and produce a comprehensive analysis.\n\n"
                "Return JSON with these sections:\n"
                "{{\n"
                '  "executive_summary": "2-3 paragraph overview of key themes",\n'
                '  "sentiment": {{\n'
                '    "overall": "positive/negative/mixed",\n'
                '    "positive_themes": [{{"theme": "...", "evidence": "direct quote"}}],\n'
                '    "negative_themes": [{{"theme": "...", "evidence": "direct quote"}}]\n'
                "  }},\n"
                '  "key_insights": [\n'
                '    {{"insight": "...", "evidence": "quote", "priority": "high/medium/low", "category": "pain_point/opportunity/risk/request"}}\n'
                "  ],\n"
                '  "decisions_and_action_items": [\n'
                '    {{"item": "...", "owner": "...", "status": "decided/pending/open"}}\n'
                "  ],\n"
                '  "key_moments": [\n'
                '    {{"moment": "...", "significance": "why it matters"}}\n'
                "  ],\n"
                '  "recommendations": ["..."],\n'
                '  "data_gaps": ["what additional data would strengthen the analysis"]\n'
                "}}\n\n"
                "Rules:\n"
                "- Every insight MUST include a direct quote as evidence\n"
                "- Prioritize specificity over generality\n"
                "- Flag contradictions between different speakers/documents\n"
                "- Separate observations from interpretations",
            ),
            ("human", "{focus}Transcript excerpts:\n\n{context}"),
        ])

        llm = get_llm("context_analysis")
        chain = prompt | llm | StrOutputParser()

        try:
            raw = chain.invoke({"focus": focus_instruction, "context": context})
            result = _parse_json(raw)
            if result:
                result["status"] = "complete"
                result["excerpts_analyzed"] = len(all_content)
                return result
        except Exception as e:
            logger.warning("analyze_transcript failed: %s", e)

        return {"status": "failed", "error": "Analysis failed", "excerpts_analyzed": len(all_content)}

    # ── Competitive intelligence ───────────────────────────────────────

    @tool
    def competitive_intelligence(focus: Optional[str] = None) -> Dict[str, Any]:
        """Extract competitive intelligence from the knowledge base.

        Searches for competitor mentions, positioning, win/loss signals,
        and market dynamics. Returns structured competitive analysis with:
        - Competitor profiles (who they are, what they offer)
        - Positioning gaps (where we differ)
        - Win/loss signals (why customers choose us or them)
        - Competitive threats and opportunities

        Use 'focus' to narrow (e.g., 'pricing vs competitors',
        'feature comparison', 'specific competitor name').
        """
        queries = [
            "competitor competition alternative market position",
            "vs compare comparison advantage disadvantage",
            "win loss chose switch churn reason",
            focus or "competitive landscape pricing differentiation",
        ]

        all_content: List[str] = []
        seen: set = set()

        for q in queries:
            try:
                docs = core_client.search_graph(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    query=q,
                    top_k=10,
                    hop_limit=1,
                    boost_pinned=True,
                    exclude_status=["archived", "deprecated"],
                )
                for d in docs:
                    nid = d.metadata.get("node_id")
                    if nid and nid not in seen:
                        seen.add(nid)
                        all_content.append(d.page_content)
            except Exception:
                pass

        # Also pull context summary for market positioning
        context_summary = ""
        try:
            summary = core_client.get_context_summary(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            if summary:
                context_summary = f"Business Context:\n{summary.get('summary', '')}\n\n"
        except Exception:
            pass

        if not all_content and not context_summary:
            return {
                "status": "no_data",
                "message": "No competitive data found in the knowledge base. Ingest documents that mention competitors.",
            }

        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{c}" for i, c in enumerate(all_content[:20])
        )

        focus_instruction = f"Focus on: {focus}\n" if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a competitive intelligence analyst. Extract all competitive "
                "insights from the provided sources.\n\n"
                "Return JSON:\n"
                "{{\n"
                '  "competitors": [\n'
                '    {{"name": "...", "description": "what they do", "strengths": ["..."], "weaknesses": ["..."], "evidence": "quote"}}\n'
                "  ],\n"
                '  "our_positioning": {{\n'
                '    "strengths": [{{"point": "...", "evidence": "quote"}}],\n'
                '    "gaps": [{{"gap": "...", "impact": "high/medium/low", "evidence": "quote"}}]\n'
                "  }},\n"
                '  "win_signals": [{{"signal": "...", "evidence": "quote"}}],\n'
                '  "loss_signals": [{{"signal": "...", "evidence": "quote"}}],\n'
                '  "opportunities": [{{"opportunity": "...", "priority": "high/medium/low"}}],\n'
                '  "threats": [{{"threat": "...", "severity": "high/medium/low"}}]\n'
                "}}\n\n"
                "Rules:\n"
                "- Only include competitors explicitly mentioned in the sources\n"
                "- Every claim needs evidence (direct quote or specific reference)\n"
                "- Distinguish between direct competitors and adjacent players\n"
                "- If data is sparse, say so — don't invent competitors",
            ),
            ("human", "{context_summary}{focus}Sources:\n\n{context}"),
        ])

        llm = get_llm("context_analysis")
        chain = prompt | llm | StrOutputParser()

        try:
            raw = chain.invoke({
                "context_summary": context_summary,
                "focus": focus_instruction,
                "context": context,
            })
            result = _parse_json(raw)
            if result:
                result["status"] = "complete"
                result["sources_analyzed"] = len(all_content)
                return result
        except Exception as e:
            logger.warning("competitive_intelligence failed: %s", e)

        return {"status": "failed", "error": "Analysis failed", "sources_analyzed": len(all_content)}

    # ── Cross-document synthesis ───────────────────────────────────────

    @tool
    def cross_document_synthesis(focus: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize patterns across ALL documents in the knowledge base.

        Searches multiple angles, cross-references findings, and surfaces:
        - Recurring themes that appear across different document types
        - Contradictions between sources
        - Evidence convergence (multiple sources supporting same conclusion)
        - Blind spots (topics mentioned once but not corroborated)

        Use 'focus' to narrow (e.g., 'pricing strategy', 'customer needs',
        'product direction').
        """
        # Cast a wide net with diverse queries
        base_queries = [
            "overview strategy direction goals",
            "customer feedback pain points needs",
            "product features capabilities roadmap",
            "market trends industry changes",
            "pricing revenue business model",
        ]
        if focus:
            base_queries.append(focus)

        all_content: List[Dict[str, Any]] = []
        seen: set = set()

        for q in base_queries:
            try:
                docs = core_client.search_graph(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    query=q,
                    top_k=8,
                    hop_limit=1,
                    boost_pinned=True,
                    exclude_status=["archived", "deprecated"],
                    recency_weight=0.15,
                )
                for d in docs:
                    nid = d.metadata.get("node_id")
                    if nid and nid not in seen:
                        seen.add(nid)
                        all_content.append({
                            "content": d.page_content,
                            "document_id": d.metadata.get("document_id"),
                            "source_type": d.metadata.get("source_type", "unknown"),
                        })
            except Exception:
                pass

        # Also get all summaries for high-level patterns
        summaries_text = ""
        try:
            all_summaries = core_client.list_summaries(
                tenant_id=tenant_id,
                client_id=client_id,
            )
            for s in all_summaries.get("summaries", []):
                if s.get("summary"):
                    st = s.get("source_type", "")
                    label = s.get("topic") or s.get("document_id") or st
                    summaries_text += f"[{st}: {label}] {s['summary'][:300]}\n\n"
        except Exception:
            pass

        if not all_content and not summaries_text:
            return {
                "status": "no_data",
                "message": "Not enough documents in the knowledge base for cross-document synthesis. Ingest more content.",
            }

        # Tag each chunk with its source for cross-referencing
        context = "\n\n---\n\n".join(
            f"[Source {i+1} | doc={c['document_id'][:8] if c.get('document_id') else '?'} | type={c['source_type']}]\n{c['content']}"
            for i, c in enumerate(all_content[:25])
        )

        focus_instruction = f"Focus the synthesis on: {focus}\n" if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior research synthesist. Cross-reference ALL provided "
                "sources to find patterns, contradictions, and insights that no single "
                "document reveals on its own.\n\n"
                "Return JSON:\n"
                "{{\n"
                '  "recurring_themes": [\n'
                '    {{"theme": "...", "frequency": "how many sources mention it", "sources": ["doc IDs"], "evidence": ["quotes"]}}\n'
                "  ],\n"
                '  "contradictions": [\n'
                '    {{"topic": "...", "source_a": "what one source says", "source_b": "what another says", "implication": "why it matters"}}\n'
                "  ],\n"
                '  "convergent_evidence": [\n'
                '    {{"conclusion": "...", "supporting_sources": 3, "confidence": "high/medium/low", "evidence": ["quotes"]}}\n'
                "  ],\n"
                '  "blind_spots": [\n'
                '    {{"topic": "...", "mentioned_in": 1, "missing_from": "what types of sources lack this", "risk": "what could be missed"}}\n'
                "  ],\n"
                '  "executive_summary": "3-4 paragraph synthesis of the most important cross-document findings",\n'
                '  "recommendations": ["actionable next steps based on the synthesis"]\n'
                "}}\n\n"
                "Rules:\n"
                "- CROSS-REFERENCE: every finding must reference 2+ sources unless it's a blind spot\n"
                "- Track which document types (web, pdf, transcript, summary) support each finding\n"
                "- Contradictions are as valuable as agreements — surface them prominently\n"
                "- Blind spots: topics mentioned in only one source that deserve corroboration\n"
                "- Be specific with quotes and document references",
            ),
            (
                "human",
                "{focus}Summaries:\n{summaries}\n\nSource chunks:\n\n{context}",
            ),
        ])

        llm = get_llm("context_analysis")
        chain = prompt | llm | StrOutputParser()

        try:
            raw = chain.invoke({
                "focus": focus_instruction,
                "summaries": summaries_text or "(no summaries available)",
                "context": context,
            })
            result = _parse_json(raw)
            if result:
                result["status"] = "complete"
                result["sources_analyzed"] = len(all_content)
                result["document_types"] = list({c["source_type"] for c in all_content})
                return result
        except Exception as e:
            logger.warning("cross_document_synthesis failed: %s", e)

        return {"status": "failed", "error": "Synthesis failed", "sources_analyzed": len(all_content)}

    return [analyze_transcript, competitive_intelligence, cross_document_synthesis]


# ── Helpers ────────────────────────────────────────────────────────────────

import re


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None

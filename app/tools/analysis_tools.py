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

    # ── Customer objections ────────────────────────────────────────────

    @tool
    def extract_objections(focus: Optional[str] = None) -> Dict[str, Any]:
        """Extract customer objections, hesitations, and blockers from transcripts.

        ONLY analyzes transcript/interview content — objections from real
        customer conversations, not marketing copy or decks. If no transcripts
        are in the KB, returns a skip message. Distinct from pain points:
        objections are reasons people HESITATE (not problems they have).

        Use 'focus' to narrow (e.g., 'pricing objections', 'onboarding blockers').
        """
        # First check if any transcripts exist at all — skip if not
        try:
            transcript_count = core_client.count_transcripts(
                tenant_id=tenant_id, client_id=client_id,
            )
        except Exception:
            transcript_count = 0

        if transcript_count == 0:
            return {
                "status": "skipped",
                "message": (
                    "No transcripts or interviews in the knowledge base. "
                    "Objection analysis requires real customer conversations — "
                    "ingest VTT transcripts or interview recordings to enable this."
                ),
                "objections": [],
            }

        # Pull transcript chunks only — restricted to VideoTranscript node type
        queries = [
            "objection hesitation concern reason not buy",
            "blocker barrier friction stuck confused",
            "too expensive too complicated not sure worried",
            focus or "why don't customers buy adopt switch",
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
                    node_types=["VideoTranscript"],  # transcripts only
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

        # Fallback: pull transcript chunks via get_transcript_chunks if search
        # didn't find VideoTranscript nodes (may differ in KG labeling)
        if not all_content:
            try:
                transcript_chunks = core_client.get_transcript_chunks(
                    tenant_id=tenant_id, client_id=client_id, limit=30,
                )
                for c in transcript_chunks:
                    text = c.get("content") or c.get("text", "")
                    if text:
                        all_content.append(text)
            except Exception:
                pass

        if not all_content:
            return {
                "status": "skipped",
                "message": "No transcript content retrieved from the knowledge base.",
                "objections": [],
            }

        context = "\n\n---\n\n".join(
            f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(all_content[:20])
        )

        focus_instruction = f"Focus on: {focus}\n" if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a sales research analyst. Extract customer objections, "
                "hesitations, and blockers from the sources. Distinguish these from "
                "pain points (existing problems) — objections are REASONS people "
                "HESITATE to adopt, buy, or engage.\n\n"
                "Return JSON:\n"
                "{{\n"
                '  "objections": [\n'
                '    {{"objection": "...", "category": "price/trust/fit/complexity/timing/authority", '
                '"severity": "high/medium/low", "frequency": "number of mentions", '
                '"evidence": "direct quote", "how_to_address": "suggested response"}}\n'
                "  ],\n"
                '  "top_blocker": "the single biggest adoption blocker",\n'
                '  "patterns": ["..."]\n'
                "}}\n\n"
                "Rules:\n"
                "- Only include objections explicitly stated or strongly implied\n"
                "- Every objection MUST have a direct quote as evidence\n"
                "- Categorize by the type of hesitation\n"
                "- Skip this if no real objections exist — don't invent them",
            ),
            ("human", "{focus}Sources:\n\n{context}"),
        ])

        llm = get_llm("context_analysis")
        chain = prompt | llm | StrOutputParser()

        try:
            raw = chain.invoke({"focus": focus_instruction, "context": context})
            result = _parse_json(raw)
            if result:
                result["status"] = "complete"
                result["sources_analyzed"] = len(all_content)
                return result
        except Exception as e:
            logger.warning("extract_objections failed: %s", e)

        return {"status": "failed", "error": "Analysis failed"}

    # ── Hypotheses to test ─────────────────────────────────────────────

    @tool
    def generate_hypotheses(focus: Optional[str] = None) -> Dict[str, Any]:
        """Generate research hypotheses to test based on KB evidence.

        Looks at convergent themes, contradictions, and blind spots to
        suggest specific things worth validating — with proposed survey
        questions. Useful for planning the next research cycle.

        Use 'focus' to narrow (e.g., 'pricing hypotheses', 'messaging tests').
        """
        # Gather diverse evidence
        queries = [
            "assumption belief hypothesis what we think",
            "unclear uncertain might may could possibly",
            "customer expects wants needs prefers",
            focus or "decisions to validate questions to answer",
        ]
        all_content: List[str] = []
        seen: set = set()

        for q in queries:
            try:
                docs = core_client.search_graph(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    query=q,
                    top_k=8,
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

        # Pull context summary for positioning
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
            return {"status": "no_data", "message": "Not enough data to generate hypotheses."}

        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{c}" for i, c in enumerate(all_content[:20])
        )

        focus_instruction = f"Focus on: {focus}\n" if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a research strategist. Based on the KB evidence, generate "
                "specific hypotheses worth testing in the next research cycle. "
                "Each hypothesis should be testable, specific, and tied to a business "
                "decision.\n\n"
                "Return JSON:\n"
                "{{\n"
                '  "hypotheses": [\n'
                '    {{\n'
                '      "hypothesis": "If X, then Y because Z",\n'
                '      "category": "pricing/messaging/feature/positioning/segment",\n'
                '      "why_test_this": "what decision this unlocks",\n'
                '      "evidence_supporting": "quote or finding that motivated this",\n'
                '      "suggested_questions": ["survey question 1", "survey question 2"],\n'
                '      "priority": "high/medium/low",\n'
                '      "expected_effort": "quick (1 survey) / medium / deep"\n'
                '    }}\n'
                "  ],\n"
                '  "top_hypothesis": "the single most valuable thing to validate next"\n'
                "}}\n\n"
                "Rules:\n"
                "- Each hypothesis MUST be a specific, falsifiable statement\n"
                "- Every hypothesis must tie to evidence in the sources\n"
                "- Suggested questions must be concrete, not abstract\n"
                "- Prioritize hypotheses that unlock big decisions\n"
                "- 3-6 hypotheses is ideal — don't pad with weak ones",
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
                return result
        except Exception as e:
            logger.warning("generate_hypotheses failed: %s", e)

        return {"status": "failed", "error": "Analysis failed"}

    return [
        analyze_transcript,
        competitive_intelligence,
        cross_document_synthesis,
        extract_objections,
        generate_hypotheses,
    ]


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

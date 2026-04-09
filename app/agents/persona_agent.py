"""
Persona discovery agent -- extracts audience personas from a tenant's KG context.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.prompts.persona_prompts import PERSONA_EXTRACTION_PROMPT
from app.analysis.base import BaseAnalysisService
from app import core_client

logger = logging.getLogger(__name__)

_DISCOVERY_QUERIES = [
    "customers audience demographics target market user segments preferences",
    "pain points challenges frustrations needs goals motivations",
    "buyer persona consumer behavior decision making purchasing patterns",
]


def run_persona_agent(
    tenant_id: str,
    client_id: str,
    request: Optional[str] = None,
    client_profile: Optional[Dict[str, Any]] = None,
    max_personas: int = 5,
    top_k: int = 15,
    hop_limit: int = 1,
) -> Dict[str, Any]:
    all_docs = {}
    for query in _DISCOVERY_QUERIES:
        try:
            docs = core_client.search_graph(tenant_id=tenant_id, client_id=client_id, query=query, top_k=top_k, hop_limit=hop_limit)
            for doc in docs:
                nid = doc.metadata.get("node_id")
                if nid and nid not in all_docs:
                    all_docs[nid] = doc
        except Exception as e:
            logger.warning("Discovery query failed (%s): %s", query[:40], e)

    if request:
        try:
            docs = core_client.search_graph(tenant_id=tenant_id, client_id=client_id, query=request, top_k=top_k, hop_limit=hop_limit)
            for doc in docs:
                nid = doc.metadata.get("node_id")
                if nid and nid not in all_docs:
                    all_docs[nid] = doc
        except Exception as e:
            logger.warning("Focused query failed: %s", e)

    if not all_docs:
        return {"personas": [], "context_used": 0, "status": "complete", "error": "No relevant content found in knowledge graph."}

    sorted_docs = sorted(all_docs.values(), key=lambda d: d.metadata.get("similarity_score", 0.0), reverse=True)[:top_k]

    summary_section = ""
    row = core_client.get_context_summary(tenant_id=tenant_id, client_id=client_id)
    if row:
        summary_section = f"\nExisting context summary:\nSummary: {row.get('summary', '')}\nTopics: {', '.join(row.get('topics', []))}\n\n"

    context = "\n\n---\n\n".join(f"[Source {i + 1}]\n{doc.page_content}" for i, doc in enumerate(sorted_docs) if doc.page_content.strip())
    profile_section = BaseAnalysisService._build_profile_section(client_profile)
    user_request = request or "Identify the key audience personas for this organization."

    llm = get_llm("persona")
    chain = PERSONA_EXTRACTION_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "context": context, "profile_section": profile_section,
            "summary_section": summary_section, "user_request": user_request,
            "max_personas": str(max_personas), "feedback_section": "",
        })
    except Exception as e:
        logger.exception("LLM persona extraction failed")
        return {"personas": [], "context_used": len(sorted_docs), "status": "failed", "error": f"LLM generation failed: {e}"}

    personas = _parse_personas(raw, max_personas)
    return {"personas": personas, "context_used": len(sorted_docs), "status": "complete", "error": None}


def _parse_personas(raw: str, max_personas: int) -> List[Dict[str, Any]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    personas = []
    for item in parsed[:max_personas]:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        demo = item.get("demographics", {}) if isinstance(item.get("demographics"), dict) else {}
        personas.append({
            "name": item["name"], "description": item.get("description", ""),
            "demographics": {"age_range": demo.get("age_range"), "income_level": demo.get("income_level"), "location": demo.get("location"), "occupation": demo.get("occupation"), "education": demo.get("education")},
            "motivations": item.get("motivations", []), "pain_points": item.get("pain_points", []),
            "behaviors": item.get("behaviors", []),
            "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
        })
    return personas

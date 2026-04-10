"""
Tools for the survey generation workflow.

Each tool wraps a service call and is stateless — tenant/client
context is passed as parameters, not captured in closures.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from app import core_client
from app.llm_config import get_llm
from app.prompts.survey_prompts import CONTEXT_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(
    tenant_id: str,
    client_id: str,
    query: str,
    top_k: int = 10,
    hop_limit: int = 1,
) -> List[Dict[str, Any]]:
    """Search the knowledge base using graph-expanded vector search.

    Returns a list of relevant content chunks with similarity scores.
    Use this to retrieve context for survey question generation.
    """
    try:
        docs = core_client.search_graph(
            tenant_id=tenant_id,
            client_id=client_id,
            query=query,
            top_k=top_k,
            hop_limit=hop_limit,
        )
    except Exception as e:
        logger.warning("search_knowledge_base failed: %s", e)
        return []

    return [
        {
            "content": doc.page_content,
            "similarity_score": doc.metadata.get("similarity_score", 0.0),
            "node_id": doc.metadata.get("node_id"),
            "document_id": doc.metadata.get("document_id"),
            "source": doc.metadata.get("source", "vector"),
        }
        for doc in docs
    ]


@tool
def get_prior_survey_outputs(
    tenant_id: str,
    client_id: str,
    output_type: str = "survey",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Fetch previously generated survey outputs for this tenant+client.

    Returns prior survey questions to avoid duplication and build on existing work.
    """
    try:
        return core_client.get_survey_outputs(
            tenant_id=tenant_id,
            client_id=client_id,
            output_type=output_type,
            limit=limit,
        )
    except Exception as e:
        logger.warning("get_prior_survey_outputs failed: %s", e)
        return []


@tool
def analyze_survey_context(
    request: str,
    context: str,
    tenant_profile: str,
) -> str:
    """Analyze knowledge base context to extract survey-relevant insights.

    Takes the survey request, retrieved KB context, and tenant profile.
    Returns a structured analysis of industry context, themes, and focus areas
    that should inform survey question design.
    """
    if not context.strip() and tenant_profile == "No profile provided.":
        return "No context or profile available. Generate general-purpose survey questions."

    llm = get_llm("context_analysis")
    chain = CONTEXT_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        analysis = chain.invoke({
            "tenant_profile": tenant_profile,
            "request": request,
            "context": context if context.strip() else "No knowledge base context available.",
        })
    except Exception as e:
        logger.exception("analyze_survey_context failed")
        analysis = f"Analysis unavailable: {e}. Proceed with general survey design."

    return analysis

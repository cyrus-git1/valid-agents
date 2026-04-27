"""
/clusters router — respondent cluster analysis.

POST /clusters/analyze — Per-session deterministic-first clustering across
                         the tenant's KB, with optional LLM cluster naming.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agents.clustering.respondent_clustering_agent import run_cluster_analysis
from app.models.api.clustering import (
    ClustersAnalyzeRequest,
    ClustersAnalyzeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.post("/analyze", response_model=ClustersAnalyzeResponse)
def clusters_analyze(req: ClustersAnalyzeRequest) -> ClustersAnalyzeResponse:
    """Run per-session respondent clustering.

    Pipeline: discover sessions → harmonise tags → extract features →
    embed (optional) → HDBSCAN/KMeans → characterise clusters → optionally
    label via LLM.

    Returns deterministic cluster boundaries + structured characterisation.
    """
    try:
        result = run_cluster_analysis(
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            survey_ids=[str(s) for s in (req.survey_ids or [])] or None,
            study_id=str(req.study_id) if req.study_id else None,
            k=req.k,
            min_cluster_size=req.min_cluster_size,
            min_word_count=req.min_word_count,
            include_text_embedding=req.include_text_embedding,
            focus=req.focus,
            produce_labels=req.produce_labels,
        )
    except Exception as e:
        logger.exception("clusters_analyze failed")
        raise HTTPException(status_code=500, detail=f"Cluster analysis failed: {e}")
    return result

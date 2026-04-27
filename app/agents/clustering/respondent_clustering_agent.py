"""
Respondent cluster analysis orchestrator.

Pipeline:
  1. Discover sessions: list_documents → group by metadata.{survey_id, session_id}
     Filter by optional survey_ids[] and study_id.
  2. Per-session feature extraction (parallel, deterministic):
     tag harmonization, statistical features, lexicon traits, entity counts.
  3. Vectorize: standardised numeric features + (optional) OpenAI embedding
     of concatenated text.
  4. Cluster: HDBSCAN by default, KMeans if k provided.
  5. Characterise each cluster: chi-square defining tags, top TF-IDF terms,
     mean VADER, dominant traits, sample sessions near centroid.
  6. (Optional) LLM labels: one call per cluster.

Returns the full unified response dict.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from app import core_client
from app.agents.clustering import cluster_labeler
from app.agents.clustering.feature_extraction import extract_session_features
from app.tools.clustering_tools import (
    assign_clusters,
    chi_square_overrepresented_tags,
    cluster_centroids,
    closest_to_centroid,
    dominant_traits_per_cluster,
    embed_in_batches,
    standardise_numeric_matrix,
    tfidf_top_terms_per_cluster,
)

logger = logging.getLogger(__name__)


_INTERNAL_SOURCE_TYPES = {"ContextSummary", "DocumentSummary", "TopicSummary"}


def run_cluster_analysis(
    *,
    tenant_id: str,
    client_id: str,
    survey_ids: Optional[List[str]] = None,
    study_id: Optional[str] = None,
    k: Optional[int] = None,
    min_cluster_size: int = 3,
    include_text_embedding: bool = True,
    focus: Optional[str] = None,
    produce_labels: bool = True,
    min_word_count: int = 30,
) -> Dict[str, Any]:
    """End-to-end clustering. Returns the full response dict."""
    t0 = time.monotonic()

    # ── 1. Discover sessions ────────────────────────────────────────────
    sessions = _discover_sessions(
        tenant_id=tenant_id,
        client_id=client_id,
        survey_ids=set(survey_ids) if survey_ids else None,
        study_id=study_id,
        min_word_count=min_word_count,
    )

    if not sessions:
        return {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "session_count": 0,
            "noise_count": 0,
            "n_clusters": 0,
            "algorithm": "none",
            "clusters": [],
            "noise_session_ids": [],
            "feature_summary": {
                "numeric_dimensions": 0,
                "embedding_dimensions": 0,
                "tags_normalised": 0,
                "tags_via_llm_fallback": 0,
            },
            "errors": [{"agent": "discovery", "error": "No sessions matched the filters"}],
            "status": "complete",
            "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
        }

    # ── 2. Per-session feature extraction (parallel) ────────────────────
    features = _extract_features_parallel(sessions)

    # ── 3. Vectorize ────────────────────────────────────────────────────
    X, numeric_dim, embed_dim = _build_feature_matrix(
        features, include_text_embedding=include_text_embedding,
    )

    # ── 4. Cluster ──────────────────────────────────────────────────────
    labels, algorithm, _probs = assign_clusters(
        X, k=k, min_cluster_size=min_cluster_size,
    )

    # ── 5. Characterise each cluster ────────────────────────────────────
    population_tags = [f["normalized_tags"] for f in features]
    population_traits = [f["lexicon_traits"] for f in features]
    population_texts = [f["concatenated_text"] for f in features]
    population_sentiments = [
        f["statistical_features"].get("vader_compound", 0.0) for f in features
    ]

    # Group session indices by cluster id
    cluster_members: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        cluster_members.setdefault(lbl, []).append(i)

    noise_indices = cluster_members.pop(-1, []) if -1 in cluster_members else []

    # Compute centroids for sample-selection
    centroids = cluster_centroids(X, labels) if X else {}

    # TF-IDF top terms per cluster
    cluster_texts: Dict[int, List[str]] = {
        cid: [population_texts[i] for i in members if i < len(population_texts)]
        for cid, members in cluster_members.items()
    }
    top_terms_by_cluster = tfidf_top_terms_per_cluster(cluster_texts)

    cluster_summaries: List[Dict[str, Any]] = []
    for cluster_id, members in sorted(cluster_members.items()):
        size = len(members)
        defining_tags = chi_square_overrepresented_tags(members, population_tags)
        dominant_traits = dominant_traits_per_cluster(members, population_traits)
        # Mean VADER
        if members:
            mean_vader = sum(population_sentiments[i] for i in members) / len(members)
        else:
            mean_vader = 0.0
        # Sample sessions: closest to centroid
        sample_idx = (
            closest_to_centroid(members, X, centroids[cluster_id])
            if cluster_id in centroids else members[:10]
        )
        sample_session_ids = [features[i]["session_id"] for i in sample_idx]

        cluster_summaries.append({
            "cluster_id": cluster_id,
            "size": size,
            "defining_tags": defining_tags,
            "top_terms": top_terms_by_cluster.get(cluster_id, []),
            "mean_vader": round(mean_vader, 4),
            "dominant_traits": dominant_traits,
            "sample_session_ids": sample_session_ids,
        })

    # ── 6. (Optional) LLM labels ────────────────────────────────────────
    if produce_labels and cluster_summaries:
        label_inputs = [
            {
                "cluster_id": cs["cluster_id"],
                "size": cs["size"],
                "defining_tags": cs["defining_tags"],
                "top_terms": cs["top_terms"],
                "dominant_traits": cs["dominant_traits"],
                "mean_vader": cs["mean_vader"],
            }
            for cs in cluster_summaries
        ]
        labels_by_cid = cluster_labeler.label_clusters_bulk(label_inputs, focus=focus)
        for cs in cluster_summaries:
            lbl = labels_by_cid.get(cs["cluster_id"], {})
            cs["label"] = lbl.get("label", f"Cluster {cs['cluster_id']}")
            cs["description"] = lbl.get("description", "")
    else:
        for cs in cluster_summaries:
            cs["label"] = f"Cluster {cs['cluster_id']} (n={cs['size']})"
            cs["description"] = ""

    # ── Feature summary ────────────────────────────────────────────────
    tags_normalised = sum(len(f["normalized_tags"]) for f in features)
    tags_via_llm = sum(
        1
        for f in features
        for src in f["tag_sources"].values()
        if src == "llm"
    )

    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "session_count": len(features),
        "noise_count": len(noise_indices),
        "n_clusters": len(cluster_summaries),
        "algorithm": algorithm,
        "clusters": cluster_summaries,
        "noise_session_ids": [features[i]["session_id"] for i in noise_indices],
        "feature_summary": {
            "numeric_dimensions": numeric_dim,
            "embedding_dimensions": embed_dim,
            "tags_normalised": tags_normalised,
            "tags_via_llm_fallback": tags_via_llm,
        },
        "errors": [],
        "status": "complete",
        "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
    }


# ── Helpers ──────────────────────────────────────────────────────────────


def _discover_sessions(
    *,
    tenant_id: str,
    client_id: str,
    survey_ids: Optional[set],
    study_id: Optional[str],
    min_word_count: int,
) -> List[Dict[str, Any]]:
    """Find documents matching scope. Returns list of {document, survey_id, session_id}."""
    try:
        data = core_client.list_documents(tenant_id=tenant_id, client_id=client_id)
    except Exception as e:
        logger.warning("list_documents failed: %s", e)
        return []
    items = data.get("items", []) or []

    out: List[Dict[str, Any]] = []
    for d in items:
        md = d.get("metadata") or {}
        # Skip internal summary documents
        if d.get("source_type") in _INTERNAL_SOURCE_TYPES:
            continue
        session_id = md.get("session_id")
        survey_id = md.get("survey_id")
        if not session_id or not survey_id:
            continue
        if study_id and md.get("study_id") != study_id:
            continue
        if survey_ids and survey_id not in survey_ids:
            continue
        # Min word count filter (cheap pre-filter to avoid noisy short sessions)
        chunks = d.get("chunks") or []
        text = "\n".join(
            (ch.get("content") or ch.get("text") or "")
            for ch in chunks
        )
        if len(text.split()) < min_word_count:
            continue
        out.append({
            "document": d,
            "survey_id": str(survey_id),
            "session_id": str(session_id),
        })
    return out


def _extract_features_parallel(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run extract_session_features in parallel. Returns features list (same order as sessions)."""
    n = len(sessions)
    if n == 0:
        return []
    max_workers = min(8, n)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    def _work(i: int):
        s = sessions[i]
        return i, extract_session_features(
            s["document"],
            survey_id=s["survey_id"],
            session_id=s["session_id"],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in [ex.submit(_work, i) for i in range(n)]:
            try:
                idx, feat = fut.result()
                results[idx] = feat
            except Exception as e:
                logger.warning("feature extraction failed: %s", e)

    return [r for r in results if r is not None]


def _build_feature_matrix(
    features: List[Dict[str, Any]],
    *,
    include_text_embedding: bool,
) -> tuple:
    """Build the X matrix for clustering. Returns (X, numeric_dim, embed_dim)."""
    if not features:
        return [], 0, 0

    # Numeric portion: standardised statistical features
    numeric_keys = [
        "word_count", "duration_seconds", "wpm", "filler_rate",
        "vader_compound", "vader_pos", "vader_neg", "vader_neu",
        "question_count", "speaker_count",
    ]
    numeric_matrix = [
        [float(f["statistical_features"].get(k, 0.0)) for k in numeric_keys]
        for f in features
    ]
    standardised = standardise_numeric_matrix(numeric_matrix)
    numeric_dim = len(numeric_keys)

    # Optional embedding portion
    embed_dim = 0
    if include_text_embedding:
        texts = [(f.get("concatenated_text") or "")[:8000] for f in features]
        try:
            embeddings = embed_in_batches(texts)
            if embeddings and len(embeddings) == len(features):
                embed_dim = len(embeddings[0])
                # Concatenate per row
                X = [
                    list(emb) + list(num)
                    for emb, num in zip(embeddings, standardised)
                ]
                return X, numeric_dim, embed_dim
            else:
                logger.warning("embedding produced wrong length; skipping embedding")
        except Exception as e:
            logger.warning("embedding failed; clustering on numeric features only: %s", e)

    return standardised, numeric_dim, embed_dim

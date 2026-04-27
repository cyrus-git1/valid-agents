"""
Pure deterministic helpers for cluster analysis.

  - embed_texts / embed_in_batches: lifted pattern from kg_router.py
  - standardise_features: z-score numeric features
  - chi_square_overrepresented_tags: per-cluster tag over-representation
  - tfidf_top_terms_per_cluster: top TF-IDF terms within each cluster
  - assign_clusters: HDBSCAN by default, k-means when k provided
"""
from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Embeddings ───────────────────────────────────────────────────────────


_EMBED_MODEL = "text-embedding-3-small"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts via OpenAI. Pattern lifted from kg_router.py."""
    if not texts:
        return []
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(f"openai package required for embeddings: {e}") from e
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_in_batches(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Embed texts in batches to respect API limits."""
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(embed_texts(texts[i:i + batch_size]))
    return out


# ── Numeric standardisation ──────────────────────────────────────────────


def standardise_numeric_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Z-score each column. Returns same shape. Uses sklearn StandardScaler."""
    if not matrix:
        return []
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError(f"sklearn required for standardisation: {e}") from e
    arr = np.array(matrix, dtype=float)
    if arr.size == 0:
        return matrix
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    return scaled.tolist()


# ── Clustering ───────────────────────────────────────────────────────────


def assign_clusters(
    X: List[List[float]],
    *,
    k: Optional[int] = None,
    min_cluster_size: int = 3,
) -> Tuple[List[int], str, Optional[List[float]]]:
    """Assign cluster labels.

    Returns (labels, algorithm, probabilities|None).

    - If k is provided: KMeans with k clusters. probabilities=None.
    - Else: HDBSCAN; noise points labelled -1. probabilities = membership probabilities.
    """
    if not X:
        return [], "none", None

    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"numpy required for clustering: {e}") from e

    X_arr = np.array(X, dtype=float)
    n = len(X_arr)

    if k is not None and k > 0:
        try:
            from sklearn.cluster import KMeans
        except ImportError as e:
            raise RuntimeError(f"sklearn required for KMeans: {e}") from e
        k = min(k, n)
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_arr).tolist()
        return labels, "kmeans", None

    # HDBSCAN default
    try:
        import hdbscan
    except ImportError as e:
        raise RuntimeError(f"hdbscan required for default clustering: {e}") from e

    mcs = max(min_cluster_size, max(3, n // 20))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X_arr).tolist()
    probabilities = clusterer.probabilities_.tolist() if clusterer.probabilities_ is not None else None
    return labels, "hdbscan", probabilities


# ── Cluster characterisation ─────────────────────────────────────────────


def chi_square_overrepresented_tags(
    cluster_member_indices: List[int],
    population_tags: List[Dict[str, str]],
    *,
    p_value_threshold: float = 0.05,
    rate_ratio_threshold: float = 1.5,
) -> List[Dict[str, Any]]:
    """For one cluster, find tag values significantly over-represented vs the
    rest of the population.

    Returns a list of {field, value, in_cluster_count, in_cluster_rate,
                       baseline_count, baseline_rate, p_value}.

    Uses scipy chi2_contingency on a 2x2 (in_cluster vs out, value vs other).
    Tags with p < threshold AND rate_ratio >= threshold are returned.
    """
    if not cluster_member_indices or not population_tags:
        return []
    try:
        from scipy.stats import chi2_contingency
    except ImportError as e:
        raise RuntimeError(f"scipy required for chi-square: {e}") from e

    n_total = len(population_tags)
    n_in = len(cluster_member_indices)
    n_out = n_total - n_in
    if n_in == 0 or n_out == 0:
        return []

    in_set = set(cluster_member_indices)

    # Discover every (field, value) pair seen in the population
    field_value_pairs: List[Tuple[str, str]] = []
    seen = set()
    for tags in population_tags:
        for f, v in tags.items():
            if not v or v == "unknown":
                continue
            key = (f, v)
            if key not in seen:
                seen.add(key)
                field_value_pairs.append(key)

    results: List[Dict[str, Any]] = []
    for field, value in field_value_pairs:
        in_count = sum(1 for i in cluster_member_indices if population_tags[i].get(field) == value)
        out_count = sum(
            1 for i, tags in enumerate(population_tags)
            if i not in in_set and tags.get(field) == value
        )

        if in_count == 0:
            continue

        in_rate = in_count / n_in if n_in else 0.0
        baseline_count = in_count + out_count
        baseline_rate = baseline_count / n_total if n_total else 0.0

        if baseline_rate == 0.0:
            continue

        rate_ratio = in_rate / baseline_rate
        if rate_ratio < rate_ratio_threshold:
            continue

        # 2x2 contingency: rows = in/out, cols = value/other
        in_other = n_in - in_count
        out_other = n_out - out_count
        table = [[in_count, in_other], [out_count, out_other]]
        try:
            chi2, p_value, _dof, _expected = chi2_contingency(table)
        except Exception:
            continue

        if p_value > p_value_threshold:
            continue

        results.append({
            "field": field,
            "value": value,
            "in_cluster_count": int(in_count),
            "in_cluster_rate": round(in_rate, 4),
            "baseline_count": int(baseline_count),
            "baseline_rate": round(baseline_rate, 4),
            "p_value": round(float(p_value), 6),
            "rate_ratio": round(rate_ratio, 3),
        })

    # Strongest first
    results.sort(key=lambda r: (r["p_value"], -r["rate_ratio"]))
    return results


def tfidf_top_terms_per_cluster(
    cluster_texts: Dict[int, List[str]],
    *,
    top_n: int = 10,
    max_features: int = 2000,
) -> Dict[int, List[str]]:
    """For each cluster id → list of top-TF-IDF tokens.

    Concatenates each cluster's texts into a single document, then runs
    TF-IDF across cluster-documents.
    """
    if not cluster_texts:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        raise RuntimeError(f"sklearn required for TF-IDF: {e}") from e

    cluster_ids = sorted(cluster_texts.keys())
    docs = [" ".join(cluster_texts[c]) for c in cluster_ids]
    if not any(docs):
        return {c: [] for c in cluster_ids}

    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:
        return {c: [] for c in cluster_ids}

    feature_names = vec.get_feature_names_out()
    out: Dict[int, List[str]] = {}
    for row, cluster_id in enumerate(cluster_ids):
        scores = matrix[row].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        out[cluster_id] = [feature_names[i] for i in top_idx if scores[i] > 0]
    return out


def dominant_traits_per_cluster(
    cluster_member_indices: List[int],
    population_traits: List[List[str]],
    *,
    threshold: float = 0.5,
) -> List[str]:
    """Traits present in >= threshold fraction of cluster members."""
    if not cluster_member_indices:
        return []
    n = len(cluster_member_indices)
    counts: Counter = Counter()
    for i in cluster_member_indices:
        if i < len(population_traits):
            counts.update(set(population_traits[i]))
    out = [trait for trait, c in counts.items() if c / n >= threshold]
    out.sort(key=lambda t: -counts[t])
    return out


def cluster_centroids(
    X: List[List[float]],
    labels: List[int],
) -> Dict[int, List[float]]:
    """Compute per-cluster centroid (mean vector). Excludes label -1 (noise)."""
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"numpy required for centroids: {e}") from e
    arr = np.array(X, dtype=float)
    grouped: Dict[int, List[List[float]]] = defaultdict(list)
    for vec, lbl in zip(arr.tolist(), labels):
        if lbl == -1:
            continue
        grouped[lbl].append(vec)
    return {lbl: np.array(vecs).mean(axis=0).tolist() for lbl, vecs in grouped.items()}


def closest_to_centroid(
    cluster_member_indices: List[int],
    X: List[List[float]],
    centroid: List[float],
    *,
    top_k: int = 10,
) -> List[int]:
    """Return up to top_k indices closest to the cluster centroid by L2 distance."""
    if not cluster_member_indices:
        return []
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"numpy required for closest_to_centroid: {e}") from e
    cent = np.array(centroid, dtype=float)
    distances = []
    for idx in cluster_member_indices:
        v = np.array(X[idx], dtype=float)
        distances.append((idx, float(np.linalg.norm(v - cent))))
    distances.sort(key=lambda kv: kv[1])
    return [idx for idx, _ in distances[:top_k]]

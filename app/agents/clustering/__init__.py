"""Respondent cluster analysis package.

Components:
- taxonomy: static canonical vocabulary for tag harmonization
- cache: TTL cache for LLM-resolved unknown tags
- tag_harmonizer: static + RapidFuzz + LLM fallback
- feature_extraction: per-session deterministic features
- cluster_labeler: one LLM call per cluster
- respondent_clustering_agent: full orchestrator
"""

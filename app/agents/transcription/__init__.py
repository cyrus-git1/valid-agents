"""Transcription orchestrator package.

Components:
- discriminate_agent: deterministic VTT analysis (stats + spaCy NER + VADER)
- llm_agents: 5 LLM-backed sub-agents (sentiment, themes, summary, insights, quotes)
- orchestrator: single-session pipeline (discriminate + parallel LLM agents)
- session_loader: pull KB-stored transcript chunks back into vtt_content
- aggregator: cross-session synthesis within a survey
- cache: TTL cache for orchestrator results
"""

"""
Lightweight base for analysis services in the agent service.

Replaces src/services/base_service.py — uses core_client HTTP calls
instead of direct Supabase access.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.llm_config import LLMConfig
from app import core_client

logger = logging.getLogger(__name__)


class BaseAnalysisService:
    """Base class providing shared helpers for analysis services."""

    def __init__(self, tenant_id: str = "", client_id: str = ""):
        self.tenant_id = tenant_id
        self.client_id = client_id

    # ── LLM ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_llm(model: str = LLMConfig.DEFAULT, temperature: float = 0.1) -> ChatOpenAI:
        return ChatOpenAI(model=model, temperature=temperature)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_json(raw_output: str, fallback_keys: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        logger.warning("LLM returned non-JSON — using fallback structure")
        result = dict(fallback_keys or {})
        result["raw_output"] = raw_output
        return result

    # ── Data access (via core API) ────────────────────────────────────────────

    def _count_transcripts(self, tenant_id: str, client_id: str) -> int:
        return core_client.count_transcripts(tenant_id=tenant_id, client_id=client_id)

    def _get_transcript_chunks(self, tenant_id: str, client_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return core_client.get_transcript_chunks(tenant_id=tenant_id, client_id=client_id, limit=limit)

    # ── Profile formatting ────────────────────────────────────────────────────

    @staticmethod
    def _build_profile_section(client_profile: Optional[Dict[str, Any]]) -> str:
        if not client_profile:
            return ""
        parts: List[str] = []
        for key in ("industry", "headcount", "revenue", "company_name", "persona"):
            if client_profile.get(key):
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {client_profile[key]}")
        demo = client_profile.get("demographic", {})
        if isinstance(demo, dict):
            for key in ("age_range", "income_bracket", "occupation", "location"):
                if demo.get(key):
                    parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
        if not parts:
            return ""
        return "Company / Client Profile:\n" + "\n".join(parts) + "\n\n"

    # ── Transcript context building ───────────────────────────────────────────

    def _build_transcript_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "(No video transcript data available. Ingest .vtt transcripts to enable analysis.)"
        return "\n\n---\n\n".join(
            f"[Transcript Excerpt {i + 1}] {c['content']}"
            for i, c in enumerate(chunks)
            if c.get("content", "").strip()
        )

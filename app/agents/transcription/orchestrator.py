"""
Single-session transcription orchestrator.

Validates a WebVTT, runs the discriminate agent, fans out to LLM sub-agents
in parallel via ThreadPoolExecutor, caches by (scope, vtt_hash, requested
analyses), and assembles a unified response.

Used by `POST /transcripts/individual` and indirectly by the cross-session
aggregator.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from app.agents.transcription import cache as orchestrator_cache
from app.agents.transcription.discriminate_agent import run_discriminate
from app.agents.transcription.llm_agents import AGENT_REGISTRY

logger = logging.getLogger(__name__)


# Default analyses run if the caller doesn't specify
DEFAULT_LLM_ANALYSES = ["sentiment", "themes", "summary", "insights", "quotes"]
ALL_ANALYSES = ["discriminate"] + DEFAULT_LLM_ANALYSES


def run_orchestration(
    *,
    vtt_content: str,
    tenant_id: str,
    client_id: str,
    survey_id: str,
    session_id: str,
    analyses: Optional[List[str]] = None,
    focus: Optional[str] = None,
    summary_type: str = "general",
    language: str = "en",
    **extra: Any,
) -> Dict[str, Any]:
    """Run the single-session pipeline. Returns a unified response dict.

    Validates the VTT, checks the TTL cache, runs discriminate first
    (synchronous), then fans out to requested LLM agents in parallel.
    Per-agent failures don't block other agents — collected in `errors`.
    """
    t0 = time.monotonic()

    # ── Validate ────────────────────────────────────────────────────────
    if not vtt_content or not vtt_content.strip().startswith("WEBVTT"):
        raise ValueError("vtt_content must start with 'WEBVTT'")

    # ── Resolve scope ───────────────────────────────────────────────────
    requested = analyses if analyses is not None else ALL_ANALYSES
    requested_llm = [a for a in requested if a in AGENT_REGISTRY]
    include_discriminate = ("discriminate" in requested) or (analyses is None)
    scope = list(requested_llm) + (["discriminate"] if include_discriminate else [])

    # ── Cache lookup ────────────────────────────────────────────────────
    vh = orchestrator_cache.vtt_hash(vtt_content)
    key = orchestrator_cache.make_key(
        tenant_id, client_id, survey_id, session_id, vh, scope,
    )
    hit = orchestrator_cache.get(key)
    if hit is not None:
        # Return a copy with from_cache=True and fresh elapsed_ms
        out = dict(hit)
        out["from_cache"] = True
        out["elapsed_ms"] = round((time.monotonic() - t0) * 1000.0, 2)
        return out

    # ── Discriminate (synchronous, fast) ────────────────────────────────
    discriminate = run_discriminate(vtt_content) if include_discriminate else None

    # ── LLM agents in parallel ──────────────────────────────────────────
    agent_inputs = {
        "vtt_content": vtt_content,
        "tenant_id": tenant_id,
        "client_id": client_id,
        "survey_id": survey_id,
        "session_id": session_id,
        "focus": focus,
        "summary_type": summary_type,
        **extra,
    }

    section_results: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    if requested_llm:
        max_workers = min(8, len(requested_llm))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_name = {
                ex.submit(AGENT_REGISTRY[name], agent_inputs): name
                for name in requested_llm
            }
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    out = fut.result()
                    section_results[name] = out.get("result") if isinstance(out, dict) else out
                    if isinstance(out, dict) and out.get("error"):
                        errors.append({"agent": name, "error": str(out["error"])})
                except Exception as e:
                    logger.exception("agent %s crashed", name)
                    section_results[name] = {"status": "failed", "error": str(e)}
                    errors.append({"agent": name, "error": str(e)})

    # ── Assemble response ───────────────────────────────────────────────
    duration = None
    if discriminate and isinstance(discriminate, dict):
        duration = (discriminate.get("totals") or {}).get("duration_seconds")

    response: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "survey_id": survey_id,
        "session_id": session_id,
        "vtt_hash": vh,
        "from_cache": False,
        "language": language,
        "duration_seconds": duration,
        "discriminate": discriminate,
        "sentiment": section_results.get("sentiment"),
        "themes":    section_results.get("themes"),
        "summary":   section_results.get("summary"),
        "insights":  section_results.get("insights"),
        "quotes":    section_results.get("quotes"),
        "errors": errors,
        "status": "complete" if not errors else "partial",
        "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
    }

    orchestrator_cache.put(key, response)
    return response

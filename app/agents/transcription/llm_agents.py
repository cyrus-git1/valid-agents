"""
LLM sub-agents for the transcription orchestrator.

Uniform signature contract (load-bearing for harness wrapping later):

    def run_<agent>(inputs: dict) -> dict:
        '''
        inputs:  { vtt_content, tenant_id, client_id, survey_id, session_id,
                   focus?, summary_type? }
        returns: { result, raw_llm_output, prompt_sent, model_name, error }
        '''

The output keys (`result`, `raw_llm_output`, `prompt_sent`, `model_name`)
are isomorphic to `app.harness_pkg.engine.StepOutput`. A future harness
wrapper is mechanical — see plan file.

Three of the agents (sentiment / summary / insights) wrap existing services
that don't surface `raw_llm_output` / `prompt_sent`. For v1 we re-render
the prompt template with the same vars to populate `prompt_sent` and
leave `raw_llm_output=""`. The two new agents (themes / quotes) build
their chains directly so they capture both fields natively.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict
from uuid import UUID, uuid4

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm, LLMConfig
from app.prompts.transcript_theme_prompts import THEME_EXTRACTION_PROMPT
from app.prompts.transcript_quote_prompts import QUOTE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def _coerce_uuid(value: Any) -> UUID:
    """Best-effort coerce string → UUID, generating a random one if invalid."""
    try:
        return UUID(str(value)) if value else uuid4()
    except Exception:
        return uuid4()


def _parse_json_loose(raw: str) -> Dict[str, Any]:
    """Parse JSON, stripping markdown fences if present."""
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _empty_output(error: str) -> Dict[str, Any]:
    return {
        "result": {},
        "raw_llm_output": "",
        "prompt_sent": "",
        "model_name": "",
        "error": error,
    }


# ── Sentiment agent (wraps SentimentAnalysisService) ────────────────────


def run_sentiment_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap SentimentAnalysisService.generate_from_vtt() in the uniform contract."""
    vtt = inputs.get("vtt_content") or ""
    if not vtt.strip():
        return _empty_output("vtt_content is empty")

    try:
        from app.analysis.sentiment import SentimentAnalysisService
        svc = SentimentAnalysisService()
        result = svc.generate_from_vtt(
            tenant_id=_coerce_uuid(inputs.get("tenant_id")),
            survey_id=_coerce_uuid(inputs.get("survey_id")),
            vtt_content=vtt,
        )
    except Exception as e:
        logger.exception("sentiment agent failed")
        return _empty_output(str(e))

    # Re-render the prompt template for v1 trace fidelity. The underlying
    # service uses SENTIMENT_ANALYSIS_PROMPT — we render with the same vars.
    prompt_sent = ""
    try:
        from app.prompts.sentiment_prompts import SENTIMENT_ANALYSIS_PROMPT
        prompt_sent = SENTIMENT_ANALYSIS_PROMPT.format(
            focus_instructions="",
            profile_section="",
            transcript_count="1",
            chunk_count="1",
            transcript_context=vtt,
            context_summary="(Not applicable — raw VTT provided.)",
        )
    except Exception as e:
        logger.debug("sentiment trace re-render failed: %s", e)

    return {
        "result": result,
        "raw_llm_output": "",  # service doesn't surface; documented gap
        "prompt_sent": prompt_sent,
        "model_name": LLMConfig.DEFAULT,
        "error": result.get("error") if isinstance(result, dict) else None,
    }


# ── Summary agent (wraps generate_summary_from_vtt) ─────────────────────


def run_summary_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap TranscriptInsightsService.generate_summary_from_vtt()."""
    vtt = inputs.get("vtt_content") or ""
    if not vtt.strip():
        return _empty_output("vtt_content is empty")

    summary_type = inputs.get("summary_type") or "general"

    try:
        from app.analysis.transcript_insights import TranscriptInsightsService
        svc = TranscriptInsightsService()
        result = svc.generate_summary_from_vtt(
            tenant_id=_coerce_uuid(inputs.get("tenant_id")),
            vtt_content=vtt,
            summary_type=summary_type,
        )
    except Exception as e:
        logger.exception("summary agent failed")
        return _empty_output(str(e))

    prompt_sent = ""
    try:
        from app.prompts.transcript_summary_prompts import TRANSCRIPT_SUMMARY_PROMPTS
        tmpl = TRANSCRIPT_SUMMARY_PROMPTS.get(summary_type) or TRANSCRIPT_SUMMARY_PROMPTS.get("general")
        if tmpl is not None:
            prompt_sent = tmpl.format(
                transcript_count="1", chunk_count="1", transcript_context=vtt,
            )
    except Exception as e:
        logger.debug("summary trace re-render failed: %s", e)

    return {
        "result": result,
        "raw_llm_output": "",
        "prompt_sent": prompt_sent,
        "model_name": LLMConfig.DEFAULT,
        "error": result.get("error") if isinstance(result, dict) else None,
    }


# ── Insights agent (wraps TranscriptInsightsService.generate_from_vtt) ──


def run_insights_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    vtt = inputs.get("vtt_content") or ""
    if not vtt.strip():
        return _empty_output("vtt_content is empty")

    try:
        from app.analysis.transcript_insights import TranscriptInsightsService
        svc = TranscriptInsightsService()
        result = svc.generate_from_vtt(
            tenant_id=_coerce_uuid(inputs.get("tenant_id")),
            survey_id=_coerce_uuid(inputs.get("survey_id")),
            vtt_content=vtt,
        )
    except Exception as e:
        logger.exception("insights agent failed")
        return _empty_output(str(e))

    prompt_sent = ""
    try:
        from app.prompts.transcript_insights_prompts import TRANSCRIPT_INSIGHTS_PROMPT
        prompt_sent = TRANSCRIPT_INSIGHTS_PROMPT.format(
            transcript_count="1", chunk_count="1", transcript_context=vtt,
        )
    except Exception as e:
        logger.debug("insights trace re-render failed: %s", e)

    return {
        "result": result,
        "raw_llm_output": "",
        "prompt_sent": prompt_sent,
        "model_name": LLMConfig.DEFAULT,
        "error": result.get("error") if isinstance(result, dict) else None,
    }


# ── Theme agent (NEW prompt) ────────────────────────────────────────────


def run_theme_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    vtt = inputs.get("vtt_content") or ""
    if not vtt.strip():
        return _empty_output("vtt_content is empty")

    focus = inputs.get("focus") or "(none)"
    llm = get_llm("context_analysis")
    chain = THEME_EXTRACTION_PROMPT | llm | StrOutputParser()

    prompt_sent = ""
    try:
        prompt_sent = THEME_EXTRACTION_PROMPT.format(
            focus=focus, transcript_context=vtt,
        )
    except Exception:
        pass

    try:
        raw = chain.invoke({"focus": focus, "transcript_context": vtt})
        parsed = _parse_json_loose(raw)
        return {
            "result": parsed if parsed else {"themes": []},
            "raw_llm_output": raw or "",
            "prompt_sent": prompt_sent,
            "model_name": getattr(llm, "model_name", LLMConfig.CONTEXT_ANALYSIS),
            "error": None if parsed else "Could not parse JSON",
        }
    except Exception as e:
        logger.exception("theme agent failed")
        return {
            "result": {"themes": []},
            "raw_llm_output": "",
            "prompt_sent": prompt_sent,
            "model_name": getattr(llm, "model_name", LLMConfig.CONTEXT_ANALYSIS),
            "error": str(e),
        }


# ── Quote agent (NEW prompt) ────────────────────────────────────────────


def run_quote_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    vtt = inputs.get("vtt_content") or ""
    if not vtt.strip():
        return _empty_output("vtt_content is empty")

    focus = inputs.get("focus") or "(none)"
    llm = get_llm("context_analysis")
    chain = QUOTE_EXTRACTION_PROMPT | llm | StrOutputParser()

    prompt_sent = ""
    try:
        prompt_sent = QUOTE_EXTRACTION_PROMPT.format(
            focus=focus, transcript_context=vtt,
        )
    except Exception:
        pass

    try:
        raw = chain.invoke({"focus": focus, "transcript_context": vtt})
        parsed = _parse_json_loose(raw)
        return {
            "result": parsed if parsed else {"notable_quotes": []},
            "raw_llm_output": raw or "",
            "prompt_sent": prompt_sent,
            "model_name": getattr(llm, "model_name", LLMConfig.CONTEXT_ANALYSIS),
            "error": None if parsed else "Could not parse JSON",
        }
    except Exception as e:
        logger.exception("quote agent failed")
        return {
            "result": {"notable_quotes": []},
            "raw_llm_output": "",
            "prompt_sent": prompt_sent,
            "model_name": getattr(llm, "model_name", LLMConfig.CONTEXT_ANALYSIS),
            "error": str(e),
        }


# ── Public registry ──────────────────────────────────────────────────────

# Names map onto the public `analyses` array in the API request body
AGENT_REGISTRY = {
    "sentiment": run_sentiment_agent,
    "themes":    run_theme_agent,
    "summary":   run_summary_agent,
    "insights":  run_insights_agent,
    "quotes":    run_quote_agent,
}

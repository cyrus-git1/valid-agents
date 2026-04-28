"""
Discriminate (deterministic) transcript analysis agent.

Composes the pure tools in `app/tools/discriminate_transcript_tools.py`
into one synchronous run. No LLM calls. Typical latency: ~50-200ms for a
short transcript.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.services.chunking_service import ChunkingService
from app.tools import discriminate_transcript_tools as T

logger = logging.getLogger(__name__)


def run_discriminate(vtt_content: str) -> Dict[str, Any]:
    """Parse VTT and run all discriminate analyses. Returns a structured dict.

    On parse failure or empty input, returns a fallback with empty sections
    and an `error` key — does not raise.
    """
    try:
        cues = ChunkingService.parse_vtt(vtt_content) or []
    except Exception as e:
        logger.warning("discriminate: parse_vtt failed: %s", e)
        return {
            "speakers": [],
            "totals": {"total_words": 0, "total_cues": 0, "total_speakers": 0, "duration_seconds": 0.0},
            "questions": [],
            "filler_words": {},
            "entities": [],
            "vader_sentiment": {"overall": {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}, "per_speaker": {}},
            "error": f"Could not parse VTT: {e}",
        }

    speakers = T.speaker_turn_stats(cues)
    tot = T.totals(cues)
    questions = T.detect_questions(cues)
    fillers = T.count_fillers(cues)
    entities = T.extract_entities_spacy(cues)
    vader_per_cue = T.vader_sentiment_per_cue(cues)
    vader_agg = T.vader_aggregate(vader_per_cue)
    sarcasm_flags = T.detect_sarcasm_markers(cues)  # Tier 1 — markers only
    # Tier 1 analytics extensions:
    sentiment_traj = T.sentiment_trajectory(cues, vader_per_cue)
    qual_nps = T.qualitative_nps(vader_per_cue)

    return {
        "speakers": speakers,
        "totals": tot,
        "questions": questions,
        "filler_words": fillers,
        "entities": entities,
        "vader_sentiment": vader_agg,
        # Sentiment trajectory: timeline + inflection points + arc summary.
        # Lets the frontend draw a sentiment-over-time chart and surface
        # the moments where sentiment shifted (e.g., when pricing came up).
        "sentiment_trajectory": sentiment_traj,
        # Qualitative NPS: -100..100 score derived from VADER cue buckets
        # so qualitative sessions can be compared directly to NPS surveys.
        "qualitative_nps": qual_nps,
        # vader_per_cue can be large; include but the orchestrator may strip
        # it from the API response for size if requested.
        "vader_per_cue": vader_per_cue,
        # Tier 1 sarcasm flags (deterministic markers). The orchestrator
        # will run Tier 2 (cross-layer VADER vs LLM) after the sentiment
        # agent completes and may upgrade these flags' confidence.
        "sarcasm_flags": sarcasm_flags,
        # cues kept for the orchestrator's reconcile step; not surfaced in
        # the public response.
        "_cues": cues,
    }

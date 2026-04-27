"""
Per-session feature extraction for cluster analysis.

Deterministic — no LLM calls except via tag_harmonizer's fallback for
unseen tags. For each session, builds a fixed-shape dict that the
orchestrator can vectorize.

Session shape (from list_documents + filter):
  - One Document with metadata containing survey_id, session_id, plus
    arbitrary demographic tags
  - chunks list with content/text fields

Output shape:
  {
    "session_id": str,
    "survey_id": str,
    "normalized_tags": {field: canonical_value, ...},
    "tag_sources": {field: "static|fuzzy|cache|llm|unknown", ...},
    "statistical_features": {word_count, duration_seconds, wpm,
                             filler_rate, vader_compound, vader_pos,
                             vader_neg, question_count, speaker_count},
    "lexicon_traits": [trait_name, ...],
    "entity_counts": {ORG: int, PRODUCT: int, PERSON: int, ...},
    "concatenated_text": str,
  }
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.agents.clustering.tag_harmonizer import harmonize_tag
from app.agents.clustering.taxonomy import TAG_FIELDS

logger = logging.getLogger(__name__)


# ── Behavioural trait lexicon ─────────────────────────────────────────────
# Each trait → list of word/phrase patterns. Word-boundary matched.

TRAIT_LEXICON: Dict[str, List[str]] = {
    "price_sensitive": [
        "expensive", "cost", "costly", "budget", "afford", "cheap",
        "pricing", "save money", "too much", "overpriced",
    ],
    "feature_oriented": [
        "feature", "functionality", "capability", "integration",
        "support for", "missing", "wish it had", "would love",
    ],
    "technically_proficient": [
        "api", "sdk", "webhook", "documentation", "endpoint",
        "schema", "json", "rest", "graphql", "deployment",
    ],
    "decision_maker": [
        "i decide", "we chose", "i picked", "my call", "approve",
        "i sign off", "final say", "i decided",
    ],
    "champion_seeking": [
        "pilot", "proof of concept", "poc", "trial", "test it out",
        "try it", "evaluate",
    ],
    "data_driven": [
        "metrics", "data", "kpi", "kpis", "measure", "benchmark",
        "analytics", "dashboard", "report", "numbers",
    ],
    "risk_averse": [
        "risk", "concerned", "worried", "compliance", "security",
        "audit", "policy", "approval", "legal", "vetting",
    ],
    "growth_focused": [
        "growth", "scale", "expansion", "acquire", "user acquisition",
        "ramp up", "fast growing", "growth stage",
    ],
}

_TRAIT_REGEXES: Dict[str, List[re.Pattern]] = {
    trait: [re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in patterns]
    for trait, patterns in TRAIT_LEXICON.items()
}

_TRAIT_THRESHOLD = 3  # min hits for trait to be assigned


# ── Public API ───────────────────────────────────────────────────────────


def extract_session_features(
    document: Dict[str, Any],
    *,
    survey_id: str,
    session_id: str,
) -> Dict[str, Any]:
    """Extract all per-session features from a single document.

    Caller passes the document (one row from list_documents.items) and the
    survey_id/session_id already known from metadata. We pull text from
    the document's chunks.
    """
    metadata = document.get("metadata") or {}
    chunks = document.get("chunks") or []

    # Concatenated text for embedding + lexicon analysis
    text_parts: List[str] = []
    for ch in chunks:
        t = (ch.get("content") or ch.get("text") or "").strip()
        if t:
            text_parts.append(t)
    concatenated_text = "\n\n".join(text_parts)

    # Tag harmonization
    normalized_tags: Dict[str, str] = {}
    tag_sources: Dict[str, str] = {}
    for field in TAG_FIELDS.keys():
        raw = metadata.get(field)
        if raw is None:
            continue
        canonical, source = harmonize_tag(field, str(raw))
        normalized_tags[field] = canonical
        tag_sources[field] = source

    # Statistical features
    statistical_features = _compute_statistical_features(
        concatenated_text=concatenated_text,
        chunks=chunks,
        metadata=metadata,
    )

    # Lexicon traits
    lexicon_traits = _detect_traits(concatenated_text)

    # Entity counts (spaCy)
    entity_counts = _count_entities(concatenated_text)

    return {
        "session_id": session_id,
        "survey_id": survey_id,
        "normalized_tags": normalized_tags,
        "tag_sources": tag_sources,
        "statistical_features": statistical_features,
        "lexicon_traits": lexicon_traits,
        "entity_counts": entity_counts,
        "concatenated_text": concatenated_text,
    }


# ── Internals ────────────────────────────────────────────────────────────


def _compute_statistical_features(
    *,
    concatenated_text: str,
    chunks: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Statistical features computed deterministically from text + chunks."""
    text = concatenated_text or ""
    word_count = len(text.split()) if text else 0

    # Try to use the existing transcript discriminate tools when the text
    # looks like it came from a VTT-shaped transcript (has speakers).
    duration_seconds = float(metadata.get("duration_seconds", 0.0)) if metadata else 0.0
    wpm = (word_count / (duration_seconds / 60.0)) if duration_seconds > 0 else 0.0
    filler_rate = _compute_filler_rate(text, word_count)
    question_count = _count_questions(text)
    speaker_count = _estimate_speaker_count(text)

    # VADER aggregate over the full text
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        scorer = _vader_singleton()
        scores = scorer.polarity_scores(text) if text else {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    except Exception as e:
        logger.warning("VADER scoring failed: %s", e)
        scores = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    return {
        "word_count": word_count,
        "duration_seconds": round(duration_seconds, 2),
        "wpm": round(wpm, 1),
        "filler_rate": round(filler_rate, 4),
        "vader_compound": round(scores.get("compound", 0.0), 4),
        "vader_pos": round(scores.get("pos", 0.0), 4),
        "vader_neg": round(scores.get("neg", 0.0), 4),
        "vader_neu": round(scores.get("neu", 0.0), 4),
        "question_count": question_count,
        "speaker_count": speaker_count,
    }


_FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "so", "actually", "basically", "literally", "right",
]
_FILLER_REGEXES = [re.compile(rf"\b{re.escape(f)}\b", re.IGNORECASE) for f in _FILLER_WORDS]


def _compute_filler_rate(text: str, word_count: int) -> float:
    if not text or word_count == 0:
        return 0.0
    total_fillers = sum(len(rx.findall(text)) for rx in _FILLER_REGEXES)
    return total_fillers / word_count


_QUESTION_START_RE = re.compile(
    r"\b(who|what|when|where|why|how|which|do|does|did|is|are|was|were|will|"
    r"would|could|should|can|may|might)\b",
    re.IGNORECASE,
)


def _count_questions(text: str) -> int:
    if not text:
        return 0
    # Sentence-ish split — periods, question marks, newlines
    sentences = re.split(r"[.!?\n]+", text)
    n = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if "?" in s or _QUESTION_START_RE.match(s):
            n += 1
    return n


_SPEAKER_RE = re.compile(r"\b([A-Z][a-z]{1,20})\s*:")


def _estimate_speaker_count(text: str) -> int:
    """Best-effort speaker count from 'Name: ' patterns at line start.

    Falls back to 1 if no clear speaker markers found.
    """
    if not text:
        return 0
    speakers = set()
    for line in text.splitlines():
        m = _SPEAKER_RE.match(line.strip())
        if m:
            speakers.add(m.group(1))
    return len(speakers) or 1


def _detect_traits(text: str) -> List[str]:
    if not text:
        return []
    traits: List[str] = []
    for trait, regexes in _TRAIT_REGEXES.items():
        hits = sum(len(rx.findall(text)) for rx in regexes)
        if hits >= _TRAIT_THRESHOLD:
            traits.append(trait)
    return traits


def _count_entities(text: str) -> Dict[str, int]:
    if not text:
        return {}
    try:
        from app.services.chunking_service import nlp
    except Exception as e:
        logger.warning("spaCy unavailable for entity counts: %s", e)
        return {}
    try:
        # Cap input size to avoid quadratic spaCy slowdowns
        doc = nlp(text[:50_000])
    except Exception as e:
        logger.warning("spaCy NER failed: %s", e)
        return {}
    counts: Dict[str, int] = {}
    for ent in doc.ents:
        counts[ent.label_] = counts.get(ent.label_, 0) + 1
    return counts


_VADER_INSTANCE = None


def _vader_singleton():
    """Lazy-init VADER scorer (avoid module-load cost when unused)."""
    global _VADER_INSTANCE
    if _VADER_INSTANCE is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER_INSTANCE = SentimentIntensityAnalyzer()
    return _VADER_INSTANCE

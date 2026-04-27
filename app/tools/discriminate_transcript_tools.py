"""
Deterministic ("discriminate") transcript analysis tools.

Pure functions that operate on already-parsed VTT cues. No LLM calls.
Used by `app/agents/transcription/discriminate_agent.py` and ultimately
the `/transcripts/individual` orchestrator.

Cue shape (from `ChunkingService.parse_vtt`):
    {"index": int, "start": float, "end": float, "speaker": str|None, "text": str}
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reuse the already-loaded spaCy pipeline
from app.services.chunking_service import nlp


# Module-level singletons (thread-safe for read-only use)
_VADER = SentimentIntensityAnalyzer()


# ── Speaker turn statistics ───────────────────────────────────────────────


def _speaker_label(speaker: Optional[str]) -> str:
    return (speaker or "Unknown").strip() or "Unknown"


def speaker_turn_stats(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-speaker counts: cue_count, total_seconds, word_count, wpm."""
    grouped: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"cue_count": 0, "total_seconds": 0.0, "word_count": 0}
    )
    for c in cues:
        spk = _speaker_label(c.get("speaker"))
        duration = max(0.0, float(c.get("end", 0.0)) - float(c.get("start", 0.0)))
        words = len((c.get("text") or "").split())
        grouped[spk]["cue_count"] += 1
        grouped[spk]["total_seconds"] += duration
        grouped[spk]["word_count"] += words

    result = []
    for spk, stats in grouped.items():
        seconds = stats["total_seconds"]
        wpm = round((stats["word_count"] / (seconds / 60.0)), 1) if seconds > 0 else 0.0
        result.append({
            "speaker": spk,
            "cue_count": int(stats["cue_count"]),
            "total_seconds": round(seconds, 2),
            "word_count": int(stats["word_count"]),
            "wpm": wpm,
        })
    # Sort by speaking time descending for stable ordering
    result.sort(key=lambda r: r["total_seconds"], reverse=True)
    return result


def totals(cues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate totals: total_words, total_cues, total_speakers, duration_seconds."""
    if not cues:
        return {
            "total_words": 0, "total_cues": 0,
            "total_speakers": 0, "duration_seconds": 0.0,
        }
    total_words = sum(len((c.get("text") or "").split()) for c in cues)
    speakers = {_speaker_label(c.get("speaker")) for c in cues}
    duration = max(float(c.get("end", 0.0)) for c in cues)
    return {
        "total_words": total_words,
        "total_cues": len(cues),
        "total_speakers": len(speakers),
        "duration_seconds": round(duration, 2),
    }


# ── Question detection ────────────────────────────────────────────────────

# WH-words and aux verbs that commonly start questions when at sentence start
_QUESTION_START_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|which|do|does|did|is|are|was|were|will|"
    r"would|could|should|can|may|might|have|has|had)\b",
    re.I,
)


def detect_questions(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return cues that look like questions (end with '?' OR start with WH/aux)."""
    out = []
    for c in cues:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        is_q = text.endswith("?") or bool(_QUESTION_START_RE.match(text))
        if is_q:
            out.append({
                "cue_index": c.get("index"),
                "speaker": _speaker_label(c.get("speaker")),
                "start": round(float(c.get("start", 0.0)), 2),
                "text": text,
            })
    return out


# ── Filler word counting ──────────────────────────────────────────────────

_FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "so", "actually", "basically", "literally", "right",
]
# Build word-boundary regexes per filler so "like" doesn't match "alike"
_FILLER_RES = [
    (f, re.compile(rf"\b{re.escape(f)}\b", re.I))
    for f in _FILLER_WORDS
]


def count_fillers(cues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count filler words per speaker. Returns {speaker: {filler: count}, _total: {...}}."""
    per_speaker: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals_dict: Dict[str, int] = defaultdict(int)
    for c in cues:
        spk = _speaker_label(c.get("speaker"))
        text = (c.get("text") or "")
        if not text:
            continue
        for filler, regex in _FILLER_RES:
            n = len(regex.findall(text))
            if n > 0:
                per_speaker[spk][filler] += n
                totals_dict[filler] += n

    return {
        **{spk: dict(d) for spk, d in per_speaker.items()},
        "_total": dict(totals_dict),
    }


# ── spaCy NER ─────────────────────────────────────────────────────────────


def extract_entities_spacy(
    cues: List[Dict[str, Any]],
    *,
    max_sample_contexts: int = 3,
    min_count: int = 1,
) -> List[Dict[str, Any]]:
    """Run spaCy NER on the full transcript text. Dedupe by (text, label).

    Returns: [{text, label, count, sample_contexts}].
    """
    if not cues:
        return []
    # Build a single text blob with cue boundaries preserved as line breaks so
    # spaCy can use surrounding context.
    full_text = "\n".join((c.get("text") or "").strip() for c in cues if c.get("text"))
    if not full_text.strip():
        return []

    doc = nlp(full_text)
    aggregated: Dict[tuple, Dict[str, Any]] = {}
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if not key[0]:
            continue
        if key not in aggregated:
            aggregated[key] = {
                "text": key[0],
                "label": key[1],
                "count": 0,
                "sample_contexts": [],
            }
        aggregated[key]["count"] += 1
        if len(aggregated[key]["sample_contexts"]) < max_sample_contexts:
            sent = ent.sent.text.strip() if ent.sent is not None else ""
            if sent and sent not in aggregated[key]["sample_contexts"]:
                aggregated[key]["sample_contexts"].append(sent[:240])

    out = [v for v in aggregated.values() if v["count"] >= min_count]
    out.sort(key=lambda e: e["count"], reverse=True)
    return out


# ── VADER sentiment ───────────────────────────────────────────────────────


def vader_sentiment_per_cue(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-cue VADER scores (compound, pos, neu, neg)."""
    out = []
    for c in cues:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        scores = _VADER.polarity_scores(text)
        out.append({
            "cue_index": c.get("index"),
            "speaker": _speaker_label(c.get("speaker")),
            "compound": round(scores["compound"], 4),
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
        })
    return out


def vader_aggregate(per_cue: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate VADER scores: overall mean + per-speaker mean."""
    if not per_cue:
        return {
            "overall": {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0},
            "per_speaker": {},
        }

    def _mean(items, key):
        return round(sum(it[key] for it in items) / len(items), 4) if items else 0.0

    overall = {
        "compound": _mean(per_cue, "compound"),
        "pos": _mean(per_cue, "pos"),
        "neu": _mean(per_cue, "neu"),
        "neg": _mean(per_cue, "neg"),
    }

    per_spk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in per_cue:
        per_spk[s["speaker"]].append(s)
    per_speaker = {
        spk: {
            "compound": _mean(items, "compound"),
            "pos": _mean(items, "pos"),
            "neu": _mean(items, "neu"),
            "neg": _mean(items, "neg"),
        }
        for spk, items in per_spk.items()
    }

    return {"overall": overall, "per_speaker": per_speaker}

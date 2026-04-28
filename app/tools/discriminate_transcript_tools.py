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


def sentiment_trajectory(
    cues: List[Dict[str, Any]],
    per_cue: List[Dict[str, Any]],
    *,
    inflection_threshold: float = 0.4,
    smoothing_window: int = 3,
) -> Dict[str, Any]:
    """Compute a sentiment timeline + inflection points across a session.

    Pairs each VADER per-cue score with the cue's start time so the frontend
    can render a line chart. Detects inflection points where the smoothed
    compound score shifts by `inflection_threshold` between consecutive cues.

    Returns:
        {
          "timeline": [{time_seconds, compound, smoothed_compound, speaker, cue_index}],
          "inflection_points": [{
              from_time, to_time, from_compound, to_compound, delta,
              direction, speaker_at_shift, surrounding_text
          }],
          "arc_summary": "starts positive, dips at 04:32, recovers"
        }
    """
    if not cues or not per_cue:
        return {"timeline": [], "inflection_points": [], "arc_summary": ""}

    # Index cues by index for quick lookup of start times + text
    cue_by_index = {c.get("index"): c for c in cues}

    # Build raw timeline (preserve per-cue ordering as they appear in cues)
    raw_timeline: List[Dict[str, Any]] = []
    for s in per_cue:
        ci = s.get("cue_index")
        cue = cue_by_index.get(ci)
        if cue is None:
            continue
        raw_timeline.append({
            "cue_index": ci,
            "time_seconds": round(float(cue.get("start", 0.0)), 2),
            "compound": s.get("compound", 0.0),
            "speaker": s.get("speaker", "Unknown"),
            "text": (cue.get("text") or "")[:280],
        })
    raw_timeline.sort(key=lambda p: p["time_seconds"])

    # Apply moving-average smoothing so a single sarcastic outlier doesn't
    # register as an inflection point
    half = max(1, smoothing_window // 2)
    smoothed: List[float] = []
    for i, point in enumerate(raw_timeline):
        lo = max(0, i - half)
        hi = min(len(raw_timeline), i + half + 1)
        window = [p["compound"] for p in raw_timeline[lo:hi]]
        smoothed.append(round(sum(window) / len(window), 4))

    timeline = [
        {**p, "smoothed_compound": smoothed[i]}
        for i, p in enumerate(raw_timeline)
    ]

    # Detect inflection points: |smoothed[i] - smoothed[i-1]| >= threshold
    inflection_points: List[Dict[str, Any]] = []
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        delta = curr["smoothed_compound"] - prev["smoothed_compound"]
        if abs(delta) < inflection_threshold:
            continue
        inflection_points.append({
            "from_time": prev["time_seconds"],
            "to_time": curr["time_seconds"],
            "from_compound": prev["smoothed_compound"],
            "to_compound": curr["smoothed_compound"],
            "delta": round(delta, 4),
            "direction": "positive_shift" if delta > 0 else "negative_shift",
            "speaker_at_shift": curr["speaker"],
            "surrounding_text": curr.get("text", "")[:240],
        })

    return {
        "timeline": timeline,
        "inflection_points": inflection_points,
        "arc_summary": _describe_arc(timeline, inflection_points),
    }


def _describe_arc(
    timeline: List[Dict[str, Any]],
    inflection_points: List[Dict[str, Any]],
) -> str:
    """Plain-text summary of the sentiment arc. Deterministic."""
    if not timeline:
        return ""

    def _label(c: float) -> str:
        if c >= 0.3:
            return "positive"
        if c <= -0.2:
            return "negative"
        return "neutral"

    start_label = _label(timeline[0]["smoothed_compound"])
    end_label = _label(timeline[-1]["smoothed_compound"])

    if not inflection_points:
        if start_label == end_label:
            return f"Stays {start_label} throughout."
        return f"Drifts from {start_label} to {end_label}."

    # Format mm:ss for shifts
    def _fmt(t: float) -> str:
        m = int(t // 60)
        s = int(t % 60)
        return f"{m:02d}:{s:02d}"

    parts = [f"Starts {start_label}"]
    for ip in inflection_points[:5]:  # cap so summary isn't a wall of text
        verb = "rises" if ip["direction"] == "positive_shift" else "drops"
        parts.append(f"{verb} at {_fmt(ip['to_time'])}")
    parts.append(f"ends {end_label}")
    return ", ".join(parts) + "."


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


# ── Qualitative NPS / net sentiment ──────────────────────────────────────


def qualitative_nps(
    per_cue: List[Dict[str, Any]],
    *,
    promoter_threshold: float = 0.5,
    detractor_threshold: float = -0.2,
) -> Dict[str, Any]:
    """Compute an NPS-style score from VADER per-cue compound scores.

    Treats each cue as a "vote":
      - compound >= promoter_threshold → promoter
      - compound <= detractor_threshold → detractor
      - otherwise → passive

    Returns:
      {
        "qualitative_nps": float,        # promoter_pct - detractor_pct, range -100..100
        "promoter_pct": float,
        "passive_pct": float,
        "detractor_pct": float,
        "promoters": int, "passives": int, "detractors": int,
        "n_cues": int,
        "per_speaker": {speaker: {qualitative_nps, ...}}
      }

    Lets researchers compare qualitative interview sentiment on the same
    -100..100 scale as their NPS surveys.
    """
    if not per_cue:
        return {
            "qualitative_nps": 0.0, "promoter_pct": 0.0, "passive_pct": 0.0,
            "detractor_pct": 0.0, "promoters": 0, "passives": 0, "detractors": 0,
            "n_cues": 0, "per_speaker": {},
        }

    def _bucket(score: float) -> str:
        if score >= promoter_threshold:
            return "promoter"
        if score <= detractor_threshold:
            return "detractor"
        return "passive"

    promoters = passives = detractors = 0
    per_spk_buckets: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"promoters": 0, "passives": 0, "detractors": 0, "n": 0}
    )
    for s in per_cue:
        bucket = _bucket(float(s.get("compound", 0.0)))
        spk = s.get("speaker", "Unknown")
        if bucket == "promoter":
            promoters += 1
            per_spk_buckets[spk]["promoters"] += 1
        elif bucket == "detractor":
            detractors += 1
            per_spk_buckets[spk]["detractors"] += 1
        else:
            passives += 1
            per_spk_buckets[spk]["passives"] += 1
        per_spk_buckets[spk]["n"] += 1

    n = len(per_cue)
    promoter_pct = round(100.0 * promoters / n, 2)
    passive_pct = round(100.0 * passives / n, 2)
    detractor_pct = round(100.0 * detractors / n, 2)
    qual_nps = round(promoter_pct - detractor_pct, 2)

    per_speaker = {}
    for spk, b in per_spk_buckets.items():
        sn = b["n"]
        p_pct = round(100.0 * b["promoters"] / sn, 2) if sn else 0.0
        d_pct = round(100.0 * b["detractors"] / sn, 2) if sn else 0.0
        per_speaker[spk] = {
            "promoter_pct": p_pct,
            "passive_pct": round(100.0 * b["passives"] / sn, 2) if sn else 0.0,
            "detractor_pct": d_pct,
            "qualitative_nps": round(p_pct - d_pct, 2),
            "n_cues": sn,
        }

    return {
        "qualitative_nps": qual_nps,
        "promoter_pct": promoter_pct,
        "passive_pct": passive_pct,
        "detractor_pct": detractor_pct,
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "n_cues": n,
        "per_speaker": per_speaker,
    }


# ── Sarcasm flagging ─────────────────────────────────────────────────────
#
# Two-tier detection:
#   1. detect_sarcasm_markers — deterministic lexical/typographical markers
#      (regex against the cue text). Fast, no LLM needed.
#   2. reconcile_sarcasm — cross-layer signal that compares VADER per-cue
#      scores against LLM sentiment output. When the lexicon and the LLM
#      disagree on the same utterance, that's a strong sarcasm signal.
#
# Both produce flags with a `signals` array and a `confidence` rating
# ("low" | "medium" | "high") so consumers can decide where to act.


# Multi-word lexical markers that strongly imply sarcasm in transcripts
_SARCASM_PHRASES = [
    "yeah right", "yeah, right",
    "oh great", "oh, great",
    "oh sure", "oh, sure",
    "sure thing",
    "yeah no", "yeah, no",
    "fine whatever", "fine, whatever",
    "oh wonderful", "oh, wonderful",
    "oh fantastic", "oh, fantastic",
    "oh perfect", "oh, perfect",
    "what a surprise",
    "as if",
    "give me a break",
    "tell me about it",
    "very funny",
    "how convenient",
    "lucky me",
    "just great",
]
_SARCASM_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _SARCASM_PHRASES) + r")\b",
    re.IGNORECASE,
)

# Single positive words that, when isolated and period-terminated, often
# signal sarcasm ("Wonderful." vs "Wonderful!")
_FLAT_POSITIVE_WORDS = {
    "wonderful", "fantastic", "amazing", "great", "perfect", "brilliant",
    "lovely", "excellent", "marvelous", "splendid",
}
_FLAT_POSITIVE_RE = re.compile(
    r"^(?:oh,?\s+|well,?\s+)?(" + "|".join(_FLAT_POSITIVE_WORDS) + r")\.+$",
    re.IGNORECASE,
)

# All-caps emphasis: 3+ ALL-CAPS letters in a single word, ignoring
# common abbreviations (we treat the word as caps-emphatic if it's not
# a known acronym pattern)
_CAPS_EMPHASIS_RE = re.compile(r"\b[A-Z]{4,}\b")

# Scare quotes: text inside double quotes within a longer utterance
_SCARE_QUOTE_RE = re.compile(r'"[^"]{2,40}"')


def detect_sarcasm_markers(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic sarcasm marker detection (Tier 1).

    Returns a list of flagged cues with the signal types found.
    Confidence is "low" for single signals and "medium" for multiple.
    """
    flags: List[Dict[str, Any]] = []
    for c in cues:
        text = (c.get("text") or "").strip()
        if not text:
            continue

        signals: List[str] = []

        if _SARCASM_PHRASE_RE.search(text):
            signals.append("lexical_marker")

        if _FLAT_POSITIVE_RE.match(text):
            signals.append("flat_positive_period")

        # Caps emphasis is signal only when the cue isn't entirely uppercase
        # (avoid flagging acronym-heavy cues) AND has a caps word
        if _CAPS_EMPHASIS_RE.search(text) and text != text.upper():
            signals.append("caps_emphasis")

        if _SCARE_QUOTE_RE.search(text):
            signals.append("scare_quotes")

        if not signals:
            continue

        confidence = "medium" if len(signals) >= 2 else "low"

        flags.append({
            "cue_index": c.get("index"),
            "speaker": _speaker_label(c.get("speaker")),
            "start": round(float(c.get("start", 0.0)), 2),
            "text": text,
            "signals": signals,
            "confidence": confidence,
        })
    return flags


def _normalize_text_for_match(s: str) -> str:
    """Normalise text for substring matching across LLM-cited quotes vs cues."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def reconcile_sarcasm(
    *,
    existing_flags: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
    vader_per_cue: List[Dict[str, Any]],
    llm_sentiment: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Tier 2 cross-layer reconciliation.

    Compares VADER per-cue compound scores against the LLM sentiment
    agent's output. When VADER reads strongly positive but the LLM has
    cited the same utterance under a negative theme or notable quote,
    that's a strong sarcasm signal.

    Boosts confidence of an existing marker-based flag, or adds a NEW
    flag for cues that markers missed.

    Args:
        existing_flags: output of detect_sarcasm_markers (may be empty)
        cues: parsed VTT cues
        vader_per_cue: VADER score per cue
        llm_sentiment: result dict from the sentiment LLM agent (or None)

    Returns:
        Updated list of sarcasm flags (de-duplicated by cue_index).
    """
    if not llm_sentiment or not isinstance(llm_sentiment, dict):
        return existing_flags

    # Build a quick cue lookup by index
    cue_by_index = {c.get("index"): c for c in cues}
    vader_by_index = {v.get("cue_index"): v for v in vader_per_cue}

    # Collect quotes the LLM cited as negative-leaning. We pull from:
    # - sentiment.themes where sentiment field == "negative"
    # - sentiment.notable_quotes where sentiment field == "negative"
    negative_quotes: List[str] = []
    for theme in (llm_sentiment.get("themes") or []):
        if not isinstance(theme, dict):
            continue
        if str(theme.get("sentiment", "")).lower() == "negative":
            desc = theme.get("description") or ""
            if desc:
                negative_quotes.append(desc)
    for nq in (llm_sentiment.get("notable_quotes") or []):
        if not isinstance(nq, dict):
            continue
        if str(nq.get("sentiment", "")).lower() == "negative":
            q = nq.get("quote") or ""
            if q:
                negative_quotes.append(q)

    if not negative_quotes:
        return existing_flags

    normalized_negatives = [_normalize_text_for_match(q) for q in negative_quotes]

    # Index existing flags so we can boost rather than duplicate
    flags_by_index: Dict[Any, Dict[str, Any]] = {
        f["cue_index"]: f for f in existing_flags if "cue_index" in f
    }

    # For each cue with strongly-positive VADER, check if its text appears
    # in any negative-cited LLM quote. If yes → sarcasm.
    _STRONG_POS_THRESHOLD = 0.5
    for cue_idx, vader in vader_by_index.items():
        compound = float(vader.get("compound", 0.0))
        if compound < _STRONG_POS_THRESHOLD:
            continue
        cue = cue_by_index.get(cue_idx)
        if not cue:
            continue
        cue_norm = _normalize_text_for_match(cue.get("text") or "")
        if len(cue_norm) < 4:
            continue
        # substring match either direction (LLM may have abbreviated the quote
        # or pulled a sentence fragment)
        matched = any(
            cue_norm in nq or nq in cue_norm
            for nq in normalized_negatives
            if nq
        )
        if not matched:
            continue

        if cue_idx in flags_by_index:
            existing = flags_by_index[cue_idx]
            if "vader_llm_disagreement" not in existing.get("signals", []):
                existing.setdefault("signals", []).append("vader_llm_disagreement")
            existing["confidence"] = "high"
            existing["vader_compound"] = compound
        else:
            flags_by_index[cue_idx] = {
                "cue_index": cue_idx,
                "speaker": _speaker_label(cue.get("speaker")),
                "start": round(float(cue.get("start", 0.0)), 2),
                "text": (cue.get("text") or "").strip(),
                "signals": ["vader_llm_disagreement"],
                "confidence": "medium",
                "vader_compound": compound,
            }

    return sorted(
        flags_by_index.values(),
        key=lambda f: (f.get("start", 0.0), f.get("cue_index") or 0),
    )

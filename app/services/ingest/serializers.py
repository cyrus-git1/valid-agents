"""Serialization helpers for inline ingest payloads."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from fastapi import HTTPException

from app.services.chunking_service import ChunkingService

_MAX_INLINE_TEXT_LENGTH = 1_000_000
_MAX_SURVEY_JSON_LENGTH = 1_000_000
_MAX_SURVEY_FIELDS = 200
_MAX_NESTED_KEYS = 100
_WEBVTT_TIMESTAMP_RE = re.compile(
    r"^(?:(?:\d{2}:)?\d{2}:\d{2}\.\d{3})\s+-->\s+(?:(?:\d{2}:)?\d{2}:\d{2}\.\d{3})"
)
_SURVEY_KEY_RE = re.compile(r"^q\d+_[a-z0-9_]+$")


def serialize_webvtt_content(content: str) -> List[Dict[str, Any]]:
    content = content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="webvtt_content cannot be empty.")
    if len(content) > _MAX_INLINE_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="webvtt_content is too large.")
    if not content.startswith("WEBVTT"):
        raise HTTPException(status_code=400, detail="webvtt_content must start with 'WEBVTT'.")

    chunks: List[Dict[str, Any]] = []
    cue_lines: List[str] = []
    cue_text_lines: List[str] = []

    def flush_cue() -> None:
        if not cue_text_lines:
            return
        text = "\n".join(cue_text_lines).strip()
        if not text:
            return
        if cue_lines and _WEBVTT_TIMESTAMP_RE.match(cue_lines[0]):
            text = f"{cue_lines[0]}\n{text}"
        chunks.append(_make_chunk(text))

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line:
            flush_cue()
            cue_lines.clear()
            cue_text_lines.clear()
            continue
        if line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "REGION")):
            continue
        if _WEBVTT_TIMESTAMP_RE.match(line) or line.isdigit():
            cue_lines.append(line)
            continue
        cue_text_lines.append(line)

    flush_cue()
    if not chunks:
        raise HTTPException(status_code=400, detail="webvtt_content did not contain any valid cues.")
    return chunks


def parse_and_serialize_survey_results(raw_json: str) -> List[Dict[str, Any]]:
    if not raw_json or not raw_json.strip():
        raise HTTPException(status_code=400, detail="survey_results_json cannot be empty.")
    if len(raw_json) > _MAX_SURVEY_JSON_LENGTH:
        raise HTTPException(status_code=400, detail="survey_results_json is too large.")
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"survey_results_json is not valid JSON: {exc}")
    _validate_survey_payload(payload)
    return _serialize_survey_payload(payload)


def _validate_survey_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="survey_results_json must be a JSON object.")
    if not payload:
        raise HTTPException(status_code=400, detail="survey_results_json cannot be empty.")
    if len(payload) > _MAX_SURVEY_FIELDS:
        raise HTTPException(status_code=400, detail="survey_results_json has too many top-level fields.")

    for key, value in payload.items():
        if key == "_heatmap":
            _validate_heatmap(value)
            continue
        if not _SURVEY_KEY_RE.match(key):
            raise HTTPException(status_code=400, detail=f"Unsupported survey question key: {key}")
        _validate_survey_value(key, value, depth=0)


def _validate_survey_value(key: str, value: Any, depth: int) -> None:
    if depth > 3:
        raise HTTPException(status_code=400, detail=f"Survey value for {key} is nested too deeply.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return
    if isinstance(value, list):
        if len(value) > _MAX_NESTED_KEYS:
            raise HTTPException(status_code=400, detail=f"Survey list for {key} is too large.")
        for item in value:
            if not isinstance(item, (str, int, float, bool)):
                raise HTTPException(status_code=400, detail=f"Survey list for {key} contains unsupported values.")
        return
    if isinstance(value, dict):
        if len(value) > _MAX_NESTED_KEYS:
            raise HTTPException(status_code=400, detail=f"Survey object for {key} is too large.")
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise HTTPException(status_code=400, detail=f"Survey object for {key} must use string keys.")
            _validate_survey_value(f"{key}.{nested_key}", nested_value, depth + 1)
        return
    raise HTTPException(status_code=400, detail=f"Survey value for {key} has an unsupported type.")


def _validate_heatmap(value: Any) -> None:
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="_heatmap must be a JSON object.")
    for question_key, entry in value.items():
        if not _SURVEY_KEY_RE.match(question_key):
            raise HTTPException(status_code=400, detail=f"Unsupported heatmap question key: {question_key}")
        if not isinstance(entry, dict):
            raise HTTPException(status_code=400, detail=f"_heatmap entry for {question_key} must be an object.")
        sessions = entry.get("sessions")
        page_url = entry.get("page_url")
        if not isinstance(sessions, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in sessions.items()):
            raise HTTPException(status_code=400, detail=f"_heatmap entry for {question_key} must contain string session IDs.")
        if not isinstance(page_url, str) or not page_url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail=f"_heatmap entry for {question_key} must include a valid page_url.")


def _serialize_survey_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunk_texts: List[str] = []
    current_lines: List[str] = []
    item_count = 0

    for key, value in payload.items():
        serialized = _format_heatmap(value) if key == "_heatmap" else _format_question(key, value)
        current_lines.append(serialized)
        item_count += 1
        if item_count >= 10:
            chunk_texts.append("\n\n".join(current_lines))
            current_lines = []
            item_count = 0

    if current_lines:
        chunk_texts.append("\n\n".join(current_lines))

    return [_make_chunk(text) for text in chunk_texts if text.strip()]


def _format_question(key: str, value: Any) -> str:
    if isinstance(value, list):
        formatted = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        formatted = "\n".join(
            f"- {nested_key}: {json.dumps(nested_value, ensure_ascii=True)}"
            for nested_key, nested_value in value.items()
        )
    else:
        formatted = str(value)
    return f"{key}:\n{formatted}" if isinstance(value, dict) else f"{key}: {formatted}"


def _format_heatmap(value: Dict[str, Any]) -> str:
    lines = ["_heatmap:"]
    for question_key, entry in value.items():
        lines.append(f"{question_key}:")
        lines.append(f"- page_url: {entry['page_url']}")
        for session_name, session_id in entry["sessions"].items():
            lines.append(f"- session {session_name}: {session_id}")
    return "\n".join(lines)


def _make_chunk(text: str) -> Dict[str, Any]:
    return {
        "text": text,
        "start_page": None,
        "end_page": None,
        "token_count": ChunkingService.llm_token_len(text),
    }

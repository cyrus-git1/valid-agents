"""
Pull KB-stored transcript chunks back into a synthetic vtt_content string
so the existing transcript services and parser can analyse them.

Used by the cross-session aggregator. Documents are matched on
metadata.survey_id + metadata.session_id (both written at ingest time).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app import core_client

logger = logging.getLogger(__name__)


def list_sessions_for_survey(
    tenant_id: str,
    client_id: str,
    survey_id: str,
) -> List[str]:
    """Return the unique session_ids tagged to this survey within tenant/client.

    Calls list_documents (cached), filters by metadata.survey_id, deduplicates
    session_ids.
    """
    try:
        data = core_client.list_documents(tenant_id=tenant_id, client_id=client_id)
    except Exception as e:
        logger.warning("list_sessions_for_survey: list_documents failed: %s", e)
        return []
    items = data.get("items", []) or []
    seen: set = set()
    out: List[str] = []
    for d in items:
        md = d.get("metadata") or {}
        if md.get("survey_id") != survey_id:
            continue
        sid = md.get("session_id")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(str(sid))
    return out


def load_session_transcript(
    tenant_id: str,
    client_id: str,
    survey_id: str,
    session_id: str,
) -> Tuple[Optional[str], List[str]]:
    """Return (vtt_content, warnings).

    Pulls all chunks from documents matching metadata.survey_id +
    metadata.session_id, joins them into a synthetic minimal-WEBVTT string
    that downstream `parse_vtt`/services accept.

    Returns (None, [...]) if no matching chunks exist.
    """
    warnings: List[str] = []
    try:
        data = core_client.list_documents(tenant_id=tenant_id, client_id=client_id)
    except Exception as e:
        return None, [f"list_documents failed: {e}"]
    items = data.get("items", []) or []

    matching_chunks: List[Dict[str, Any]] = []
    for d in items:
        md = d.get("metadata") or {}
        if md.get("survey_id") != survey_id:
            continue
        if md.get("session_id") != session_id:
            continue
        for ch in d.get("chunks") or []:
            matching_chunks.append(ch)

    if not matching_chunks:
        return None, [f"No chunks found for session_id={session_id}"]

    # Order by chunk_index where available so the reconstructed text is stable
    matching_chunks.sort(key=lambda c: c.get("chunk_index", 0))

    # Concatenate text; we don't have original cue timestamps so we
    # synthesize a single-cue WEBVTT wrapper. This is fine because the
    # downstream services treat the body as text content.
    body = "\n".join(
        (ch.get("content") or ch.get("text") or "").strip()
        for ch in matching_chunks
        if (ch.get("content") or ch.get("text") or "").strip()
    )
    if not body.strip():
        return None, [f"All chunks for session_id={session_id} were empty"]

    vtt_content = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 99:59:59.999\n"
        f"{body}\n"
    )
    return vtt_content, warnings

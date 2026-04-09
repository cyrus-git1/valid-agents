"""
Persist harness traces to Supabase — table for metadata, bucket for full JSON.

Called from harness_store.record_trace() on every harness run.
Failures are logged but never break the request path.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

BUCKET = "harness-traces"
TABLE = "harness_traces"


def persist_trace(sb: Any, trace_dict: dict[str, Any]) -> None:
    """Write trace metadata to table and full JSON to storage bucket."""
    job_id = trace_dict.get("job_id", "")
    step_name = trace_dict.get("step_name", "unknown")

    # ── Row in harness_traces table ─────────────────────────────────────
    row = {
        "job_id": job_id,
        "step_name": step_name,
        "status": trace_dict.get("status", "unknown"),
        "attempts": trace_dict.get("attempts", 0),
        "final_score": trace_dict.get("final_score"),
        "cheap_check_failures": trace_dict.get("cheap_check_failures", 0),
        "total_latency_ms": trace_dict.get("total_latency_ms"),
        "inputs_summary": _extract_inputs_summary(trace_dict),
        "genome_version": trace_dict.get("genome_version"),
    }

    try:
        sb.table(TABLE).insert(row).execute()
    except Exception as e:
        logger.warning("Failed to persist trace row for %s: %s", job_id, e)

    # ── Full JSON to bucket ─────────────────────────────────────────────
    blob = json.dumps(trace_dict, default=str).encode("utf-8")
    path = f"{step_name}/{job_id}.json"

    try:
        sb.storage.from_(BUCKET).upload(
            path, blob, file_options={"upsert": "true", "content-type": "application/json"},
        )
    except Exception as e:
        logger.warning("Failed to upload trace blob for %s: %s", job_id, e)


def load_traces(
    sb: Any,
    step_name: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Read trace metadata rows from the table."""
    try:
        res = (
            sb.table(TABLE)
            .select("*")
            .eq("step_name", step_name)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.warning("Failed to load traces for %s: %s", step_name, e)
        return []


def load_full_trace(
    sb: Any,
    step_name: str,
    job_id: str,
) -> dict[str, Any] | None:
    """Download the full trace JSON from the bucket."""
    path = f"{step_name}/{job_id}.json"
    try:
        data = sb.storage.from_(BUCKET).download(path)
        return json.loads(data)
    except Exception as e:
        logger.warning("Failed to download trace %s: %s", path, e)
        return None


def load_all_full_traces(
    sb: Any,
    step_name: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Bulk download full traces for the optimizer.

    Fetches job_ids from the table, then downloads each from the bucket.
    """
    rows = load_traces(sb, step_name, limit=limit)
    traces = []
    for row in rows:
        job_id = row.get("job_id")
        if not job_id:
            continue
        full = load_full_trace(sb, step_name, job_id)
        if full:
            traces.append(full)
    return traces


def _extract_inputs_summary(trace_dict: dict[str, Any]) -> str | None:
    """Extract a short summary from inputs_snapshot for the metadata row."""
    snapshot = trace_dict.get("inputs_snapshot", {})
    if not snapshot:
        return None
    request = snapshot.get("request", "")
    if request:
        return request[:500]
    return json.dumps(snapshot, default=str)[:500]

"""
Persist harness traces to Supabase as JSONL logs.

One bucket, one file per step per day:
  harness-traces/{step_name}/{YYYY-MM-DD}.jsonl

No table — the JSONL files are the single source of truth for the optimizer.
Redis handles fast queries for the /jobs/ API.

Called from harness_store.record_trace() on every harness run.
Failures are logged but never break the request path.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

BUCKET = "harness-traces"


def persist_trace(sb: Any, trace_dict: dict[str, Any]) -> None:
    """Append a trace to the daily JSONL file in the bucket."""
    step_name = trace_dict.get("step_name", "unknown")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = f"{step_name}/{date_str}.jsonl"
    line = json.dumps(trace_dict, default=str) + "\n"

    try:
        # Download existing content and append
        try:
            existing = sb.storage.from_(BUCKET).download(path)
            if isinstance(existing, (bytes, bytearray)):
                content = existing.decode("utf-8") + line
            else:
                content = line
        except Exception:
            # File doesn't exist yet
            content = line

        sb.storage.from_(BUCKET).upload(
            path,
            content.encode("utf-8"),
            file_options={"upsert": "true", "content-type": "application/x-ndjson"},
        )
    except Exception as e:
        logger.warning("Failed to persist trace for %s: %s", trace_dict.get("job_id", "?"), e)


def load_all_full_traces(
    sb: Any,
    step_name: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Load full traces from JSONL files in the bucket.

    Lists all JSONL files for the step, downloads and parses them,
    returns traces newest first (up to limit).
    """
    try:
        files = sb.storage.from_(BUCKET).list(step_name)
    except Exception as e:
        logger.warning("Failed to list trace files for %s: %s", step_name, e)
        return []

    if not files:
        return []

    # Sort by name desc (YYYY-MM-DD.jsonl sorts chronologically)
    jsonl_files = [f for f in files if f.get("name", "").endswith(".jsonl")]
    jsonl_files.sort(key=lambda f: f["name"], reverse=True)

    traces: list[dict[str, Any]] = []

    for file_info in jsonl_files:
        if len(traces) >= limit:
            break

        path = f"{step_name}/{file_info['name']}"
        try:
            data = sb.storage.from_(BUCKET).download(path)
            content = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)

            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    traces.append(trace)
                except json.JSONDecodeError:
                    continue

                if len(traces) >= limit:
                    break
        except Exception as e:
            logger.warning("Failed to read trace file %s: %s", path, e)

    # Newest first (traces within a file are in append order = oldest first)
    traces.reverse()
    return traces[:limit]


def load_full_trace(
    sb: Any,
    step_name: str,
    job_id: str,
) -> dict[str, Any] | None:
    """Find a specific trace by job_id across all JSONL files.

    Scans files newest-first. Not efficient for random access —
    use Redis for fast lookups, this is for the optimizer.
    """
    try:
        files = sb.storage.from_(BUCKET).list(step_name)
    except Exception:
        return None

    jsonl_files = [f for f in files if f.get("name", "").endswith(".jsonl")]
    jsonl_files.sort(key=lambda f: f["name"], reverse=True)

    for file_info in jsonl_files:
        path = f"{step_name}/{file_info['name']}"
        try:
            data = sb.storage.from_(BUCKET).download(path)
            content = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)

            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    if trace.get("job_id") == job_id:
                        return trace
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue

    return None

"""
Redis-backed chunk processing queue.

Chunks are enqueued after chunking + language filtering. A worker
processes them (embed → store) with per-chunk tracking so nothing
is lost on crash.

Each chunk job is a JSON blob in a Redis list. The worker pops jobs,
processes them, and tracks results in a Redis hash. The ingest service
can poll the hash to check progress.

Queue keys:
  chunks:{job_id}:pending    — list of chunk jobs waiting to be processed
  chunks:{job_id}:status     — hash with per-chunk status + overall progress
  chunks:{job_id}:results    — hash with chunk_id results

Usage
-----
    from app.services.chunk_queue import ChunkQueue

    queue = ChunkQueue()
    job_id = queue.enqueue_chunks(chunks, context)
    # ... later ...
    status = queue.get_job_status(job_id)
    results = queue.get_job_results(job_id)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CHUNK_TTL = 86400  # 24 hours — auto-cleanup for completed jobs


def _get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


@dataclass
class ChunkJob:
    """A single chunk to be embedded and stored."""
    chunk_index: int
    text: str
    start_page: int | None
    end_page: int | None
    token_count: int | None
    # Context needed by the worker
    tenant_id: str
    client_id: str
    document_id: str
    source_uri: str
    source_type: str
    extra_metadata: Dict[str, Any]
    embed_model: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, data: str) -> ChunkJob:
        return cls(**json.loads(data))


@dataclass
class ChunkJobStatus:
    """Overall status of a chunk processing job."""
    job_id: str
    total: int
    processed: int
    failed: int
    status: str  # "pending" | "processing" | "complete" | "partial_failure"
    chunk_ids: List[str]
    warnings: List[str]


class ChunkQueue:
    """Redis-backed queue for chunk embed+store operations."""

    def __init__(self, redis_client: redis.Redis | None = None):
        self.r = redis_client or _get_redis()

    # ── Enqueue ─────────────────────────────────────────────────────────────

    def enqueue_chunks(
        self,
        *,
        chunks: List[Dict[str, Any]],
        tenant_id: str,
        client_id: str,
        document_id: str,
        source_uri: str,
        source_type: str,
        extra_metadata: Dict[str, Any],
        embed_model: str = "text-embedding-3-small",
    ) -> str:
        """Enqueue chunks for processing. Returns a job_id for tracking.

        Chunks are persisted to Redis before returning, so even if the
        calling process dies, the chunks are safe in the queue.
        """
        job_id = str(uuid.uuid4())
        pending_key = f"chunks:{job_id}:pending"
        status_key = f"chunks:{job_id}:status"

        pipe = self.r.pipeline()

        for idx, chunk in enumerate(chunks):
            job = ChunkJob(
                chunk_index=idx,
                text=chunk["text"],
                start_page=chunk.get("start_page"),
                end_page=chunk.get("end_page"),
                token_count=chunk.get("token_count"),
                tenant_id=tenant_id,
                client_id=client_id,
                document_id=document_id,
                source_uri=source_uri,
                source_type=source_type,
                extra_metadata=extra_metadata,
                embed_model=embed_model,
            )
            pipe.rpush(pending_key, job.to_json())

        # Initialize status
        pipe.hset(status_key, mapping={
            "job_id": job_id,
            "total": str(len(chunks)),
            "processed": "0",
            "failed": "0",
            "status": "pending",
            "created_at": str(time.time()),
        })

        # TTL for auto-cleanup
        pipe.expire(pending_key, CHUNK_TTL)
        pipe.expire(status_key, CHUNK_TTL)

        pipe.execute()

        logger.info(
            "Enqueued %d chunks for job %s (doc=%s)",
            len(chunks), job_id, document_id,
        )
        return job_id

    # ── Worker ──────────────────────────────────────────────────────────────

    def process_next(
        self,
        job_id: str,
        process_fn: Any,  # Callable[[ChunkJob], str | None] — returns chunk_id or None
    ) -> bool:
        """Pop and process one chunk from the queue.

        process_fn receives a ChunkJob and should:
          - Embed the text
          - Store the chunk
          - Return the chunk_id (UUID string) on success, or raise on failure

        Returns True if a chunk was processed, False if queue is empty.
        """
        pending_key = f"chunks:{job_id}:pending"
        status_key = f"chunks:{job_id}:status"
        results_key = f"chunks:{job_id}:results"

        raw = self.r.lpop(pending_key)
        if raw is None:
            return False

        job = ChunkJob.from_json(raw)

        try:
            chunk_id = process_fn(job)

            # Track success
            self.r.hset(results_key, str(job.chunk_index), json.dumps({
                "status": "ok",
                "chunk_id": chunk_id,
            }))
            self.r.hincrby(status_key, "processed", 1)

        except Exception as e:
            logger.warning("Chunk %d failed for job %s: %s", job.chunk_index, job_id, e)

            # Track failure — push back to a dead letter field, don't lose it
            self.r.hset(results_key, str(job.chunk_index), json.dumps({
                "status": "failed",
                "error": str(e),
                "chunk_data": job.to_json(),  # preserve for retry
            }))
            self.r.hincrby(status_key, "failed", 1)

        # Check if all chunks are done
        status = self.r.hgetall(status_key)
        total = int(status.get("total", 0))
        processed = int(status.get("processed", 0))
        failed = int(status.get("failed", 0))

        if processed + failed >= total:
            final_status = "complete" if failed == 0 else "partial_failure"
            self.r.hset(status_key, "status", final_status)
            self.r.expire(results_key, CHUNK_TTL)

            logger.info(
                "Job %s finished: %d/%d processed, %d failed",
                job_id, processed, total, failed,
            )

        return True

    def process_all(
        self,
        job_id: str,
        process_fn: Any,
    ) -> ChunkJobStatus:
        """Process all chunks in a job until the queue is empty."""
        self.r.hset(f"chunks:{job_id}:status", "status", "processing")

        while self.process_next(job_id, process_fn):
            pass

        return self.get_job_status(job_id)

    # ── Status ──────────────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> ChunkJobStatus:
        """Get the current status of a chunk processing job."""
        status_key = f"chunks:{job_id}:status"
        results_key = f"chunks:{job_id}:results"

        status = self.r.hgetall(status_key)
        if not status:
            return ChunkJobStatus(
                job_id=job_id, total=0, processed=0, failed=0,
                status="not_found", chunk_ids=[], warnings=[],
            )

        # Collect successful chunk_ids and failure warnings
        results = self.r.hgetall(results_key)
        chunk_ids: List[str] = []
        warnings: List[str] = []

        for idx_str, result_json in sorted(results.items(), key=lambda x: int(x[0])):
            result = json.loads(result_json)
            if result["status"] == "ok":
                chunk_ids.append(result["chunk_id"])
            else:
                warnings.append(f"chunk {idx_str}: {result.get('error', 'unknown error')}")

        return ChunkJobStatus(
            job_id=job_id,
            total=int(status.get("total", 0)),
            processed=int(status.get("processed", 0)),
            failed=int(status.get("failed", 0)),
            status=status.get("status", "unknown"),
            chunk_ids=chunk_ids,
            warnings=warnings,
        )

    # ── Retry ───────────────────────────────────────────────────────────────

    def retry_failed(
        self,
        job_id: str,
        process_fn: Any,
    ) -> int:
        """Re-process failed chunks from a job. Returns count retried."""
        results_key = f"chunks:{job_id}:results"
        pending_key = f"chunks:{job_id}:pending"
        status_key = f"chunks:{job_id}:status"

        results = self.r.hgetall(results_key)
        retried = 0

        for idx_str, result_json in results.items():
            result = json.loads(result_json)
            if result["status"] == "failed" and result.get("chunk_data"):
                # Re-enqueue the failed chunk
                self.r.rpush(pending_key, result["chunk_data"])
                # Clear the failed result so it can be reprocessed
                self.r.hdel(results_key, idx_str)
                self.r.hincrby(status_key, "failed", -1)
                retried += 1

        if retried > 0:
            self.r.hset(status_key, "status", "processing")
            logger.info("Re-enqueued %d failed chunks for job %s", retried, job_id)

            # Process them
            while self.process_next(job_id, process_fn):
                pass

        return retried

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def cleanup_job(self, job_id: str) -> None:
        """Delete all Redis keys for a job."""
        self.r.delete(
            f"chunks:{job_id}:pending",
            f"chunks:{job_id}:status",
            f"chunks:{job_id}:results",
        )

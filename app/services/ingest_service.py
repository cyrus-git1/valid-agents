"""
src/services/ingest_service.py
-------------------------------
Orchestrates the full ingest pipeline for PDF, DOCX, and web sources.
No FastAPI / HTTP coupling — call from a router, background task, or worker.

Supported source types
----------------------
  pdf / docx  — file_bytes + file_name → upload to bucket → chunk → embed → store
  xlsx / xls  — file_bytes + file_name → upload to bucket → pandas parse → chunk → embed → store
  vtt         — file_bytes + file_name → upload to bucket → parse WebVTT → chunk → embed → store
  web         — web_url → scrape subprocess → chunk → embed → store

Import
------
    from src.services.ingest_service import IngestService, IngestInput, IngestOutput
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from supabase import Client

from app.models.api.ingest import IngestInput, IngestOutput
from app.services.chunking_service import ChunkingService, document_bytes_to_chunks, web_scraped_json_to_chunks
from app.services.scraper_service import ScraperService
from app.services.context_summary_service import ContextSummaryService
from app.services.chunk_queue import ChunkQueue, ChunkJob

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
PDF_BUCKET = "pdf"
_SUPPORTED_FILE_TYPES = {"pdf", "docx", "vtt", "xlsx", "xls"}


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class IngestService:
    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Storage ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_storage_key(name: str) -> str:
        """Sanitize a filename into a valid Supabase Storage key.

        Removes brackets, parentheses, commas, and other special chars
        that Supabase rejects.  Spaces are replaced with underscores.
        """
        stem, dot, ext = name.rpartition(".")
        if not dot:
            stem, ext = name, ""
        # Replace spaces and special chars with underscores
        stem = re.sub(r"[\s\[\]\(\),!@#$%^&+={}|;:'\"<>?]+", "_", stem)
        # Collapse multiple underscores
        stem = re.sub(r"_+", "_", stem).strip("_")
        return f"{stem}.{ext}" if ext else stem

    def upload_to_bucket(self, file_bytes: bytes, file_name: str, bucket: str = PDF_BUCKET) -> str:
        path = self._sanitize_storage_key(file_name.lstrip("/"))
        self.sb.storage.from_(bucket).upload(path, file_bytes, file_options={"upsert": "true"})
        logger.info("Uploaded %d bytes → bucket '%s' path '%s'", len(file_bytes), bucket, path)
        return path

    def download_from_storage(self, source_uri: str) -> Tuple[bytes, str, str, str]:
        """
        Download from Supabase Storage by source_uri ("bucket:pdf/file.pdf").
        Returns (file_bytes, file_type, bucket, path).
        """
        uri = source_uri.removeprefix("bucket:")
        bucket, path = uri.split("/", 1)
        file_type = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        data = self.sb.storage.from_(bucket).download(path)
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(f"Unexpected storage download type: {type(data)}")
        return bytes(data), file_type, bucket, path

    def _storage_uri(self, bucket: str, path: str) -> str:
        return f"bucket:{bucket}/{path}"

    # ── Documents ─────────────────────────────────────────────────────────────

    def _upsert_document(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        source_type: str,
        source_uri: str,
        title: Optional[str],
        metadata: JsonDict,
    ) -> UUID:
        # Check if document already exists for this tenant+client+source
        existing = (
            self.sb.table("documents")
            .select("id")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .eq("source_uri", source_uri)
            .limit(1)
            .execute()
        )
        if existing.data:
            doc_id = existing.data[0]["id"]
            self.sb.table("documents").update({
                "source_type": source_type,
                "title": title,
                "metadata": metadata or {},
            }).eq("id", doc_id).execute()
            logger.info("Updated existing document %s", doc_id)
            return UUID(doc_id)

        res = (
            self.sb.table("documents")
            .insert({
                "tenant_id": str(tenant_id),
                "client_id": str(client_id),
                "source_type": source_type,
                "source_uri": source_uri,
                "title": title,
                "metadata": metadata or {},
            })
            .execute()
        )
        if not res.data:
            raise RuntimeError("documents insert returned no rows")
        return UUID(res.data[0]["id"])

    # ── Chunks ────────────────────────────────────────────────────────────────

    def _upsert_chunk(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        chunk_index: int,
        start_page: Optional[int],
        end_page: Optional[int],
        text: str,
        token_count: Optional[int],
        metadata: JsonDict,
        embedding: Optional[List[float]],
    ) -> UUID:
        res = self.sb.rpc(
            "upsert_chunk",
            {
                "p_tenant_id": str(tenant_id),
                "p_document_id": str(document_id),
                "p_chunk_index": chunk_index,
                "p_page_start": start_page,
                "p_page_end": end_page,
                "p_content": text,
                "p_content_tokens": token_count,
                "p_metadata": metadata or {},
                "p_embedding": embedding,
            },
        ).execute()
        return UUID(str(res.data))

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_kg(self, *, tenant_id: UUID, client_id: UUID) -> JsonDict:
        res = self.sb.rpc(
            "prune_kg",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_edge_stale_days": 90,
                "p_node_stale_days": 180,
                "p_min_degree": 3,
                "p_keep_edge_evidence": 5,
                "p_keep_node_evidence": 10,
            },
        ).execute()
        return res.data or {}

    # ── Shared chunk processing ────────────────────────────────────────────────

    def _store_chunks(
        self,
        *,
        chunks: List[JsonDict],
        tenant_id: UUID,
        document_id: UUID,
        source_uri: str,
        source_type: str,
        extra_metadata: JsonDict,
        embed_model: str,
        embed_batch_size: int,
    ) -> Tuple[List[UUID], List[str]]:
        """Filter chunks for English content, then enqueue for embed+store via Redis.

        Chunks are persisted to the Redis queue before any processing starts,
        so nothing is lost if the process dies mid-way. The worker processes
        each chunk independently — one failure doesn't block the others.
        """
        warnings: List[str] = []

        if not chunks:
            return [], warnings

        # ── Language filter: strip non-English tokens, drop weak chunks ──
        filtered_chunks = []
        skipped_count = 0
        for c in chunks:
            filtered_text, should_keep = ChunkingService.filter_english_tokens(c["text"])
            if should_keep:
                c["text"] = filtered_text
                c["token_count"] = ChunkingService.llm_token_len(filtered_text)
                filtered_chunks.append(c)
            else:
                skipped_count += 1

        if skipped_count > 0:
            warnings.append(f"Skipped {skipped_count} chunk(s) with insufficient English content.")
            logger.info("Language filter: skipped %d/%d chunks", skipped_count, len(chunks))

        chunks = filtered_chunks
        if not chunks:
            warnings.append("All chunks were filtered out — document may not contain English content.")
            return [], warnings

        # ── Enqueue chunks to Redis ─────────────────────────────────────
        queue = ChunkQueue()
        job_id = queue.enqueue_chunks(
            chunks=chunks,
            tenant_id=str(tenant_id),
            client_id=str(document_id).split("-")[0],  # not used in upsert, just context
            document_id=str(document_id),
            source_uri=source_uri,
            source_type=source_type,
            extra_metadata=extra_metadata,
            embed_model=embed_model,
        )

        # ── Process all chunks from the queue ───────────────────────────
        # Each chunk is processed independently — if one fails, the rest
        # still succeed and the failed one is tracked for retry.
        def _process_chunk(job: ChunkJob) -> str:
            """Embed and store a single chunk. Returns chunk_id string."""
            # TODO: wire to core API embedding endpoint when available
            # For now, store without embedding
            chunk_id = self._upsert_chunk(
                tenant_id=UUID(job.tenant_id),
                document_id=UUID(job.document_id),
                chunk_index=job.chunk_index,
                start_page=job.start_page,
                end_page=job.end_page,
                text=job.text,
                token_count=job.token_count,
                metadata={
                    "source_uri": job.source_uri,
                    "source_type": job.source_type,
                    "chunk_start_page": job.start_page,
                    "chunk_end_page": job.end_page,
                    **job.extra_metadata,
                },
                embedding=None,  # TODO: embed via core API
            )
            return str(chunk_id)

        result = queue.process_all(job_id, _process_chunk)

        chunk_ids = [UUID(cid) for cid in result.chunk_ids]
        warnings.extend(result.warnings)

        if result.failed > 0:
            warnings.append(
                f"{result.failed}/{result.total} chunks failed. "
                f"Job ID for retry: {job_id}"
            )

        logger.info(
            "Chunk queue job %s: %d/%d processed, %d failed",
            job_id, result.processed, result.total, result.failed,
        )

        return chunk_ids, warnings

    # ── File ingest ───────────────────────────────────────────────────────────

    def _ingest_file(self, inp: IngestInput) -> IngestOutput:
        if not inp.file_bytes:
            raise ValueError("file_bytes is required for PDF/DOCX ingest")
        if not inp.file_name:
            raise ValueError("file_name is required for PDF/DOCX ingest")

        file_name = inp.file_name
        file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if file_type not in _SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type '{file_type}'. Supported: pdf, docx, vtt, xlsx.")

        storage_path = self.upload_to_bucket(inp.file_bytes, file_name)
        source_uri = self._storage_uri(PDF_BUCKET, storage_path)

        document_id = self._upsert_document(
            tenant_id=inp.tenant_id,
            client_id=inp.client_id,
            source_type=file_type,
            source_uri=source_uri,
            title=inp.title or file_name,
            metadata={
                **(inp.metadata or {}),
                "bucket": PDF_BUCKET,
                "object_path": storage_path,
                "file_type": file_type,
                "file_name": file_name,
            },
        )
        logger.info("Upserted document %s (%s)", document_id, file_name)

        try:
            chunks = document_bytes_to_chunks(inp.file_bytes, file_type=file_type)
            logger.info("Tokenized %d chunks from %s", len(chunks), file_name)
        except Exception as e:
            logger.exception("Chunking failed for %s: %s", file_name, e)
            raise RuntimeError(f"Chunking failed for {file_name}: {e}") from e

        if not chunks:
            return IngestOutput(
                document_id=document_id,
                source_type=file_type,
                source_uri=source_uri,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["Tokenizer produced no chunks — document may be empty or unreadable."],
            )

        chunk_ids, warnings = self._store_chunks(
            chunks=chunks,
            tenant_id=inp.tenant_id,
            document_id=document_id,
            source_uri=source_uri,
            source_type=file_type,
            extra_metadata={"file_name": file_name},
            embed_model=inp.embed_model,
            embed_batch_size=inp.embed_batch_size,
        )

        out = IngestOutput(
            document_id=document_id,
            source_type=file_type,
            source_uri=source_uri,
            chunks_upserted=len(chunk_ids),
            chunk_ids=chunk_ids,
            warnings=warnings,
        )
        out._chunks_data = chunks
        return out

    # ── Web ingest ────────────────────────────────────────────────────────────

    def _ingest_web(self, inp: IngestInput) -> IngestOutput:
        if not inp.web_url:
            raise ValueError("web_url is required for web ingest")

        url = inp.web_url
        source_type = "web"

        logger.info("Starting web scrape of %s", url)
        scraper = ScraperService()
        scraped = scraper.scrape(url)
        logger.info("Scraper collected %d pages from %s", scraped.total_pages, url)

        document_id = self._upsert_document(
            tenant_id=inp.tenant_id,
            client_id=inp.client_id,
            source_type=source_type,
            source_uri=url,
            title=inp.title or (scraped.pages[0].title if scraped.pages else "") or url,
            metadata={
                **(inp.metadata or {}),
                "scraped_pages": scraped.total_pages,
                "scraped_at": scraped.scraped_at,
            },
        )

        if scraped.total_pages == 0:
            return IngestOutput(
                document_id=document_id,
                source_type=source_type,
                source_uri=url,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["Scraper returned no pages — site may block crawling."],
            )

        scraped_json = scraped.to_dict()
        try:
            chunks = web_scraped_json_to_chunks(scraped_json)
            logger.info("Tokenized %d chunks from %s", len(chunks), url)
        except Exception as e:
            logger.exception("Chunking failed for web scrape %s: %s", url, e)
            raise RuntimeError(f"Chunking failed for {url}: {e}") from e

        if not chunks:
            return IngestOutput(
                document_id=document_id,
                source_type=source_type,
                source_uri=url,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["Tokenizer produced no chunks from scraped content."],
            )

        chunk_ids, warnings = self._store_chunks(
            chunks=chunks,
            tenant_id=inp.tenant_id,
            document_id=document_id,
            source_uri=url,
            source_type=source_type,
            extra_metadata={"scraped_url": url},
            embed_model=inp.embed_model,
            embed_batch_size=inp.embed_batch_size,
        )

        out = IngestOutput(
            document_id=document_id,
            source_type=source_type,
            source_uri=url,
            chunks_upserted=len(chunk_ids),
            chunk_ids=chunk_ids,
            warnings=warnings,
        )
        out._chunks_data = chunks
        return out

    # ── Entity linking ────────────────────────────────────────────────────────

    def _link_entities(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        entities: list,
        chunk_ids: List[UUID],
        chunks: List[JsonDict],
        kg_svc: "KGService",
    ) -> int:
        """Link submitted entities to chunks that mention them.

        For each entity:
          1. Upsert an Entity KG node
          2. Scan each chunk for mentions (case-insensitive)
          3. Create 'mentions' edge from chunk node to entity node
        For entity pairs that co-occur in the same chunk:
          4. Create 'co_occurs' edge between entity nodes

        Returns the total number of mention edges created.
        """
        if not entities or not chunk_ids or not chunks:
            return 0

        # ── Embed entity names ─────────────────────────────────────────
        embed_texts_list = [f"{ent.type}: {ent.name}" for ent in entities]
        try:
            entity_embeddings = embed_texts(embed_texts_list)
        except Exception as e:
            logger.warning("Entity embedding failed, proceeding without: %s", e)
            entity_embeddings = [None] * len(entities)

        # ── Upsert entity nodes ─────────────────────────────────────────
        entity_node_ids: dict[str, UUID] = {}
        for ent, embedding in zip(entities, entity_embeddings):
            entity_key = f"entity:{tenant_id}:{ent.name.lower().strip()}"
            try:
                node_id = kg_svc.upsert_node(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    node_key=entity_key,
                    type_value="Entity",
                    name=ent.name,
                    description=f"{ent.type}: {ent.name}",
                    properties={
                        "entity_type": ent.type,
                        **ent.properties,
                    },
                    embedding=embedding,
                    status="active",
                )
                entity_node_ids[ent.name.lower().strip()] = node_id
            except Exception as e:
                logger.warning("Entity upsert failed for '%s': %s", ent.name, e)

        if not entity_node_ids:
            return 0

        # ── Scan chunks for mentions ────────────────────────────────────
        total_mentions = 0

        # We need chunk node IDs — fetch them by node_key pattern
        chunk_node_ids: dict[int, UUID] = {}
        for idx, chunk_id in enumerate(chunk_ids):
            # Chunk nodes use node_key=f"chunk:{chunk_id}"
            try:
                res = (
                    self.sb.table("kg_nodes")
                    .select("id")
                    .eq("tenant_id", str(tenant_id))
                    .eq("node_key", f"chunk:{chunk_id}")
                    .limit(1)
                    .execute()
                )
                rows = res.data or []
                if rows:
                    chunk_node_ids[idx] = UUID(rows[0]["id"])
            except Exception:
                pass

        for idx, (chunk_data, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            chunk_text_lower = chunk_data.get("text", "").lower()
            chunk_node_id = chunk_node_ids.get(idx)

            # Track which entities appear in this chunk for co-occurrence
            entities_in_chunk: list[str] = []

            for entity_name_lower, entity_node_id in entity_node_ids.items():
                if entity_name_lower in chunk_text_lower:
                    entities_in_chunk.append(entity_name_lower)

                    if chunk_node_id:
                        # Create mentions edge: chunk → entity
                        try:
                            kg_svc.upsert_edge(
                                tenant_id=tenant_id,
                                client_id=client_id,
                                src_id=chunk_node_id,
                                dst_id=entity_node_id,
                                rel_type="mentions",
                                weight=1.0,
                                properties={"entity_type": next(
                                    (e.type for e in entities if e.name.lower().strip() == entity_name_lower), ""
                                )},
                            )
                            total_mentions += 1
                        except Exception as e:
                            logger.warning("Mentions edge failed: %s", e)

                        # Link evidence
                        try:
                            # Extract a short quote around the mention
                            pos = chunk_text_lower.find(entity_name_lower)
                            start = max(0, pos - 50)
                            end = min(len(chunk_data.get("text", "")), pos + len(entity_name_lower) + 50)
                            quote = chunk_data.get("text", "")[start:end].strip()

                            kg_svc.upsert_node_evidence(
                                tenant_id=tenant_id,
                                client_id=client_id,
                                node_id=entity_node_id,
                                chunk_id=chunk_id,
                                quote=quote,
                                score=1.0,
                            )
                        except Exception:
                            pass

            # ── Co-occurrence edges between entities in the same chunk ──
            if len(entities_in_chunk) > 1:
                for i_ent in range(len(entities_in_chunk)):
                    for j_ent in range(i_ent + 1, len(entities_in_chunk)):
                        a = entities_in_chunk[i_ent]
                        b = entities_in_chunk[j_ent]
                        try:
                            kg_svc.upsert_edge(
                                tenant_id=tenant_id,
                                client_id=client_id,
                                src_id=entity_node_ids[a],
                                dst_id=entity_node_ids[b],
                                rel_type="co_occurs",
                                weight=1.0,
                                properties={"source_chunk_id": str(chunk_id)},
                            )
                        except Exception:
                            pass

        logger.info(
            "Entity linking: %d entities, %d mention edges for tenant=%s",
            len(entity_node_ids), total_mentions, tenant_id,
        )
        return total_mentions

    # ── Entry point ───────────────────────────────────────────────────────────

    def ingest(self, inp: IngestInput) -> IngestOutput:
        if inp.file_bytes is not None and inp.file_name is not None:
            result = self._ingest_file(inp)
        elif inp.web_url is not None:
            result = self._ingest_web(inp)
        else:
            raise ValueError(
                "IngestInput requires either (file_bytes + file_name) or web_url."
            )

        # Build / update KG nodes + similarity edges
        # TODO: replace with core API endpoint call (KG build has moved to memory service)
        if result.chunks_upserted > 0:
            logger.info(
                "KG build skipped — KGService has moved to memory service. "
                "Wire to core API endpoint when available."
            )
            result.warnings.append("KG build pending — waiting for memory service endpoint.")

            # Entity linking requires KG service — skip until endpoint is available
            if inp.entities:
                result.warnings.append(
                    f"Entity linking skipped ({len(inp.entities)} entities) — "
                    "waiting for memory service endpoint."
                )

            # Auto-generate / update context summary
            try:
                summary_svc = ContextSummaryService(self.sb)
                summary_svc.generate_summary(
                    tenant_id=inp.tenant_id,
                    client_id=inp.client_id,
                    force_regenerate=True,
                )
                logger.info(
                    "Context summary upserted for tenant=%s client=%s",
                    inp.tenant_id, inp.client_id,
                )
            except Exception as e:
                result.warnings.append(f"Context summary generation failed: {e}")
                logger.warning("Context summary generation failed: %s", e)

        if inp.prune_after_ingest:
            try:
                result.prune_result = self._prune_kg(
                    tenant_id=inp.tenant_id,
                    client_id=inp.client_id,
                )
            except Exception as e:
                result.warnings.append(f"prune_kg failed: {e}")

        logger.info(
            "Ingest complete — document=%s chunks=%d warnings=%d",
            result.document_id, result.chunks_upserted, len(result.warnings),
        )
        return result

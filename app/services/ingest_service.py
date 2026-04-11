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

import numpy as np
from openai import OpenAI
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

    # ── Embedding ─────────────────────────────────────────────────────────────

    @staticmethod
    def _embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Embed texts via OpenAI."""
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    @staticmethod
    def _embed_in_batches(
        texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 64,
    ) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(IngestService._embed_texts(texts[i:i + batch_size], model=model))
        return out

    # ── KG build (chunk nodes + similarity edges) ────────────────────────────

    def _build_kg_for_chunks(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        chunk_ids: List[UUID],
        chunks: List[JsonDict],
        similarity_threshold: float = 0.82,
        max_edges_per_chunk: int = 10,
    ) -> JsonDict:
        """Create KG nodes for chunks and draw cosine-similarity edges.

        Calls Supabase RPCs directly (no KGService dependency).
        Returns summary dict with counts.
        """
        if not chunk_ids or not chunks:
            return {"nodes_upserted": 0, "edges_upserted": 0}

        # Fetch embeddings for stored chunks
        all_chunk_rows = []
        offset = 0
        while True:
            batch = self.sb.rpc("fetch_chunks_with_embeddings", {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_document_id": None,
                "p_limit": 500,
                "p_offset": offset,
            }).execute()
            rows = batch.data or []
            if not rows:
                break
            all_chunk_rows.extend(rows)
            if len(rows) < 500:
                break
            offset += 500

        # Filter to valid embeddings
        valid_chunks: List[JsonDict] = []
        valid_embeddings: List[List[float]] = []
        for c in all_chunk_rows:
            emb = c.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                    c["embedding"] = emb
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(emb, list) and len(emb) == 1536:
                valid_chunks.append(c)
                valid_embeddings.append(emb)

        if not valid_chunks:
            return {"nodes_upserted": 0, "edges_upserted": 0, "note": "No chunks with valid embeddings"}

        # Upsert chunk nodes
        chunk_id_to_node_id: Dict[str, UUID] = {}
        nodes_upserted = 0
        for c in valid_chunks:
            cid = c["id"]
            preview = (c.get("content") or "")[:80].strip().replace("\n", " ")
            try:
                res = self.sb.rpc("upsert_kg_node", {
                    "p_tenant_id": str(tenant_id),
                    "p_client_id": str(client_id),
                    "p_node_key": f"chunk:{cid}",
                    "p_type": "Chunk",
                    "p_name": f"Chunk {c.get('chunk_index', 0)}",
                    "p_description": preview + ("…" if len(c.get("content", "")) > 80 else ""),
                    "p_properties": {
                        "chunk_id": cid,
                        "document_id": c.get("document_id"),
                        "chunk_index": c.get("chunk_index"),
                    },
                    "p_embedding": c["embedding"],
                    "p_status": "active",
                }).execute()
                node_id = UUID(str(res.data))
                chunk_id_to_node_id[cid] = node_id
                nodes_upserted += 1

                # Node evidence
                self.sb.table("kg_node_evidence").upsert({
                    "tenant_id": str(tenant_id),
                    "client_id": str(client_id),
                    "node_id": str(node_id),
                    "chunk_id": cid,
                    "quote": (c.get("content") or "")[:200].strip() or None,
                    "score": 1.0,
                }, on_conflict="tenant_id,client_id,node_id,chunk_id").execute()
            except Exception as e:
                logger.warning("Chunk node upsert failed for %s: %s", cid, e)

        # Similarity edges
        if len(valid_chunks) < 2:
            return {"nodes_upserted": nodes_upserted, "edges_upserted": 0}

        vectors = np.array(valid_embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vectors / norms
        sim = normed @ normed.T

        edges_upserted = 0
        n = len(valid_chunks)
        for i in range(n):
            sims_i = sim[i].copy()
            sims_i[i] = -1.0
            cand_idx = np.where(sims_i >= similarity_threshold)[0]
            if cand_idx.size == 0:
                continue
            cand_sorted = cand_idx[np.argsort(sims_i[cand_idx])[::-1]][:max_edges_per_chunk]
            src_cid = valid_chunks[i]["id"]
            src_nid = chunk_id_to_node_id.get(src_cid)
            if not src_nid:
                continue
            for j in cand_sorted:
                dst_cid = valid_chunks[j]["id"]
                dst_nid = chunk_id_to_node_id.get(dst_cid)
                if not dst_nid:
                    continue
                try:
                    self.sb.rpc("upsert_kg_edge", {
                        "p_tenant_id": str(tenant_id),
                        "p_client_id": str(client_id),
                        "p_src_id": str(src_nid),
                        "p_dst_id": str(dst_nid),
                        "p_rel_type": "related_to",
                        "p_weight": float(sims_i[j]),
                        "p_properties": {"method": "chunk_embedding_cosine"},
                    }).execute()
                    edges_upserted += 1
                except Exception as e:
                    logger.warning("Similarity edge failed: %s", e)

        logger.info("KG build: %d nodes, %d edges for tenant=%s", nodes_upserted, edges_upserted, tenant_id)
        return {"nodes_upserted": nodes_upserted, "edges_upserted": edges_upserted}

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
            embedding = None
            try:
                embedding = self._embed_texts([job.text], model=embed_model)[0]
            except Exception as e:
                logger.warning("Chunk embedding failed (index %d): %s", job.chunk_index, e)

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
                embedding=embedding,
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

    # ── LLM entity extraction ──────────────────────────────────────────────────

    _NER_SYSTEM_PROMPT = (
        "You are a named entity extraction system. You will receive text chunks "
        "from a document. Extract all named entities from the text.\n\n"
        "For each entity, return:\n"
        "- name: the entity name as it appears in the text\n"
        "- type: one of: person, organization, location, product, topic, concept, event, technology\n\n"
        "Return a JSON array of objects. If no entities are found, return an empty array.\n"
        "Only return the JSON array, no other text.\n\n"
        "Example output:\n"
        '[{"name": "Acme Corp", "type": "organization"}, '
        '{"name": "John Smith", "type": "person"}, '
        '{"name": "Toronto", "type": "location"}]'
    )

    _NER_BATCH_SIZE = 10

    def _extract_entities_llm(
        self,
        chunks: List[JsonDict],
        model: str = "gpt-4o-mini",
    ) -> List[JsonDict]:
        """Extract named entities from chunks using batched LLM calls.

        Sends chunks in batches of 10 to the LLM. Returns a deduplicated
        list of entity dicts: [{"name": "...", "type": "..."}, ...]
        """
        if not chunks:
            return []

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        all_entities: List[JsonDict] = []

        for i in range(0, len(chunks), self._NER_BATCH_SIZE):
            batch = chunks[i:i + self._NER_BATCH_SIZE]

            # Format chunks for the prompt
            chunk_texts = []
            for j, c in enumerate(batch):
                text = c.get("text", "").strip()
                if text:
                    chunk_texts.append(f"[Chunk {i + j + 1}]\n{text}")

            if not chunk_texts:
                continue

            user_content = "\n\n---\n\n".join(chunk_texts)

            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self._NER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                )
                raw = resp.choices[0].message.content or "[]"

                # Parse JSON — handle markdown code blocks
                raw = raw.strip()
                if raw.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                    if match:
                        raw = match.group(1).strip()

                entities = json.loads(raw)
                if isinstance(entities, list):
                    all_entities.extend(entities)
            except json.JSONDecodeError:
                logger.warning("NER batch %d-%d returned non-JSON, skipping", i, i + len(batch))
            except Exception as e:
                logger.warning("NER batch %d-%d failed: %s", i, i + len(batch), e)

        # Deduplicate by (lowercase name, type)
        seen: set[tuple[str, str]] = set()
        unique: List[JsonDict] = []
        for ent in all_entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "").strip().lower()
            if not name or not etype:
                continue
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                unique.append({"name": name, "type": etype})

        logger.info("NER extraction: %d entities from %d chunks (%d batches)",
                    len(unique), len(chunks), (len(chunks) + self._NER_BATCH_SIZE - 1) // self._NER_BATCH_SIZE)
        return unique

    # ── Entity linking ────────────────────────────────────────────────────────

    def _link_entities(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        entities: list,
        chunk_ids: List[UUID],
        chunks: List[JsonDict],
    ) -> int:
        """Link submitted entities to chunks that mention them.

        For each entity:
          1. Upsert an Entity KG node (with embedding)
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
            entity_embeddings = self._embed_texts(embed_texts_list)
        except Exception as e:
            logger.warning("Entity embedding failed, proceeding without: %s", e)
            entity_embeddings = [None] * len(entities)

        # ── Upsert entity nodes ─────────────────────────────────────────
        entity_node_ids: dict[str, UUID] = {}
        for ent, embedding in zip(entities, entity_embeddings):
            entity_key = f"entity:{tenant_id}:{ent.name.lower().strip()}"
            try:
                res = self.sb.rpc("upsert_kg_node", {
                    "p_tenant_id": str(tenant_id),
                    "p_client_id": str(client_id),
                    "p_node_key": entity_key,
                    "p_type": "Entity",
                    "p_name": ent.name,
                    "p_description": f"{ent.type}: {ent.name}",
                    "p_properties": {"entity_type": ent.type, **ent.properties},
                    "p_embedding": embedding,
                    "p_status": "active",
                }).execute()
                entity_node_ids[ent.name.lower().strip()] = UUID(str(res.data))
            except Exception as e:
                logger.warning("Entity upsert failed for '%s': %s", ent.name, e)

        if not entity_node_ids:
            return 0

        # ── Scan chunks for mentions ────────────────────────────────────
        total_mentions = 0

        # Fetch chunk node IDs
        chunk_node_ids: dict[int, UUID] = {}
        for idx, chunk_id in enumerate(chunk_ids):
            try:
                res = (
                    self.sb.table("kg_nodes")
                    .select("id")
                    .eq("tenant_id", str(tenant_id))
                    .eq("node_key", f"chunk:{chunk_id}")
                    .limit(1)
                    .execute()
                )
                if res.data:
                    chunk_node_ids[idx] = UUID(res.data[0]["id"])
            except Exception:
                pass

        for idx, (chunk_data, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            chunk_text_lower = chunk_data.get("text", "").lower()
            chunk_node_id = chunk_node_ids.get(idx)

            entities_in_chunk: list[str] = []

            for entity_name_lower, entity_node_id in entity_node_ids.items():
                if entity_name_lower in chunk_text_lower:
                    entities_in_chunk.append(entity_name_lower)

                    if chunk_node_id:
                        # mentions edge: chunk → entity
                        try:
                            self.sb.rpc("upsert_kg_edge", {
                                "p_tenant_id": str(tenant_id),
                                "p_client_id": str(client_id),
                                "p_src_id": str(chunk_node_id),
                                "p_dst_id": str(entity_node_id),
                                "p_rel_type": "mentions",
                                "p_weight": 1.0,
                                "p_properties": {"entity_type": next(
                                    (e.type for e in entities if e.name.lower().strip() == entity_name_lower), ""
                                )},
                            }).execute()
                            total_mentions += 1
                        except Exception as e:
                            logger.warning("Mentions edge failed: %s", e)

                        # Evidence quote
                        try:
                            pos = chunk_text_lower.find(entity_name_lower)
                            start = max(0, pos - 50)
                            end = min(len(chunk_data.get("text", "")), pos + len(entity_name_lower) + 50)
                            quote = chunk_data.get("text", "")[start:end].strip()

                            self.sb.table("kg_node_evidence").upsert({
                                "tenant_id": str(tenant_id),
                                "client_id": str(client_id),
                                "node_id": str(entity_node_id),
                                "chunk_id": str(chunk_id),
                                "quote": quote,
                                "score": 1.0,
                            }, on_conflict="tenant_id,client_id,node_id,chunk_id").execute()
                        except Exception:
                            pass

            # Co-occurrence edges
            if len(entities_in_chunk) > 1:
                for i_ent in range(len(entities_in_chunk)):
                    for j_ent in range(i_ent + 1, len(entities_in_chunk)):
                        a = entities_in_chunk[i_ent]
                        b = entities_in_chunk[j_ent]
                        try:
                            self.sb.rpc("upsert_kg_edge", {
                                "p_tenant_id": str(tenant_id),
                                "p_client_id": str(client_id),
                                "p_src_id": str(entity_node_ids[a]),
                                "p_dst_id": str(entity_node_ids[b]),
                                "p_rel_type": "co_occurs",
                                "p_weight": 1.0,
                                "p_properties": {"source_chunk_id": str(chunk_id)},
                            }).execute()
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

        # Build KG chunk nodes + similarity edges
        if result.chunks_upserted > 0:
            try:
                kg_result = self._build_kg_for_chunks(
                    tenant_id=inp.tenant_id,
                    client_id=inp.client_id,
                    chunk_ids=result.chunk_ids,
                    chunks=result._chunks_data,
                )
                logger.info(
                    "KG build — nodes=%d edges=%d",
                    kg_result.get("nodes_upserted", 0),
                    kg_result.get("edges_upserted", 0),
                )
            except Exception as e:
                result.warnings.append(f"KG build failed: {e}")
                logger.warning("KG build failed: %s", e)

            # Extract entities via LLM NER + merge with submitted entities
            from app.models.api.ingest import IngestEntity

            all_entities = list(inp.entities)  # start with submitted entities

            if inp.extract_entities and result._chunks_data:
                try:
                    extracted = self._extract_entities_llm(result._chunks_data)
                    # Merge: deduplicate against submitted entities
                    existing_keys = {
                        (e.name.lower().strip(), e.type.lower().strip())
                        for e in all_entities
                    }
                    for ent in extracted:
                        key = (ent["name"].lower().strip(), ent["type"].lower().strip())
                        if key not in existing_keys:
                            existing_keys.add(key)
                            all_entities.append(IngestEntity(
                                name=ent["name"],
                                type=ent["type"],
                            ))
                    logger.info(
                        "NER: %d extracted + %d submitted = %d total entities",
                        len(extracted), len(inp.entities), len(all_entities),
                    )
                except Exception as e:
                    result.warnings.append(f"NER extraction failed: {e}")
                    logger.warning("NER extraction failed: %s", e)

            # Link all entities (submitted + extracted) to chunks
            if all_entities and result._chunks_data:
                try:
                    entities_linked = self._link_entities(
                        tenant_id=inp.tenant_id,
                        client_id=inp.client_id,
                        entities=all_entities,
                        chunk_ids=result.chunk_ids,
                        chunks=result._chunks_data,
                    )
                    result.entities_linked = entities_linked
                except Exception as e:
                    result.warnings.append(f"Entity linking failed: {e}")
                    logger.warning("Entity linking failed: %s", e)

            # Auto-generate / update context summary via context agent
            try:
                from app.agents.context_agent import run_context_agent
                ctx_result = run_context_agent(
                    tenant_id=str(inp.tenant_id),
                    client_id=str(inp.client_id),
                    client_profile=inp.metadata.get("client_profile") if inp.metadata else None,
                    force_regenerate=True,
                )
                if ctx_result.get("has_summary"):
                    logger.info(
                        "Context summary generated via agent for tenant=%s client=%s (status=%s)",
                        inp.tenant_id, inp.client_id, ctx_result.get("status"),
                    )
                else:
                    result.warnings.append(
                        f"Context summary not generated: {ctx_result.get('error', ctx_result.get('status', 'unknown'))}"
                    )
            except Exception as e:
                result.warnings.append(f"Context agent failed: {e}")
                logger.warning("Context agent failed: %s", e)

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

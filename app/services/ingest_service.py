"""
Stateless ingest service — processes files/URLs and sends results to core API.

The agent service handles:
  - File parsing (PDF, DOCX, VTT, XLSX)
  - Web scraping
  - Token-aware chunking
  - English language filtering
  - NER entity extraction (LLM)
  - Context summary generation (via context agent)

The core API (memory layer) handles:
  - File storage (bucket)
  - Document/chunk persistence
  - Embedding generation
  - KG node/edge creation
  - Entity linking

No direct Supabase access — all storage goes through core_client HTTP calls.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from openai import OpenAI

from app import core_client
from app.models.api.ingest import IngestInput, IngestOutput
from app.services.chunking_service import ChunkingService, document_bytes_to_chunks, web_scraped_json_to_chunks
from app.services.scraper_service import ScraperService

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
_SUPPORTED_FILE_TYPES = {"pdf", "docx", "vtt", "xlsx", "xls"}


class IngestService:
    """Stateless ingest — processes content locally, stores via core API."""

    def __init__(self):
        pass  # No Supabase client needed

    # ── Chunking + filtering ─────────────────────────────────────────────

    def _process_chunks(
        self,
        chunks: List[JsonDict],
    ) -> Tuple[List[JsonDict], List[str]]:
        """Filter chunks for English content. Returns (filtered_chunks, warnings)."""
        warnings: List[str] = []

        if not chunks:
            return [], warnings

        filtered = []
        skipped = 0
        for c in chunks:
            filtered_text, should_keep = ChunkingService.filter_english_tokens(c["text"])
            if should_keep:
                c["text"] = filtered_text
                c["token_count"] = ChunkingService.llm_token_len(filtered_text)
                filtered.append(c)
            else:
                skipped += 1

        if skipped > 0:
            warnings.append(f"Skipped {skipped} chunk(s) with insufficient English content.")

        if not filtered:
            warnings.append("All chunks were filtered out — document may not contain English content.")

        return filtered, warnings

    # ── NER extraction ───────────────────────────────────────────────────

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
        """Extract named entities from chunks using batched LLM calls."""
        if not chunks:
            return []

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, skipping NER")
            return []

        client = OpenAI(api_key=api_key)
        all_entities: List[JsonDict] = []

        for i in range(0, len(chunks), self._NER_BATCH_SIZE):
            batch = chunks[i:i + self._NER_BATCH_SIZE]
            chunk_texts = []
            for j, c in enumerate(batch):
                text = c.get("text", "").strip()
                if text:
                    chunk_texts.append(f"[Chunk {i + j + 1}]\n{text}")

            if not chunk_texts:
                continue

            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self._NER_SYSTEM_PROMPT},
                        {"role": "user", "content": "\n\n---\n\n".join(chunk_texts)},
                    ],
                )
                raw = resp.choices[0].message.content or "[]"
                raw = raw.strip()
                if raw.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                    if match:
                        raw = match.group(1).strip()
                entities = json.loads(raw)
                if isinstance(entities, list):
                    all_entities.extend(entities)
            except json.JSONDecodeError:
                logger.warning("NER batch %d returned non-JSON", i)
            except Exception as e:
                logger.warning("NER batch %d failed: %s", i, e)

        # Deduplicate
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

        logger.info("NER: %d unique entities from %d chunks", len(unique), len(chunks))
        return unique

    # ── File ingest ──────────────────────────────────────────────────────

    def _ingest_file(self, inp: IngestInput) -> IngestOutput:
        if not inp.file_bytes:
            raise ValueError("file_bytes is required for file ingest")
        if not inp.file_name:
            raise ValueError("file_name is required for file ingest")

        file_name = inp.file_name
        file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if file_type not in _SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type '{file_type}'. Supported: {', '.join(_SUPPORTED_FILE_TYPES)}.")

        # Chunk locally
        try:
            chunks = document_bytes_to_chunks(inp.file_bytes, file_type=file_type)
            logger.info("Chunked %d chunks from %s", len(chunks), file_name)
        except Exception as e:
            raise RuntimeError(f"Chunking failed for {file_name}: {e}") from e

        if not chunks:
            return IngestOutput(
                document_id=UUID("00000000-0000-0000-0000-000000000000"),
                source_type=file_type,
                source_uri="",
                chunks_upserted=0,
                warnings=["Chunking produced no output — document may be empty."],
            )

        # Filter + NER
        chunks, warnings = self._process_chunks(chunks)
        entities = []
        if inp.extract_entities and chunks:
            try:
                extracted = self._extract_entities_llm(chunks)
                # Merge with submitted entities
                submitted = [{"name": e.name, "type": e.type, **e.properties} for e in (inp.entities or [])]
                seen_keys = {(e["name"].lower(), e["type"]) for e in extracted}
                for s in submitted:
                    key = (s["name"].lower(), s["type"])
                    if key not in seen_keys:
                        extracted.append(s)
                        seen_keys.add(key)
                entities = extracted
            except Exception as e:
                warnings.append(f"NER extraction failed: {e}")

        # Send to core API
        try:
            result = core_client.ingest_document(
                tenant_id=str(inp.tenant_id),
                client_id=str(inp.client_id),
                file_name=file_name,
                file_bytes=inp.file_bytes,
                source_type=file_type,
                title=inp.title or file_name,
                metadata=inp.metadata or {},
                chunks=chunks,
                entities=entities,
            )
            doc_id = result.get("document_id", "00000000-0000-0000-0000-000000000000")
            chunks_stored = result.get("chunks_upserted", len(chunks))
            warnings.extend(result.get("warnings", []))
        except Exception as e:
            warnings.append(f"Core API storage failed: {e}")
            doc_id = "00000000-0000-0000-0000-000000000000"
            chunks_stored = 0

        out = IngestOutput(
            document_id=UUID(doc_id) if isinstance(doc_id, str) else doc_id,
            source_type=file_type,
            source_uri=f"file:{file_name}",
            chunks_upserted=chunks_stored,
            entities_linked=len(entities),
            warnings=warnings,
        )
        out._processed_chunks = chunks  # for context agent
        return out

    # ── Web ingest ───────────────────────────────────────────────────────

    def _ingest_web(self, inp: IngestInput) -> IngestOutput:
        if not inp.web_url:
            raise ValueError("web_url is required for web ingest")

        url = inp.web_url

        # Scrape
        scraper = ScraperService()
        scraped = scraper.scrape(url)
        logger.info("Scraped %d pages from %s", scraped.total_pages, url)

        if scraped.total_pages == 0:
            return IngestOutput(
                document_id=UUID("00000000-0000-0000-0000-000000000000"),
                source_type="web",
                source_uri=url,
                chunks_upserted=0,
                warnings=["Scraper returned no pages."],
            )

        # Chunk
        try:
            chunks = web_scraped_json_to_chunks(scraped.to_dict())
            logger.info("Chunked %d chunks from %s", len(chunks), url)
        except Exception as e:
            raise RuntimeError(f"Chunking failed for {url}: {e}") from e

        if not chunks:
            return IngestOutput(
                document_id=UUID("00000000-0000-0000-0000-000000000000"),
                source_type="web",
                source_uri=url,
                chunks_upserted=0,
                warnings=["Chunking produced no output from scraped content."],
            )

        # Filter + NER
        chunks, warnings = self._process_chunks(chunks)
        entities = []
        if inp.extract_entities and chunks:
            try:
                extracted = self._extract_entities_llm(chunks)
                submitted = [{"name": e.name, "type": e.type, **e.properties} for e in (inp.entities or [])]
                seen_keys = {(e["name"].lower(), e["type"]) for e in extracted}
                for s in submitted:
                    key = (s["name"].lower(), s["type"])
                    if key not in seen_keys:
                        extracted.append(s)
                        seen_keys.add(key)
                entities = extracted
            except Exception as e:
                warnings.append(f"NER extraction failed: {e}")

        # Send to core API
        try:
            result = core_client.ingest_web_scraped(
                tenant_id=str(inp.tenant_id),
                client_id=str(inp.client_id),
                url=url,
                title=inp.title or (scraped.pages[0].title if scraped.pages else "") or url,
                metadata={
                    **(inp.metadata or {}),
                    "scraped_pages": scraped.total_pages,
                    "scraped_at": scraped.scraped_at,
                },
                chunks=chunks,
                entities=entities,
            )
            doc_id = result.get("document_id", "00000000-0000-0000-0000-000000000000")
            chunks_stored = result.get("chunks_upserted", len(chunks))
            warnings.extend(result.get("warnings", []))
        except Exception as e:
            warnings.append(f"Core API storage failed: {e}")
            doc_id = "00000000-0000-0000-0000-000000000000"
            chunks_stored = 0

        out = IngestOutput(
            document_id=UUID(doc_id) if isinstance(doc_id, str) else doc_id,
            source_type="web",
            source_uri=url,
            chunks_upserted=chunks_stored,
            entities_linked=len(entities),
            warnings=warnings,
        )
        out._processed_chunks = chunks  # for context agent
        return out

    # ── Entry point ──────────────────────────────────────────────────────

    def ingest(self, inp: IngestInput) -> IngestOutput:
        if inp.file_bytes is not None and inp.file_name is not None:
            result = self._ingest_file(inp)
        elif inp.web_url is not None:
            result = self._ingest_web(inp)
        else:
            raise ValueError("IngestInput requires either (file_bytes + file_name) or web_url.")

        # Auto-generate context summary via context agent (skip in batch mode)
        if result.chunks_upserted > 0 and not inp.skip_context_generation:
            try:
                from app.agents.context_agent import run_context_agent
                # Pass the chunks we just processed so the context summary
                # includes new content that may not be indexed in the KB yet
                processed_chunks = getattr(result, '_processed_chunks', [])
                ctx_result = run_context_agent(
                    tenant_id=str(inp.tenant_id),
                    client_id=str(inp.client_id),
                    client_profile=inp.metadata.get("client_profile") if inp.metadata else None,
                    force_regenerate=True,
                    new_chunks=processed_chunks,
                )
                if not ctx_result.get("has_summary"):
                    result.warnings.append(
                        f"Context summary not generated: {ctx_result.get('error', ctx_result.get('status', 'unknown'))}"
                    )
            except Exception as e:
                result.warnings.append(f"Context agent failed: {e}")
                logger.warning("Context agent failed: %s", e)

        return result

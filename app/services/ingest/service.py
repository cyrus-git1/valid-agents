"""Stateless ingest service for file, web, and inline content."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple
from uuid import UUID

from openai import OpenAI

from app import core_client
from app.models.ingest import IngestInput, IngestOutput
from app.models.ingest_api import BatchIngestResponse, BatchItemStatus
from app.services.chunking_service import ChunkingService, document_bytes_to_chunks, web_scraped_json_to_chunks
from app.services.ingest.constants import SUPPORTED_FILE_TYPES
from app.services.ingest.helpers import (
    build_file_ingest_input,
    build_serialized_ingest_input,
    build_web_ingest_input,
    choose_ingest_mode,
    init_batch_item,
    inline_source_type,
    parse_entities_json,
    validate_file_upload,
)
from app.services.ingest.serializers import parse_and_serialize_survey_results, serialize_webvtt_content
from app.services.scraper_service import ScraperService

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


class IngestService:
    """Process content locally, then store it via the core API."""

    _NER_SYSTEM_PROMPT = (
        "You are a named entity extraction system. You will receive text chunks "
        "from a document. Extract all named entities from the text.\n\n"
        "For each entity, return:\n"
        "- name: the canonical, properly-cased entity name\n"
        "- type: one of: Person, Organization, Location, Product, Topic, Concept, Event, Technology, Metric\n"
        "- properties: an object with:\n"
        "  - confidence: 0.0-1.0 — how certain you are this is a real entity (not a generic word)\n"
        "  - aliases: array of alternate names/abbreviations found in the text "
        "(e.g., 'IBM' for 'International Business Machines')\n"
        "  - canonical_name: the normalized, most complete form of the name\n"
        "  - Additional context fields when available:\n"
        "    - For Person: role, organization\n"
        "    - For Organization: industry, type (startup, enterprise, agency, etc.)\n"
        "    - For Product: category, vendor\n"
        "    - For Metric: value, unit, timeframe\n\n"
        "Rules:\n"
        "- Use Title Case for type values (Person, not person)\n"
        "- Set confidence < 0.5 for ambiguous or generic terms\n"
        "- Merge duplicates: if 'Acme' and 'Acme Corp' appear, return one entity "
        "with canonical_name='Acme Corp' and aliases=['Acme']\n"
        "- Do NOT extract common English words as entities\n"
        "- Do NOT extract generic industry terms as entities unless they are specific "
        "to the company (e.g., 'SaaS' is a concept, 'Salesforce' is an Organization)\n\n"
        "Return ONLY a JSON array. If no entities are found, return [].\n\n"
        "Example output:\n"
        '[\n'
        '  {"name": "Acme Corp", "type": "Organization", "properties": {'
        '"confidence": 0.95, "aliases": ["Acme", "Acme Corporation"], '
        '"canonical_name": "Acme Corp", "industry": "SaaS"}},\n'
        '  {"name": "Jane Smith", "type": "Person", "properties": {'
        '"confidence": 0.9, "aliases": ["Jane"], '
        '"canonical_name": "Jane Smith", "role": "CEO", "organization": "Acme Corp"}},\n'
        '  {"name": "Toronto", "type": "Location", "properties": {'
        '"confidence": 0.85, "aliases": [], "canonical_name": "Toronto"}},\n'
        '  {"name": "Customer Churn Rate", "type": "Metric", "properties": {'
        '"confidence": 0.8, "aliases": ["churn rate"], '
        '"canonical_name": "Customer Churn Rate", "value": "5.2%", "timeframe": "monthly"}}\n'
        ']'
    )
    _NER_BATCH_SIZE = 10

    def _build_batch_response(self, items: List[Dict[str, Any]]) -> BatchIngestResponse:
        batch_items = [BatchItemStatus(**item) for item in items]
        total = len(batch_items)
        failed = sum(item.status == "failed" for item in batch_items)
        completed = sum(item.status == "complete" for item in batch_items)

        return BatchIngestResponse(
            total=total,
            completed=completed,
            failed=failed,
            status=self._batch_status(total=total, failed=failed),
            items=batch_items,
        )

    def _batch_status(self, *, total: int, failed: int) -> str:
        if failed == total:
            return "failed"
        if failed > 0:
            return "partial_failure"
        return "complete"

    def _run_context_regeneration(
        self,
        *,
        tenant_id: str,
        client_id: str,
        metadata: Dict[str, Any] | None = None,
        processed_chunks: List[JsonDict] | None = None,
    ) -> List[str]:
        warnings: List[str] = []
        try:
            from app.agents.context_agent import run_context_agent

            ctx_result = run_context_agent(
                tenant_id=tenant_id,
                client_id=client_id,
                client_profile=metadata.get("client_profile") if metadata else None,
                force_regenerate=True,
                new_chunks=processed_chunks or [],
            )
            if not ctx_result.get("has_summary"):
                warnings.append(
                    f"Context summary not generated: {ctx_result.get('error', ctx_result.get('status', 'unknown'))}"
                )
        except Exception as exc:
            warnings.append(f"Context agent failed: {exc}")
            logger.warning("Context agent failed: %s", exc)
        return warnings

    def ingest_file_request(
        self,
        *,
        tenant_id,
        client_id,
        file_bytes: bytes | None,
        file_name: str | None,
        content_type: str | None,
        webvtt_content: str | None,
        survey_results_json: str | None,
        title: str | None = None,
        entities_json: str | None = None,
        extract_entities: bool = True,
        prune_after_ingest: bool = False,
        skip_context_generation: bool = False,
    ) -> IngestOutput:
        parsed_entities = parse_entities_json(entities_json)
        mode = choose_ingest_mode(
            file=object() if file_bytes is not None else None,
            webvtt_content=webvtt_content,
            survey_results_json=survey_results_json,
        )

        if mode == "file":
            resolved_name = file_name or "upload.bin"
            validate_file_upload(resolved_name, content_type)
            ingest_input = build_file_ingest_input(
                tenant_id=tenant_id,
                client_id=client_id,
                file_bytes=file_bytes or b"",
                file_name=resolved_name,
                title=title,
                entities=parsed_entities,
                extract_entities=extract_entities,
                prune_after_ingest=prune_after_ingest,
                skip_context_generation=skip_context_generation,
            )
        elif mode == "webvtt":
            chunks = serialize_webvtt_content(webvtt_content or "")
            ingest_input = build_serialized_ingest_input(
                tenant_id=tenant_id,
                client_id=client_id,
                chunks=chunks,
                source_type=inline_source_type(mode),
                source_uri="inline:webvtt",
                title=title or "Inline WebVTT",
                metadata={"ingest_mode": "inline_webvtt"},
                entities=parsed_entities,
                extract_entities=extract_entities,
                prune_after_ingest=prune_after_ingest,
                skip_context_generation=skip_context_generation,
            )
        else:
            chunks = parse_and_serialize_survey_results(survey_results_json or "")
            ingest_input = build_serialized_ingest_input(
                tenant_id=tenant_id,
                client_id=client_id,
                chunks=chunks,
                source_type=inline_source_type(mode),
                source_uri="inline:survey_results",
                title=title or "Survey Results",
                metadata={"ingest_mode": "survey_results_json"},
                entities=parsed_entities,
                extract_entities=extract_entities,
                prune_after_ingest=prune_after_ingest,
                skip_context_generation=skip_context_generation,
            )

        return self.ingest(ingest_input)

    def ingest_uploaded_file(
        self,
        *,
        tenant_id,
        client_id,
        file_bytes: bytes,
        file_name: str,
        content_type: str | None = None,
        title: str | None = None,
        entities_json: str | None = None,
        extract_entities: bool = True,
        prune_after_ingest: bool = False,
        skip_context_generation: bool = False,
    ) -> IngestOutput:
        return self.ingest_file_request(
            tenant_id=tenant_id,
            client_id=client_id,
            file_bytes=file_bytes,
            file_name=file_name,
            content_type=content_type,
            webvtt_content=None,
            survey_results_json=None,
            title=title,
            entities_json=entities_json,
            extract_entities=extract_entities,
            prune_after_ingest=prune_after_ingest,
            skip_context_generation=skip_context_generation,
        )

    def ingest_web_request(
        self,
        *,
        tenant_id,
        client_id,
        url: str,
        title: str | None = None,
        metadata: Dict[str, Any] | None = None,
        entities=None,
        extract_entities: bool = True,
        prune_after_ingest: bool = False,
        skip_context_generation: bool = False,
    ) -> IngestOutput:
        return self.ingest(
            build_web_ingest_input(
                tenant_id=tenant_id,
                client_id=client_id,
                url=url,
                title=title,
                metadata=metadata,
                entities=entities,
                extract_entities=extract_entities,
                prune_after_ingest=prune_after_ingest,
                skip_context_generation=skip_context_generation,
            )
        )

    def ingest_uploaded_file_batch(
        self,
        *,
        tenant_id,
        client_id,
        files: List[Dict[str, Any]],
        prune_after_ingest: bool = False,
    ) -> BatchIngestResponse:
        items: List[Dict[str, Any]] = []
        total_chunks = 0

        for index, upload in enumerate(files):
            file_name = upload.get("file_name") or f"upload_{index}.bin"
            item = init_batch_item(index, file_name)
            try:
                result = self.ingest_uploaded_file(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    file_bytes=upload["file_bytes"],
                    file_name=file_name,
                    content_type=upload.get("content_type"),
                    prune_after_ingest=prune_after_ingest and (index == len(files) - 1),
                    skip_context_generation=True,
                )
                total_chunks += result.chunks_upserted
                item.update(
                    status="complete",
                    document_id=str(result.document_id),
                    chunks_upserted=result.chunks_upserted,
                    warnings=result.warnings,
                )
            except Exception as exc:
                item.update(status="failed", detail=str(exc))
            items.append(item)

        if total_chunks > 0:
            batch_warnings = self._run_context_regeneration(
                tenant_id=str(tenant_id),
                client_id=str(client_id),
            )
            if batch_warnings:
                for item in items:
                    if item["status"] == "complete":
                        item["warnings"].extend(batch_warnings)
        return self._build_batch_response(items)

    def ingest_web_batch(
        self,
        *,
        tenant_id,
        client_id,
        items: List[Dict[str, Any]],
        prune_after_ingest: bool = False,
    ) -> BatchIngestResponse:
        batch_items: List[Dict[str, Any]] = []
        total_chunks = 0

        for index, raw_item in enumerate(items):
            item = init_batch_item(index, raw_item["url"])
            try:
                result = self.ingest_web_request(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    url=raw_item["url"],
                    title=raw_item.get("title"),
                    metadata=raw_item.get("metadata"),
                    prune_after_ingest=prune_after_ingest and (index == len(items) - 1),
                    skip_context_generation=True,
                )
                total_chunks += result.chunks_upserted
                item.update(
                    status="complete",
                    document_id=str(result.document_id),
                    chunks_upserted=result.chunks_upserted,
                    warnings=result.warnings,
                )
            except Exception as exc:
                item.update(status="failed", detail=str(exc))
            batch_items.append(item)

        if total_chunks > 0:
            batch_warnings = self._run_context_regeneration(
                tenant_id=str(tenant_id),
                client_id=str(client_id),
            )
            if batch_warnings:
                for item in batch_items:
                    if item["status"] == "complete":
                        item["warnings"].extend(batch_warnings)
        return self._build_batch_response(batch_items)

    def _process_chunks(self, chunks: List[JsonDict]) -> Tuple[List[JsonDict], List[str]]:
        warnings: List[str] = []
        if not chunks:
            return [], warnings

        filtered: List[JsonDict] = []
        skipped = 0
        for chunk in chunks:
            filtered_text, should_keep = ChunkingService.filter_english_tokens(chunk["text"])
            if should_keep:
                chunk["text"] = filtered_text
                chunk["token_count"] = ChunkingService.llm_token_len(filtered_text)
                filtered.append(chunk)
            else:
                skipped += 1

        if skipped > 0:
            warnings.append(f"Skipped {skipped} chunk(s) with insufficient English content.")
        if not filtered:
            warnings.append("All chunks were filtered out; document may not contain English content.")
        return filtered, warnings

    def _extract_entities_llm(self, chunks: List[JsonDict], model: str = "gpt-4o-mini") -> List[JsonDict]:
        if not chunks:
            return []

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, skipping NER")
            return []

        client = OpenAI(api_key=api_key)
        all_entities: List[JsonDict] = []

        for index in range(0, len(chunks), self._NER_BATCH_SIZE):
            batch = chunks[index:index + self._NER_BATCH_SIZE]
            chunk_texts = []
            for offset, chunk in enumerate(batch):
                text = chunk.get("text", "").strip()
                if text:
                    chunk_texts.append(f"[Chunk {index + offset + 1}]\n{text}")
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
                raw = (resp.choices[0].message.content or "[]").strip()
                if raw.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                    if match:
                        raw = match.group(1).strip()
                entities = json.loads(raw)
                if isinstance(entities, list):
                    all_entities.extend(entities)
            except json.JSONDecodeError:
                logger.warning("NER batch %d returned non-JSON", index)
            except Exception as exc:
                logger.warning("NER batch %d failed: %s", index, exc)

        # Dedup by canonical_name (or name) + type, merging aliases
        merged: dict[tuple[str, str], JsonDict] = {}
        for entity in all_entities:
            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "").strip()
            if not name or not entity_type:
                continue

            props = entity.get("properties", {})
            if not isinstance(props, dict):
                props = {}

            canonical = (props.get("canonical_name") or name).strip()
            key = (canonical.lower(), entity_type.lower())

            if key in merged:
                # Merge aliases from duplicate
                existing = merged[key]
                existing_props = existing.get("properties", {})
                existing_aliases = set(existing_props.get("aliases", []))
                new_aliases = set(props.get("aliases", []))
                # Add the current name as an alias if it differs from canonical
                if name.lower() != canonical.lower():
                    new_aliases.add(name)
                existing_props["aliases"] = sorted(existing_aliases | new_aliases)
                # Keep higher confidence
                if props.get("confidence", 0) > existing_props.get("confidence", 0):
                    existing_props["confidence"] = props["confidence"]
                # Merge any extra properties (role, industry, etc.)
                for k, v in props.items():
                    if k not in ("confidence", "aliases", "canonical_name") and v and k not in existing_props:
                        existing_props[k] = v
            else:
                # Normalize: ensure canonical_name and aliases are set
                if "canonical_name" not in props:
                    props["canonical_name"] = canonical
                if "aliases" not in props:
                    props["aliases"] = []
                if name.lower() != canonical.lower() and name not in props["aliases"]:
                    props["aliases"].append(name)
                if "confidence" not in props:
                    props["confidence"] = 0.7
                merged[key] = {
                    "name": canonical,
                    "type": entity_type,
                    "properties": props,
                }

        # Filter out low-confidence entities to reduce variance
        _MIN_CONFIDENCE = 0.5
        unique = [
            e for e in merged.values()
            if e.get("properties", {}).get("confidence", 0) >= _MIN_CONFIDENCE
        ]
        filtered_count = len(merged) - len(unique)
        if filtered_count:
            logger.info("NER: filtered %d low-confidence entities (< %.1f)", filtered_count, _MIN_CONFIDENCE)
        logger.info("NER: %d entities from %d chunks", len(unique), len(chunks))
        return unique

    def _merge_entities(self, extracted: List[JsonDict], submitted_entities) -> List[JsonDict]:
        """Merge user-submitted entities with LLM-extracted ones.

        Submitted entities are converted to the rich format (with properties)
        and deduped against extracted entities by (canonical_name, type).
        """
        seen_keys = set()
        for e in extracted:
            props = e.get("properties", {})
            canonical = (props.get("canonical_name") or e.get("name", "")).lower()
            etype = e.get("type", "").lower()
            seen_keys.add((canonical, etype))

        for entity in (submitted_entities or []):
            canonical = entity.name.strip()
            etype = entity.type.strip()
            key = (canonical.lower(), etype.lower())
            if key not in seen_keys:
                extracted.append({
                    "name": canonical,
                    "type": etype,
                    "properties": {
                        "canonical_name": canonical,
                        "aliases": [],
                        "confidence": 1.0,  # user-submitted = high confidence
                        **entity.properties,
                    },
                })
                seen_keys.add(key)
        return extracted

    def _ingest_file(self, inp: IngestInput) -> IngestOutput:
        if not inp.file_bytes:
            raise ValueError("file_bytes is required for file ingest")
        if not inp.file_name:
            raise ValueError("file_name is required for file ingest")

        file_name = inp.file_name
        file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if file_type not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type '{file_type}'. Supported: {', '.join(SUPPORTED_FILE_TYPES)}.")

        try:
            chunks = document_bytes_to_chunks(inp.file_bytes, file_type=file_type)
            logger.info("Chunked %d chunks from %s", len(chunks), file_name)
        except Exception as exc:
            raise RuntimeError(f"Chunking failed for {file_name}: {exc}") from exc

        if not chunks:
            return IngestOutput(
                document_id=UUID("00000000-0000-0000-0000-000000000000"),
                source_type=file_type,
                source_uri="",
                chunks_upserted=0,
                warnings=["Chunking produced no output; document may be empty."],
            )

        chunks, warnings = self._process_chunks(chunks)
        entities: List[JsonDict] = []
        if inp.extract_entities and chunks:
            try:
                entities = self._merge_entities(self._extract_entities_llm(chunks), inp.entities)
            except Exception as exc:
                warnings.append(f"NER extraction failed: {exc}")

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
        except Exception as exc:
            warnings.append(f"Core API storage failed: {exc}")
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
        out._processed_chunks = chunks
        return out

    def _ingest_web(self, inp: IngestInput) -> IngestOutput:
        if not inp.web_url:
            raise ValueError("web_url is required for web ingest")

        url = inp.web_url
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

        try:
            chunks = web_scraped_json_to_chunks(scraped.to_dict())
            logger.info("Chunked %d chunks from %s", len(chunks), url)
        except Exception as exc:
            raise RuntimeError(f"Chunking failed for {url}: {exc}") from exc

        if not chunks:
            return IngestOutput(
                document_id=UUID("00000000-0000-0000-0000-000000000000"),
                source_type="web",
                source_uri=url,
                chunks_upserted=0,
                warnings=["Chunking produced no output from scraped content."],
            )

        chunks, warnings = self._process_chunks(chunks)
        entities: List[JsonDict] = []
        if inp.extract_entities and chunks:
            try:
                entities = self._merge_entities(self._extract_entities_llm(chunks), inp.entities)
            except Exception as exc:
                warnings.append(f"NER extraction failed: {exc}")

        try:
            result = core_client.ingest_web_scraped(
                tenant_id=str(inp.tenant_id),
                client_id=str(inp.client_id),
                url=url,
                title=inp.title or (scraped.pages[0].title if scraped.pages else "") or url,
                metadata={**(inp.metadata or {}), "scraped_pages": scraped.total_pages, "scraped_at": scraped.scraped_at},
                chunks=chunks,
                entities=entities,
            )
            doc_id = result.get("document_id", "00000000-0000-0000-0000-000000000000")
            chunks_stored = result.get("chunks_upserted", len(chunks))
            warnings.extend(result.get("warnings", []))
        except Exception as exc:
            warnings.append(f"Core API storage failed: {exc}")
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
        out._processed_chunks = chunks
        return out

    def _ingest_serialized(self, inp: IngestInput) -> IngestOutput:
        if not inp.serialized_chunks:
            raise ValueError("serialized_chunks is required for serialized ingest")
        if not inp.serialized_source_type:
            raise ValueError("serialized_source_type is required for serialized ingest")

        chunks, warnings = self._process_chunks(inp.serialized_chunks)
        entities: List[JsonDict] = []
        if inp.extract_entities and chunks:
            try:
                entities = self._merge_entities(self._extract_entities_llm(chunks), inp.entities)
            except Exception as exc:
                warnings.append(f"NER extraction failed: {exc}")

        source_uri = inp.serialized_source_uri or f"inline:{inp.serialized_source_type}"
        try:
            result = core_client.ingest_web_scraped(
                tenant_id=str(inp.tenant_id),
                client_id=str(inp.client_id),
                url=source_uri,
                title=inp.title or inp.serialized_source_type,
                metadata={**(inp.metadata or {}), "source_type": inp.serialized_source_type},
                chunks=chunks,
                entities=entities,
            )
            doc_id = result.get("document_id", "00000000-0000-0000-0000-000000000000")
            chunks_stored = result.get("chunks_upserted", len(chunks))
            warnings.extend(result.get("warnings", []))
        except Exception as exc:
            warnings.append(f"Core API storage failed: {exc}")
            doc_id = "00000000-0000-0000-0000-000000000000"
            chunks_stored = 0

        out = IngestOutput(
            document_id=UUID(doc_id) if isinstance(doc_id, str) else doc_id,
            source_type=inp.serialized_source_type,
            source_uri=source_uri,
            chunks_upserted=chunks_stored,
            entities_linked=len(entities),
            warnings=warnings,
        )
        out._processed_chunks = chunks
        return out

    def ingest(self, inp: IngestInput) -> IngestOutput:
        if inp.file_bytes is not None and inp.file_name is not None:
            result = self._ingest_file(inp)
        elif inp.web_url is not None:
            result = self._ingest_web(inp)
        elif inp.serialized_chunks:
            result = self._ingest_serialized(inp)
        else:
            raise ValueError("IngestInput requires file, web URL, or serialized chunks.")

        if result.chunks_upserted > 0 and not inp.skip_context_generation:
            result.warnings.extend(
                self._run_context_regeneration(
                    tenant_id=str(inp.tenant_id),
                    client_id=str(inp.client_id),
                    metadata=inp.metadata,
                    processed_chunks=getattr(result, "_processed_chunks", []),
                )
            )

        return result

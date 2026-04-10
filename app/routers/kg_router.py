"""
/kg router
----------
Standalone KG entity and relationship management for external NER pipelines.

POST /kg/entities       -- Batch upsert entity nodes (with auto-embedding)
POST /kg/relationships  -- Batch upsert typed relationship edges

Ordering contract: entities must be upserted before relationships,
because relationships resolve entity names to node IDs.
"""
from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter
from openai import OpenAI

from app.supabase_client import get_supabase
from app.models.api.kg import (
    EntityBatchRequest,
    EntityBatchResponse,
    EntityUpsertResult,
    RelationshipBatchRequest,
    RelationshipBatchResponse,
    RelationshipUpsertResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kg", tags=["kg"])

_EMBED_MODEL = "text-embedding-3-small"


# -- Helpers --


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts via OpenAI."""
    import os
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _embed_in_batches(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_texts(texts[i:i + batch_size]))
    return out


def _upsert_kg_node(
    sb, *, tenant_id, client_id, node_key, type_value, name,
    description=None, properties=None, embedding=None, status="active",
) -> UUID:
    res = sb.rpc("upsert_kg_node", {
        "p_tenant_id": str(tenant_id),
        "p_client_id": str(client_id) if client_id else None,
        "p_node_key": node_key,
        "p_type": type_value,
        "p_name": name,
        "p_description": description,
        "p_properties": properties or {},
        "p_embedding": embedding,
        "p_status": status,
    }).execute()
    return UUID(str(res.data))


def _upsert_kg_edge(
    sb, *, tenant_id, client_id, src_id, dst_id, rel_type,
    weight=None, properties=None,
) -> UUID:
    res = sb.rpc("upsert_kg_edge", {
        "p_tenant_id": str(tenant_id),
        "p_client_id": str(client_id) if client_id else None,
        "p_src_id": str(src_id),
        "p_dst_id": str(dst_id),
        "p_rel_type": rel_type,
        "p_weight": weight,
        "p_properties": properties or {},
    }).execute()
    return UUID(str(res.data))


def _upsert_node_evidence(
    sb, *, tenant_id, client_id, node_id, chunk_id, quote=None, score=None,
) -> None:
    sb.table("kg_node_evidence").upsert({
        "tenant_id": str(tenant_id),
        "client_id": str(client_id) if client_id else None,
        "node_id": str(node_id),
        "chunk_id": str(chunk_id),
        "quote": quote,
        "score": score,
    }, on_conflict="tenant_id,client_id,node_id,chunk_id").execute()


def _upsert_edge_evidence(
    sb, *, tenant_id, client_id, edge_id, chunk_id, quote=None, score=None,
) -> None:
    sb.table("kg_edge_evidence").upsert({
        "tenant_id": str(tenant_id),
        "client_id": str(client_id) if client_id else None,
        "edge_id": str(edge_id),
        "chunk_id": str(chunk_id),
        "quote": quote,
        "score": score,
    }, on_conflict="tenant_id,client_id,edge_id,chunk_id").execute()


def _resolve_node_id(sb, tenant_id: UUID, client_id: UUID, node_key: str) -> Optional[UUID]:
    res = (
        sb.table("kg_nodes")
        .select("id")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .eq("node_key", node_key)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    if res.data:
        return UUID(res.data[0]["id"])
    return None


# -- Endpoints --


@router.post("/entities", response_model=EntityBatchResponse)
def upsert_entities(req: EntityBatchRequest) -> EntityBatchResponse:
    """
    Batch upsert entity nodes into the KG.

    Each entity is embedded (name + type + description) and stored as a
    node with type 'Entity'. Duplicate entities (same name per tenant/client)
    are updated via upsert.
    """
    sb = get_supabase()

    # Compose embedding texts and batch embed
    embed_input = []
    for ent in req.entities:
        parts = [f"{ent.type}: {ent.name}"]
        if ent.description:
            parts.append(ent.description)
        embed_input.append(". ".join(parts))

    try:
        embeddings = _embed_in_batches(embed_input)
    except Exception as e:
        logger.warning("Entity batch embedding failed: %s", e)
        embeddings = [None] * len(req.entities)

    results: List[EntityUpsertResult] = []
    succeeded = 0

    for ent, embedding in zip(req.entities, embeddings):
        node_key = f"entity:{req.tenant_id}:{ent.name.lower().strip()}"
        try:
            node_id = _upsert_kg_node(
                sb,
                tenant_id=req.tenant_id,
                client_id=req.client_id,
                node_key=node_key,
                type_value="Entity",
                name=ent.name,
                description=ent.description or f"{ent.type}: {ent.name}",
                properties={"entity_type": ent.type, **ent.properties},
                embedding=embedding,
            )

            for chunk_id in ent.source_chunk_ids:
                try:
                    _upsert_node_evidence(
                        sb, tenant_id=req.tenant_id, client_id=req.client_id,
                        node_id=node_id, chunk_id=chunk_id, score=1.0,
                    )
                except Exception as e:
                    logger.warning("Entity evidence failed: node=%s chunk=%s: %s", node_id, chunk_id, e)

            results.append(EntityUpsertResult(
                name=ent.name, type=ent.type,
                node_id=str(node_id), node_key=node_key, status="ok",
            ))
            succeeded += 1
        except Exception as e:
            logger.warning("Entity upsert failed for '%s': %s", ent.name, e)
            results.append(EntityUpsertResult(
                name=ent.name, type=ent.type,
                node_key=node_key, status="failed", error=str(e),
            ))

    return EntityBatchResponse(
        total=len(req.entities), succeeded=succeeded,
        failed=len(req.entities) - succeeded, results=results,
    )


@router.post("/relationships", response_model=RelationshipBatchResponse)
def upsert_relationships(req: RelationshipBatchRequest) -> RelationshipBatchResponse:
    """
    Batch upsert typed relationship edges between entity nodes.

    Entities must already exist (call POST /kg/entities first).
    Source and target are resolved by entity name + tenant_id.
    """
    sb = get_supabase()

    results: List[RelationshipUpsertResult] = []
    succeeded = 0

    for rel in req.relationships:
        src_key = f"entity:{req.tenant_id}:{rel.source_entity_name.lower().strip()}"
        dst_key = f"entity:{req.tenant_id}:{rel.target_entity_name.lower().strip()}"

        try:
            src_id = _resolve_node_id(sb, req.tenant_id, req.client_id, src_key)
            if src_id is None:
                raise ValueError(f"Source entity not found: '{rel.source_entity_name}'")

            dst_id = _resolve_node_id(sb, req.tenant_id, req.client_id, dst_key)
            if dst_id is None:
                raise ValueError(f"Target entity not found: '{rel.target_entity_name}'")

            edge_id = _upsert_kg_edge(
                sb,
                tenant_id=req.tenant_id, client_id=req.client_id,
                src_id=src_id, dst_id=dst_id,
                rel_type=rel.rel_type, weight=rel.weight,
                properties=rel.properties,
            )

            for chunk_id in rel.evidence_chunk_ids:
                try:
                    _upsert_edge_evidence(
                        sb, tenant_id=req.tenant_id, client_id=req.client_id,
                        edge_id=edge_id, chunk_id=chunk_id,
                    )
                except Exception as e:
                    logger.warning("Relationship evidence failed: edge=%s chunk=%s: %s", edge_id, chunk_id, e)

            results.append(RelationshipUpsertResult(
                source_entity_name=rel.source_entity_name,
                target_entity_name=rel.target_entity_name,
                rel_type=rel.rel_type,
                edge_id=str(edge_id), status="ok",
            ))
            succeeded += 1
        except Exception as e:
            logger.warning("Relationship upsert failed '%s'->'%s': %s",
                          rel.source_entity_name, rel.target_entity_name, e)
            results.append(RelationshipUpsertResult(
                source_entity_name=rel.source_entity_name,
                target_entity_name=rel.target_entity_name,
                rel_type=rel.rel_type,
                status="failed", error=str(e),
            ))

    return RelationshipBatchResponse(
        total=len(req.relationships), succeeded=succeeded,
        failed=len(req.relationships) - succeeded, results=results,
    )

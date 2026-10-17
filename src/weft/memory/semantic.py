"""Semantic memory: entities, typed relations and provenance-tracked facts.

Model
-----
* ``(:Entity)`` — anything that can be named (people, systems, invoices...).
  ``key`` = normalised ``type|name`` is unique per tenant graph.
* ``(:Entity)-[:REL {type, confidence, evidence_count, first_seen, last_seen}]->(:Entity)``
  — direct relations (e.g. ``works_on``, ``owns``, ``part_of``).
* ``(:Entity)-[:SUBJ]->(:Fact)-[:OBJ]->(:Entity)`` — reified assertions so each
  fact keeps its own confidence, evidence counter and list of source episodes.
  Facts also carry an embedding over their natural-language ``text`` so recall
  can find them without knowing the entities involved.
"""

from __future__ import annotations

import re
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.ids import ulid
from weft.providers.embeddings import EmbeddingProvider
from weft.timeutil import now_iso

_SLUG = re.compile(r"[^a-z0-9]+")


def entity_key(name: str, type_: str = "Concept") -> str:
    n = _SLUG.sub("_", name.strip().lower()).strip("_")
    t = _SLUG.sub("_", (type_ or "Concept").strip().lower()).strip("_")
    return f"{t}|{n}"


def fact_text(subject: str, predicate: str, obj: str) -> str:
    return f"{subject} {predicate.replace('_', ' ')} {obj}"


async def upsert_entity(
    graph: Graph,
    embedder: EmbeddingProvider,
    *,
    name: str,
    type_: str = "Concept",
    description: str = "",
) -> str:
    """Create or refresh an entity; returns its stable id."""
    key = entity_key(name, type_)
    eid = ulid()
    embedding = await embedder.embed(f"{name} {description}".strip())
    res = await graph.query(
        """
        MERGE (e:Entity {key: $key})
          ON CREATE SET e.id = $id, e.name = $name, e.type = $type,
                        e.created_at = $now, e.mention_count = 0,
                        e.embedding = vecf32($embedding),
                        e.description = $description
          ON MATCH SET e.last_seen = $now,
                       e.embedding = vecf32($embedding)
        SET e:Memory, e.search_text = $search_text
        RETURN e.id AS id
        """,
        params={
            "key": key,
            "id": eid,
            "name": name,
            "type": type_,
            "description": description,
            "search_text": f"{name}\n{description}".strip()[:4000],
            "embedding": embedding,
            "now": now_iso(),
        },
    )
    return str(res.result_set[0][0])


async def link_mentions(
    graph: Graph,
    *,
    episode_id: str,
    entity_ids: list[str],
) -> None:
    if not entity_ids:
        return
    await graph.query(
        """
        MATCH (e:Episode {id: $eid}), (ent:Entity)
        WHERE ent.id IN $eids
        MERGE (e)-[:MENTIONS]->(ent)
        SET ent.mention_count = coalesce(ent.mention_count, 0) + 1
        """,
        params={"eid": episode_id, "eids": entity_ids},
    )


async def upsert_relation(
    graph: Graph,
    *,
    subject_id: str,
    object_id: str,
    rel_type: str,
    confidence: float = 0.7,
) -> None:
    """MERGE a typed relation and aggregate evidence (repeat hits raise confidence)."""
    rel = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper()) or "REL"
    now = now_iso()
    await graph.query(
        f"""
        MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
        MERGE (a)-[r:{rel}]->(b)
        ON CREATE SET r.evidence_count = 1, r.first_seen = $now, r.confidence = $conf
        ON MATCH SET r.evidence_count = coalesce(r.evidence_count, 1) + 1,
                     r.last_seen = $now,
                     r.confidence = CASE
                         WHEN coalesce(r.confidence, 0) > $conf THEN coalesce(r.confidence, 0)
                         ELSE $conf
                     END
        """,
        params={"a": subject_id, "b": object_id, "conf": confidence, "now": now},
    )


async def upsert_fact(
    graph: Graph,
    embedder: EmbeddingProvider,
    *,
    subject: str,
    predicate: str,
    obj: str,
    subject_type: str = "Concept",
    object_type: str = "Concept",
    confidence: float = 0.7,
    source_episode_id: str | None = None,
    source: str = "api",
    promote_at_evidence: int = 3,
) -> str:
    """Upsert a reified fact; evidence aggregation increases confidence over time."""
    s_key = entity_key(subject, subject_type)
    o_key = entity_key(obj, object_type)
    pred = predicate.strip().lower().replace(" ", "_")
    text = fact_text(subject, pred, obj)
    embedding = await embedder.embed(text)
    fid = ulid()
    now = now_iso()
    res = await graph.query(
        """
        MERGE (s:Entity {key: $s_key})
          ON CREATE SET s.id = $s_id, s.name = $s_name, s.type = $s_type,
                        s.created_at = $now, s.mention_count = 0,
                        s.embedding = vecf32($embedding)
        MERGE (o:Entity {key: $o_key})
          ON CREATE SET o.id = $o_id, o.name = $o_name, o.type = $o_type,
                        o.created_at = $now, o.mention_count = 0,
                        o.embedding = vecf32($embedding)
        WITH s, o
        MERGE (s)-[:SUBJ]->(f:Fact {predicate: $pred})-[:OBJ]->(o)
        ON CREATE SET f.id = $fid, f.text = $text, f.confidence = $conf,
                      f.evidence_count = 1, f.first_seen = $now, f.last_seen = $now,
                      f.valid_to = NULL, f.source = $source, f.archived = false,
                      f.embedding = vecf32($embedding)
        ON MATCH SET f.evidence_count = coalesce(f.evidence_count, 1) + 1,
                     f.last_seen = $now,
                     f.confidence = CASE
                         WHEN (CASE WHEN f.confidence > $conf THEN f.confidence ELSE $conf END)
                               + $confidence_gain > 1.0
                         THEN 1.0
                         ELSE (CASE WHEN f.confidence > $conf THEN f.confidence ELSE $conf END)
                              + $confidence_gain
                     END,
                     f.valid_to = NULL,
                     f.archived = false
        SET s:Memory, o:Memory, f:Memory,
            s.search_text = coalesce(s.search_text, $s_name),
            o.search_text = coalesce(o.search_text, $o_name),
            f.search_text = $text
        RETURN f.id AS id, f.evidence_count AS evidence
        """,
        params={
            "s_key": s_key,
            "o_key": o_key,
            "s_id": ulid(),
            "o_id": ulid(),
            "s_name": subject,
            "o_name": obj,
            "s_type": subject_type,
            "o_type": object_type,
            "pred": pred,
            "fid": fid,
            "text": text,
            "conf": confidence,
            "confidence_gain": 1.0 / max(1, promote_at_evidence),
            "embedding": embedding,
            "source": source,
            "now": now,
        },
    )
    fact_id, _evidence = res.result_set[0] if res.result_set else (fid, 1)
    if source_episode_id:
        await graph.query(
            """
            MATCH (f:Fact {id: $fid}), (e:Episode {id: $eid})
            MERGE (e)-[:ASSERTED]->(f)
            """,
            params={"fid": fact_id, "eid": source_episode_id},
        )
    return str(fact_id)


async def invalidate_fact(graph: Graph, fact_id: str) -> None:
    """Soft-delete: keep the node for provenance but mark it invalid."""
    await graph.query(
        "MATCH (f:Fact {id: $id}) SET f.valid_to = $now, f.archived = true",
        params={"id": fact_id, "now": now_iso()},
    )


async def forget(graph: Graph, *, entity_id: str | None = None, fact_id: str | None = None) -> int:
    """Hard-delete for right-to-erasure flows. Returns removed node count."""
    if fact_id:
        res = await graph.query(
            "MATCH (f:Fact {id: $id}) DELETE f RETURN 1", params={"id": fact_id}
        )
        return len(res.result_set)
    if entity_id:
        res = await graph.query(
            """
            MATCH (e:Entity {id: $id})
            OPTIONAL MATCH (e)-[r]-()
            DELETE r, e
            RETURN 1
            """,
            params={"id": entity_id},
        )
        return len(res.result_set)
    return 0


async def active_facts(
    graph: Graph,
    *,
    subject: str | None = None,
    predicate: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    where = ["(f.valid_to IS NULL OR f.valid_to = '')", "coalesce(f.archived, false) = false"]
    params: dict[str, Any] = {"limit": limit}
    if subject:
        where.append("s.name = $subject")
        params["subject"] = subject
    if predicate:
        where.append("f.predicate = $predicate")
        params["predicate"] = predicate.replace(" ", "_").lower()
    res = await graph.ro_query(
        f"""
        MATCH (s:Entity)-[:SUBJ]->(f:Fact)-[:OBJ]->(o:Entity)
        WHERE {" AND ".join(where)}
        OPTIONAL MATCH (src:Episode)-[:ASSERTED]->(f)
        RETURN f.id, s.name, s.type, f.predicate, o.name, o.type, f.text,
               f.confidence, f.evidence_count, f.first_seen, f.last_seen, f.valid_to,
               collect(DISTINCT src.id)
        ORDER BY f.last_seen DESC
        LIMIT $limit
        """,
        params=params,
    )
    return [
        {
            "id": r[0],
            "subject": r[1],
            "subject_type": r[2],
            "predicate": r[3],
            "object": r[4],
            "object_type": r[5],
            "text": r[6],
            "confidence": float(r[7] or 0),
            "evidence_count": int(r[8] or 1),
            "first_seen": r[9],
            "last_seen": r[10],
            "valid_to": r[11],
            "source_episode_ids": [x for x in (r[12] or []) if x],
        }
        for r in res.result_set
    ]


async def entity_neighbors(
    graph: Graph,
    *,
    entity_id: str,
    depth: int = 2,
    limit: int = 25,
) -> dict[str, Any]:
    """Bounded traversal used for explanation and agent context expansion."""
    depth = max(1, min(depth, 4))
    res = await graph.ro_query(
        f"""
        MATCH path = (start:Entity {{id: $id}})-[*1..{depth}]-(other:Entity)
        RETURN path
        LIMIT $limit
        """,
        params={"id": entity_id, "limit": limit},
    )
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for (path,) in res.result_set:
        for n in path.nodes:
            props = n.properties
            nodes[props.get("id", str(n.id))] = {
                "id": props.get("id"),
                "name": props.get("name"),
                "type": props.get("type"),
            }
        for e in path.edges:
            edges.append(
                {
                    "type": e.relationship,
                    "source": e.node_id,
                    "target": e.target_id,
                    "properties": e.properties,
                }
            )
    return {"nodes": list(nodes.values()), "edges": edges}


__all__ = [
    "active_facts",
    "entity_key",
    "entity_neighbors",
    "fact_text",
    "forget",
    "invalidate_fact",
    "link_mentions",
    "upsert_entity",
    "upsert_fact",
    "upsert_relation",
]

"""Episodic memory: what happened, when, and how important it still is.

Episodes are the primary unit of recall. Each carries an embedding for vector
search, an importance prior set at write time, a mutable salience that decays
half-life style, and access counters that boost re-remembered events
(learning-interference model: usage fights decay).
"""

from __future__ import annotations

import json
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.ids import ulid
from weft.providers.embeddings import EmbeddingProvider
from weft.timeutil import age_days, now_iso


async def create_episode(
    graph: Graph,
    embedder: EmbeddingProvider,
    *,
    kind: str,
    title: str,
    summary: str,
    content: str = "",
    importance: float = 0.5,
    ts: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    action_signature: str | None = None,
    outcome: str | None = None,
    extra: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
) -> str:
    eid = ulid()
    text = f"{title}\n{summary}\n{content}".strip()
    if embedding is None:
        embedding = await embedder.embed(text)
    now = now_iso()
    await graph.query(
        """
        CREATE (e:Episode:Memory {
            id: $id, kind: $kind, title: $title, summary: $summary, content: $content,
            search_text: $search_text,
            importance: $importance, salience: 1.0, ts: $ts, created_at: $now,
            session_id: $session_id, user_id: $user_id, agent_id: $agent_id,
            action_signature: $action_signature, outcome: $outcome,
            access_count: 0, last_accessed_at: $ts, archived: false,
            embedding: vecf32($embedding), extra: $extra
        })
        """,
        params={
            "id": eid,
            "kind": kind,
            "title": title[:300],
            "summary": summary[:2000],
            "content": content[:12000],
            "search_text": text[:16000],
            "importance": float(importance),
            "ts": ts or now,
            "now": now,
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "action_signature": action_signature,
            "outcome": outcome,
            "embedding": embedding,
            "extra": json.dumps(extra or {}),
        },
    )
    if session_id:
        await graph.query(
            """
            MATCH (s:Session {id: $sid}), (e:Episode {id: $eid})
            CREATE (e)-[:PART_OF]->(s)
            """,
            params={"sid": session_id, "eid": eid},
        )
        await _chain_temporal(graph, session_id, eid)
    return eid


async def _chain_temporal(graph: Graph, session_id: str, episode_id: str) -> None:
    """Link the newest episode to its predecessor for narrative order."""
    await graph.query(
        """
        MATCH (prev:Episode)-[:PART_OF]->(s:Session {id: $sid})
        WHERE prev.id <> $eid AND prev.ts <= $ts
        WITH prev ORDER BY prev.ts DESC, prev.id DESC LIMIT 1
        WITH prev
        MATCH (cur:Episode {id: $eid})
        MERGE (prev)-[:PRECEDES]->(cur)
        """,
        params={"sid": session_id, "eid": episode_id, "ts": now_iso()},
    )


async def get_episode(graph: Graph, episode_id: str) -> dict[str, Any] | None:
    res = await graph.ro_query(
        """
        MATCH (e:Episode {id: $id})
        OPTIONAL MATCH (e)-[:MENTIONS]->(ent:Entity)
        RETURN e.id, e.kind, e.title, e.summary, e.content, e.importance, e.ts,
               e.session_id, e.user_id, e.agent_id, e.salience, e.access_count,
               e.archived, collect(DISTINCT ent.name) AS entities
        """,
        params={"id": episode_id},
    )
    if not res.result_set:
        return None
    r = res.result_set[0]
    return {
        "id": r[0],
        "kind": r[1],
        "title": r[2],
        "summary": r[3],
        "content": r[4],
        "importance": float(r[5] or 0),
        "ts": r[6],
        "session_id": r[7],
        "user_id": r[8],
        "agent_id": r[9],
        "salience": float(r[10] if r[10] is not None else 1.0),
        "access_count": int(r[11] or 0),
        "archived": bool(r[12]),
        "entities": [e for e in (r[13] or []) if e],
    }


async def touch_episodes(graph: Graph, episode_ids: list[str]) -> None:
    """Boost salience and access counters for memories that were just used."""
    if not episode_ids:
        return
    await graph.query(
        """
        UNWIND $ids AS id
        MATCH (e:Episode {id: id})
        SET e.access_count = coalesce(e.access_count, 0) + 1,
            e.last_accessed_at = $now,
            e.salience = CASE
                WHEN coalesce(e.salience, 1.0) * 1.05 > 2.0 THEN 2.0
                ELSE coalesce(e.salience, 1.0) * 1.05
            END
        """,
        params={"ids": episode_ids, "now": now_iso()},
    )


async def episodes_for_session(graph: Graph, session_id: str) -> list[dict[str, Any]]:
    res = await graph.ro_query(
        """
        MATCH (e:Episode)-[:PART_OF]->(:Session {id: $sid})
        RETURN e.id, e.title, e.summary, e.kind, e.importance, e.ts
        ORDER BY e.ts ASC
        """,
        params={"sid": session_id},
    )
    return [
        {
            "id": r[0],
            "title": r[1],
            "summary": r[2],
            "kind": r[3],
            "importance": float(r[4] or 0),
            "ts": r[5],
        }
        for r in res.result_set
    ]


async def apply_decay_and_archive(
    graph: Graph,
    *,
    half_life_days: float,
    archive_after_days: float,
    archive_salience: float,
) -> dict[str, int]:
    """Decay salience toward zero and archive stale low-salience episodes.

    FalkorDB has no date arithmetic, so ages are computed from the stored
    RFC-3339 timestamps and the resulting salience/archival decisions are
    written back in one batched Cypher pass. Archived episodes keep their
    nodes and edges (provenance stays intact) but are excluded from recall.
    """
    now = now_iso()
    res = await graph.query(
        """
        MATCH (e:Episode)
        WHERE e.archived = false OR e.archived IS NULL
        RETURN e.id, e.ts, e.salience
        """,
    )
    to_update: list[dict[str, Any]] = []
    archived = 0
    for eid, ts, salience in res.result_set:
        age = age_days(ts or now)
        s = float(salience if salience is not None else 1.0) * 0.5 ** (
            age / max(half_life_days, 0.001)
        )
        s = max(0.0, s)
        entry: dict[str, Any] = {"id": eid, "salience": s}
        if age >= archive_after_days and s < archive_salience:
            entry["archived"] = True
            archived += 1
        to_update.append(entry)
    if to_update:
        await graph.query(
            """
            UNWIND $rows AS row
            MATCH (e:Episode {id: row.id})
            SET e.salience = row.salience
            WITH e, row
            WHERE row.archived = true
            SET e.archived = true, e.archived_at = $now
            """,
            params={"rows": to_update, "now": now},
        )
    return {"evaluated": len(to_update), "archived": archived}


__all__ = [
    "apply_decay_and_archive",
    "create_episode",
    "episodes_for_session",
    "get_episode",
    "touch_episodes",
]

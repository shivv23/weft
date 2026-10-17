"""Persistent conversation history.

Messages are written in one round trip (``UNWIND`` over the batch), sessions
track a monotonic sequence counter, and every ingest enqueues an idempotent
``session.consolidate`` job so background extraction is exactly-once per
message batch even under concurrent writers.
"""

from __future__ import annotations

import json
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.errors import NotFoundError, ValidationError
from weft.ids import ulid
from weft.processing import queue
from weft.timeutil import now_iso


async def create_session(
    graph: Graph,
    *,
    user_id: str,
    agent_id: str | None = None,
    title: str | None = None,
    meta: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    if not user_id:
        raise ValidationError("user_id is required")
    sid = session_id or ulid()
    now = now_iso()
    await graph.query(
        """
        MERGE (u:User {id: $user_id})
          ON CREATE SET u.created_at = $now
        WITH u
        CREATE (s:Session {id: $sid, status: 'open', started_at: $now,
                           last_activity_at: $now, message_count: 0,
                           consolidated_seq: 0})
        SET s.user_id = $user_id, s.agent_id = $agent_id,
            s.title = $title, s.meta = $meta
        CREATE (u)-[:HAS_SESSION]->(s)
        FOREACH (_ IN CASE WHEN $agent_id IS NULL OR $agent_id = '' THEN [] ELSE [1] END |
            MERGE (a:Agent {id: $agent_id})
              ON CREATE SET a.name = $agent_id, a.created_at = $now, a.kind = 'worker',
                            a.description = '', a.capabilities = '[]'
            CREATE (a)-[:RUNS]->(s))
        RETURN s.id AS id
        """,
        params={
            "sid": sid,
            "user_id": user_id,
            "agent_id": agent_id,
            "title": title,
            "meta": json.dumps(meta or {}),
            "now": now,
        },
    )
    session = await get_session(graph, sid)
    if session is None:
        raise NotFoundError(f"session {sid} could not be created")
    return session


async def get_session(graph: Graph, session_id: str) -> dict[str, Any] | None:
    res = await graph.ro_query(
        """
        MATCH (s:Session {id: $id})
        RETURN s.id, s.user_id, s.agent_id, s.title, s.status, s.started_at,
               s.last_activity_at, s.message_count, s.meta
        """,
        params={"id": session_id},
    )
    if not res.result_set:
        return None
    row = res.result_set[0]
    return {
        "id": row[0],
        "user_id": row[1],
        "agent_id": row[2],
        "title": row[3],
        "status": row[4],
        "started_at": row[5],
        "last_activity_at": row[6],
        "message_count": int(row[7] or 0),
        "meta": _loads(row[8]),
    }


async def close_session(graph: Graph, session_id: str) -> dict[str, Any]:
    await graph.query(
        "MATCH (s:Session {id: $id}) SET s.status = 'closed', s.ended_at = $now",
        params={"id": session_id, "now": now_iso()},
    )
    session = await get_session(graph, session_id)
    if session is None:
        raise NotFoundError(f"session {session_id} not found")
    return session


async def append_messages(
    graph: Graph,
    session_id: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append a batch of messages, bump counters, enqueue consolidation.

    Single Cypher statement => single atomic write, which keeps concurrent
    writers from interleaving sequence numbers.
    """
    if not messages:
        return []
    session = await get_session(graph, session_id)
    if session is None:
        raise NotFoundError(f"session {session_id} not found")

    rows = []
    for m in messages:
        content = m.get("content")
        if not content or not str(content).strip():
            raise ValidationError("message content must be non-empty")
        rows.append(
            {
                "id": m.get("id") or ulid(),
                "role": m.get("role", "user"),
                "content": str(content),
                "meta": json.dumps(m.get("meta") or {}),
                "ts": m.get("ts") or now_iso(),
            }
        )

    await graph.query(
        """
        MATCH (s:Session {id: $sid})
        WITH s
        UNWIND range(0, size($rows) - 1) AS i
        WITH s, i, $rows[i] AS row
        MERGE (msg:Message {id: row.id})
          ON CREATE SET msg.seq = s.message_count + i + 1, msg.created = true
        WITH s, msg, row
        WHERE msg.created = true
        SET msg.role = row.role, msg.content = row.content,
            msg.meta = row.meta, msg.ts = row.ts, msg.consolidated = false
        REMOVE msg.created
        CREATE (s)-[:HAS_MESSAGE]->(msg)
        WITH s, count(msg) AS added
        SET s.message_count = s.message_count + added,
            s.last_activity_at = $now
        """,
        params={"sid": session_id, "rows": rows, "now": now_iso()},
    )

    await queue.enqueue(
        graph,
        "session.consolidate",
        {"session_id": session_id},
        dedupe_key=f"consolidate:{session_id}",
    )
    return await list_messages(graph, session_id, limit=len(rows), ascending=False)


async def list_messages(
    graph: Graph,
    session_id: str,
    *,
    limit: int = 100,
    before_seq: int | None = None,
    ascending: bool = True,
) -> list[dict[str, Any]]:
    order = "ASC" if ascending else "DESC"
    params: dict[str, Any] = {"sid": session_id, "limit": limit}
    before = ""
    if before_seq is not None:
        before = "AND msg.seq < $before_seq"
        params["before_seq"] = before_seq
    res = await graph.ro_query(
        f"""
        MATCH (s:Session {{id: $sid}})-[:HAS_MESSAGE]->(msg:Message)
        WHERE true {before}
        RETURN msg.id, msg.seq, msg.role, msg.content, msg.ts, msg.meta
        ORDER BY msg.seq {order}
        LIMIT $limit
        """,
        params=params,
    )
    out = [
        {
            "id": r[0],
            "seq": int(r[1]),
            "role": r[2],
            "content": r[3],
            "ts": r[4],
            "meta": _loads(r[5]),
        }
        for r in res.result_set
    ]
    if not ascending:
        out.reverse()
    return out


async def unconsolidated_messages(
    graph: Graph, session_id: str, *, limit: int = 200
) -> list[dict[str, Any]]:
    res = await graph.ro_query(
        """
        MATCH (s:Session {id: $sid})-[:HAS_MESSAGE]->(msg:Message)
        WHERE msg.consolidated IS NULL OR msg.consolidated = false
        RETURN msg.id, msg.seq, msg.role, msg.content, msg.ts, msg.meta
        ORDER BY msg.seq ASC
        LIMIT $limit
        """,
        params={"sid": session_id, "limit": limit},
    )
    return [
        {
            "id": r[0],
            "seq": int(r[1]),
            "role": r[2],
            "content": r[3],
            "ts": r[4],
            "meta": _loads(r[5]),
        }
        for r in res.result_set
    ]


async def mark_consolidated(graph: Graph, message_ids: list[str]) -> None:
    if not message_ids:
        return
    await graph.query(
        "UNWIND $ids AS id MATCH (m:Message {id: id}) SET m.consolidated = true",
        params={"ids": message_ids},
    )


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


__all__ = [
    "append_messages",
    "close_session",
    "create_session",
    "get_session",
    "list_messages",
    "mark_consolidated",
    "unconsolidated_messages",
]

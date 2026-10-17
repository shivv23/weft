"""Agent-to-agent handoffs.

A handoff is a durable work item plus a context packet: free-form JSON
payload, priority, session/user linkage and typed ``ref_ids`` pointing at
episodes/facts/procedures the receiving agent should load first.

Claiming is a compare-and-set, so two racing agents can never both win the
same handoff. Lifecycle: ``pending → claimed → completed|rejected``.
"""

from __future__ import annotations

import json
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.errors import ClaimError, NotFoundError
from weft.ids import ulid
from weft.timeutil import now_iso


async def create_handoff(
    graph: Graph,
    *,
    from_agent: str,
    to_agent: str,
    title: str = "",
    reason: str = "",
    context: dict[str, Any] | None = None,
    priority: int = 5,
    session_id: str | None = None,
    user_id: str | None = None,
    ref_ids: list[str] | None = None,
) -> dict[str, Any]:
    if from_agent == to_agent:
        raise ClaimError("an agent cannot hand off to itself")
    hid = ulid()
    now = now_iso()
    await graph.query(
        """
        MERGE (a:Agent {id: $from})
          ON CREATE SET a.name = $from, a.created_at = $now, a.kind = 'worker',
                        a.description = '', a.capabilities = '[]'
        MERGE (b:Agent {id: $to})
          ON CREATE SET b.name = $to, b.created_at = $now, b.kind = 'worker',
                        b.description = '', b.capabilities = '[]'
        CREATE (h:Handoff {
            id: $id, title: $title, reason: $reason, context: $context,
            priority: $priority, status: 'pending', created_at: $now,
            session_id: $session_id, user_id: $user_id, from_agent: $from,
            to_agent: $to
        })
        CREATE (a)-[:SENT]->(h)-[:TO]->(b)
        RETURN h.id AS id
        """,
        params={
            "id": hid,
            "from": from_agent,
            "to": to_agent,
            "title": title[:300],
            "reason": reason[:2000],
            "context": json.dumps(context or {}, ensure_ascii=False),
            "priority": priority,
            "session_id": session_id,
            "user_id": user_id,
            "now": now,
        },
    )
    if session_id:
        await graph.query(
            """
            MATCH (h:Handoff {id: $hid}), (s:Session {id: $sid})
            CREATE (h)-[:DURING]->(s)
            """,
            params={"hid": hid, "sid": session_id},
        )
    for ref in ref_ids or []:
        await graph.query(
            """
            MATCH (h:Handoff {id: $hid})
            MATCH (n) WHERE n.id = $ref
            MERGE (h)-[:REFERENCES]->(n)
            """,
            params={"hid": hid, "ref": ref},
        )
    handoff = await get_handoff(graph, hid)
    assert handoff is not None
    return handoff


async def get_handoff(graph: Graph, handoff_id: str) -> dict[str, Any] | None:
    res = await graph.ro_query(
        """
        MATCH (h:Handoff {id: $id})
        OPTIONAL MATCH (h)-[:REFERENCES]->(ref)
        RETURN h.id, h.from_agent, h.to_agent, h.title, h.reason, h.context,
               h.priority, h.status, h.session_id, h.user_id, h.created_at,
               h.claimed_at, h.completed_at, h.result,
               collect(DISTINCT ref.id)
        """,
        params={"id": handoff_id},
    )
    return _row(res.result_set[0]) if res.result_set else None


async def inbox(
    graph: Graph,
    *,
    agent_id: str,
    status: str | None = "pending",
    limit: int = 20,
) -> list[dict[str, Any]]:
    where = ["h.to_agent = $agent"]
    params: dict[str, Any] = {"agent": agent_id, "limit": limit}
    if status:
        where.append("h.status = $status")
        params["status"] = status
    res = await graph.ro_query(
        f"""
        MATCH (:Agent {{id: $agent}})<-[:TO]-(h:Handoff)
        WHERE {" AND ".join(where)}
        OPTIONAL MATCH (h)-[:REFERENCES]->(ref)
        RETURN h.id, h.from_agent, h.to_agent, h.title, h.reason, h.context,
               h.priority, h.status, h.session_id, h.user_id, h.created_at,
               h.claimed_at, h.completed_at, h.result, collect(DISTINCT ref.id)
        ORDER BY h.priority ASC, h.created_at ASC
        LIMIT $limit
        """,
        params=params,
    )
    return [_row(r) for r in res.result_set]


async def claim(graph: Graph, *, handoff_id: str, agent_id: str) -> dict[str, Any]:
    now = now_iso()
    res = await graph.query(
        """
        MATCH (h:Handoff {id: $id, status: 'pending'})
        SET h.status = 'claimed', h.claimed_by = $agent, h.claimed_at = $now
        RETURN h.id
        """,
        params={"id": handoff_id, "agent": agent_id, "now": now},
    )
    if not res.result_set:
        current = await get_handoff(graph, handoff_id)
        if current is None:
            raise NotFoundError(f"handoff {handoff_id} not found")
        raise ClaimError(
            f"handoff {handoff_id} is not claimable (status={current['status']})",
            details={"status": current["status"]},
        )
    await graph.query(
        """
        MATCH (h:Handoff {id: $id}), (a:Agent {id: $agent})
        MERGE (a)-[:CLAIMED]->(h)
        """,
        params={"id": handoff_id, "agent": agent_id},
    )
    handoff = await get_handoff(graph, handoff_id)
    assert handoff is not None
    return handoff


async def complete(
    graph: Graph,
    *,
    handoff_id: str,
    agent_id: str,
    result: Any = None,
) -> dict[str, Any]:
    now = now_iso()
    res = await graph.query(
        """
        MATCH (h:Handoff {id: $id, status: 'claimed', claimed_by: $agent})
        SET h.status = 'completed', h.completed_at = $now, h.result = $result
        RETURN h.id
        """,
        params={
            "id": handoff_id,
            "agent": agent_id,
            "now": now,
            "result": json.dumps(result, ensure_ascii=False) if result is not None else None,
        },
    )
    if not res.result_set:
        current = await get_handoff(graph, handoff_id)
        if current is None:
            raise NotFoundError(f"handoff {handoff_id} not found")
        raise ClaimError(
            f"handoff {handoff_id} cannot be completed by {agent_id} (status={current['status']})",
            details={"status": current["status"]},
        )
    handoff = await get_handoff(graph, handoff_id)
    assert handoff is not None
    return handoff


async def reject(
    graph: Graph, *, handoff_id: str, agent_id: str, reason: str = ""
) -> dict[str, Any]:
    now = now_iso()
    res = await graph.query(
        """
        MATCH (h:Handoff {id: $id, status: 'pending'})
        SET h.status = 'rejected', h.completed_at = $now, h.result = $reason
        RETURN h.id
        """,
        params={"id": handoff_id, "agent": agent_id, "now": now, "result": reason},
    )
    if not res.result_set:
        raise ClaimError(f"handoff {handoff_id} cannot be rejected")
    handoff = await get_handoff(graph, handoff_id)
    assert handoff is not None
    return handoff


def _row(r: Any) -> dict[str, Any]:
    context = r[5]
    if isinstance(context, str) and context:
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            context = {"raw": context}
    result = r[13]
    if isinstance(result, str) and result:
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass
    return {
        "id": r[0],
        "from_agent": r[1],
        "to_agent": r[2],
        "title": r[3],
        "reason": r[4],
        "context": context or {},
        "priority": int(r[6] or 5),
        "status": r[7],
        "session_id": r[8],
        "user_id": r[9],
        "created_at": r[10],
        "claimed_at": r[11],
        "completed_at": r[12],
        "result": result,
        "ref_ids": [x for x in (r[14] or []) if x],
    }


__all__ = ["claim", "complete", "create_handoff", "get_handoff", "inbox", "reject"]

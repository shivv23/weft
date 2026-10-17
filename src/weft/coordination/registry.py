"""Agent registry: lightweight directory used by handoffs and the API."""

from __future__ import annotations

import json
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.timeutil import now_iso


async def register_agent(
    graph: Graph,
    *,
    agent_id: str,
    name: str,
    kind: str = "worker",
    description: str = "",
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    now = now_iso()
    await graph.query(
        """
        MERGE (a:Agent {id: $id})
          ON CREATE SET a.created_at = $now
        SET a.name = $name, a.kind = $kind, a.description = $description,
            a.capabilities = $capabilities
        """,
        params={
            "id": agent_id,
            "name": name,
            "kind": kind,
            "description": description,
            "capabilities": json.dumps(capabilities or []),
            "now": now,
        },
    )
    agent = await get_agent(graph, agent_id)
    assert agent is not None
    return agent


async def get_agent(graph: Graph, agent_id: str) -> dict[str, Any] | None:
    res = await graph.ro_query(
        """
        MATCH (a:Agent {id: $id})
        OPTIONAL MATCH (a)-[:SENT]->(out:Handoff)
        OPTIONAL MATCH (in:Handoff)-[:TO]->(a)
        RETURN a.id, a.name, a.kind, a.description, a.capabilities, a.created_at,
               count(DISTINCT out), count(DISTINCT in)
        """,
        params={"id": agent_id},
    )
    return _row(res.result_set[0]) if res.result_set else None


async def list_agents(graph: Graph, *, limit: int = 100) -> list[dict[str, Any]]:
    res = await graph.ro_query(
        """
        MATCH (a:Agent)
        OPTIONAL MATCH (a)-[:SENT]->(out:Handoff)
        OPTIONAL MATCH (in:Handoff)-[:TO]->(a)
        RETURN a.id, a.name, a.kind, a.description, a.capabilities, a.created_at,
               count(DISTINCT out), count(DISTINCT in)
        ORDER BY a.created_at ASC
        LIMIT $limit
        """,
        params={"limit": limit},
    )
    return [_row(r) for r in res.result_set]


def _row(r: Any) -> dict[str, Any]:
    caps = r[4]
    if isinstance(caps, str) and caps:
        try:
            caps = json.loads(caps)
        except json.JSONDecodeError:
            caps = []
    return {
        "id": r[0],
        "name": r[1],
        "kind": r[2],
        "description": r[3],
        "capabilities": caps if isinstance(caps, list) else [],
        "created_at": r[5],
        "handoffs_sent": int(r[6] or 0),
        "handoffs_received": int(r[7] or 0),
    }


__all__ = ["get_agent", "list_agents", "register_agent"]

"""Procedural memory: how things get done.

Two write paths:

1. **Explicit** — a developer or agent registers a playbook via the API/MCP.
2. **Mined** — the consolidation worker watches for repeated successful action
   sequences in episodes (``action_signature``) and promotes them to
   procedures once they hit the configured support threshold.

Feedback (success/failure + duration) updates the procedure's success rate so
retrieval can prefer proven playbooks, closing the loop: experience becomes
routine.
"""

from __future__ import annotations

import json
import re
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.ids import ulid
from weft.providers.embeddings import EmbeddingProvider
from weft.timeutil import now_iso

_SLUG = re.compile(r"[^a-z0-9]+")


def procedure_key(name: str) -> str:
    return _SLUG.sub("_", name.strip().lower()).strip("_")


def _proc_text(name: str, description: str, steps: list[str]) -> str:
    return f"{name}\n{description}\n" + "\n".join(steps)


async def upsert_procedure(
    graph: Graph,
    embedder: EmbeddingProvider,
    *,
    name: str,
    description: str = "",
    steps: list[str] | None = None,
    tags: list[str] | None = None,
    derived_from_episode: str | None = None,
) -> str:
    steps = steps or []
    key = procedure_key(name)
    pid = ulid()
    now = now_iso()
    proc_text = _proc_text(name, description, steps)
    embedding = await embedder.embed(proc_text)
    res = await graph.query(
        """
        MERGE (p:Procedure {key: $key})
          ON CREATE SET p.id = $id, p.name = $name, p.description = $description,
                        p.steps = $steps, p.tags = $tags, p.version = 1,
                        p.success_count = 0, p.failure_count = 0,
                        p.created_at = $now, p.last_used_at = NULL,
                        p.embedding = vecf32($embedding)
          ON MATCH SET p.description = $description, p.steps = $steps, p.tags = $tags,
                       p.embedding = vecf32($embedding)
        SET p:Memory, p.search_text = $search_text
        RETURN p.id AS id
        """,
        params={
            "key": key,
            "id": pid,
            "name": name,
            "description": description,
            "steps": json.dumps(steps),
            "tags": json.dumps(tags or []),
            "search_text": proc_text[:8000],
            "embedding": embedding,
            "now": now,
        },
    )
    proc_id = str(res.result_set[0][0])
    if derived_from_episode:
        await graph.query(
            """
            MATCH (p:Procedure {id: $pid}), (e:Episode {id: $eid})
            MERGE (p)-[:DERIVED_FROM]->(e)
            """,
            params={"pid": proc_id, "eid": derived_from_episode},
        )
    return proc_id


async def record_feedback(
    graph: Graph,
    *,
    procedure_id: str,
    success: bool,
    duration_ms: float | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    now = now_iso()
    await graph.query(
        """
        MATCH (p:Procedure {id: $pid})
        SET p.success_count = coalesce(p.success_count, 0) + $s,
            p.failure_count = coalesce(p.failure_count, 0) + $f,
            p.last_used_at = $now,
            p.duration_ema = CASE
                WHEN $dur IS NULL THEN p.duration_ema
                WHEN p.duration_ema IS NULL THEN $dur
                ELSE (p.duration_ema * 0.7 + $dur * 0.3)
            END
        """,
        params={
            "pid": procedure_id,
            "s": 1 if success else 0,
            "f": 0 if success else 1,
            "dur": duration_ms,
            "now": now,
        },
    )
    if episode_id:
        await graph.query(
            """
            MATCH (p:Procedure {id: $pid}), (e:Episode {id: $eid})
            MERGE (p)-[u:USED_IN]->(e)
            SET u.success = $success, u.duration_ms = $dur, u.ts = $now
            """,
            params={
                "pid": procedure_id,
                "eid": episode_id,
                "success": success,
                "dur": duration_ms,
                "now": now,
            },
        )
    proc = await get_procedure(graph, procedure_id)
    if proc is None:
        return {}
    return proc


async def get_procedure(graph: Graph, procedure_id: str) -> dict[str, Any] | None:
    res = await graph.ro_query(
        """
        MATCH (p:Procedure {id: $id})
        RETURN p.id, p.name, p.description, p.steps, p.tags, p.success_count,
               p.failure_count, p.duration_ema, p.version, p.last_used_at
        """,
        params={"id": procedure_id},
    )
    if not res.result_set:
        return None
    r = res.result_set[0]
    return {
        "id": r[0],
        "name": r[1],
        "description": r[2],
        "steps": _loads_list(r[3]),
        "tags": _loads_list(r[4]),
        "success_count": int(r[5] or 0),
        "failure_count": int(r[6] or 0),
        "avg_duration_ms": float(r[7]) if r[7] is not None else None,
        "version": int(r[8] or 1),
        "last_used_at": r[9],
    }


async def list_procedures(graph: Graph, *, limit: int = 50) -> list[dict[str, Any]]:
    res = await graph.ro_query(
        """
        MATCH (p:Procedure)
        RETURN p.id, p.name, p.description, p.steps, p.tags, p.success_count,
               p.failure_count, p.duration_ema, p.version, p.last_used_at
        ORDER BY (coalesce(p.success_count,0) - coalesce(p.failure_count,0)) DESC
        LIMIT $limit
        """,
        params={"limit": limit},
    )
    out = []
    for r in res.result_set:
        out.append(
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "steps": _loads_list(r[3]),
                "tags": _loads_list(r[4]),
                "success_count": int(r[5] or 0),
                "failure_count": int(r[6] or 0),
                "avg_duration_ms": float(r[7]) if r[7] is not None else None,
                "version": int(r[8] or 1),
                "last_used_at": r[9],
            }
        )
    return out


async def mine_procedures(
    graph: Graph,
    embedder: EmbeddingProvider,
    *,
    min_support: int = 2,
) -> list[str]:
    """Promote repeatedly successful action signatures into procedures."""
    res = await graph.query(
        """
        MATCH (e:Episode)
        WHERE e.action_signature IS NOT NULL AND e.outcome = 'success'
          AND coalesce(e.archived, false) = false
        WITH e.action_signature AS sig, count(*) AS support,
             min(e.ts) AS first_ts, head(collect(e.id)) AS sample_id,
             head(collect(e.title)) AS sample_title
        WHERE support >= $min_support
        RETURN sig, support, sample_id, sample_title, first_ts
        """,
        params={"min_support": min_support},
    )
    created: list[str] = []
    for sig, support, sample_id, sample_title, _first_ts in res.result_set:
        steps = [s for s in (sig or "").split(">") if s]
        if len(steps) < 2:
            continue
        name = f"learned: {' → '.join(steps[:4])}"
        existing = await graph.ro_query(
            """
            MATCH (p:Procedure {key: $key})
            RETURN p.id, p.success_count, p.derived_sig_support
            """,
            params={"key": procedure_key(name)},
        )
        if existing.result_set:
            pid, _successes, _ = existing.result_set[0]
            await graph.query(
                """
                MATCH (p:Procedure {id: $pid})
                SET p.success_count = CASE
                        WHEN coalesce(p.success_count, 0) > $support
                        THEN coalesce(p.success_count, 0) ELSE $support END,
                    p.derived_sig_support = $support, p.last_used_at = $now
                """,
                params={"pid": pid, "support": support, "now": now_iso()},
            )
            continue
        pid = await upsert_procedure(
            graph,
            embedder,
            name=name,
            description=(
                f"Learned from {support} successful episodes "
                f"(e.g. '{sample_title}'). Sequence: {' → '.join(steps)}."
            ),
            steps=steps,
            tags=["learned"],
            derived_from_episode=sample_id,
        )
        await graph.query(
            "MATCH (p:Procedure {id: $pid}) SET p.success_count = $support, "
            "p.derived_sig_support = $support",
            params={"pid": pid, "support": support},
        )
        created.append(pid)
    return created


def _loads_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


__all__ = [
    "get_procedure",
    "list_procedures",
    "mine_procedures",
    "procedure_key",
    "record_feedback",
    "upsert_procedure",
]

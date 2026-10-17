"""Durable job queue implemented on FalkorDB itself.

Weft deliberately has **no extra infrastructure**: the same graph that stores
memory also stores work. Jobs are ``(:Job)`` nodes claimed with an atomic
``SET ... WHERE status='pending'`` compare-and-set — FalkorDB executes each
query as a single unit, so two workers can never claim the same job.

At-least-once delivery: a claimed job that crashes the worker is reclaimed by
the janitor once ``locked_at`` exceeds the lease. Handlers must be idempotent
(consolidation is: it only processes messages not yet marked consolidated).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.ids import ulid
from weft.timeutil import iso_from_ms, now_iso, now_ms

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 120.0


async def enqueue(
    graph: Graph,
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    run_at_ms: int | None = None,
    dedupe_key: str | None = None,
) -> str:
    """Enqueue a job. ``dedupe_key`` makes enqueue idempotent (MERGE)."""
    job_id = ulid()
    run_at = iso_from_ms(run_at_ms if run_at_ms is not None else now_ms())
    if dedupe_key:
        await graph.query(
            """
            MERGE (j:Job {dedupe_key: $dedupe_key})
              ON CREATE SET j.id = $id, j.kind = $kind, j.payload = $payload,
                            j.status = 'pending', j.attempts = 0,
                            j.created_at = $now, j.run_at = $run_at
            RETURN j.id AS id
            """,
            params={
                "dedupe_key": dedupe_key,
                "id": job_id,
                "kind": kind,
                "payload": json.dumps(payload or {}),
                "run_at": run_at,
                "now": now_iso(),
            },
        )
        res = await graph.ro_query(
            "MATCH (j:Job {dedupe_key: $k}) RETURN j.id", params={"k": dedupe_key}
        )
        return res.result_set[0][0] if res.result_set else job_id
    await graph.query(
        """
        CREATE (j:Job {id: $id, kind: $kind, payload: $payload, status: 'pending',
                       attempts: 0, created_at: $now, run_at: $run_at,
                       dedupe_key: $dedupe_key})
        """,
        params={
            "id": job_id,
            "kind": kind,
            "payload": json.dumps(payload or {}),
            "run_at": run_at,
            "now": now_iso(),
            "dedupe_key": None,
        },
    )
    return job_id


async def claim(
    graph: Graph,
    worker_id: str,
    *,
    kinds: list[str] | None = None,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Atomically claim up to ``limit`` runnable jobs for this worker."""
    params: dict[str, Any] = {
        "worker": worker_id,
        "now": now_iso(),
        "limit": limit,
    }
    kind_filter = ""
    if kinds:
        kind_filter = "AND j.kind IN $kinds"
        params["kinds"] = kinds
    res = await graph.query(
        f"""
        MATCH (j:Job {{status: 'pending'}})
        WHERE j.run_at <= $now {kind_filter}
        WITH j ORDER BY j.created_at LIMIT $limit
        SET j.status = 'running', j.locked_by = $worker, j.locked_at = $now,
            j.attempts = j.attempts + 1
        RETURN j.id, j.kind, j.payload, j.attempts
        """,
        params=params,
    )
    jobs = []
    for jid, kind, payload, attempts in res.result_set:
        try:
            body = json.loads(payload) if isinstance(payload, str) else (payload or {})
        except json.JSONDecodeError:
            body = {}
        jobs.append({"id": jid, "kind": kind, "payload": body, "attempts": attempts})
    return jobs


async def complete(graph: Graph, job_id: str) -> None:
    await graph.query(
        "MATCH (j:Job {id: $id}) SET j.status = 'done', j.finished_at = $now REMOVE j.payload",
        params={"id": job_id, "now": now_iso()},
    )


async def fail(
    graph: Graph,
    job_id: str,
    error: str,
    *,
    max_attempts: int = 5,
    backoff_seconds: float = 5.0,
) -> None:
    res = await graph.ro_query("MATCH (j:Job {id: $id}) RETURN j.attempts", params={"id": job_id})
    attempts = int(res.result_set[0][0]) if res.result_set else max_attempts
    common = {
        "id": job_id,
        "err": error[:2000],
        "now": now_iso(),
    }
    if attempts >= max_attempts:
        await graph.query(
            "MATCH (j:Job {id: $id}) SET j.status = 'dead', j.last_error = $err, "
            "j.finished_at = $now",
            params=common,
        )
        return
    backoff = backoff_seconds * (2 ** max(0, attempts - 1))
    await graph.query(
        "MATCH (j:Job {id: $id}) SET j.status = 'pending', j.run_at = $run_at, "
        "j.last_error = $err, j.locked_by = NULL, j.locked_at = NULL",
        params={**common, "run_at": iso_from_ms(now_ms() + int(backoff * 1000))},
    )


async def reap_stale(graph: Graph, lease_seconds: float = DEFAULT_LEASE_SECONDS) -> int:
    """Requeue jobs whose worker died (lock older than the lease)."""
    cutoff = iso_from_ms(now_ms() - int(lease_seconds * 1000))
    res = await graph.query(
        """
        MATCH (j:Job {status: 'running'})
        WHERE j.locked_at < $cutoff
        SET j.status = 'pending', j.run_at = $now, j.locked_by = NULL
        RETURN count(j) AS n
        """,
        params={"cutoff": cutoff, "now": now_iso()},
    )
    return int(res.result_set[0][0]) if res.result_set else 0


async def queue_depth(graph: Graph) -> dict[str, int]:
    res = await graph.query("MATCH (j:Job) RETURN j.status, count(j)")
    return {str(status): int(n) for status, n in res.result_set}


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "claim",
    "complete",
    "enqueue",
    "fail",
    "queue_depth",
    "reap_stale",
]

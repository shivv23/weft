"""Shared agent state — the blackboard.

Arbitrary JSON state keyed by ``(scope, key)`` with optimistic concurrency:

* ``expected_version`` omitted → last-write-wins (version still increments).
* ``expected_version`` given → compare-and-set; a stale writer gets Conflict.

Scopes: ``global`` | ``user:<id>`` | ``session:<id>`` | ``agent:<id>`` | ``team:<name>``.
Every write stamps ``updated_by`` and ``updated_at`` so the SSE event stream
and audit trails can follow state changes.
"""

from __future__ import annotations

import json
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.errors import ConflictError, NotFoundError
from weft.timeutil import now_iso

_VALID_SCOPES = ("global", "user:", "session:", "agent:", "team:")


def _validate_scope(scope: str) -> None:
    if not any(scope == s or scope.startswith(s) for s in _VALID_SCOPES):
        raise NotFoundError(
            f"invalid blackboard scope {scope!r}; expected one of "
            f"'global', 'user:<id>', 'session:<id>', 'agent:<id>', 'team:<name>'"
        )


async def set_entry(
    graph: Graph,
    *,
    scope: str,
    key: str,
    value: Any,
    updated_by: str | None = None,
    expected_version: int | None = None,
) -> dict[str, Any]:
    _validate_scope(scope)
    now = now_iso()
    payload = json.dumps(value, ensure_ascii=False)
    if expected_version is None:
        await graph.query(
            """
            MERGE (b:Blackboard {scope: $scope, key: $key})
              ON CREATE SET b.version = 1, b.value = $value, b.updated_by = $by,
                            b.updated_at = $now
              ON MATCH SET b.version = coalesce(b.version, 0) + 1, b.value = $value,
                           b.updated_by = $by, b.updated_at = $now
            """,
            params={
                "scope": scope,
                "key": key,
                "value": payload,
                "by": updated_by,
                "now": now,
            },
        )
    else:
        if expected_version == 0:
            res = await graph.query(
                """
                CREATE (b:Blackboard {scope: $scope, key: $key, version: 1,
                                      value: $value, updated_by: $by, updated_at: $now})
                RETURN b.version AS v
                """,
                params={
                    "scope": scope,
                    "key": key,
                    "value": payload,
                    "by": updated_by,
                    "now": now,
                },
            )
            if not res.result_set:
                raise ConflictError(
                    f"blackboard entry {scope}/{key} already exists",
                    details={"scope": scope, "key": key},
                )
        else:
            res = await graph.query(
                """
                MATCH (b:Blackboard {scope: $scope, key: $key})
                WHERE b.version = $expected
                SET b.version = $expected + 1, b.value = $value,
                    b.updated_by = $by, b.updated_at = $now
                RETURN b.version AS v
                """,
                params={
                    "scope": scope,
                    "key": key,
                    "expected": expected_version,
                    "value": payload,
                    "by": updated_by,
                    "now": now,
                },
            )
            if not res.result_set:
                current = await get_entry(graph, scope=scope, key=key)
                if current is None:
                    raise NotFoundError(f"blackboard entry {scope}/{key} not found")
                raise ConflictError(
                    f"version mismatch for {scope}/{key}: "
                    f"expected {expected_version}, current {current['version']}",
                    details={"current_version": current["version"]},
                )
    entry = await get_entry(graph, scope=scope, key=key)
    assert entry is not None
    return entry


async def get_entry(graph: Graph, *, scope: str, key: str) -> dict[str, Any] | None:
    _validate_scope(scope)
    res = await graph.ro_query(
        """
        MATCH (b:Blackboard {scope: $scope, key: $key})
        RETURN b.scope, b.key, b.value, b.version, b.updated_by, b.updated_at
        """,
        params={"scope": scope, "key": key},
    )
    return _row(res.result_set[0]) if res.result_set else None


async def get_scope(
    graph: Graph, *, scopes: list[str], prefix: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    """Read all entries visible to an agent: its scopes (plus optional global)."""
    for s in scopes:
        _validate_scope(s)
    res = await graph.ro_query(
        """
        MATCH (b:Blackboard)
        WHERE b.scope IN $scopes
        RETURN b.scope, b.key, b.value, b.version, b.updated_by, b.updated_at
        ORDER BY b.updated_at DESC
        LIMIT $limit
        """,
        params={"scopes": scopes, "limit": limit},
    )
    rows = [_row(r) for r in res.result_set]
    if prefix:
        rows = [r for r in rows if r["key"].startswith(prefix)]
    return rows


async def delete_entry(graph: Graph, *, scope: str, key: str) -> bool:
    _validate_scope(scope)
    res = await graph.query(
        "MATCH (b:Blackboard {scope: $scope, key: $key}) DELETE b RETURN 1",
        params={"scope": scope, "key": key},
    )
    return bool(res.result_set)


def _row(r: Any) -> dict[str, Any]:
    value = r[2]
    if isinstance(value, str) and value:
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return {
        "scope": r[0],
        "key": r[1],
        "value": value,
        "version": int(r[3] or 0),
        "updated_by": r[4],
        "updated_at": r[5],
    }


__all__ = ["delete_entry", "get_entry", "get_scope", "set_entry"]

"""Idempotent schema provisioning: indexes and constraints for a tenant graph.

Everything here is safe to run repeatedly. Index kinds:

* **range / unique** — fast equality and uniqueness on ids (sessions, agents,
  messages, handoffs, blackboard keys).
* **full-text**   — RediSearch-backed keyword recall over summaries, facts,
  entity names and procedure docs.
* **vector**      — HNSW ANN over embeddings written with ``vecf32(...)``.

Every recallable memory node (Episode, Fact, Entity, Procedure) also carries a
shared ``:Memory`` label with a normalised ``search_text`` property. The ANN and
full-text indexes are built on that single label, so one hybrid recall issues
**one** vector query and **one** full-text query instead of one per kind — the
difference between ~14 and ~60 recalls/second on a single FalkorDB core.

The schema is deliberately label-additive: new optional properties never
require a migration, and DDL runs once per tenant graph at first touch.
"""

from __future__ import annotations

import logging

from falkordb.asyncio.graph import Graph

logger = logging.getLogger(__name__)

MEMORY_LABEL = "Memory"

_UNIQUE_CONSTRAINTS: list[tuple[str, str]] = [
    ("User", "id"),
    ("Agent", "id"),
    ("Session", "id"),
    ("Message", "id"),
    ("Episode", "id"),
    ("Entity", "key"),  # key = normalised "type|name", unique per tenant
    ("Fact", "id"),
    ("Procedure", "id"),
    ("Handoff", "id"),
    ("Job", "id"),
]

_RANGE_INDEXES: list[tuple[str, str]] = [
    ("Blackboard", "scope"),
    ("Blackboard", "updated_at"),
    ("Job", "status"),
    ("Job", "run_at"),
    ("Episode", "ts"),
    ("Episode", "session_id"),
    ("Episode", "user_id"),
    ("Episode", "archived"),
    ("Fact", "predicate"),
    ("Fact", "last_seen"),
    ("Handoff", "status"),
    ("Handoff", "to_agent"),
    ("Handoff", "created_at"),
    ("Session", "user_id"),
    ("Message", "session_id"),
    ("Procedure", "last_used_at"),
]


async def ensure_schema(graph: Graph, *, dim: int = 256) -> None:
    """Create every index/constraint the memory engine relies on."""
    for label, prop in _UNIQUE_CONSTRAINTS:
        try:
            await graph.create_node_unique_constraint(label, prop)
        except Exception as exc:
            logger.debug("unique constraint %s.%s: %s", label, prop, exc)

    for label, prop in _RANGE_INDEXES:
        try:
            await graph.create_node_range_index(label, prop)
        except Exception as exc:
            logger.debug("range index %s.%s: %s", label, prop, exc)

    try:
        await graph.create_node_fulltext_index(MEMORY_LABEL, "search_text")
    except Exception as exc:
        logger.debug("fulltext index %s.search_text: %s", MEMORY_LABEL, exc)

    try:
        await graph.create_node_vector_index(
            MEMORY_LABEL, "embedding", dim=dim, similarity_function="cosine"
        )
    except Exception as exc:
        logger.debug("vector index %s: %s", MEMORY_LABEL, exc)


__all__ = ["MEMORY_LABEL", "ensure_schema"]

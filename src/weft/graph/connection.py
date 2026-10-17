"""FalkorDB connection management and per-tenant graph routing."""

from __future__ import annotations

import logging
from typing import Any

from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import Graph

from weft.config import Settings, get_settings
from weft.errors import StorageError
from weft.ids import tenant_graph_name

logger = logging.getLogger(__name__)


def to_plain(value: Any) -> Any:
    """Convert falkordb result objects (Node/Edge/Path) into plain JSON-safe data."""
    from falkordb.edge import Edge
    from falkordb.node import Node
    from falkordb.path import Path

    if isinstance(value, Node):
        return {
            "_type": "node",
            "id": value.id,
            "labels": list(value.labels),
            "properties": {k: to_plain(v) for k, v in value.properties.items()},
        }
    if isinstance(value, Edge):
        return {
            "_type": "edge",
            "id": value.id,
            "relationship": value.relationship,
            "properties": {k: to_plain(v) for k, v in value.properties.items()},
            "source": value.node_id,
            "target": value.target_id,
        }
    if isinstance(value, Path):
        return {
            "_type": "path",
            "nodes": [to_plain(n) for n in value.nodes],
            "relationships": [to_plain(e) for e in value.edges],
        }
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    return value


class GraphClient:
    """Owns the async FalkorDB client and hands out per-tenant graph handles.

    Tenant isolation model: every tenant gets its own FalkorDB graph
    (``weft_<tenant>``). Graph-level indexes, full-text catalogs and HNSW vector
    indexes are therefore scoped to a single tenant, which makes cross-tenant
    leakage structurally impossible and lets operators move or shard any tenant
    onto its own FalkorDB instance without a schema change.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._db: FalkorDB | None = None
        self._graphs: dict[str, Graph] = {}

    async def connect(self) -> FalkorDB:
        if self._db is None:
            s = self.settings
            try:
                if s.falkor_dsn:
                    self._db = await FalkorDB.from_url(
                        s.falkor_dsn, max_connections=s.falkor_max_connections
                    )
                else:
                    self._db = FalkorDB(
                        host=s.falkor_host,
                        port=s.falkor_port,
                        password=s.falkor_password,
                        max_connections=s.falkor_max_connections,
                    )
            except Exception as exc:
                raise StorageError(
                    f"Cannot reach FalkorDB at "
                    f"{s.falkor_dsn or f'{s.falkor_host}:{s.falkor_port}'}: {exc}"
                ) from exc
        return self._db

    async def graph_for(self, tenant: str) -> Graph:
        """Return (and cache) the graph handle for a tenant, auto-provisioning it."""
        from weft.graph.schema import ensure_schema

        name = tenant_graph_name(tenant)
        graph = self._graphs.get(name)
        if graph is None:
            db = await self.connect()
            graph = db.select_graph(name)
            await ensure_schema(graph, dim=self.settings.embeddings_dim)
            self._graphs[name] = graph
        return graph

    async def list_tenants(self) -> list[str]:
        db = await self.connect()
        names = await db.list_graphs()
        return sorted(n.removeprefix("weft_") for n in names if n.startswith("weft_"))

    async def drop_tenant(self, tenant: str) -> bool:
        """Delete a tenant's entire memory graph. Destructive; used by tests/ops."""
        name = tenant_graph_name(tenant)
        db = await self.connect()
        graph = self._graphs.pop(name, None) or db.select_graph(name)
        try:
            await graph.delete()
            return True
        except Exception:
            return False

    async def ping(self) -> bool:
        try:
            db = await self.connect()
            await db.list_graphs()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.aclose()
            finally:
                self._db = None
                self._graphs.clear()


__all__ = ["Graph", "GraphClient", "to_plain"]

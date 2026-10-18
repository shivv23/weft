"""Multi-tenant worker pool.

The API is stateless: it writes raw messages and enqueues a consolidation job
into the *same* FalkorDB graph. This pool is what actually turns that raw
traffic into durable memory. It is deliberately decoupled from the API process
so that it can be scaled independently (``docker compose up --scale weft-worker=4``)
and so that request latency never depends on extraction work.

One :class:`ConsolidationWorker` runs per tenant graph. The pool discovers
tenants from FalkorDB (``weft_*`` graphs), so a brand-new tenant is picked up
automatically within ``WEFT_WORKER_TENANT_REFRESH_SECONDS`` — no registration
step, no config reload. A periodic maintenance job decays/archives episodes,
mines procedures and purges stale jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from functools import partial

from weft.config import Settings, get_settings
from weft.graph.connection import GraphClient
from weft.processing.consolidator import ConsolidationWorker

logger = logging.getLogger(__name__)


class WorkerPool:
    """Supervises one consolidation loop per tenant, forever."""

    def __init__(
        self,
        client: GraphClient,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self._workers: dict[str, ConsolidationWorker] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._known: set[str] = set()
        self._stopped = asyncio.Event()
        self._maintenance_task: asyncio.Task[None] | None = None

    async def run_forever(self) -> None:
        logger.info(
            "worker pool starting (refresh=%.0fs)", self.settings.worker_tenant_refresh_seconds
        )
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        try:
            while not self._stopped.is_set():
                try:
                    await self._reconcile()
                except Exception:
                    logger.exception("worker pool reconcile failed")
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stopped.wait(),
                        timeout=self.settings.worker_tenant_refresh_seconds,
                    )
        finally:
            await self.stop()

    async def _reconcile(self) -> None:
        tenants = set(await self.client.list_tenants())
        for tenant in sorted(tenants - set(self._tasks)):
            graph = await self.client.graph_for(tenant)
            worker = ConsolidationWorker(
                graph=graph,
                settings=self.settings,
                graph_exists=partial(self.client.exists, tenant),
            )
            self._workers[tenant] = worker
            self._tasks[tenant] = asyncio.create_task(
                worker.run_forever(), name=f"weft-worker:{tenant}"
            )
            self._known.add(tenant)
            logger.info("worker attached to tenant %r", tenant)
        for tenant in sorted(set(self._tasks) - tenants):
            await self._detach(tenant)

    async def _detach(self, tenant: str) -> None:
        """Stop and forget the worker for a tenant whose graph no longer exists.

        FalkorDB auto-creates a graph on any query, so a worker that outlives
        its tenant would silently resurrect the deleted graph on the next poll.
        """
        task = self._tasks.pop(tenant, None)
        worker = self._workers.pop(tenant, None)
        self._known.discard(tenant)
        if worker is not None:
            worker.stop()
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        logger.info("worker detached from removed tenant %r", tenant)

    async def _maintenance_loop(self) -> None:
        interval = max(30.0, self.settings.maintenance_interval_seconds)
        while not self._stopped.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), timeout=interval)
            if self._stopped.is_set():
                break
            current = set(await self.client.list_tenants())
            for tenant, worker in list(self._workers.items()):
                if tenant not in current:
                    continue
                try:
                    stats = await worker.run_maintenance()
                    logger.info("maintenance ran for tenant %r: %s", tenant, stats)
                except Exception:
                    logger.exception("maintenance failed for tenant %r", tenant)

    async def stop(self) -> None:
        self._stopped.set()
        for worker in self._workers.values():
            worker.stop()
        tasks = list(self._tasks.values())
        if self._maintenance_task is not None:
            tasks.append(self._maintenance_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        self._workers.clear()
        logger.info("worker pool stopped")


__all__ = ["WorkerPool"]

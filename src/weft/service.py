"""Public facade: ``Weft`` (process-wide) and ``WeftHandle`` (per tenant).

All application code — REST API, MCP server, demos, benchmarks — goes through
these two objects, so behaviour is identical everywhere and tests can drive
the library directly.
"""

from __future__ import annotations

import logging
from typing import Any

from weft.config import Settings, get_settings
from weft.coordination import blackboard as bb
from weft.coordination import handoff as handoff_mod
from weft.coordination import registry
from weft.graph.connection import GraphClient
from weft.memory import conversation, episodic, procedural, semantic
from weft.models import (
    AgentOut,
    BlackboardEntry,
    ContextPack,
    EpisodeOut,
    FactOut,
    HandoffOut,
    MessageIn,
    MessageOut,
    ProcedureOut,
    RecallResponse,
    SessionOut,
    StatsOut,
)
from weft.processing import queue
from weft.processing.consolidator import ConsolidationWorker, consolidate_now
from weft.providers.embeddings import EmbeddingProvider, build_embedder
from weft.retrieval.hybrid import HybridRetriever, entity_path

logger = logging.getLogger(__name__)


class Weft:
    """Process-wide entry point: owns connections and providers."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = GraphClient(self.settings)
        self.embedder = embedder or build_embedder(self.settings)

    async def ping(self) -> bool:
        return await self.client.ping()

    async def list_tenants(self) -> list[str]:
        return await self.client.list_tenants()

    async def drop_tenant(self, tenant: str) -> bool:
        return await self.client.drop_tenant(tenant)

    def for_tenant(self, tenant: str) -> WeftHandle:
        return WeftHandle(self, tenant)

    def start_worker(self, tenant: str) -> ConsolidationWorker:
        """Background consolidation worker for one tenant graph."""
        return _LazyWorker(self, tenant)  # type: ignore[return-value]

    async def aclose(self) -> None:
        await self.client.close()


class _LazyWorker:
    """Small wrapper so a worker can be created before the event loop runs."""

    def __init__(self, weft: Weft, tenant: str) -> None:
        self._weft = weft
        self._tenant = tenant
        self._inner: ConsolidationWorker | None = None
        self._task: asyncio.Task | None = None

    async def _ensure(self) -> ConsolidationWorker:
        if self._inner is None:
            graph = await self._weft.client.graph_for(self._tenant)
            self._inner = ConsolidationWorker(
                graph=graph,
                embedder=self._weft.embedder,
                settings=self._weft.settings,
                worker_id=f"worker-{self._tenant}",
            )
        return self._inner

    async def start(self) -> None:
        import asyncio

        worker = await self._ensure()
        self._task = asyncio.create_task(worker.run_forever(), name=f"weft-worker-{self._tenant}")

    async def stop(self) -> None:
        if self._inner:
            self._inner.stop()
        if self._task:
            self._task.cancel()
            import asyncio

            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def run_once(self) -> int:
        return await (await self._ensure()).run_once()


import asyncio  # noqa: E402  (used by _LazyWorker)


class WeftHandle:
    """Tenant-scoped API surface. Every call is isolated to that tenant's graph."""

    def __init__(self, weft: Weft, tenant: str) -> None:
        self.weft = weft
        self.tenant = tenant
        self.settings = weft.settings

    async def _graph(self):
        return await self.weft.client.graph_for(self.tenant)

    # ------------------------------------------------------------------ agents
    async def register_agent(
        self,
        *,
        id: str,
        name: str,
        kind: str = "worker",
        description: str = "",
        capabilities: list[str] | None = None,
    ) -> AgentOut:
        g = await self._graph()
        data = await registry.register_agent(
            g,
            agent_id=id,
            name=name,
            kind=kind,
            description=description,
            capabilities=capabilities,
        )
        return AgentOut(**data)

    async def list_agents(self) -> list[AgentOut]:
        g = await self._graph()
        return [AgentOut(**a) for a in await registry.list_agents(g)]

    # --------------------------------------------------------------- sessions
    async def create_session(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        title: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> SessionOut:
        g = await self._graph()
        data = await conversation.create_session(
            g, user_id=user_id, agent_id=agent_id, title=title, meta=meta
        )
        return SessionOut(**data)

    async def get_session(self, session_id: str) -> SessionOut | None:
        g = await self._graph()
        data = await conversation.get_session(g, session_id)
        return SessionOut(**data) if data else None

    async def list_sessions(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[SessionOut]:
        g = await self._graph()
        rows = await conversation.list_sessions(
            g, user_id=user_id, agent_id=agent_id, status=status, limit=limit
        )
        return [SessionOut(**r) for r in rows]

    async def close_session(self, session_id: str) -> SessionOut:
        g = await self._graph()
        return SessionOut(**await conversation.close_session(g, session_id))

    async def append_messages(
        self, session_id: str, messages: list[MessageIn | dict[str, Any]]
    ) -> list[MessageOut]:
        g = await self._graph()
        rows = [
            m.model_dump(exclude_none=True) if isinstance(m, MessageIn) else m for m in messages
        ]
        out = await conversation.append_messages(g, session_id, rows)
        return [MessageOut(**m) for m in out]

    async def list_messages(
        self, session_id: str, *, limit: int = 100, before_seq: int | None = None
    ) -> list[MessageOut]:
        g = await self._graph()
        out = await conversation.list_messages(g, session_id, limit=limit, before_seq=before_seq)
        return [MessageOut(**m) for m in out]

    # ----------------------------------------------------------- consolidation
    async def consolidate_session(self, session_id: str) -> dict[str, Any]:
        g = await self._graph()
        return await consolidate_now(
            g, session_id, embedder=self.weft.embedder, settings=self.settings
        )

    async def enqueue_consolidation(self, session_id: str) -> str:
        g = await self._graph()
        return await queue.enqueue(
            g,
            "session.consolidate",
            {"session_id": session_id},
            dedupe_key=f"consolidate:{session_id}",
        )

    async def run_maintenance(self) -> dict[str, Any]:
        g = await self._graph()
        worker = ConsolidationWorker(
            graph=g,
            embedder=self.weft.embedder,
            settings=self.settings,
            worker_id="adhoc",
        )
        return await worker.run_maintenance()

    # ----------------------------------------------------------------- recall
    async def recall(
        self,
        *,
        query: str,
        k: int = 8,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        kinds: list[str] | None = None,
        since: str | None = None,
        include_explanation: bool = True,
        touch: bool = True,
    ) -> RecallResponse:
        import time

        started = time.perf_counter()
        g = await self._graph()
        retriever = HybridRetriever(
            graph=g,
            embedder=self.weft.embedder,
            weights=self.settings.retrieval_weights,
            half_life_days=self.settings.episode_half_life_days,
        )
        items = await retriever.recall(
            query=query,
            k=k,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            kinds=kinds,
            since=since,
            include_explanation=include_explanation,
            touch=touch,
        )
        return RecallResponse(
            query=query,
            items=items,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # -------------------------------------------------------- memory: direct
    async def add_fact(self, **kwargs: Any) -> str:
        g = await self._graph()
        obj = kwargs.pop("object", None)
        if obj is not None:
            kwargs["obj"] = obj
        return await semantic.upsert_fact(
            g,
            self.weft.embedder,
            promote_at_evidence=self.settings.fact_promote_evidence,
            **kwargs,
        )

    async def list_facts(
        self, *, subject: str | None = None, predicate: str | None = None, limit: int = 100
    ) -> list[FactOut]:
        g = await self._graph()
        rows = await semantic.active_facts(g, subject=subject, predicate=predicate, limit=limit)
        return [FactOut(**r) for r in rows]

    async def add_episode(
        self,
        *,
        kind: str = "interaction",
        title: str,
        summary: str,
        content: str = "",
        importance: float = 0.5,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        entity_names: list[str] | None = None,
    ) -> str:
        g = await self._graph()
        eid = await episodic.create_episode(
            g,
            self.weft.embedder,
            kind=kind,
            title=title,
            summary=summary,
            content=content,
            importance=importance,
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        for name in entity_names or []:
            ent_id = await semantic.upsert_entity(g, self.weft.embedder, name=name)
            await semantic.link_mentions(g, episode_id=eid, entity_ids=[ent_id])
        return eid

    async def get_episode(self, episode_id: str) -> EpisodeOut | None:
        g = await self._graph()
        data = await episodic.get_episode(g, episode_id)
        return EpisodeOut(**data) if data else None

    async def forget_entity(self, entity_id: str) -> int:
        g = await self._graph()
        return await semantic.forget(g, entity_id=entity_id)

    async def forget_fact(self, fact_id: str) -> int:
        g = await self._graph()
        return await semantic.forget(g, fact_id=fact_id)

    # ------------------------------------------------------ memory: procedural
    async def add_procedure(
        self,
        *,
        name: str,
        description: str = "",
        steps: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        g = await self._graph()
        return await procedural.upsert_procedure(
            g, self.weft.embedder, name=name, description=description, steps=steps, tags=tags
        )

    async def get_procedure(self, procedure_id: str) -> ProcedureOut | None:
        g = await self._graph()
        data = await procedural.get_procedure(g, procedure_id)
        return ProcedureOut(**data) if data else None

    async def list_procedures(self, *, limit: int = 50) -> list[ProcedureOut]:
        g = await self._graph()
        return [ProcedureOut(**p) for p in await procedural.list_procedures(g, limit=limit)]

    async def procedure_feedback(
        self,
        procedure_id: str,
        *,
        success: bool,
        duration_ms: float | None = None,
        episode_id: str | None = None,
    ) -> ProcedureOut:
        g = await self._graph()
        data = await procedural.record_feedback(
            g,
            procedure_id=procedure_id,
            success=success,
            duration_ms=duration_ms,
            episode_id=episode_id,
        )
        return ProcedureOut(**data)

    # ------------------------------------------------------------ coordination
    async def set_state(
        self,
        *,
        scope: str,
        key: str,
        value: Any,
        updated_by: str | None = None,
        expected_version: int | None = None,
    ) -> BlackboardEntry:
        g = await self._graph()
        data = await bb.set_entry(
            g,
            scope=scope,
            key=key,
            value=value,
            updated_by=updated_by,
            expected_version=expected_version,
        )
        return BlackboardEntry(**data)

    async def get_state(self, *, scope: str, key: str) -> BlackboardEntry | None:
        g = await self._graph()
        data = await bb.get_entry(g, scope=scope, key=key)
        return BlackboardEntry(**data) if data else None

    async def scope_state(self, scopes: list[str]) -> list[BlackboardEntry]:
        g = await self._graph()
        return [BlackboardEntry(**e) for e in await bb.get_scope(g, scopes=scopes)]

    async def all_state(self, *, limit: int = 500) -> list[BlackboardEntry]:
        g = await self._graph()
        return [BlackboardEntry(**e) for e in await bb.list_all(g, limit=limit)]

    async def delete_state(self, *, scope: str, key: str) -> bool:
        g = await self._graph()
        return await bb.delete_entry(g, scope=scope, key=key)

    async def create_handoff(self, **kwargs: Any) -> HandoffOut:
        g = await self._graph()
        return HandoffOut(**await handoff_mod.create_handoff(g, **kwargs))

    async def claim_handoff(self, *, handoff_id: str, agent_id: str) -> HandoffOut:
        g = await self._graph()
        return HandoffOut(**await handoff_mod.claim(g, handoff_id=handoff_id, agent_id=agent_id))

    async def complete_handoff(
        self, *, handoff_id: str, agent_id: str, result: Any = None
    ) -> HandoffOut:
        g = await self._graph()
        return HandoffOut(
            **await handoff_mod.complete(g, handoff_id=handoff_id, agent_id=agent_id, result=result)
        )

    async def reject_handoff(
        self, *, handoff_id: str, agent_id: str, reason: str = ""
    ) -> HandoffOut:
        g = await self._graph()
        return HandoffOut(
            **await handoff_mod.reject(g, handoff_id=handoff_id, agent_id=agent_id, reason=reason)
        )

    async def handoff_inbox(
        self, *, agent_id: str, status: str | None = "pending", limit: int = 20
    ) -> list[HandoffOut]:
        g = await self._graph()
        return [
            HandoffOut(**h)
            for h in await handoff_mod.inbox(g, agent_id=agent_id, status=status, limit=limit)
        ]

    async def get_handoff(self, handoff_id: str) -> HandoffOut | None:
        g = await self._graph()
        data = await handoff_mod.get_handoff(g, handoff_id)
        return HandoffOut(**data) if data else None

    # ----------------------------------------------------------- context pack
    async def context_pack(
        self,
        *,
        agent_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        query: str = "",
        k: int = 8,
    ) -> ContextPack:
        """Everything an agent needs at the start of a turn: memories, pending
        handoffs addressed to it, shared blackboard state and session summary."""
        import time

        started = time.perf_counter()
        g = await self._graph()
        retriever = HybridRetriever(
            graph=g,
            embedder=self.weft.embedder,
            weights=self.settings.retrieval_weights,
            half_life_days=self.settings.episode_half_life_days,
        )
        memories = await retriever.recall(
            query=query or f"user {user_id or ''} agent {agent_id}",
            k=k,
            user_id=user_id,
            session_id=None,
            include_explanation=True,
        )
        pending = await handoff_mod.inbox(g, agent_id=agent_id, status="pending")
        scopes = [f"agent:{agent_id}", "global"]
        if user_id:
            scopes.append(f"user:{user_id}")
        if session_id:
            scopes.append(f"session:{session_id}")
        state = await bb.get_scope(g, scopes=scopes)
        session = await conversation.get_session(g, session_id) if session_id else None
        return ContextPack(
            agent_id=agent_id,
            query=query,
            memories=memories,
            pending_handoffs=[HandoffOut(**h) for h in pending],
            blackboard=[BlackboardEntry(**e) for e in state],
            session=SessionOut(**session) if session else None,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # ------------------------------------------------------------------ graph
    async def entity_path(self, *, frm: str, to: str, max_depth: int = 4):
        g = await self._graph()
        return await entity_path(g, from_entity=frm, to_entity=to, max_depth=max_depth)

    async def entity_neighbors(self, *, entity_id: str, depth: int = 2, limit: int = 25):
        g = await self._graph()
        return await semantic.entity_neighbors(g, entity_id=entity_id, depth=depth, limit=limit)

    # ------------------------------------------------------------------ stats
    async def stats(self) -> StatsOut:
        g = await self._graph()
        res = await g.ro_query(
            """
            MATCH (n)
            RETURN labels(n)[0], count(n)
            """
        )
        counts = {str(label): int(n) for label, n in res.result_set if label}
        depth = await queue.queue_depth(g)
        graphs = await self.weft.client.list_tenants()
        return StatsOut(
            tenant=self.tenant,
            counts=counts,
            job_queue=depth,
            graphs=len(graphs),
        )

    async def session_timeline(self, session_id: str) -> dict[str, Any]:
        """Episodes + handoffs of a session in order — the narrative view."""
        g = await self._graph()
        eps = await episodic.episodes_for_session(g, session_id)
        hands = await g.ro_query(
            """
            MATCH (h:Handoff {session_id: $sid})
            RETURN h.id, h.from_agent, h.to_agent, h.title, h.status, h.created_at
            ORDER BY h.created_at ASC
            """,
            params={"sid": session_id},
        )
        return {
            "session_id": session_id,
            "episodes": eps,
            "handoffs": [
                {
                    "id": r[0],
                    "from": r[1],
                    "to": r[2],
                    "title": r[3],
                    "status": r[4],
                    "created_at": r[5],
                }
                for r in hands.result_set
            ],
        }


__all__ = ["Weft", "WeftHandle"]

"""Consolidation: raw conversation becomes durable, linked memory.

Pipeline per session (idempotent — only unprocessed messages are read):

1. Pull new messages.
2. Extract structure (LLM or heuristic): episodes, entities, relations, facts.
3. Write semantic layer (entities MERGEd by key, facts with evidence counts).
4. Write episodic layer (episodes with embeddings, MENTION links, PRECEDES chain).
5. Mine repeated successful action signatures into procedures.
6. Mark messages consolidated; clear the enqueue dedupe key.

The background :class:`ConsolidationWorker` drains a durable queue that lives
in the same FalkorDB graph — no Redis/SQS/Kafka required.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.config import Settings, get_settings
from weft.memory import conversation, episodic, procedural, semantic
from weft.processing import queue
from weft.processing.extractors import Extraction, Extractor, build_extractor
from weft.providers.embeddings import EmbeddingProvider, build_embedder
from weft.providers.llm import LLMProvider, build_llm
from weft.timeutil import iso_from_ms, now_ms

logger = logging.getLogger(__name__)


class ConsolidationService:
    def __init__(
        self,
        *,
        graph: Graph,
        embedder: EmbeddingProvider,
        extractor: Extractor,
        settings: Settings | None = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.extractor = extractor
        self.settings = settings or get_settings()

    async def consolidate_session(self, session_id: str) -> dict[str, Any]:
        messages = await conversation.unconsolidated_messages(self.graph, session_id)
        if not messages:
            await self._clear_dedupe(session_id)
            return {"session_id": session_id, "messages": 0, "episodes": 0}

        session = await conversation.get_session(self.graph, session_id)
        user_id = session["user_id"] if session else None
        agent_id = session.get("agent_id") if session else None
        extraction = await self.extractor.extract(messages, user_id=user_id)

        entity_ids = await self._write_entities(extraction)
        episode_ids = await self._write_episodes(
            session_id, user_id, agent_id, extraction, messages, entity_ids
        )
        await self._write_relations(extraction, entity_ids)
        await self._write_facts(session_id, extraction, entity_ids)
        mined = await procedural.mine_procedures(
            self.graph, self.embedder, min_support=self.settings.procedure_min_support
        )

        await conversation.mark_consolidated(self.graph, [m["id"] for m in messages])
        await self._clear_dedupe(session_id)
        return {
            "session_id": session_id,
            "messages": len(messages),
            "episodes": len(episode_ids),
            "entities": len(entity_ids),
            "facts": len(extraction.facts),
            "relations": len(extraction.relations),
            "procedures_mined": len(mined),
            "extractor": self.extractor.name,
        }

    async def _clear_dedupe(self, session_id: str) -> None:
        await self.graph.query(
            "MATCH (j:Job {dedupe_key: $k}) DELETE j",
            params={"k": f"consolidate:{session_id}"},
        )

    async def _write_entities(self, extraction: Extraction) -> dict[str, str]:
        ids: dict[str, str] = {}
        for e in extraction.entities:
            eid = await semantic.upsert_entity(
                self.graph,
                self.embedder,
                name=e["name"],
                type_=e["type"],
                description=e.get("description", ""),
            )
            ids[e["name"].lower()] = eid
        return ids

    async def _resolve_entity(
        self, name: str, ids: dict[str, str], *, type_: str = "Concept"
    ) -> str:
        key = name.lower()
        if key in ids:
            return ids[key]
        eid = await semantic.upsert_entity(self.graph, self.embedder, name=name, type_=type_)
        ids[key] = eid
        return eid

    async def _write_episodes(
        self,
        session_id: str,
        user_id: str | None,
        agent_id: str | None,
        extraction: Extraction,
        messages: list[dict[str, Any]],
        entity_ids: dict[str, str] | None = None,
    ) -> list[str]:
        if entity_ids is None:
            entity_ids = {}
        specs = extraction.episodes or [
            {
                "title": "conversation",
                "summary": " ".join(str(m.get("content", "")) for m in messages)[:1500],
                "kind": "interaction",
                "importance": 0.4,
                "entities": [],
                "action_signature": None,
                "outcome": None,
            }
        ]
        texts = [f"{e['title']}\n{e['summary']}" for e in specs]
        embeddings = await self.embedder.embed_many(texts)
        created: list[str] = []
        prev_id: str | None = None
        for spec, emb in zip(specs, embeddings, strict=True):
            eid = await episodic.create_episode(
                self.graph,
                self.embedder,
                kind=str(spec.get("kind", "interaction")),
                title=str(spec.get("title", "conversation")),
                summary=str(spec.get("summary", "")),
                content="",
                importance=float(spec.get("importance", 0.5)),
                ts=messages[-1].get("ts") if messages else None,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                action_signature=spec.get("action_signature"),
                outcome=spec.get("outcome"),
                embedding=emb,
            )
            mention_names = list(spec.get("entities") or [])
            mention_ids = []
            for name in mention_names:
                eid_for_name = entity_ids.get(name.lower())
                if eid_for_name is None:
                    eid_for_name = await semantic.upsert_entity(
                        self.graph, self.embedder, name=name
                    )
                    entity_ids[name.lower()] = eid_for_name
                mention_ids.append(eid_for_name)
            await semantic.link_mentions(self.graph, episode_id=eid, entity_ids=mention_ids)
            if prev_id:
                await self.graph.query(
                    """
                    MATCH (a:Episode {id: $a}), (b:Episode {id: $b})
                    MERGE (a)-[:PRECEDES]->(b)
                    """,
                    params={"a": prev_id, "b": eid},
                )
            prev_id = eid
            created.append(eid)
        return created

    async def _write_relations(self, extraction: Extraction, entity_ids: dict[str, str]) -> None:
        for rel in extraction.relations:
            a = await self._resolve_entity(rel["subject"], entity_ids)
            b = await self._resolve_entity(rel["object"], entity_ids)
            await semantic.upsert_relation(
                self.graph,
                subject_id=a,
                object_id=b,
                rel_type=rel["predicate"],
                confidence=rel.get("confidence", 0.6),
            )

    async def _write_facts(
        self,
        session_id: str,
        extraction: Extraction,
        entity_ids: dict[str, str],
    ) -> None:
        source_episode = await self.graph.ro_query(
            """
            MATCH (e:Episode)-[:PART_OF]->(:Session {id: $sid})
            RETURN e.id ORDER BY e.ts DESC LIMIT 1
            """,
            params={"sid": session_id},
        )
        src = source_episode.result_set[0][0] if source_episode.result_set else None
        for fact in extraction.facts:
            await semantic.upsert_fact(
                self.graph,
                self.embedder,
                subject=fact["subject"],
                predicate=fact["predicate"],
                obj=fact["object"],
                confidence=fact.get("confidence", 0.6),
                source_episode_id=src,
                source="consolidation",
                promote_at_evidence=self.settings.fact_promote_evidence,
            )


class ConsolidationWorker:
    """Background loop: claim jobs from the in-graph queue, process, ack."""

    def __init__(
        self,
        *,
        graph: Graph,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        worker_id: str | None = None,
        graph_exists: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.graph = graph
        self.embedder = embedder or build_embedder(self.settings)
        self.llm = llm if llm is not None else build_llm(self.settings)
        self.extractor = build_extractor(self.llm)
        self.service = ConsolidationService(
            graph=graph,
            embedder=self.embedder,
            extractor=self.extractor,
            settings=self.settings,
        )
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.graph_exists = graph_exists
        self._stopped = asyncio.Event()

    async def run_forever(self) -> None:
        logger.info("consolidation worker %s started", self.worker_id)
        while not self._stopped.is_set():
            try:
                processed = await self.run_once()
            except Exception:
                logger.exception("consolidation cycle failed")
                processed = 0
            if processed == 0:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stopped.wait(), timeout=self.settings.consolidation_poll_seconds
                    )
        logger.info("consolidation worker %s stopped", self.worker_id)

    def stop(self) -> None:
        self._stopped.set()

    async def run_once(self) -> int:
        if self.graph_exists is not None and not await self.graph_exists():
            logger.info(
                "consolidation worker %s: tenant graph is gone, stopping",
                self.worker_id,
            )
            self._stopped.set()
            return 0
        await queue.reap_stale(self.graph)
        jobs = await queue.claim(
            self.graph,
            self.worker_id,
            limit=self.settings.consolidation_batch,
        )
        handled = 0
        for job in jobs:
            try:
                await self._handle(job)
                await self.graph.query("MATCH (j:Job {id: $id}) DELETE j", params={"id": job["id"]})
                handled += 1
            except Exception as exc:
                logger.exception("job %s (%s) failed", job["id"], job["kind"])
                await queue.fail(
                    self.graph, job["id"], str(exc), max_attempts=5, backoff_seconds=5.0
                )
        return handled

    async def _handle(self, job: dict[str, Any]) -> None:
        kind = job["kind"]
        payload = job["payload"]
        if kind == "session.consolidate":
            await self.service.consolidate_session(payload["session_id"])
        elif kind == "maintenance.decay":
            await self.run_maintenance()
        else:
            logger.warning("unknown job kind %r", kind)

    async def run_maintenance(self) -> dict[str, Any]:
        """Decay/archive episodes, mine procedures, purge stale jobs."""
        stats = await episodic.apply_decay_and_archive(
            self.graph,
            half_life_days=self.settings.episode_half_life_days,
            archive_after_days=self.settings.episode_archive_after_days,
            archive_salience=self.settings.episode_archive_salience,
        )
        mined = await procedural.mine_procedures(
            self.graph, self.embedder, min_support=self.settings.procedure_min_support
        )
        reaped = await queue.reap_stale(self.graph)
        await self.graph.query(
            """
            MATCH (j:Job)
            WHERE j.status = 'done' AND j.finished_at < $cutoff
            DELETE j
            """,
            params={"cutoff": iso_from_ms(now_ms() - 3_600_000)},
        )
        stats["procedures_mined"] = len(mined)
        stats["jobs_reaped"] = reaped
        return stats


async def consolidate_now(
    graph: Graph,
    session_id: str,
    *,
    embedder: EmbeddingProvider | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Synchronous consolidation (used by tests, CLI and the demo API)."""
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    llm = build_llm(settings)
    service = ConsolidationService(
        graph=graph,
        embedder=embedder,
        extractor=build_extractor(llm),
        settings=settings,
    )
    return await service.consolidate_session(session_id)


__all__ = ["ConsolidationService", "ConsolidationWorker", "consolidate_now"]

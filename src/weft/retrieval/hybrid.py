"""Hybrid recall: vector + full-text + graph traversal, with explanations.

FalkorDB gives us three complementary channels inside one query surface:

* **vector**   — HNSW ANN over the shared ``:Memory.embedding`` index (one query
  returns episodes, facts, entities and procedures together)
* **full-text** — RediSearch over the shared ``:Memory.search_text`` field
* **graph**    — entity mentions and 1-hop relation traversal (the channel the
  other databases cannot do: following *relationships* the text never spells
  out)

Channels are fused with a weighted score that also factors recency decay,
write-time importance and usage frequency (a memory that keeps getting used
resists decay). Every returned item carries a score breakdown and, when it
came via the graph channel, the traversal path that produced it — so an agent
or a human can always answer *"why did I get this memory?"*.

Tenant note: vector/full-text indexes are per-tenant graphs, so ANN search
never crosses a tenant boundary; filters (user/session/kind) are applied
post-ANN on the candidate set.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any

from falkordb.asyncio.graph import Graph

from weft.ids import ulid
from weft.memory import episodic as episodic_mem
from weft.models import MemoryItem, ScoreBreakdown
from weft.providers.embeddings import EmbeddingProvider
from weft.timeutil import age_days

logger = logging.getLogger(__name__)

_TOKENS_MAX = 12
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]{1,}")

# Question words carry no meaning for entity-name matching: "was" would match
# any entity whose name happens to contain those letters.
_GRAPH_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it of on or "
    "the to was were what when where which who whom why will with".split()
)

# Graph-channel weights. A name match only says "this query is about X"; the
# episodes and facts reached *through* X are what actually answer it, so they
# rank above the bare entity.
_GRAPH_ENTITY = 0.6
_GRAPH_EPISODE = 0.85
_GRAPH_FACT = 0.9

_KIND_BY_LABEL = {
    "Episode": "episode",
    "Fact": "fact",
    "Entity": "entity",
    "Procedure": "procedure",
}


def _kind_from_labels(labels: Any) -> str | None:
    """Resolve a shared ``:Memory`` node back to its concrete memory kind."""
    for label in labels or []:
        kind = _KIND_BY_LABEL.get(str(label))
        if kind:
            return kind
    return None


def _sanitize_fulltext(query: str) -> str:
    tokens = _TOKEN_RE.findall(query)[:_TOKENS_MAX]
    if not tokens:
        return ""
    return " | ".join(f'"{t}"' for t in tokens)


class HybridRetriever:
    def __init__(
        self,
        *,
        graph: Graph,
        embedder: EmbeddingProvider,
        weights: dict[str, float],
        half_life_days: float = 14.0,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.weights = weights
        self.half_life_days = half_life_days

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
    ) -> list[MemoryItem]:
        started = time.perf_counter()
        kinds = kinds or ["episode", "fact", "entity", "procedure"]
        query_vec = await self.embedder.embed(query)
        overfetch = max(k * 4, 16)

        candidates: dict[tuple[str, str], dict[str, Any]] = {}

        async def _vector_channel() -> None:
            try:
                res = await self.graph.ro_query(
                    """
                    CALL db.idx.vector.queryNodes('Memory', 'embedding', $k, vecf32($q))
                    YIELD node, score
                    RETURN node, score
                    """,
                    params={"q": query_vec, "k": overfetch},
                )
            except Exception as exc:
                logger.debug("vector channel unavailable: %s", exc)
                return
            for node, dist in res.result_set:
                props = node.properties
                kind = _kind_from_labels(node.labels)
                if kind is None or kind not in kinds:
                    continue
                _add(
                    candidates,
                    kind,
                    str(props.get("id") or ulid()),
                    props,
                    channel="vector",
                    vector_distance=float(dist),
                )

        async def _fulltext_channel() -> None:
            term = _sanitize_fulltext(query)
            if not term:
                return
            try:
                res = await self.graph.ro_query(
                    """
                    CALL db.idx.fulltext.queryNodes('Memory', $term)
                    YIELD node, score
                    RETURN node, score
                    LIMIT $k
                    """,
                    params={"term": term, "k": overfetch},
                )
            except Exception as exc:
                logger.debug("fulltext channel unavailable: %s", exc)
                return
            for node, score in res.result_set:
                props = node.properties
                kind = _kind_from_labels(node.labels)
                if kind is None or kind not in kinds:
                    continue
                _add(
                    candidates,
                    kind,
                    str(props.get("id") or ulid()),
                    props,
                    channel="fulltext",
                    fulltext_score=float(score),
                )

        async def _graph_channel() -> None:
            tokens = [
                t.lower()
                for t in _TOKEN_RE.findall(query)
                if len(t) > 2 and t.lower() not in _GRAPH_STOPWORDS
            ][:8]
            if not tokens:
                return
            conds = " OR ".join(f"toLower(ent.name) CONTAINS $t{i}" for i in range(len(tokens)))
            params: dict[str, Any] = {f"t{i}": t for i, t in enumerate(tokens)}
            params["limit"] = overfetch
            try:
                # facts are reified as (s)-[:SUBJ]->(Fact)-[:OBJ]->(o): look at both
                # sides, otherwise "Acme Corp tier gold" is missed when asking about Acme
                res = await self.graph.ro_query(
                    f"""
                    MATCH (ent:Entity)
                    WHERE {conds}
                    OPTIONAL MATCH (ep:Episode)-[:MENTIONS]->(ent)
                    OPTIONAL MATCH (f:Fact)-[:OBJ]->(ent)
                    OPTIONAL MATCH (ent)-[:SUBJ]->(sf:Fact)
                    OPTIONAL MATCH (ent)-[r]-(neighbor:Entity)
                    RETURN ent, collect(DISTINCT ep)[..5],
                           collect(DISTINCT f)[..5] + collect(DISTINCT sf)[..5],
                           collect(DISTINCT neighbor)[..5], collect(DISTINCT type(r))[..5]
                    LIMIT $limit
                    """,
                    params=params,
                )
            except Exception as exc:
                logger.debug("graph channel unavailable: %s", exc)
                return
            for ent_node, eps, facts, neighbors, rel_types in res.result_set:
                ent_props = ent_node.properties
                ent_id = str(ent_props.get("id") or ulid())
                ent_name = str(ent_props.get("name") or "")
                _add(
                    candidates,
                    "entity",
                    ent_id,
                    ent_props,
                    channel="graph",
                    graph_score=_GRAPH_ENTITY,
                )
                path_base = f"query → entity '{ent_name}'"
                # an entity with no description reads as "Acme Corp (Organization) —";
                # show what the graph knows about it instead
                known = [
                    str(f.properties.get("text"))
                    for f in facts or []
                    if f is not None
                    and f.properties.get("text")
                    and not f.properties.get("archived")
                    and not f.properties.get("valid_to")
                ]
                if known:
                    candidates[("entity", ent_id)]["known_facts"] = known[:4]
                for ep in eps or []:
                    if ep is None:
                        continue
                    props = ep.properties
                    if props.get("archived"):
                        continue
                    _add(
                        candidates,
                        "episode",
                        str(props.get("id") or ulid()),
                        props,
                        channel="graph",
                        graph_score=_GRAPH_EPISODE,
                        path=f"{path_base} → MENTIONS ← episode",
                    )
                for f in facts or []:
                    if f is None:
                        continue
                    props = f.properties
                    if props.get("archived") or props.get("valid_to"):
                        continue
                    _add(
                        candidates,
                        "fact",
                        str(props.get("id") or ulid()),
                        props,
                        channel="graph",
                        graph_score=_GRAPH_FACT,
                        path=f"{path_base} → fact via '{props.get('predicate', '')}'",
                    )
                if neighbors:
                    pairs = ", ".join(
                        f"{rel or 'REL'}→ {n.properties.get('name', '?')}"
                        for n, rel in zip(neighbors, rel_types, strict=False)
                        if n is not None
                    )
                    if pairs:
                        _add(
                            candidates,
                            "entity",
                            ent_id,
                            ent_props,
                            channel="graph",
                            graph_score=_GRAPH_ENTITY,
                            path=f"{path_base} → neighbors: {pairs}",
                        )

        import asyncio

        await asyncio.gather(_vector_channel(), _fulltext_channel(), _graph_channel())

        items: list[MemoryItem] = []
        for (kind, cid), cand in candidates.items():
            props = cand["props"]
            text, ts, base_quality = _text_ts_quality(kind, props)
            if kind == "entity" and not props.get("description") and cand.get("known_facts"):
                text = f"{text.rstrip(' —')} — {'; '.join(cand['known_facts'])}"
            if user_id and kind == "episode" and props.get("user_id") not in (None, user_id):
                continue
            if agent_id and kind == "episode" and props.get("agent_id") not in (None, agent_id):
                continue
            if (
                session_id
                and kind == "episode"
                and props.get("session_id")
                not in (
                    None,
                    session_id,
                )
            ):
                continue
            if since and ts and ts < since:
                continue
            if kind == "episode" and props.get("archived"):
                continue
            if kind == "fact" and (props.get("archived") or props.get("valid_to")):
                continue

            scores = ScoreBreakdown()
            vec_sims = [
                1.0 / (1.0 + cand["vector_distance"])
                for c in [cand]
                if c.get("vector_distance") is not None
            ]
            scores.vector = vec_sims[0] if vec_sims else 0.0
            scores.fulltext = min(1.0, cand.get("fulltext_score", 0.0) / 10.0)
            scores.graph = float(cand.get("graph_score", 0.0))
            age = age_days(ts) if ts else 365.0
            scores.recency = 0.5 ** (age / max(self.half_life_days, 0.001))
            scores.importance = base_quality
            access = int(props.get("access_count") or 0)
            scores.frequency = min(1.0, math.log1p(access) / 4.0)
            total = scores.total(self.weights)

            item = MemoryItem(
                id=cid,
                kind=kind,  # type: ignore[arg-type]
                text=text,
                score=round(total, 6),
                scores=scores,
                ts=ts,
                matched_via=sorted(cand["channels"]),
            )
            if include_explanation:
                item.explanation = _explanation(cand, scores, self.weights)
            items.append(item)

        items.sort(key=lambda i: i.score, reverse=True)
        selected = _mmr(items, k)

        if touch:
            touch_ids = [i.id for i in selected if i.kind == "episode"]
            if touch_ids:
                await episodic_mem.touch_episodes(self.graph, touch_ids)
        logger.debug(
            "recall %r -> %d items in %.1fms",
            query,
            len(selected),
            (time.perf_counter() - started) * 1000,
        )
        return selected


def _add(
    candidates: dict[tuple[str, str], dict[str, Any]],
    kind: str,
    cid: str,
    props: dict[str, Any],
    *,
    channel: str,
    vector_distance: float | None = None,
    fulltext_score: float | None = None,
    graph_score: float | None = None,
    path: str | None = None,
) -> None:
    key = (kind, cid)
    cand = candidates.setdefault(
        key,
        {"props": props, "channels": set(), "paths": [], "props_by_channel": {}},
    )
    cand["channels"].add(channel)
    if vector_distance is not None:
        cand["vector_distance"] = vector_distance
    if fulltext_score is not None:
        cand["fulltext_score"] = max(cand.get("fulltext_score", 0.0), fulltext_score)
    if graph_score is not None:
        cand["graph_score"] = max(cand.get("graph_score", 0.0), graph_score)
    if path:
        cand["paths"].append(path)


def _text_ts_quality(kind: str, props: dict[str, Any]) -> tuple[str, str | None, float]:
    if kind == "episode":
        text = f"{props.get('title', '')}\n{props.get('summary', '')}".strip()
        return text, props.get("ts"), float(props.get("importance") or 0.5)
    if kind == "fact":
        text = str(props.get("text") or "")
        conf = float(props.get("confidence") or 0.5)
        return text, props.get("last_seen"), conf
    if kind == "entity":
        text = (
            f"{props.get('name', '')} ({props.get('type', 'Concept')}) — "
            f"{props.get('description', '')}".strip()
        )
        mentions = int(props.get("mention_count") or 0)
        return (
            text,
            props.get("last_seen") or props.get("created_at"),
            min(1.0, 0.3 + mentions / 20.0),
        )
    steps = props.get("steps")
    if isinstance(steps, str):
        text = f"{props.get('name', '')}\n{props.get('description', '')}\n{steps}"
    else:
        text = f"{props.get('name', '')}\n{props.get('description', '')}"
    successes = int(props.get("success_count") or 0)
    failures = int(props.get("failure_count") or 0)
    trials = successes + failures
    quality = 0.5 if trials == 0 else successes / trials
    return text.strip(), props.get("last_used_at"), quality


def _explanation(
    cand: dict[str, Any], scores: ScoreBreakdown, weights: dict[str, float]
) -> dict[str, Any]:
    contrib = {k: round(getattr(scores, k) * weights.get(k, 0.0), 5) for k in weights}
    return {
        "matched_via": sorted(cand["channels"]),
        "score_contributions": contrib,
        "paths": cand["paths"][:5],
        "components": {
            "vector_distance": cand.get("vector_distance"),
            "fulltext_score": cand.get("fulltext_score"),
            "graph_score": cand.get("graph_score"),
        },
    }


def _mmr(items: list[MemoryItem], k: int, lambda_: float = 0.75) -> list[MemoryItem]:
    """Maximal-marginal-relevance pick: relevance with a light redundancy guard.

    Without fetching embeddings again we treat exact-duplicate texts as the
    redundancy measure (cheap and effective for memory recall).
    """
    selected: list[MemoryItem] = []
    pool = list(items)
    seen_norm: set[str] = set()
    while pool and len(selected) < k:
        best: MemoryItem | None = None
        best_score = -1e9
        for cand in pool:
            norm = " ".join(cand.text.lower().split())[:120]
            redundancy = 1.0 if norm in seen_norm else 0.0
            score = lambda_ * cand.score - (1 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best = cand
        assert best is not None
        pool.remove(best)
        selected.append(best)
        seen_norm.add(" ".join(best.text.lower().split())[:120])
    return selected


async def entity_path(
    graph: Graph, *, from_entity: str, to_entity: str, max_depth: int = 4
) -> list[dict[str, Any]]:
    """Shortest undirected relationship path between two entities."""
    res = await graph.ro_query(
        f"""
        MATCH path = shortestPath(
            (a:Entity {{name: $from}})-[*1..{max_depth}]-(b:Entity {{name: $to}})
        )
        RETURN path LIMIT 1
        """,
        params={"from": from_entity, "to": to_entity},
    )
    if not res.result_set:
        return []
    path = res.result_set[0][0]
    steps: list[dict[str, Any]] = []
    for n in path.nodes:
        props = n.properties
        steps.append(
            {
                "kind": "entity",
                "name": props.get("name"),
                "type": props.get("type"),
                "id": props.get("id"),
            }
        )
    for e in path.edges:
        steps.append({"kind": "relation", "type": e.relationship, "properties": e.properties})
    return steps


async def touch_facts(graph: Graph, fact_ids: list[str]) -> None:
    if not fact_ids:
        return
    await graph.query(
        "UNWIND $ids AS id MATCH (f:Fact {id: id}) "
        "SET f.access_count = coalesce(f.access_count, 0) + 1",
        params={"ids": fact_ids},
    )


__all__ = ["HybridRetriever", "entity_path", "touch_facts"]

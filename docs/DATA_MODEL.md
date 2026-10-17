# Weft Data Model

Everything Weft stores lives in a FalkorDB graph. Tenant isolation is **structural**:
each tenant gets its own graph named `weft_<tenant>`, so an ANN or full-text index
created for one tenant literally cannot see another's data. This document describes
the nodes, relationships and indexes inside a single tenant graph.

## Labels at a glance

| Label | Role | Key |
|---|---|---|
| `User` | end user | `id` |
| `Agent` | autonomous worker `/` specialist | `id` |
| `Session` | a conversation thread | `id` |
| `Message` | one turn of conversation | `id` |
| `Episode` | an event/memory of "what happened" — **episodic memory** | `id` |
| `Entity` | a person, org, product, concept — **semantic memory** | `key` = `type\|name` |
| `Fact` | a reified assertion — **semantic memory** | `id` |
| `Procedure` | a reusable action sequence — **procedural memory** | `id` |
| `Blackboard` | versioned shared state | `(scope, key)` |
| `Handoff` | a unit of work transferred agent→agent | `id` |
| `Job` | durable background work (queue) | `id` |
| `Memory` | *shared super-label* on all recallable memory | — |

### The `:Memory` super-label

Every recallable node (`Episode`, `Entity`, `Fact`, `Procedure`) also carries the
label `:Memory` and a normalised `search_text` property. Weft builds **one** HNSW
vector index and **one** RediSearch full-text index on `:Memory`, instead of one
pair per kind. Benefits:

- a hybrid recall issues 1 vector query + 1 full-text query (not 4+4);
- new memory kinds need no new index;
- results fuse uniformly, then resolve back to a concrete kind via the node's
  labels.

```
(:Episode:Memory  {embedding: vecf32([...]), search_text: "...", ...})
(:Fact:Memory     {embedding: vecf32([...]), search_text: "...", ...})
(:Entity:Memory   {embedding: vecf32([...]), search_text: "...", ...})
(:Procedure:Memory{embedding: vecf32([...]), search_text: "...", ...})
```

## Relationships

```
(User)-[:HAS_SESSION]->(Session)
(Agent)-[:RUNS]->(Session)
(Session)-[:HAS_MESSAGE]->(Message)

(Episode)-[:PART_OF]->(Session)          # which session an episode belongs to
(Episode)-[:PRECEDES]->(Episode)         # temporal chain within a session
(Episode)-[:MENTIONS]->(Entity)          # entities an episode is about
(Episode)-[:ASSERTED]->(Fact)            # provenance: episode is evidence for a fact

(Entity)-[:SUBJ]->(Fact)-[:OBJ]->(Entity)   # reified assertion (subject, predicate, object)
(Entity)-[:<PREDICATE>]->(Entity)           # optional direct typed relationship

(Procedure)-[:DERIVED_FROM]->(Episode)   # mining provenance
(Procedure)-[:USED_IN]->(Episode)        # feedback: procedure applied in an episode

(Agent)-[:SENT]->(Handoff)-[:TO]->(Agent)
(Handoff)-[:DURING]->(Session)
(Handoff)-[:REFERENCES]->(<any>)         # linked memory references
(Agent)-[:CLAIMED]->(Handoff)
```

The reified `(Entity)-[:SUBJ]->(Fact)-[:OBJ]->(Entity)` pattern is deliberate: it
lets a fact carry its own properties (`confidence`, `evidence_count`, `valid_to`)
and lets Weft traverse *from either endpoint* without losing the predicate — and
lets a single `Fact` be asserted by many episodes for evidence counting.

## Node properties

### `Episode` (episodic)
`id`, `kind`, `title`, `summary`, `content`, `importance` (0–1),
`ts` (ISO timestamp), `session_id`, `user_id`, `agent_id`,
`salience` (decayed weight), `access_count`, `archived`,
`action_signature` (for procedure mining), `outcome`,
`embedding` (vecf32), `search_text`.

### `Entity` (semantic)
`id`, `key` (`type|normalised-name`, unique), `name`, `type` (`Organization`,
`Person`, `Product`, `Concept`, …), `description`, `mention_count`,
`created_at`, `last_seen`, `embedding`, `search_text`.

### `Fact` (semantic)
`id`, `subject`, `predicate`, `object`, `text` ("S P O"),
`confidence` (0–1), `evidence_count`, `first_seen`, `last_seen`,
`valid_to` (bitemporal invalidation), `source_episode_ids`,
`archived`, `embedding`, `search_text`.

### `Procedure` (procedural)
`id`, `key`, `name`, `description`, `steps` (list), `tags` (list),
`success_count`, `failure_count`, `avg_duration_ms`, `version`,
`last_used_at`, `embedding`, `search_text`.

### `Session` / `Message`
`Session`: `id`, `user_id`, `agent_id`, `title`, `status`, `started_at`,
`last_activity_at`, `message_count`, `meta`.
`Message`: `id`, `session_id`, `seq`, `role` (`system|user|assistant|tool`),
`content`, `ts`, `meta`, `consolidated` (bool — drives incremental consolidation).

### `Blackboard` (coordination)
`scope`, `key`, `value`, `version`, `updated_by`, `updated_at`.
Unique on `(scope, key)`. `version` enables optimistic concurrency: a write with
`expected_version` performs an atomic compare-and-swap.

### `Handoff` (coordination)
`id`, `from_agent`, `to_agent`, `title`, `reason`, `context`, `priority` (0–9),
`status` (`pending|claimed|completed|rejected`), `session_id`, `user_id`,
`created_at`, `claimed_at`, `completed_at`, `result`.

### `Job` (durable queue)
`id`, `kind` (`session.consolidate`, `maintenance.decay`), `payload`,
`status` (`pending|running|done|dead`), `attempts`, `created_at`, `run_at`,
`locked_by`, `locked_at`, `finished_at`, `dedupe_key`, `last_error`.

## Indexes & constraints

Created idempotently per tenant graph at first touch (`graph/schema.py`):

- **Unique constraints** on `id` for `User, Agent, Session, Message, Episode,
  Fact, Procedure, Handoff, Job`, and on `Entity.key`.
- **Range indexes** on the hot filter/sort/join properties: `Episode(ts,
  session_id, user_id, archived)`, `Fact(predicate, last_seen)`,
  `Handoff(status, to_agent, created_at)`, `Session(user_id)`,
  `Message(session_id)`, `Procedure(last_used_at)`, `Blackboard(scope,
  updated_at)`, `Job(status, run_at)`.
- **Full-text index** `FOR (m:Memory) ON (m.search_text)`.
- **Vector index** `FOR (m:Memory) ON (m.embedding)`
  `OPTIONS {dimension: <WEFT_EMBEDDINGS_DIM>, similarityFunction: 'cosine'}`.

> **Note on `dim`.** FalkorDB fixes vector dimension at index creation. It comes
> from `WEFT_EMBEDDINGS_DIM` (default 256, matching the offline `hash` embedder).
> Changing it requires recreating the tenant graph.

## Recall scoring

A recall fuses three channels into a candidate set, then scores each candidate:

| Component | Source | Notes |
|---|---|---|
| `vector` | `1 / (1 + cosine_distance)` | HNSW over `:Memory.embedding` |
| `fulltext` | `min(1, rediScore / 10)` | RediSearch over `:Memory.search_text` |
| `graph` | `1.0` (entity) / `0.8` (fact) / `0.7` (episode) | reached via `MENTIONS`/`OBJ`/neighbors |
| `recency` | `0.5 ** (age_days / half_life_days)` | age from the node's own timestamp |
| `importance` | write-time importance / fact confidence / entity mention count / procedure success rate | |
| `frequency` | `min(1, log1p(access_count) / 4)` | "a memory that keeps getting used resists decay" |

`score = Σ weightᵢ · componentᵢ`, with weights from `WEFT_WEIGHT_*` (normalised).
Results are de-duplicated by Maximal Marginal Relevance (MMR) before the top-`k`
is returned, and each item carries its `score` breakdown plus the graph traversal
`path` that surfaced it.

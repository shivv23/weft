# Weft Architecture

## Goals

1. **One durable store.** Memory, conversation history, shared state, handoffs and
   the background queue all live in FalkorDB. Operationally there is a single thing
   to run, back up and reason about.
2. **Write fast, think later.** Ingesting a message must not block on extraction or
   embedding. The API writes raw data and enqueues work; a worker turns raw data
   into memory.
3. **Structural tenant isolation.** A bug in a filter should not be able to leak
   one tenant's data into another's. Isolation is a property of the graph, not of a
   `WHERE` clause.
4. **Explainable recall.** An agent (or a human debugging it) must be able to ask
   *why* a memory came back.
5. **No hidden defaults.** Endpoints, models, weights and thresholds are all
   environment-configurable.

## Layering

```
┌──────────────────────────────── application surfaces ───────────────────────────────┐
│  FastAPI (api/app.py)     MCP server (mcp/server.py)     CLI + demo + bench         │
└───────────────┬───────────────────────┬───────────────────────────┬─────────────────┘
                └──────────────┬────────┴───────────────┬───────────┘
                               ▼                        ▼
                        Weft (process-wide)      _LazyWorker / WorkerPool
                               │                        │
                               ▼                        ▼
                        WeftHandle (per tenant)   ConsolidationService
                               │                        │
    ┌──────────────┬───────────┼────────────┬───────────┴──────────┬──────────────┐
    ▼              ▼           ▼            ▼                      ▼              ▼
 memory/       retrieval/   coordination/ processing/         providers/     graph/
 episodic      hybrid       blackboard    extractors          embeddings     connection
 semantic      (vec+ft+     handoffs      consolidator        llm            schema
 procedural    graph)       registry      queue/worker
    └──────────────┴───────────┴────────────┴──────────────────────┴──────────────┘
                               ▼
                        GraphClient ──▶ FalkorDB (weft_<tenant> per tenant)
```

Every surface goes through `Weft`/`WeftHandle`, so the REST API, the MCP tools, the
demo and the tests exercise **exactly the same code path** — no logic is duplicated
in an adapter.

## Request / data flows

### Ingest (fast path)

```
POST /sessions/{id}/messages
  ├─ MERGE (:Session)-[:HAS_MESSAGE]->(:Message)   (one write, seq-ordered)
  └─ MERGE (:Job {dedupe_key: "consolidate:<sid>"})  (idempotent enqueue)
```

No embedding, no LLM, no extraction on the request path. The response returns as
soon as the messages are durable.

### Consolidation (memory path)

`WorkerPool` discovers tenants (`weft_*` graphs) and runs one
`ConsolidationWorker` per tenant. Each worker claims jobs from the in-graph queue
with an atomic CAS and runs `ConsolidationService.consolidate_session`:

1. **Read** only messages not yet marked `consolidated` (incremental + idempotent).
2. **Extract** structure — `LLMExtractor` when an LLM is configured, otherwise a
   deterministic `HeuristicExtractor` (regex/grammar rules doing entity, relation,
   fact, episode and action-signature extraction). Both produce the same
   `Extraction` shape.
3. **Write semantic layer** — `MERGE` entities by `key`, upsert reified facts with
   evidence counting and confidence strengthening.
4. **Write episodic layer** — create episodes (with embeddings), link `MENTIONS`,
   chain `PRECEDES`, and `ASSERTED` provenance to facts.
5. **Mine procedures** — scan episodes carrying an `action_signature`; sequences
   seen `>= WEFT_PROCEDURE_MIN_SUPPORT` times become/strengthen `Procedure` nodes.
6. **Mark consolidated** and clear the dedupe job.

The queue is durable: claimed jobs have a lease and a janitor (`reap_stale`)
requeues them if a worker dies. Handlers are idempotent, so at-least-once delivery
is safe.

### Recall (read path)

```
HybridRetriever.recall(query)
  ├─ embed(query)
  ├─ asyncio.gather(
  │     vector channel:    queryNodes('Memory','embedding',k)     → candidates
  │     fulltext channel:  queryNodes('Memory', term) LIMIT k      → candidates
  │     graph channel:     entity name match → MENTIONS/OBJ/neighbors → candidates)
  ├─ filter (user_id / session_id / kind / since / archived)
  ├─ score (vector, fulltext, graph, recency, importance, frequency)
  ├─ MMR-select top-k
  └─ touch accessed episodes (access_count++)
```

The three channels run concurrently. Because FalkorDB serializes index access,
the shared `:Memory` index (one vector + one full-text query) is what keeps the
read path cheap — see README *Performance & scaling*.

### Coordination

- **Blackboard** (`coordination/blackboard.py`): `MERGE (b:Blackboard {scope,key})`
  with a monotonic `version`. A write may pass `expected_version`; the query
  updates only when it matches, so concurrent writers get an explicit conflict
  instead of a silent lost update.
- **Handoffs** (`coordination/handoff.py`): created as
  `(Agent)-[:SENT]->(Handoff)-[:TO]->(Agent)` with `priority` and `status`. Claim is
  a single atomic `MATCH ... WHERE status='pending' SET status='claimed'` — exactly
  one racing agent wins (the benchmark exercises this: 16 wins / 112 rejected for
  16 handoffs). Inboxes are ordered by priority then age.
- **Agent registry** (`coordination/registry.py`): the directory of agents plus
  live counts of handoffs sent/received.

## Why per-tenant graphs

Alternatives considered:

| Approach | Verdict |
|---|---|
| One graph + `tenant_id` filter everywhere | Rejected: one missed filter leaks data; indexes become shared and noisy. |
| One FalkorDB instance per tenant | Overkill for small tenants; still supported by sharding many tenants' graphs onto one instance. |
| **One graph per tenant** | Chosen: structural isolation, per-tenant indexes, and trivially movable/shardable. |

Because a tenant is just a graph named `weft_<tenant>`, scaling out is sharding
tenants across instances. The API and worker pool are stateless and can be
replicated behind a load balancer; the only state is in FalkorDB.

## Failure modes & guarantees

- **FalkorDB unreachable** → `StorageError` from `GraphClient`; API `/health`
  reports `degraded`; writes fail loudly (no silent data loss).
- **Worker crash mid-consolidation** → the job's lease expires and it is requeued;
  consolidation only touches un-marked messages, so re-running is safe.
- **Concurrent handoff claims** → atomic; exactly one winner, explicitly rejected
  losers (`ClaimError`).
- **Concurrent blackboard writes** → optimistic versioning; conflicts surfaced.
- **Duplicate enqueues** → `dedupe_key` MERGEs the job, so a flurry of messages in
  one session schedules a single consolidation.

## Extension points

- **New memory kind**: add a module under `memory/`, always label it `:Memory` with
  `embedding` + `search_text`, and add it to `retrieval/hybrid.py::_KIND_BY_LABEL`.
- **New extractor**: implement `Extractor.extract` and return an `Extraction`.
- **New provider**: implement `EmbeddingProvider`/`LLMProvider` and register it in
  the `build_*` factories.
- **New surface**: talk to `WeftHandle`; never reach into the graph directly from
  adapters.

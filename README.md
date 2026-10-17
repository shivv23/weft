# Weft

**Graph-native memory and coordination for multi-agent systems, built on FalkorDB.**

Weft gives a fleet of AI agents a *shared, durable, queryable memory* — and the
coordination primitives (a blackboard and first-class handoffs) they need to work
together. Everything lives in one FalkorDB graph per tenant: episodic, semantic
and procedural memory, conversation history, shared state, agent handoffs and the
background job queue. No Redis, no vector database, no SQS — **one graph**.

```
   User ──HAS_SESSION──▶ Session ──HAS_MESSAGE──▶ Message
                            ▲
                    PART_OF │            Episode ──MENTIONS──▶ Entity
                            │               │  ▲                  │
   Agent ──RUNS─────────────┘            PRECEDES  DERIVED_FROM    SUBJ
                                            │           │          ▼
   Agent ──SENT──▶ Handoff ──TO──▶ Agent  Procedure     └──────▶ Fact ──OBJ──▶ Entity
                     │                                              │
                     └──DURING──▶ Session        Episode ──ASSERTED─┘
```

---

## Why FalkorDB

Most "agent memory" stacks bolt a vector store onto a relational database and lose
the one thing agents actually need: **relationships**. Weft uses FalkorDB as the
single source of truth because it collapses five systems into one:

| Capability | Vector DB + SQL alternative | Weft on FalkorDB |
|---|---|---|
| Semantic recall | separate ANN index | HNSW via `db.idx.vector.queryNodes` **in the same graph** |
| Keyword recall | separate search engine | RediSearch via `db.idx.fulltext.queryNodes` |
| Relational reasoning | application-side joins | **graph traversal** (entity → mentions → episodes → facts) |
| Cross-session narrative | manual reconstruction | `PRECEDES` chains + `DERIVED_FROM` provenance |
| Multi-agent coordination | bespoke queue/locks | blackboard CAS + handoffs + durable `(:Job)` queue |
| Tenant isolation | row-level filters (easy to leak) | **one graph per tenant** (isolation is structural) |

The payoff shows up in recall: Weft reasons over *relationships the text never
spells out* — "renewals mentioned near Acme Corp's refund" finds the refund
episode through the entity graph, not string matching.

## What it does

- **Episodic memory** — timestamped events with salience decay, archival, and a
  `PRECEDES` chain so you can reconstruct the story of a session.
- **Semantic memory** — entities and reified `(subject)-[:SUBJ]->(Fact)-[:OBJ]->(object)`
  facts with evidence counts and confidence that strengthens on corroboration.
- **Procedural memory** — repeated successful action signatures are *mined* into
  reusable procedures, then ranked by their success rate.
- **Conversation history** — durable sessions/messages, consolidated
  asynchronously into memory by a background worker.
- **Shared blackboard** — agents coordinate through versioned, optionally
  compare-and-swapped key/value state scoped to `global`, `agent:*`, `user:*`,
  `session:*`.
- **Handoffs** — first-class a2a work transfer with atomic claim, priority inbox,
  references and provenance.
- **Hybrid recall with explanations** — vector + full-text + graph, fused by a
  transparent weighted score (recency, importance, frequency). Every item reports
  *why* it was returned, including the traversal path.
- **Context pack** — one call returns everything an agent needs at the start of a
  turn: relevant memories, pending handoffs, blackboard state, session summary.
- **Web console & MCP** — a dependency-free single-page console served by the API
  (at `/`) to explore recall, sessions, handoffs, blackboard and stats; plus the
  same capabilities exposed as MCP tools for any agent.

---

## Quickstart

### 1. Run the whole stack with Docker Compose

```bash
docker compose up --build
# FalkorDB on :6379 (Browser on :3000), API on :8080
```

### 2. Or run against a local FalkorDB (development)

```bash
docker run -d --name weft-falkordb -p 6379:6379 -p 3000:3000 \
  -e "FALKORDB_ARGS=MAX_QUEUED_QUERIES 1000 TIMEOUT 10000 RESULTSET_SIZE 10000 CACHE_SIZE 400 THREAD_COUNT 8 OMP_THREAD_COUNT 8" \
  falkordb/falkordb:latest

uv sync
uv run weft ping      # -> ok
```

### 3. See it think — the multi-agent demo

```bash
uv run weft demo
```

Two support sessions with three agents (triage → billing → technical). The demo
shows a handoff being created and atomically claimed, a blackboard wipe-out of an
open incident, **cross-session memory** (a *different* agent recalls what the
billing agent learned last week), procedure mining, hybrid recall explanations,
and tenant isolation — then cleans up after itself.

### 4. Hit the REST API

```bash
uv run weft api   # console at http://localhost:8080/ · OpenAPI at http://localhost:8080/docs
```

```bash
# write a session + messages
SID=$(curl -s localhost:8080/api/v1/sessions -H 'content-type: application/json' \
  -d '{"user_id":"u_nova","agent_id":"triage"}' | jq -r .id)

curl -s localhost:8080/api/v1/sessions/$SID/messages -H 'content-type: application/json' \
  -d '[{"role":"user","content":"Acme Corp was double-charged for the enterprise plan."},
       {"role":"assistant","content":"I have issued a refund of RF-9001 for 2400 USD."}]'

# consolidate (or let the worker do it) and recall across everything
curl -s localhost:8080/api/v1/sessions/$SID/consolidate -X POST
curl -s localhost:8080/api/v1/recall -H 'content-type: application/json' \
  -d '{"query":"why did Acme Corp get a refund?","k":5}' | jq
```

### 5. Use it as MCP tools from any agent

```bash
uv run weft mcp    # stdio MCP server
```

Tools: `weft_recall`, `weft_remember_episode`, `weft_add_fact`,
`weft_context_pack`, `weft_create_handoff`, `weft_claim_handoff`,
`weft_complete_handoff`, `weft_set_state`, `weft_get_state`, `weft_inbox`,
`weft_register_agent`. Same engine, same memory, as the REST API.

### 6. Or embed the library directly

```python
from weft.service import Weft

weft = Weft()
h = weft.for_tenant("acme")
await h.append_messages(session_id, [{"role": "user", "content": "..."}])
await h.consolidate_session(session_id)
res = await h.recall(query="what did we promise Acme Corp?", k=5)
```

Runnable end-to-end: [`examples/minimal.py`](examples/minimal.py).

---

## Benchmark

`uv run weft bench` drives a synthetic multi-agent workload against a **live**
FalkorDB and reports latency percentiles and error counts:

```
$ uv run weft bench --users 10 --agents 4 --writes 200 --queries 100 --concurrency 8 --tenants 2
[setup] seeded in 0.1s (662 episodes/s incl. embeddings)

  operation              count     ops/s   p50ms   p95ms   p99ms  err
  write_message            200     735.0    9.89   14.21   15.19    0
  recall                   100      10.5  771.95  888.44  922.56    0
  blackboard_cas            32     188.0    5.23    7.76    8.24    0
  handoff_claim            128     127.0    7.22   12.04   13.75    0   (16 won, 112 rejected atomically)
  cross_tenant_recall      100       1.8  547.78  688.81  961.10    0
  total ops=560  errors=0
```

A single hybrid recall is **~60–80 ms warm**; the table above runs at concurrency 8,
so its `p50` reflects queueing behind FalkorDB's serialized index access, not
per-query cost. See [Performance & scaling](#performance--scaling).

---

## Architecture

```
            ┌──────────────────────────── application ────────────────────────────┐
            │  REST API (FastAPI)      MCP server (FastMCP)      library / demo   │
            └───────────────┬───────────────────┬───────────────────────┬────────┘
                            │                   │                       │
                            └───────────────┬───┴───────────────────────┘
                                            ▼
                                     Weft / WeftHandle        (tenant-scoped facade)
             ┌──────────────┬──────────────┬──────────────┬──────────────┬─────────┐
             ▼              ▼              ▼              ▼              ▼         ▼
        memory/        retrieval/      coordination/   processing/    providers/
     episodic semantic  hybrid.py     blackboard       consolidator  embeddings
     procedural conv.   (vec+ft+graph) handoffs registry  queue      llm
             └──────────────┴──────────────┴──────────────┴──────────────┴─────────┘
                                            ▼
                     GraphClient  ──▶  FalkorDB   (one graph per tenant: weft_<tenant>)
```

The **ingest path** is write-fast and the **memory path** is asynchronous:

```
append_messages ─┐
                 ├─▶ (:Job session.consolidate)  ──▶  WorkerPool ──▶ ConsolidationService
blackboard /     │        (durable, in-graph)            │            ├─ extract (LLM or heuristic)
handoff writes ──┘                                       │            ├─ write entities/facts/episodes
                                                         │            ├─ link MENTIONS / PRECEDES / SUBJ
                                                         │            └─ mine procedures
                                                         └─ periodic maintenance: decay, archive, reap jobs
```

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
Graph model: [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) ·
Hackathon notes: [`docs/SUBMISSION.md`](docs/SUBMISSION.md).

---

## Providers & configuration

Weft runs **fully offline by default** (deterministic hashing embeddings +
rule-based extraction), so the demo and tests need no API keys. Switch providers
with environment variables:

```bash
WEFT_LLM_PROVIDER=openai            # none | openai | anthropic | ollama
WEFT_EMBEDDINGS_PROVIDER=openai     # hash | openai | ollama
WEFT_OPENAI_API_KEY=sk-...
```

Nothing is hardcoded: every host, port, weight, threshold and lifecycle window is
env-configurable. See [`.env.example`](.env.example) for the complete list. Key
knobs:

| Variable | Default | Meaning |
|---|---|---|
| `WEFT_FALKOR_HOST` / `WEFT_FALKOR_PORT` | `localhost` / `6379` | FalkorDB endpoint |
| `WEFT_EMBEDDINGS_PROVIDER` / `_DIM` | `hash` / `256` | embedding backend + vector dimension |
| `WEFT_LLM_PROVIDER` | `none` | extraction backend (heuristic when `none`) |
| `WEFT_WEIGHT_VECTOR/_FULLTEXT/_GRAPH/...` | see `.env.example` | recall fusion weights |
| `WEFT_EPISODE_HALF_LIFE_DAYS` | `14` | salience decay half-life |
| `WEFT_FACT_PROMOTE_EVIDENCE` | `3` | evidence hits before a fact is high-confidence |
| `WEFT_PROCEDURE_MIN_SUPPORT` | `2` | repeats before an action sequence becomes a procedure |
| `WEFT_WORKERS_IN_API` | `false` | run the worker pool inside the API (dev only) |
| `WEFT_API_KEYS` | *(empty)* | `key:tenant[:role]` CSV; empty disables auth |

> Embedding dimension is fixed per tenant graph at first use. Changing
> `WEFT_EMBEDDINGS_DIM` requires recreating that tenant's graph (or use
> `POST /api/v1/tenants/{tenant}/drop`).

---

## Performance & scaling

Measured against FalkorDB 8.x, single instance:

- **Writes parallelize** — message ingest and blackboard/handoff writes scale
  across connections (`~735 ops/s` of message writes with embeddings, p50 < 10 ms).
- **Index queries serialize** — FalkorDB currently executes ANN (`vector`) and
  RediSearch (`fulltext`) queries under a lock, so a hybrid recall is effectively
  serial at **~15 recalls/s per instance** regardless of `THREAD_COUNT`.
- Weft minimizes the number of index queries: every recallable memory node shares
  a `:Memory` label with one `embedding` and one `search_text` index, so a recall
  issues **one** vector query + **one** full-text query (not one per memory kind),
  and both are bounded by `LIMIT`. This cut warm recall latency by ~3× (≈180 ms → ≈65 ms).

**Horizontal scaling is built into the data model.** Because each tenant is its
own graph, the natural scale-out is to shard tenants across FalkorDB instances
(one instance per tenant-shard) and run the stateless API/worker pool in front of
them. There is no shared schema, no cross-tenant index and no coordination
metadata to migrate — moving a tenant is a graph copy.

Tuning baked into `docker-compose.yml` (server defaults are too small for an
agent fleet):

```
FALKORDB_ARGS: MAX_QUEUED_QUERIES 1000 TIMEOUT 10000 RESULTSET_SIZE 10000 CACHE_SIZE 400 THREAD_COUNT 8 OMP_THREAD_COUNT 8
```

`MAX_QUEUED_QUERIES` (default 25) is the one that bites first — a burst of agent
traffic is rejected with "Max pending queries exceeded" long before the instance
is saturated.

---

## Testing

```bash
uv run pytest tests/ -v          # integration tests against a live FalkorDB
uv run ruff check src/ tests/    # lint
uv run ruff format --check src/  # format
```

The suite provisions throwaway tenants, exercises isolation between them, and
drops them afterwards.

---

## Project layout

```
src/weft/
  service.py            Weft (process) + WeftHandle (per tenant) — the public façade
  config.py             env-driven Settings (no hardcoded values)
  models.py             Pydantic domain models (shared by API + MCP)
  graph/                connection.py (per-tenant routing), schema.py (indexes)
  memory/               conversation, episodic, semantic, procedural
  retrieval/hybrid.py   vector + full-text + graph fusion, explanations, MMR
  coordination/         blackboard (CAS), handoffs, agent registry
  processing/           extractors, consolidator, in-graph queue, worker pool
  providers/            embeddings + llm abstractions (hash/openai/anthropic/ollama)
  api/app.py            FastAPI REST surface
  mcp/server.py         MCP tool server
  demo.py               end-to-end multi-agent demo
  bench.py              concurrency / latency benchmark
  cli.py                weft <ping|api|worker|mcp|demo|bench>
docs/                   architecture, data model, submission notes
examples/               minimal.py — library quickstart
tests/                  integration tests
```

## License

MIT. Built for the **FalkorDB Graph Hacks** hackathon.

> Weft — *the graph-native memory and coordination layer for multi-agent systems.*

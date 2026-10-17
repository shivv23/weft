# Weft — Submission Notes

**Hackathon:** Graph Hacks: Context for AI Agents (WeMakeDevs × FalkorDB)
**Track:** 02 — Agent Memory and Coordination
**Dates:** Oct 15–18, 2026 · Submission closes Oct 18, 11:59 PM IST

> **One-liner.** Weft is a graph-native memory and coordination layer that gives a
> fleet of agents shared episodic/semantic/procedural memory, durable conversation
> history, a blackboard and first-class handoffs — all in a single FalkorDB graph
> per tenant, with explainable hybrid recall.

## Track 02 focus → where it lives in the code

| Track 02 focus | Weft implementation |
|---|---|
| **Episodic memory** | `memory/episodic.py` — timestamped `Episode` nodes, salience decay, archival, `PRECEDES` chains |
| **Semantic memory** | `memory/semantic.py` — `Entity` nodes + reified `(Entity)-[:SUBJ]->(Fact)-[:OBJ]->(Entity)` with evidence counting and confidence strengthening |
| **Procedural memory** | `memory/procedural.py` — `Procedure` nodes mined from repeated successful action signatures, ranked by success rate |
| **Persistent conversation history** | `memory/conversation.py` — durable `Session`/`Message` with incremental, idempotent consolidation |
| **Shared state between agents** | `coordination/blackboard.py` — versioned, CAS-capable `Blackboard` scoped to `global` / `agent:*` / `user:*` / `session:*` |
| **Handoffs between agents** | `coordination/handoff.py` — atomic claim, priority inbox, provenance via `:REFERENCES` |
| **High-concurrency reads/writes** | `processing/queue.py` (in-graph durable queue), `bench.py` (concurrency/latency benchmark), tuned FalkorDB args in `docker-compose.yml` |
| **Separate memory per user or tenant** | `graph/connection.py` — one graph (`weft_<tenant>`) per tenant; isolation is structural, not a filter |

## How FalkorDB shapes the product (it is not bolted on)

Weft would not exist in this form without FalkorDB:

1. **Vector + full-text + traversal in one store.** A recall fans out to HNSW
   (`db.idx.vector.queryNodes`), RediSearch (`db.idx.fulltext.queryNodes`) and a
   Cypher traversal of `MENTIONS`/`OBJ`/neighbor edges — all against the same
   graph. The graph channel surfaces memories the text never mentions (relations),
   which neither a vector DB nor a keyword index can do.
2. **Reified facts as graph structure.** `(Entity)-[:SUBJ]->(Fact)-[:OBJ]->(Entity)`
   lets facts carry confidence, evidence and bitemporal validity *and* be traversed
   from either endpoint — the basis for multi-hop reasoning and provenance.
3. **One graph per tenant.** FalkorDB's multigraph topology makes structural tenant
   isolation and horizontal sharding a data-model property, not an application
   concern.
4. **The database is the queue.** The durable `(:Job)` queue and the handoff/blackboard
   CAS primitives use FalkorDB's single-query atomicity instead of adding Redis/SQS.
5. **Provenance is native.** `DERIVED_FROM`, `ASSERTED`, `REFERENCES` and the graph
   channel's returned path let the agent *show how it reached an answer*.

## Demo script (what to look for)

`uv run weft demo` runs the full story end to end:

1. **Register agents** — triage, billing, technical (shared directory, per-tenant).
2. **Session 1** — a user reports a billing issue; triage creates a handoff to
   billing; billing **atomically claims** it, completes it, writes facts and an
   episode; a `user:/open_incidents` blackboard entry is written; the session is
   consolidated into memory.
3. **Session 2** — the *same user* returns to a **different agent** (technical).
   `context_pack` returns memories recalled from session 1 — including the refund
   episode reached through the `Acme Corp` entity — each with a **score breakdown
   and the traversal path** (`query → entity 'Acme Corp' → MENTIONS ← episode`).
4. **Procedure learning** — a repeated action signature is mined into a Procedure.
5. **Tenant isolation** — a `globex` tenant cannot see `acme`'s data.
6. Cleanup — demo tenants are dropped.

`uv run weft bench` proves the concurrency claims: parallel message writes,
hybrid recall under load, blackboard CAS and handoff-claim contention (with the
exactly-one-winner property), and cross-tenant isolation — reporting latency
percentiles and `errors=0`.

## Judging alignment

- **FalkorDB is primary and central** — it is the *only* datastore; removing it
  removes the product.
- **The graph shapes behaviour** — recall ranking, provenance, facts, procedures
  and coordination all derive from graph structure, not from a side index.
- **Deployable / reproducible** — `docker compose up --build` brings up FalkorDB +
  API + worker; a fully offline demo and test suite need no API keys. It is also
  **live at <https://superapp2026.duckdns.org/weft>** (opens the web console).
- **New work during the event** — the entire `weft` codebase was written for this
  submission.

## Live deployment

The stack is deployed (FalkorDB + API + worker) behind a Caddy TLS gateway:

| What | URL |
|---|---|
| Web console (landing) | `https://superapp2026.duckdns.org/weft/` |
| Base (`/weft` → 302 → console) | `https://superapp2026.duckdns.org/weft` |
| Health | `https://superapp2026.duckdns.org/weft/api/v1/health` |
| Swagger UI | `https://superapp2026.duckdns.org/weft/docs` |
| OpenAPI | `https://superapp2026.duckdns.org/weft/openapi.json` |

The `/weft/` landing page is a **single-file, dependency-free web console** (vanilla
JS, no CDN or build step — served from the API) with tabs for **Recall** (score
breakdown bars, matched channels and provenance paths per hit), **Sessions**
(create/load, append messages, timeline, consolidate), **Handoffs** (atomic
claim/complete/reject inbox, create, agent directory), **Blackboard** (scoped
compare-and-swap writes) and **Stats** (node counts, durable job queue, tenant
switcher). The tenant is selected with the `X-Tenant-Id` header, persisted
client-side.

Select a tenant with the `X-Tenant-Id` header (default `demo`). The seeded `demo`
tenant ships with a two-agent scenario (triage → billing-specialist) that has
already been consolidated and recalled:

```bash
curl -s -X POST https://superapp2026.duckdns.org/weft/api/v1/recall \
  -H 'content-type: application/json' \
  -d '{"query":"why was Acme Corp refunded?","k":5,"user_id":"u_nova"}'
```

## Run it in 30 seconds (offline, no keys)

```bash
docker compose up --build      # or the single `docker run` in the README
uv run weft demo               # the multi-agent story
uv run weft bench              # concurrency numbers
uv run pytest tests/ -v        # integration tests against live FalkorDB
```

## Media

- Demo video: _<add link>_
- Repository: <https://github.com/shivv23/weft>
- Blog post: _<add link>_

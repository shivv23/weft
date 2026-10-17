"""Concurrency & latency benchmark for Weft on FalkorDB.

Exercises the paths that matter for a shared agent platform:

* **write throughput** — many agents appending messages concurrently
* **recall latency**  — hybrid vector+fulltext+graph search under load
* **contention**      — optimistic blackboard CAS and handoff claims
* **isolation**       — mixed workloads across many tenants at once

Run::

    uv run weft bench
    uv run weft bench --users 200 --concurrency 64 --seed 500 --queries 400

Every knob has a flag, so the numbers in the README are reproducible.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
import uuid
from dataclasses import dataclass, field

from weft import Weft
from weft.config import get_settings
from weft.errors import ClaimError

WORDS = [
    "refund",
    "invoice",
    "payment",
    "charge",
    "account",
    "customer",
    "vpn",
    "tunnel",
    "latency",
    "deploy",
    "cluster",
    "node",
    "restart",
    "outage",
    "error",
    "timeout",
    "retry",
    "quota",
    "plan",
    "tier",
    "enterprise",
    "billing",
    "duplicate",
    "credit",
    "ledger",
    "webhook",
    "integration",
    "sso",
    "saml",
    "oauth",
]


@dataclass
class Stats:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    started: float = 0.0
    elapsed: float = 0.0

    def add(self, ms: float, ok: bool = True) -> None:
        self.latencies_ms.append(ms)
        if not ok:
            self.errors += 1

    def report(self) -> dict[str, float]:
        n = len(self.latencies_ms)
        if not n:
            return {
                "op": self.name,
                "count": 0,
                "throughput": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "errors": self.errors,
            }
        s = sorted(self.latencies_ms)
        p = lambda q: s[min(n - 1, int(q * n))]  # noqa: E731
        return {
            "op": self.name,
            "count": n,
            "throughput": round(n / self.elapsed, 1) if self.elapsed else 0.0,
            "p50": round(p(0.50), 2),
            "p95": round(p(0.95), 2),
            "p99": round(p(0.99), 2),
            "errors": self.errors,
        }


def _phrase(rng: random.Random, n: int = 10) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


async def _timed(coro, stats: Stats) -> object | None:
    t0 = time.perf_counter()
    try:
        out = await coro
        stats.add((time.perf_counter() - t0) * 1000, ok=True)
        return out
    except Exception as exc:
        stats.add((time.perf_counter() - t0) * 1000, ok=False)
        print(f"    ! {stats.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


async def _bounded(tasks: list, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def run(t):
        async with sem:
            return await t

    await asyncio.gather(*(run(t) for t in tasks))


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weft bench")
    parser.add_argument("--users", type=int, default=50, help="distinct users (sessions)")
    parser.add_argument("--agents", type=int, default=8, help="distinct agents")
    parser.add_argument(
        "--seed", type=int, default=300, help="episodes seeded per tenant before measuring"
    )
    parser.add_argument("--writes", type=int, default=600, help="timed message writes")
    parser.add_argument("--queries", type=int, default=300, help="timed recall queries")
    parser.add_argument("--concurrency", type=int, default=32, help="in-flight operations")
    parser.add_argument(
        "--tenants", type=int, default=3, help="tenants exercised concurrently for isolation"
    )
    parser.add_argument("--keep", action="store_true", help="keep benchmark tenants")
    args = parser.parse_args(argv)

    rng = random.Random(1337)
    settings = get_settings()
    weft = Weft(settings)
    if not await weft.ping():
        print("FalkorDB unreachable. Start it on 6379 first.", file=sys.stderr)
        return 1

    run_id = uuid.uuid4().hex[:6]
    tenants = [f"bench_{run_id}_{i}" for i in range(args.tenants)]
    handles = [weft.for_tenant(t) for t in tenants]
    print("\nWeft concurrency benchmark")
    print(
        f"  tenants={args.tenants}  users/tenant={args.users}  agents={args.agents}  "
        f"concurrency={args.concurrency}  embedder={weft.embedder.name}  "
        f"llm={settings.llm_provider}"
    )
    print(f"  falkordb={settings.falkor_host}:{settings.falkor_port}")

    # ------------------------------------------------------------------ setup
    agents = [f"agent{i}" for i in range(args.agents)]
    sessions: list[tuple[object, str]] = []  # (handle, session_id)
    for h in handles:
        for a in agents:
            await h.register_agent(id=a, name=f"Agent {a}", capabilities=["work"])
        for u in range(args.users):
            user = f"user{u}"
            s = await h.create_session(user_id=user, agent_id=rng.choice(agents))
            sessions.append((h, s.id))

    # seed memory so recall has something to rank
    print(f"\n[setup] seeding {args.seed * len(handles)} episodes ...")
    t0 = time.perf_counter()
    seed_tasks = []
    for h in handles:
        for _i in range(args.seed):
            seed_tasks.append(
                h.add_episode(
                    title=_phrase(rng, 4),
                    summary=_phrase(rng, 24),
                    importance=round(rng.random(), 2),
                    user_id=f"user{rng.randrange(args.users)}",
                    entity_names=[
                        rng.choice(
                            ["Acme Corp", "Globex", "INV-2043", "VPN", "ACCT-7781", "Starter plan"]
                        )
                    ],
                )
            )
    await _bounded(seed_tasks, args.concurrency)
    seed_s = time.perf_counter() - t0
    print(
        f"[setup] seeded in {seed_s:.1f}s "
        f"({len(seed_tasks) / seed_s:.0f} episodes/s incl. embeddings)"
    )

    reports: list[dict[str, float]] = []

    # ------------------------------------------------------- write throughput
    print(f"\n[1/4] concurrent message writes ({args.writes} ops) ...")
    w = Stats("write_message")
    w.started = time.perf_counter()
    tasks = []
    for _i in range(args.writes):
        h, sid = sessions[rng.randrange(len(sessions))]
        tasks.append(
            _timed(
                h.append_messages(
                    sid,
                    [
                        {"role": "user", "content": _phrase(rng, 16)},
                        {"role": "assistant", "content": _phrase(rng, 16)},
                    ],
                ),
                w,
            )
        )
    await _bounded(tasks, args.concurrency)
    w.elapsed = time.perf_counter() - w.started
    reports.append(w.report())

    # --------------------------------------------------------- recall latency
    print(f"[2/4] concurrent hybrid recall ({args.queries} queries) ...")
    r = Stats("recall")
    r.started = time.perf_counter()
    tasks = []
    for _i in range(args.queries):
        h = handles[rng.randrange(len(handles))]
        tasks.append(
            _timed(
                h.recall(query=_phrase(rng, 5), k=8, user_id=f"user{rng.randrange(args.users)}"),
                r,
            )
        )
    await _bounded(tasks, args.concurrency)
    r.elapsed = time.perf_counter() - r.started
    reports.append(r.report())

    # ------------------------------------------------------------- contention
    print(f"[3/4] blackboard CAS + handoff claims ({args.concurrency * 4} ops) ...")
    b = Stats("blackboard_cas")
    hd = Stats("handoff_claim")
    k = args.concurrency * 4
    tasks = []
    for i in range(k):
        h = handles[rng.randrange(len(handles))]
        tasks.append(
            _timed(
                h.set_state(scope="global", key=f"counter{i % 4}", value=i, updated_by="bench"), b
            )
        )
    await _bounded(tasks, args.concurrency)
    b.elapsed = max(0.001, sum(b.latencies_ms) / 1000)
    reports.append(b.report())

    # handoff contention: many agents race to claim the same handoff. Exactly
    # one claim per handoff may win; the rest are expected ClaimErrors (proof
    # that the contended edge is claimed atomically), so they are not failures.
    race_h = handles[0]
    n_handoffs = 16
    expected_wins = 0
    contended = 0

    async def claim(handoff_id: str, agent_id: str) -> None:
        nonlocal expected_wins, contended
        t0 = time.perf_counter()
        try:
            await race_h.claim_handoff(handoff_id=handoff_id, agent_id=agent_id)
            expected_wins += 1
            hd.add((time.perf_counter() - t0) * 1000, ok=True)
        except ClaimError:
            contended += 1
            hd.add((time.perf_counter() - t0) * 1000, ok=True)
        except Exception:
            hd.add((time.perf_counter() - t0) * 1000, ok=False)

    race_tasks = []
    for i in range(n_handoffs):
        hf = await race_h.create_handoff(
            from_agent="agent0",
            to_agent="agent1",
            title=f"race {i}",
            reason="contention test",
        )
        for a in range(args.concurrency):
            race_tasks.append(claim(hf.id, f"agent{a}"))
    await _bounded(race_tasks, args.concurrency)
    hd.elapsed = max(0.001, sum(hd.latencies_ms) / 1000)
    reports.append(hd.report())
    print(
        f"       (contention: {expected_wins} claims won, {contended} rejected atomically, "
        f"expected wins={n_handoffs})"
    )

    # ------------------------------------------------------------- isolation
    print("[4/4] mixed-tenant isolation workload ...")
    iso = Stats("cross_tenant_recall")
    tasks = []
    for _i in range(args.queries):
        h = handles[rng.randrange(len(handles))]
        tasks.append(_timed(h.recall(query=_phrase(rng, 4), k=5), iso))
    await _bounded(tasks, args.concurrency)
    iso.elapsed = max(0.001, sum(iso.latencies_ms) / 1000)
    reports.append(iso.report())

    # ---------------------------------------------------------------- results
    print("\n" + "=" * 78)
    print(
        f"  {'operation':<20} {'count':>7} {'ops/s':>9} "
        f"{'p50ms':>7} {'p95ms':>7} {'p99ms':>7} {'err':>4}"
    )
    print("-" * 78)
    for rep in reports:
        print(
            f"  {rep['op']:<20} {rep['count']:>7} {rep['throughput']:>9.1f} "
            f"{rep['p50']:>7.2f} {rep['p95']:>7.2f} {rep['p99']:>7.2f} {rep['errors']:>4}"
        )
    print("=" * 78)

    total_ops = sum(rep["count"] for rep in reports)
    total_err = sum(rep["errors"] for rep in reports)
    print(f"  total ops={total_ops}  errors={total_err}")

    if not args.keep:
        for t in tenants:
            await weft.drop_tenant(t)
        print("  (benchmark tenants dropped; use --keep to inspect them)")
    else:
        print(f"  kept tenants: {', '.join(tenants)}")
    await weft.aclose()
    print()
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

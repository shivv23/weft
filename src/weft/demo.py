"""Weft multi-agent demo: cross-session memory + handoffs between agents.

Scenario (single FalkorDB instance, two tenants):

  Tenant "acme"   — a user interacts with a Triage agent in session 1
                    (billing dispute). The triage agent stores memory,
                    writes shared blackboard state and hands off to a Billing
                    agent with the full context.

  Tenant "globex" — an unrelated user in a different tenant, to show isolation.

  Session 2 (new session, same tenant) — the user returns with a technical
                    question. A Tech agent runs a context-pack query and gets
                    the billing dispute memory, semantic facts (customer tier,
                    account id), the blackboard incident flag and the pending
                    handoff trail — without re-asking anything.

Run:  uv run python -m weft.demo
Env:  WEFT_LLM_PROVIDER=none (default) runs fully offline with the heuristic
      extractor. Set openai/anthropic/ollama for high-quality extraction.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any

from weft import Weft
from weft.config import get_settings


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _kv(rows: list[tuple[str, Any]]) -> None:
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")


def _print_memories(items: list[Any]) -> None:
    if not items:
        print("    (no memories matched)")
        return
    for m in items:
        channels = ",".join(m.matched_via)
        print(f"    [{m.kind:9s}] {m.text[:90]}")
        print(
            f"      score={m.score:.3f} via={channels} "
            f"vec={m.scores.vector:.2f} ft={m.scores.fulltext:.2f} "
            f"graph={m.scores.graph:.2f} rec={m.scores.recency:.2f}"
        )
        paths = m.explanation.get("paths", []) if m.explanation else []
        if paths:
            print(f"      path: {paths[0]}")


async def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    settings = get_settings()
    weft = Weft(settings)
    ok = await weft.ping()
    if not ok:
        print("FalkorDB is not reachable. Start it with:")
        print("  docker run -d -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest")
        return 1

    tenant = f"demo_{uuid.uuid4().hex[:6]}"
    other = f"other_{uuid.uuid4().hex[:6]}"
    acme = weft.for_tenant(tenant)
    globex = weft.for_tenant(other)
    print(f"\nFalkorDB: OK   tenants: acme={tenant}, globex={other}")
    print(
        f"embedder={weft.embedder.name}  llm={settings.llm_provider}  (offline-friendly defaults)"
    )

    # ------------------------------------------------------------------
    _banner("0. Register agents (shared directory, per-tenant graph)")
    for aid, name, kind, caps in [
        ("triage", "Triage Agent", "router", ["classify", "route", "summarise"]),
        ("billing", "Billing Agent", "specialist", ["refunds", "invoices", "payments"]),
        ("tech", "Technical Support Agent", "specialist", ["debug", "logs", "config"]),
    ]:
        await acme.register_agent(id=aid, name=name, kind=kind, description=name, capabilities=caps)
    agents = await acme.list_agents()
    for a in agents:
        print(f"  - {a.id:8s} {a.name:26s} caps={','.join(a.capabilities)}")

    # ------------------------------------------------------------------
    _banner("1. Session 1 — user reports a billing issue to Triage")
    s1 = await acme.create_session(user_id="u_nova", agent_id="triage", title="Billing dispute")
    print(f"  session: {s1.id}")

    turns = [
        {
            "role": "user",
            "content": (
                "Hi, I'm Nova from Acme Corp. We were double charged on invoice "
                "INV-2043 last month and need a refund. Our account id is ACCT-7781."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Sorry to hear that. I've logged the double-charge on INV-2043 for "
                "Acme Corp (ACCT-7781). I'm routing you to our Billing specialist."
            ),
            "meta": {
                "actions": ["classify_billing", "lookup_invoice", "create_ticket"],
                "outcome": "success",
            },
        },
    ]
    await acme.append_messages(s1.id, turns)
    print("  wrote 2 messages (user + assistant)")

    # Triage decides to hand off to Billing, passing structured context.
    h1 = await acme.create_handoff(
        from_agent="triage",
        to_agent="billing",
        title="Refund double-charge INV-2043",
        reason="Customer reports duplicate charge on invoice INV-2043; refund required.",
        context={
            "customer": "Acme Corp",
            "account_id": "ACCT-7781",
            "invoice": "INV-2043",
            "issue": "double_charge",
            "requested_action": "refund",
        },
        priority=3,
        session_id=s1.id,
        user_id="u_nova",
    )
    print(f"  handoff created: {h1.id}  triage -> billing  priority={h1.priority}")

    # Shared blackboard state for the incident.
    st = await acme.set_state(
        scope="user:u_nova",
        key="open_incidents",
        value=[{"invoice": "INV-2043", "type": "double_charge", "status": "open"}],
        updated_by="triage",
    )
    print(f"  blackboard user:u_nova/open_incidents v{st.version} written")

    # Billing agent picks up the handoff.
    inbox = await acme.handoff_inbox(agent_id="billing")
    claimed = await acme.claim_handoff(handoff_id=inbox[0].id, agent_id="billing")
    print(f"  billing claimed handoff: status={claimed.status}")

    # Billing processes and completes.
    await acme.add_fact(
        subject="Acme Corp",
        predicate="account_id",
        object="ACCT-7781",
        confidence=0.95,
        subject_type="Organization",
        object_type="Artifact",
        source="billing_agent",
    )
    await acme.add_fact(
        subject="Acme Corp",
        predicate="tier",
        object="Enterprise",
        confidence=0.9,
        subject_type="Organization",
        source="billing_agent",
    )
    done = await acme.complete_handoff(
        handoff_id=claimed.id,
        agent_id="billing",
        result={
            "refund_id": "RF-9001",
            "amount": 2400,
            "currency": "USD",
            "note": "Refund issued for duplicate charge on INV-2043",
        },
    )
    print(f"  billing completed handoff: status={done.status}")
    print("  billing stored 2 semantic facts about Acme Corp")

    # Billing stores an episode of the resolution.
    await acme.add_episode(
        kind="milestone",
        title="Refund RF-9001 issued",
        summary="Billing issued refund RF-9001 for 2400 USD to Acme Corp for the "
        "duplicate charge on invoice INV-2043. Account ACCT-7781, Enterprise tier.",
        importance=0.9,
        session_id=s1.id,
        user_id="u_nova",
        agent_id="billing",
        entity_names=["Acme Corp", "INV-2043", "ACCT-7781"],
    )
    print("  billing stored an episodic memory of the resolution")

    # Consolidate session 1 (background worker normally does this; here inline).
    stats = await acme.consolidate_session(s1.id)
    print(
        f"  consolidated session 1: {stats['episodes']} episodes, "
        f"{stats['entities']} entities, {stats['facts']} facts (extractor={stats['extractor']})"
    )

    # ------------------------------------------------------------------
    _banner("2. Session 2 — same user returns, different agent, new problem")
    s2 = await acme.create_session(user_id="u_nova", agent_id="tech", title="VPN problem")
    print(f"  session: {s2.id}")

    pack = await acme.context_pack(
        agent_id="tech",
        user_id="u_nova",
        session_id=s2.id,
        query="Nova from Acme Corp says VPN stopped working after the refund",
        k=6,
    )
    print(f"  context_pack for tech agent  ({pack.took_ms:.1f}ms)")
    print("  --- recalled memories ---")
    _print_memories(pack.memories)
    print("  --- blackboard state visible to tech ---")
    for e in pack.blackboard:
        print(f"    {e.scope}/{e.key} v{e.version}: {str(e.value)[:80]}")
    print("  --- pending handoffs in tech's inbox ---")
    for h in pack.pending_handoffs:
        print(f"    {h.id} from={h.from_agent} title={h.title!r}")
    if not pack.pending_handoffs:
        print("    (none — billing already completed its handoff)")

    # Tech answers with full context, stores new memory.
    await acme.append_messages(
        s2.id,
        [
            {
                "role": "user",
                "content": (
                    "Hi again — Nova from Acme Corp. VPN stopped working this morning. "
                    "Can you check if the refund work affected anything?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Hi Nova. I can see your Acme Corp account (ACCT-7781, Enterprise) and "
                    "last month's INV-2043 refund (RF-9001). The refund shouldn't touch VPN. "
                    "Checking tunnel logs now."
                ),
                "meta": {
                    "actions": ["recall_context", "check_vpn_tunnel", "tail_gateway_logs"],
                    "outcome": "success",
                },
            },
        ],
    )
    await acme.consolidate_session(s2.id)

    # ------------------------------------------------------------------
    _banner("3. Procedural memory — repeat a successful refund flow")
    refund_proc = await acme.add_procedure(
        name="issue_refund",
        description="Standard refund flow for duplicate charges",
        steps=[
            "verify_invoice",
            "confirm_duplicate",
            "issue_refund",
            "notify_customer",
            "update_ledger",
        ],
        tags=["billing", "refund"],
    )
    await acme.procedure_feedback(procedure_id=refund_proc, success=True, duration_ms=180_000)
    await acme.procedure_feedback(procedure_id=refund_proc, success=True, duration_ms=165_000)
    proc = await acme.get_procedure(procedure_id=refund_proc)
    print(
        f"  procedure '{proc.name}'  success={proc.success_count} failure={proc.failure_count} "
        f"avg_ms={proc.avg_duration_ms:.0f}"
    )
    print(f"  steps: {' -> '.join(proc.steps)}")

    # ------------------------------------------------------------------
    _banner("4. Hybrid recall — explainability for the agent's answer")
    rec = await acme.recall(
        query="What do we know about Acme Corp's account and refund history?",
        k=5,
        user_id="u_nova",
        include_explanation=True,
    )
    print(f"  query: {rec.query!r}   ({rec.took_ms:.1f}ms)")
    _print_memories(rec.items)

    # ------------------------------------------------------------------
    _banner("5. Tenant isolation — globex cannot see acme's memory")
    await globex.add_fact(subject="Globex", predicate="plan", object="Starter")
    globex_facts = await globex.list_facts()
    acme_facts = await acme.list_facts()
    globex_recall = await globex.recall(query="Acme Corp refund", k=3)
    print(f"  globex facts: {[f.text for f in globex_facts]}")
    print(f"  acme facts:   {[f.text for f in acme_facts]}")
    print(
        f"  globex recall for 'Acme Corp refund' -> {len(globex_recall.items)} items "
        f"(must not include acme's memories)"
    )

    # ------------------------------------------------------------------
    _banner("6. Timeline — the story behind an answer")
    timeline = await acme.session_timeline(s1.id)
    print(f"  session {s1.id}")
    for ep in timeline["episodes"]:
        print(f"    episode {ep['id'][:8]}…  [{ep['kind']}]  {ep['title']}")
    for h in timeline["handoffs"]:
        print(
            f"    handoff  {h['id'][:8]}…  {h['from']} -> {h['to']}  [{h['status']}]  {h['title']}"
        )

    st_final = await acme.stats()
    _banner("7. Graph stats (tenant acme)")
    _kv(sorted(st_final.counts.items()))

    # ------------------------------------------------------------------
    _banner("Done")
    print("  Every memory above is a node/edge in FalkorDB — query it yourself:")
    print("    docker exec -it weft-falkordb redis-cli")
    print(f'    GRAPH.QUERY weft_{tenant} "MATCH (e:Episode) RETURN e.title, e.importance"')
    print("  To run the API + MCP server instead of the in-process demo:")
    print("    uv run weft api      # REST on :8080")
    print("    uv run weft mcp      # MCP stdio server")

    # cleanup
    await weft.drop_tenant(tenant)
    await weft.drop_tenant(other)
    await weft.aclose()
    print("\n  (demo tenants dropped)\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

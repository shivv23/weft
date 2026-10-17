"""Basic end-to-end integration tests against FalkorDB.

These run only if FalkorDB is reachable on localhost:6379. They exercise:
- tenant isolation (separate graphs)
- episodic/semantic/procedural memory
- blackboard shared state + CAS
- handoffs (claim/complete)
- hybrid recall (vector + full-text + graph) + explainability
- consolidation (ingest -> extract -> link)
- high-concurrency write/read smoke test
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from weft import Weft
from weft.errors import ClaimError, ConflictError

FALKOR_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
CAN_RUN = False

try:
    import redis as r

    client = r.Redis(host=FALKOR_HOST, port=FALKOR_PORT, decode_responses=True)
    if client.ping():
        CAN_RUN = True
    client.close()
except Exception:
    CAN_RUN = False


pytestmark = pytest.mark.skipif(not CAN_RUN, reason="FalkorDB not reachable")


async def _tenant() -> str:
    return f"t_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_tenant_isolation():
    w = Weft()
    t1 = await _tenant()
    t2 = await _tenant()
    h1 = w.for_tenant(t1)
    h2 = w.for_tenant(t2)

    await h1.add_fact(subject="X", predicate="owns", object="Y")
    await h2.add_fact(subject="P", predicate="owns", object="Q")

    facts1 = await h1.list_facts()
    facts2 = await h2.list_facts()
    assert len(facts1) == 1
    assert len(facts2) == 1
    assert facts1[0].subject == "X"
    assert facts2[0].subject == "P"

    await w.drop_tenant(t1)
    await w.drop_tenant(t2)
    await w.aclose()


@pytest.mark.asyncio
async def test_conversation_consolidation_and_recall():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    sess = await h.create_session(user_id="u1", agent_id="a1", title="test")
    await h.append_messages(
        sess.id,
        [
            {
                "role": "user",
                "content": (
                    "The payment gateway has an outage. Refund charge 783 for customer ACME."
                ),
            },
            {
                "role": "assistant",
                "content": "Logged incident. Routing to on-call.",
                "meta": {"actions": ["log", "route"], "outcome": "success"},
            },
        ],
    )
    res = await h.consolidate_session(sess.id)
    assert res["episodes"] >= 1
    assert res["messages"] == 2

    rec = await h.recall(query="payment outage refund ACME", k=5)
    assert len(rec.items) >= 1
    assert any(i.explanation for i in rec.items)
    assert rec.items[0].matched_via

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_blackboard_cas_and_shared_state():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    e = await h.set_state(
        scope="session:s1", key="goal", value={"status": "planning"}, updated_by="a1"
    )
    assert e.version == 1

    got = await h.get_state(scope="session:s1", key="goal")
    assert got and got.value["status"] == "planning"

    e2 = await h.set_state(
        scope="session:s1",
        key="goal",
        value={"status": "running"},
        updated_by="a2",
        expected_version=1,
    )
    assert e2.version == 2

    with pytest.raises(ConflictError):
        await h.set_state(
            scope="session:s1", key="goal", value={"x": 1}, updated_by="a1", expected_version=1
        )

    scope = await h.scope_state(scopes=["session:s1", "global"])
    assert len(scope) >= 1

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_handoffs_claim_complete():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    await h.register_agent(id="a1", name="Planner")
    await h.register_agent(id="a2", name="Coder")

    sess = await h.create_session(user_id="u1")
    hnd = await h.create_handoff(
        from_agent="a1",
        to_agent="a2",
        title="Fix bug",
        reason="payment flow regression",
        context={"file": "pay.py"},
        session_id=sess.id,
        ref_ids=[],
    )
    assert hnd.status == "pending"

    inbox = await h.handoff_inbox(agent_id="a2")
    assert len(inbox) == 1

    claimed = await h.claim_handoff(handoff_id=hnd.id, agent_id="a2")
    assert claimed.status == "claimed"

    with pytest.raises(ClaimError):
        await h.claim_handoff(handoff_id=hnd.id, agent_id="a1")

    done = await h.complete_handoff(handoff_id=hnd.id, agent_id="a2", result={"pr": "123"})
    assert done.status == "completed"

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_semantic_procedural_and_provenance():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    fid = await h.add_fact(subject="ACME", predicate="tier", object="gold")
    facts = await h.list_facts()
    assert any(f.id == fid for f in facts)

    pid = await h.add_procedure(
        name="deploy_payment",
        description="deploy payment service",
        steps=["build", "test", "canary", "promote"],
        tags=["payments"],
    )
    proc = await h.get_procedure(pid)
    assert proc and proc.name == "deploy_payment"

    proc2 = await h.procedure_feedback(procedure_id=pid, success=True, duration_ms=1200)
    assert proc2.success_count == 1

    # build a graph: create episode that mentions entities; traversal can be queried
    eid = await h.add_episode(
        title="onboard acme",
        summary="acme onboarded, using grafana for dashboards",
        importance=0.7,
        entity_names=["ACME", "grafana"],
    )
    ep = await h.get_episode(eid)
    assert ep and ep.title == "onboard acme"

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_concurrent_writes_no_races():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    async def writer(i: int):
        s = await h.create_session(user_id=f"u{i}")
        await h.append_messages(
            s.id,
            [
                {"role": "user", "content": f"task {i}"},
                {"role": "assistant", "content": f"done {i}"},
            ],
        )
        await h.consolidate_session(s.id)
        return s.id

    ids = await asyncio.gather(*[writer(i) for i in range(10)])
    assert len(ids) == 10

    for sid in ids:
        msgs = await h.list_messages(sid)
        assert len(msgs) == 2

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_context_pack_highlights_relevant_memories():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    s = await h.create_session(user_id="u1", agent_id="agent-bot")
    await h.append_messages(
        s.id,
        [
            {"role": "user", "content": "We lost logs for tenant acme last night."},
            {
                "role": "assistant",
                "content": "Checking log pipeline for acme.",
                "meta": {"actions": ["check_pipelines"], "outcome": "success"},
            },
        ],
    )
    await h.consolidate_session(s.id)
    await h.create_handoff(
        from_agent="agent-bot",
        to_agent="sre-bot",
        title="Investigate acme logs",
        reason="customer escalated",
        session_id=s.id,
    )

    pack = await h.context_pack(
        agent_id="sre-bot", user_id="u1", session_id=s.id, query="acme logs lost"
    )
    assert pack.agent_id == "sre-bot"
    assert len(pack.pending_handoffs) >= 1
    assert any(m.score > 0 for m in pack.memories)

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_message_sequence_agent_id_and_scoped_recall():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    sess = await h.create_session(user_id="u1", agent_id="triage", title="billing")
    assert sess.agent_id == "triage"

    first = await h.append_messages(
        sess.id,
        [
            {"role": "user", "content": "Acme was double charged."},
            {"role": "assistant", "content": "Issued refund RF-9001."},
        ],
    )
    assert sorted(m.seq for m in first) == [1, 2]

    second = await h.append_messages(
        sess.id,
        [
            {"role": "user", "content": "Thanks."},
            {"role": "user", "content": "Please confirm."},
        ],
    )
    assert sorted(m.seq for m in second) == [3, 4]

    await h.consolidate_session(sess.id)

    scoped = await h.recall(query="refund Acme", k=5, agent_id="triage")
    assert any(i.kind == "episode" for i in scoped.items)

    other = await h.recall(query="refund Acme", k=5, agent_id="someone-else")
    assert not any(i.kind == "episode" for i in other.items)

    await w.drop_tenant(t)
    await w.aclose()


@pytest.mark.asyncio
async def test_recall_explanation_contains_channels_and_paths():
    w = Weft()
    t = await _tenant()
    h = w.for_tenant(t)

    await h.add_fact(subject="DB1", predicate="hosts", object="billing")
    await h.add_episode(
        title="DB1 restarted",
        summary="billing db restarted by ops",
        entity_names=["DB1", "billing"],
    )
    rec = await h.recall(query="billing DB1 restarted", k=3, include_explanation=True)
    assert rec.items
    for it in rec.items:
        assert it.matched_via
        assert it.explanation.get("matched_via")
        assert "score_contributions" in it.explanation

    await w.drop_tenant(t)
    await w.aclose()

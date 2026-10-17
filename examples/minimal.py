"""Minimal library usage — no API server, no LLM keys required.

    uv run python examples/minimal.py

Shows the full loop: write conversation history, consolidate it into memory,
then recall across sessions with an explanation of *why* each memory surfaced.

Requires a reachable FalkorDB (see README quickstart).
"""

from __future__ import annotations

import asyncio

from weft.service import Weft

TENANT = "example_minimal"


async def main() -> None:
    weft = Weft()
    if not await weft.ping():
        raise SystemExit("FalkorDB is not reachable (see README quickstart)")

    # Always start from a clean tenant so the example is deterministic.
    await weft.drop_tenant(TENANT)
    h = weft.for_tenant(TENANT)

    await h.register_agent(id="support", name="Support Agent")

    # --- Session 1: raw conversation history -------------------------------
    s1 = await h.create_session(user_id="u_nova", agent_id="support", title="Billing")
    await h.append_messages(
        s1.id,
        [
            {"role": "user", "content": "Acme Corp was double charged for the enterprise plan."},
            {"role": "assistant", "content": "Apologies — I issued refund RF-9001 for 2400 USD."},
        ],
    )

    # The API writes raw history fast; consolidation turns it into memory.
    stats = await h.consolidate_session(s1.id)
    print("consolidated:", stats)

    # --- Session 2: a NEW session recalls what session 1 learned -----------
    s2 = await h.create_session(user_id="u_nova", agent_id="support", title="Follow-up")
    response = await h.recall(query="what did we promise Acme Corp?", k=4)

    print(f"\nrecall took {response.took_ms} ms for {len(response.items)} items:")
    for item in response.items:
        print(f"  [{item.kind:9}] score={item.score:.3f}  {item.text.splitlines()[0][:70]}")
        for path in item.explanation.get("paths", []):
            print(f"      path: {path}")

    # Direct memory writes are also part of the public API.
    await h.add_fact(
        subject="Acme Corp",
        predicate="account_tier",
        object="enterprise",
        confidence=0.9,
    )
    facts = await h.list_facts(subject="Acme Corp")
    print("\nfacts about Acme Corp:")
    for f in facts:
        print(f"  {f.subject} {f.predicate} {f.object} (confidence {f.confidence})")

    print(f"\nsession timeline: {(await h.session_timeline(s2.id))['session_id']}")
    print("stats:", (await h.stats()).counts)

    await weft.drop_tenant(TENANT)  # tidy up
    await weft.aclose()


if __name__ == "__main__":
    asyncio.run(main())

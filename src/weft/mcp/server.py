"""MCP server: exposes Weft's memory/coordination as tools.

The Model Context Protocol (MCP) treats tools as callable actions. Weft
exposes a minimal but useful set: recall, remember_episode, add_fact,
context_pack, handoff, set_state, get_state, claim_handoff, complete_handoff.
Each tool call carries tenant context (header or meta). This is the same
graph engine as the REST API — shared state, shared memory.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from weft.config import get_settings
from weft.service import Weft

_settings = get_settings()
_weft = Weft()


def _handle_for(tenant: str = "demo") -> Any:
    return _weft.for_tenant(tenant)


mcp = MCPServer("weft", dependencies=[])


@mcp.tool("weft_ping")
async def ping() -> dict[str, Any]:
    return {"ok": await _weft.ping(), "tenants": await _weft.list_tenants()}


@mcp.tool("weft_recall")
async def recall(
    query: str,
    k: int = 8,
    tenant: str = "demo",
    user_id: str | None = None,
    session_id: str | None = None,
    kinds: list[str] | None = None,
    include_explanation: bool = True,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.recall(
        query=query,
        k=k,
        user_id=user_id,
        session_id=session_id,
        kinds=kinds,
        include_explanation=include_explanation,
    )
    return res.model_dump()


@mcp.tool("weft_remember_episode")
async def remember_episode(
    title: str,
    summary: str,
    tenant: str = "demo",
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    importance: float = 0.5,
    entity_names: list[str] | None = None,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    eid = await h.add_episode(
        title=title,
        summary=summary,
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        importance=importance,
        entity_names=entity_names,
    )
    return {"episode_id": eid}


@mcp.tool("weft_add_fact")
async def add_fact(
    subject: str,
    predicate: str,
    object: str,
    tenant: str = "demo",
    confidence: float = 0.7,
    source_episode_id: str | None = None,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    fid = await h.add_fact(
        subject=subject,
        predicate=predicate,
        object=object,
        confidence=confidence,
        source_episode_id=source_episode_id,
    )
    return {"fact_id": fid}


@mcp.tool("weft_context_pack")
async def context_pack(
    agent_id: str,
    tenant: str = "demo",
    user_id: str | None = None,
    session_id: str | None = None,
    query: str = "",
    k: int = 8,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.context_pack(
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        query=query,
        k=k,
    )
    return res.model_dump()


@mcp.tool("weft_create_handoff")
async def create_handoff(
    from_agent: str,
    to_agent: str,
    tenant: str = "demo",
    title: str = "",
    reason: str = "",
    context: dict[str, Any] | None = None,
    priority: int = 5,
    session_id: str | None = None,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.create_handoff(
        from_agent=from_agent,
        to_agent=to_agent,
        title=title,
        reason=reason,
        context=context,
        priority=priority,
        session_id=session_id,
    )
    return res.model_dump()


@mcp.tool("weft_claim_handoff")
async def claim_handoff(handoff_id: str, agent_id: str, tenant: str = "demo") -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.claim_handoff(handoff_id=handoff_id, agent_id=agent_id)
    return res.model_dump()


@mcp.tool("weft_complete_handoff")
async def complete_handoff(
    handoff_id: str,
    agent_id: str,
    tenant: str = "demo",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.complete_handoff(handoff_id=handoff_id, agent_id=agent_id, result=result)
    return res.model_dump()


@mcp.tool("weft_set_state")
async def set_state(
    scope: str,
    key: str,
    value: Any,
    tenant: str = "demo",
    updated_by: str | None = None,
    expected_version: int | None = None,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.set_state(
        scope=scope,
        key=key,
        value=value,
        updated_by=updated_by,
        expected_version=expected_version,
    )
    return res.model_dump()


@mcp.tool("weft_get_state")
async def get_state(scope: str, key: str, tenant: str = "demo") -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.get_state(scope=scope, key=key)
    if not res:
        return {"found": False}
    return {"found": True, **res.model_dump()}


@mcp.tool("weft_inbox")
async def inbox(agent_id: str, tenant: str = "demo", limit: int = 20) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.handoff_inbox(agent_id=agent_id, limit=limit)
    return {"items": [r.model_dump() for r in res]}


@mcp.tool("weft_register_agent")
async def register_agent(
    agent_id: str,
    name: str,
    tenant: str = "demo",
    kind: str = "worker",
    description: str = "",
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    h = _handle_for(tenant)
    res = await h.register_agent(
        id=agent_id, name=name, kind=kind, description=description, capabilities=capabilities
    )
    return res.model_dump()


async def serve_stdio() -> None:
    await mcp.run_stdio_async()


__all__ = ["mcp", "serve_stdio"]

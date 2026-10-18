"""REST API: thin HTTP surface over the service layer.

Endpoints map one-to-one to high-traffic flows (sessions/messages/recall/
handoffs/context_pack). The router is stateless and reads the tenant from
``X-Tenant-Id`` (and authorizes via ``X-API-Key`` against the in-process
map). No business logic lives here.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from weft.api.console import INDEX_HTML
from weft.config import get_settings
from weft.errors import ClaimError, ConflictError, NotFoundError, ValidationError, WeftError
from weft.models import (
    AgentIn,
    BlackboardSet,
    ContextPackRequest,
    FactIn,
    HandoffCreate,
    MessageIn,
    ProcedureFeedback,
    ProcedureIn,
    RecallRequest,
    SessionCreate,
)
from weft.service import Weft

router = APIRouter()
_settings = get_settings()


def _weft() -> Weft:
    return app.state.weft  # type: ignore[attr-defined]


def require_handle(
    request: Request,
    x_tenant_id: str = Header(default="demo"),
    x_api_key: str | None = Header(default=None),
) -> Any:
    tenant = x_tenant_id
    api_keys = _settings.api_key_map
    if api_keys:
        if not x_api_key or x_api_key not in api_keys:
            raise HTTPException(status_code=401, detail="invalid or missing x-api-key")
        tenant, _ = api_keys[x_api_key]
    return _weft().for_tenant(tenant)


@router.get("/health")
async def health(w: Weft = Depends(_weft)) -> dict[str, Any]:
    ok = await w.ping()
    return {"status": "ok" if ok else "degraded", "tenants": await w.list_tenants()}


@router.post("/agents")
async def register_agent(payload: AgentIn, h=Depends(require_handle)):
    return await h.register_agent(**payload.model_dump())


@router.get("/agents")
async def list_agents(h=Depends(require_handle)):
    return await h.list_agents()


@router.post("/sessions")
async def create_session(payload: SessionCreate, h=Depends(require_handle)):
    return await h.create_session(**payload.model_dump())


@router.get("/sessions")
async def list_sessions(
    user_id: str | None = None,
    agent_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    h=Depends(require_handle),
):
    return await h.list_sessions(user_id=user_id, agent_id=agent_id, status=status, limit=limit)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, h=Depends(require_handle)):
    s = await h.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@router.post("/sessions/{session_id}/close")
async def close_session(session_id: str, h=Depends(require_handle)):
    return await h.close_session(session_id)


@router.post("/sessions/{session_id}/messages")
async def append_messages(session_id: str, payload: list[MessageIn], h=Depends(require_handle)):
    return await h.append_messages(session_id, payload)


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: str, limit: int = 100, before_seq: int | None = None, h=Depends(require_handle)
):
    return await h.list_messages(session_id, limit=limit, before_seq=before_seq)


@router.post("/sessions/{session_id}/consolidate")
async def consolidate_session(session_id: str, h=Depends(require_handle)):
    return await h.consolidate_session(session_id)


@router.post("/maintenance/decay")
async def maintenance(h=Depends(require_handle)):
    return await h.run_maintenance()


@router.post("/recall")
async def recall(payload: RecallRequest, h=Depends(require_handle)):
    return await h.recall(**payload.model_dump())


@router.post("/facts")
async def add_fact(payload: FactIn, h=Depends(require_handle)):
    fid = await h.add_fact(**payload.model_dump())
    return {"id": fid}


@router.get("/facts")
async def list_facts(
    subject: str | None = None,
    predicate: str | None = None,
    limit: int = 100,
    h=Depends(require_handle),
):
    return await h.list_facts(subject=subject, predicate=predicate, limit=limit)


@router.post("/procedures")
async def add_procedure(payload: ProcedureIn, h=Depends(require_handle)):
    pid = await h.add_procedure(**payload.model_dump())
    return {"id": pid}


@router.get("/procedures")
async def list_procedures(limit: int = 50, h=Depends(require_handle)):
    return await h.list_procedures(limit=limit)


@router.post("/procedures/{procedure_id}/feedback")
async def procedure_feedback(
    procedure_id: str, payload: ProcedureFeedback, h=Depends(require_handle)
):
    return await h.procedure_feedback(procedure_id, **payload.model_dump())


@router.post("/state/{scope}/{key}")
async def set_state(
    scope: str,
    key: str,
    payload: BlackboardSet,
    x_tenant_id: str = Header(default="demo"),
    updated_by: str | None = Header(default=None),
    h=Depends(require_handle),
):
    return await h.set_state(
        scope=scope,
        key=key,
        value=payload.value,
        expected_version=payload.expected_version,
        updated_by=updated_by,
    )


@router.get("/state")
async def all_state(limit: int = 500, h=Depends(require_handle)):
    return await h.all_state(limit=limit)


@router.get("/state/{scope}/{key}")
async def get_state(scope: str, key: str, h=Depends(require_handle)):
    entry = await h.get_state(scope=scope, key=key)
    if not entry:
        raise HTTPException(status_code=404, detail="state not found")
    return entry


@router.post("/state/scope")
async def scope_state(scopes: list[str], h=Depends(require_handle)):
    return await h.scope_state(scopes)


@router.delete("/state/{scope}/{key}")
async def delete_state(scope: str, key: str, h=Depends(require_handle)):
    ok = await h.delete_state(scope=scope, key=key)
    return {"deleted": ok}


@router.post("/handoffs")
async def create_handoff(payload: HandoffCreate, h=Depends(require_handle)):
    return await h.create_handoff(**payload.model_dump())


@router.get("/handoffs/{handoff_id}")
async def get_handoff(handoff_id: str, h=Depends(require_handle)):
    hnd = await h.get_handoff(handoff_id)
    if not hnd:
        raise HTTPException(status_code=404, detail="handoff not found")
    return hnd


@router.post("/handoffs/{handoff_id}/claim")
async def claim_handoff(handoff_id: str, agent_id: str, h=Depends(require_handle)):
    try:
        return await h.claim_handoff(handoff_id=handoff_id, agent_id=agent_id)
    except ClaimError as exc:
        raise HTTPException(status_code=409, detail=exc.message) from exc


@router.post("/handoffs/{handoff_id}/complete")
async def complete_handoff(
    handoff_id: str, agent_id: str, result: dict | None = None, h=Depends(require_handle)
):
    try:
        return await h.complete_handoff(handoff_id=handoff_id, agent_id=agent_id, result=result)
    except ClaimError as exc:
        raise HTTPException(status_code=409, detail=exc.message) from exc


@router.post("/handoffs/{handoff_id}/reject")
async def reject_handoff(
    handoff_id: str, agent_id: str, reason: str = "", h=Depends(require_handle)
):
    try:
        return await h.reject_handoff(handoff_id=handoff_id, agent_id=agent_id, reason=reason)
    except ClaimError as exc:
        raise HTTPException(status_code=409, detail=exc.message) from exc


@router.get("/agents/{agent_id}/inbox")
async def agent_inbox(
    agent_id: str, status: str | None = "pending", limit: int = 20, h=Depends(require_handle)
):
    return await h.handoff_inbox(agent_id=agent_id, status=status, limit=limit)


@router.post("/context-pack")
async def context_pack(payload: ContextPackRequest, h=Depends(require_handle)):
    return await h.context_pack(**payload.model_dump())


@router.get("/sessions/{session_id}/timeline")
async def session_timeline(session_id: str, h=Depends(require_handle)):
    return await h.session_timeline(session_id)


@router.get("/stats")
async def stats(h=Depends(require_handle)):
    return await h.stats()


@router.get("/tenants")
async def tenants(w: Weft = Depends(_weft)):
    return {"tenants": await w.list_tenants()}


@router.post("/tenants/{tenant}/drop")
async def drop_tenant(tenant: str, w: Weft = Depends(_weft)):
    ok = await w.drop_tenant(tenant)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to drop tenant")
    return {"dropped": ok}


@asynccontextmanager
async def lifespan(app: FastAPI):
    weft = Weft()
    await weft.ping()
    app.state.weft = weft
    pool = None
    pool_task = None
    if _settings.workers_in_api and _settings.consolidation_enabled:
        from weft.processing.worker import WorkerPool

        pool = WorkerPool(weft.client, settings=_settings)
        app.state.worker_pool = pool
        pool_task = asyncio.create_task(pool.run_forever())
    try:
        yield
    finally:
        if pool is not None:
            await pool.stop()
            if pool_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pool_task
        await weft.aclose()


_root = _settings.root_path.rstrip("/")
if _root and not _root.startswith("/"):
    _root = "/" + _root

app = FastAPI(
    title="Weft",
    version="0.1.0",
    lifespan=lifespan,
    root_path=_root,
    servers=[{"url": _root or "/"}] if _root else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")


def _console() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/", include_in_schema=False)
async def console_root() -> HTMLResponse:
    return _console()


@app.get("/console", include_in_schema=False)
async def console_page() -> HTMLResponse:
    return _console()


@app.exception_handler(WeftError)
async def _weft_err(_: Request, exc: WeftError):
    status = 400
    if isinstance(exc, NotFoundError):
        status = 404
    if isinstance(exc, (ConflictError, ClaimError)):
        status = 409
    if isinstance(exc, ValidationError):
        status = 422
    return JSONResponse(status_code=status, content={"detail": exc.message})


def create_app() -> FastAPI:
    return app

"""CLI entrypoint."""

from __future__ import annotations

import asyncio
import sys


async def _ping() -> int:
    from weft.service import Weft

    w = Weft()
    try:
        ok = await w.ping()
        print("ok" if ok else "degraded")
        await w.aclose()
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv or argv[0] == "ping":
        return asyncio.run(_ping())
    if argv[0] == "api":
        import uvicorn

        from weft.api.app import app
        from weft.config import get_settings

        settings = get_settings()
        uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=False)
        return 0
    if argv[0] == "worker":
        return asyncio.run(_worker())
    if argv[0] == "mcp":
        asyncio.run(serve_mcp())
        return 0
    if argv[0] == "demo":
        from weft.demo import main as demo_main

        return asyncio.run(demo_main())
    if argv[0] == "bench":
        from weft.bench import main as bench_main

        return asyncio.run(bench_main(argv[1:]))
    print("usage: weft <ping|api|worker|mcp|demo|bench>", file=sys.stderr)
    return 2


async def _worker() -> int:
    """Run the multi-tenant consolidation worker pool in the foreground."""
    from weft.processing.worker import WorkerPool
    from weft.service import Weft

    w = Weft()
    try:
        await w.ping()
        pool = WorkerPool(w.client)
        await pool.run_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        await w.aclose()


async def serve_mcp() -> None:
    from weft.mcp.server import serve_stdio

    await serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())

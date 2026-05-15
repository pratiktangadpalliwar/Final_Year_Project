"""FastAPI app factory. Wires routers, builds Storage/RoundManager/ControlPlane,
mounts /static (if dist exists), and starts the asyncio round_loop on startup.

Tests pass start_round_loop=False to keep the loop dormant."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.routers import admin, client as client_router, metrics, ws
from server.app.storage import Storage
from server.app.ws_hub import WsHub


_LOG = logging.getLogger("fl.server")
_STATIC_DIR = Path(__file__).parent / "static"


def build_app(start_round_loop: bool = True) -> FastAPI:
    settings = Settings()
    storage = Storage(bucket=settings.s3_bucket, region=settings.aws_region)
    rm = RoundManager(min_nodes=settings.min_nodes, quorum_pct=settings.quorum_pct)
    cp = ControlPlane()
    cp.global_.inter_round_delay_s = settings.inter_round_delay_s
    hub = WsHub()

    app = FastAPI(title="fl-server")
    app.include_router(client_router.build_router(rm=rm, cp=cp, storage=storage))
    app.include_router(metrics.build_router(rm=rm, cp=cp))
    app.include_router(admin.build_router())
    app.include_router(ws.build_router(hub=hub))

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    @app.on_event("startup")
    async def _startup():
        from server.app.round_loop import restore_state, run_round_loop
        try:
            restore_state(rm=rm, cp=cp, storage=storage, settings=settings)
        except Exception as e:
            _LOG.exception("failed to restore state: %s", e)
            raise
        if start_round_loop:
            import asyncio
            app.state.round_task = asyncio.create_task(
                run_round_loop(rm=rm, cp=cp, storage=storage, hub=hub, settings=settings)
            )

    @app.on_event("shutdown")
    async def _shutdown():
        task = getattr(app.state, "round_task", None)
        if task:
            task.cancel()

    app.state.rm = rm
    app.state.cp = cp
    app.state.storage = storage
    app.state.hub = hub
    app.state.settings = settings
    return app

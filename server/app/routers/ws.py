"""/ws/live WebSocket endpoint. Subscribers receive every event the round_loop
broadcasts via WsHub."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.app.ws_hub import WsHub


def build_router(*, hub: WsHub) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        hub.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.remove(ws)

    return router

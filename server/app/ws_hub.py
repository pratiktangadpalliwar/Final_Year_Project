"""WebSocket fan-out. Pure: depends only on the .send_json(payload) protocol,
so unit tests can inject a fake. Real FastAPI WebSocket implements that method
natively."""
from __future__ import annotations

from typing import Protocol


class _Sock(Protocol):
    async def send_json(self, payload: dict) -> None: ...


class WsHub:
    def __init__(self) -> None:
        self._subs: set[_Sock] = set()

    def add(self, ws: _Sock) -> None:
        self._subs.add(ws)

    def remove(self, ws: _Sock) -> None:
        self._subs.discard(ws)

    def size(self) -> int:
        return len(self._subs)

    async def broadcast(self, payload: dict) -> None:
        dead: list[_Sock] = []
        for ws in list(self._subs):
            try:
                await ws.send_json(payload)
            except Exception:  # any send error → drop subscriber
                dead.append(ws)
        for d in dead:
            self._subs.discard(d)

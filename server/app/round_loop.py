"""Phase 1 stub. Phase 4 (Tasks 4.1/4.2) replaces this with the real loop."""
from __future__ import annotations

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.storage import Storage
from server.app.ws_hub import WsHub


def restore_state(*, rm: RoundManager, cp: ControlPlane, storage: Storage, settings: Settings) -> None:
    return None


async def run_round_loop(*, rm: RoundManager, cp: ControlPlane, storage: Storage, hub: WsHub, settings: Settings) -> None:
    return None

"""Client-facing endpoints: /register, /model/global, /model/update, /control/{bank_id}.

The pending-updates queue is module-level (a dict keyed by (round, bank_id)) so the
round_loop (Task 4.1) can drain it. Clean for a single-process server; if you ever
shard fl-server, replace with a real queue.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.storage import Storage


# Module-level pending updates queue. Keys: (round, bank_id) → payload.
_pending: dict[tuple[int, str], dict[str, Any]] = {}


def get_pending() -> dict[tuple[int, str], dict[str, Any]]:
    return _pending


def reset_pending() -> None:
    _pending.clear()


class RegisterIn(BaseModel):
    bank_id: str
    bank_name: str
    n_samples: int


class UpdateIn(BaseModel):
    bank_id: str
    round: int
    weights_key: str
    n_samples: int
    metrics: dict[str, float]


def build_router(*, rm: RoundManager, cp: ControlPlane, storage: Storage) -> APIRouter:
    router = APIRouter()

    @router.post("/register")
    def register(payload: RegisterIn):
        rm.register(payload.bank_id, payload.bank_name, payload.n_samples)
        return {"current_round": cp.global_.current_round}

    @router.get("/model/global")
    def get_global(bank_id: str):
        if bank_id not in rm.registered:
            raise HTTPException(404, "bank not registered")
        latest = storage.latest_round(prefix="models/global_round_")
        if latest is None:
            raise HTTPException(503, "no global model yet")
        key = f"models/global_round_{latest:04d}.pt"
        return {"round": latest, "weights_key": key, "weights_url": storage.presign_get(key)}

    @router.post("/model/update")
    def post_update(payload: UpdateIn):
        if payload.bank_id not in rm.registered:
            raise HTTPException(404, "bank not registered")
        _pending[(payload.round, payload.bank_id)] = payload.model_dump()
        return {"accepted": True}

    @router.get("/control/{bank_id}")
    def get_control(bank_id: str):
        bc = cp.get_bank(bank_id)
        latest = storage.latest_round(prefix="models/global_round_") or 0
        key = f"models/global_round_{latest:04d}.pt"
        return {
            "paused": cp.global_.paused,
            "current_round": cp.global_.current_round,
            "dataset_version": bc.dataset_version,
            "fault": bc.fault,
            "weights_key": key,
        }

    return router

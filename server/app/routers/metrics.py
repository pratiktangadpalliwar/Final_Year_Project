"""Metrics + status endpoints. Round history is kept in-memory; checkpointed
to S3 via RoundManager+Storage in Task 4.2. The global-eval AUC numbers come
from server/app/eval.py (computed by round_loop after each aggregation)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from fastapi import APIRouter, HTTPException

from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager


_global_history: deque[dict[str, Any]] = deque(maxlen=1000)
_bank_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))


def push_global_metrics(row: dict[str, Any]) -> None:
    _global_history.append(row)


def push_bank_metrics(bank_id: str, row: dict[str, Any]) -> None:
    _bank_history[bank_id].append(row)


def reset_history() -> None:
    _global_history.clear()
    _bank_history.clear()


def build_router(*, rm: RoundManager, cp: ControlPlane) -> APIRouter:
    router = APIRouter()

    @router.get("/round/status")
    def round_status():
        return {
            "round": cp.global_.current_round,
            "state": cp.global_.state,
            "paused": cp.global_.paused,
            "active_banks": rm.active_node_count(),
            "quorum_size": rm.quorum_size(),
        }

    @router.get("/banks")
    def banks():
        out = []
        for bid, info in rm.registered.items():
            out.append({
                "bank_id": bid,
                "bank_name": info.bank_name,
                "n_samples": info.n_samples,
                "trust": rm.trust_scores.get(bid, 1.0),
                "suspended": bid in rm.suspended,
                "dataset_version": cp.get_bank(bid).dataset_version,
                "fault": cp.get_bank(bid).fault,
                "cumulative_eps": rm.cumulative_eps_per_bank.get(bid, 0.0),
            })
        return out

    @router.get("/banks/{bank_id}/history")
    def bank_history(bank_id: str, n: int = 50):
        if bank_id not in rm.registered:
            raise HTTPException(404, "bank not registered")
        h = list(_bank_history[bank_id])
        return h[-n:]

    @router.get("/metrics")
    def metrics(n: int = 50):
        return {
            "history": list(_global_history)[-n:],
            "cumulative_eps_global": rm.cumulative_eps_global,
            "current_round": cp.global_.current_round,
        }

    @router.get("/health")
    def health():
        return {"status": "ok"}

    return router

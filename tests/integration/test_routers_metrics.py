import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.routers.metrics import (
    build_router,
    push_bank_metrics,
    push_global_metrics,
    reset_history,
)


@pytest.fixture
def app():
    rm = RoundManager()
    rm.register("bank_01", "B1", 100)
    rm.register("bank_02", "B2", 200)
    rm.current_round = 5
    cp = ControlPlane()
    cp.global_.current_round = 5
    cp.global_.state = "idle"

    reset_history()
    for r in range(1, 6):
        push_global_metrics({"round": r, "auc": 0.7 + r * 0.02, "f1": 0.5, "method": "fedavg"})
        push_bank_metrics("bank_01", {"round": r, "auc": 0.6 + r * 0.02, "loss": 0.2})
        push_bank_metrics("bank_02", {"round": r, "auc": 0.65 + r * 0.02, "loss": 0.18})

    app = FastAPI()
    app.include_router(build_router(rm=rm, cp=cp))
    yield TestClient(app)
    reset_history()


@pytest.mark.integration
def test_round_status(app):
    r = app.get("/round/status")
    assert r.status_code == 200
    body = r.json()
    assert body["round"] == 5
    assert body["state"] == "idle"


@pytest.mark.integration
def test_banks_lists_registered(app):
    r = app.get("/banks")
    assert r.status_code == 200
    body = r.json()
    ids = {b["bank_id"] for b in body}
    assert ids == {"bank_01", "bank_02"}


@pytest.mark.integration
def test_bank_history(app):
    r = app.get("/banks/bank_01/history", params={"n": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert body[-1]["round"] == 5


@pytest.mark.integration
def test_global_metrics(app):
    r = app.get("/metrics", params={"n": 50})
    body = r.json()
    assert len(body["history"]) == 5
    assert body["history"][-1]["round"] == 5

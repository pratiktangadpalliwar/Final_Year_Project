import boto3
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_aws

from server.app.control_plane import ControlPlane
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import build_router, get_pending, reset_pending
from server.app.storage import Storage


@pytest.fixture
def app_and_state():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        st = Storage(bucket="fl-test", region="us-east-1")
        rm = RoundManager()
        cp = ControlPlane()
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())
        reset_pending()
        app = FastAPI()
        app.include_router(build_router(rm=rm, cp=cp, storage=st))
        yield TestClient(app), rm, cp, st
        reset_pending()


@pytest.mark.integration
def test_register_creates_bank_and_returns_round(app_and_state):
    client, rm, cp, st = app_and_state
    r = client.post("/register", json={"bank_id": "bank_01", "bank_name": "Bank One", "n_samples": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["current_round"] == 0
    assert "bank_01" in rm.registered


@pytest.mark.integration
def test_get_global_model_returns_weights_url(app_and_state):
    client, rm, cp, st = app_and_state
    rm.register("bank_01", "Bank One", 100)
    r = client.get("/model/global", params={"bank_id": "bank_01"})
    assert r.status_code == 200
    body = r.json()
    assert body["round"] == 0
    assert body["weights_key"] == "models/global_round_0000.pt"
    assert body["weights_url"].startswith("https://")


@pytest.mark.integration
def test_post_model_update_records_in_pending(app_and_state):
    client, rm, cp, st = app_and_state
    rm.register("bank_01", "Bank One", 100)
    cp.global_.current_round = 1
    payload = {
        "bank_id": "bank_01",
        "round": 1,
        "weights_key": "updates/bank_01/round_0001.pt",
        "n_samples": 100,
        "metrics": {"val_auc": 0.85, "val_loss": 0.12},
    }
    r = client.post("/model/update", json=payload)
    assert r.status_code == 200
    pending = get_pending()
    assert (1, "bank_01") in pending


@pytest.mark.integration
def test_control_returns_pause_and_fault(app_and_state):
    client, rm, cp, st = app_and_state
    rm.register("bank_03", "B3", 100)
    cp.set_fault("bank_03", "byzantine")
    cp.global_.current_round = 5
    cp.global_.paused = True
    r = client.get("/control/bank_03")
    assert r.status_code == 200
    body = r.json()
    assert body["paused"] is True
    assert body["fault"] == "byzantine"
    assert body["current_round"] == 5
    assert body["dataset_version"] == 1
    assert body["weights_key"] == "models/global_round_0000.pt"

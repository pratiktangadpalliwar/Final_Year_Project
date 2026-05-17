import io
import pickle

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import reset_pending
from server.app.routers.metrics import push_global_metrics, reset_history
from server.app.storage import Storage
from server.app.ws_hub import WsHub
from tests.shared.fakes import fake_post_update, fake_register


def _seed_validation_set(storage):
    """8 rows of bogus features so eval can run."""
    x = torch.randn(8, 19)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    buf = io.BytesIO()
    pickle.dump({"X": x, "y": y}, buf)
    storage.put_bytes("validation/val_set.pkl", buf.getvalue())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rollback_triggers_when_metric_drops_more_than_threshold(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        # tiny threshold so any drop trips rollback
        monkeypatch.setenv("ROLLBACK_THRESHOLD", "0.01")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())
        _seed_validation_set(st)

        rm = RoundManager(min_nodes=3, quorum_pct=0.6)
        cp = ControlPlane()
        hub = WsHub()
        reset_pending(); reset_history()

        # baseline AUC at round >5 (rollback only fires after warm-up)
        rm.current_round = 6
        push_global_metrics({"round": 6, "auc": 0.95, "method": "fedavg"})

        for i in range(3):
            fake_register(rm, f"bank_0{i+1}")
            fake_post_update(f"bank_0{i+1}", round=7, storage=st, perturbation=2.0)  # noisy → AUC tanks

        from server.app.round_loop import run_one_round
        await run_one_round(rm=rm, cp=cp, storage=st, hub=hub, settings=s, target_round=7)

        # rolled back → no global_round_0007 written
        assert st.latest_round(prefix="models/global_round_") == 0

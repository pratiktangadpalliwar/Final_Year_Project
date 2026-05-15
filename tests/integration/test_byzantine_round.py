import boto3
import pytest
from moto import mock_aws

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import reset_pending
from server.app.routers.metrics import reset_history
from server.app.storage import Storage
from server.app.ws_hub import WsHub
from tests.shared.fakes import fake_post_update, fake_register


@pytest.mark.integration
@pytest.mark.asyncio
async def test_byzantine_update_drops_trust_score(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())

        rm = RoundManager(min_nodes=3, quorum_pct=0.6)
        cp = ControlPlane()
        hub = WsHub()
        reset_pending(); reset_history()

        for i in range(4):
            fake_register(rm, f"bank_0{i+1}")

        for i in range(3):
            fake_post_update(f"bank_0{i+1}", round=1, storage=st)
        fake_post_update("bank_04", round=1, storage=st, bad=True)

        from server.app.round_loop import run_one_round
        await run_one_round(rm=rm, cp=cp, storage=st, hub=hub, settings=s, target_round=1, deadline_s=2)

        # bad bank flagged; honest banks rewarded (or at worst no decay).
        assert rm.trust_scores["bank_04"] < rm.trust_scores["bank_01"]
        assert rm.trust_scores["bank_04"] < 1.0

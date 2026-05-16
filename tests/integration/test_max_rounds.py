import asyncio

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_rounds_pauses_loop(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        monkeypatch.setenv("MAX_ROUNDS", "2")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())

        rm = RoundManager()
        cp = ControlPlane()
        cp.global_.current_round = 2  # already at max
        hub = WsHub()
        reset_pending(); reset_history()

        from server.app.round_loop import run_round_loop
        task = asyncio.create_task(run_round_loop(rm=rm, cp=cp, storage=st, hub=hub, settings=s))
        await asyncio.sleep(2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert cp.global_.paused is True

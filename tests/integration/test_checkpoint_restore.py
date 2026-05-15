import boto3
import pytest
from moto import mock_aws

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.storage import Storage


@pytest.mark.integration
def test_restore_state_picks_up_latest_checkpoint(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")

        rm_old = RoundManager()
        rm_old.register("bank_a", "A", 100); rm_old.register("bank_b", "B", 200)
        rm_old.flag_node("bank_b")
        rm_old.current_round = 7
        rm_old.cumulative_eps_global = 35.0
        st.put_json("checkpoints/round_0007.json", rm_old.checkpoint_dict())

        cp_old = ControlPlane()
        cp_old.global_.current_round = 7
        cp_old.set_fault("bank_b", "byzantine")
        st.put_json("control/state.json", cp_old.snapshot_dict())

        from server.app.round_loop import restore_state
        rm_new = RoundManager(); cp_new = ControlPlane()
        restore_state(rm=rm_new, cp=cp_new, storage=st, settings=s)

        assert rm_new.current_round == 7
        assert rm_new.cumulative_eps_global == 35.0
        assert rm_new.trust_scores["bank_b"] < 1.0
        assert cp_new.global_.current_round == 7
        assert cp_new.banks["bank_b"].fault == "byzantine"

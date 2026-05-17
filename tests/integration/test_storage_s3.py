import boto3
import pytest
import torch
from moto import mock_aws

from server.app.storage import Storage


@pytest.fixture
def s3_bucket():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="fl-test")
        yield "fl-test"


@pytest.mark.integration
def test_put_and_get_weights_roundtrip(s3_bucket):
    st = Storage(bucket=s3_bucket, region="us-east-1")
    weights = {"a": torch.ones(4), "b": torch.zeros(2)}
    st.put_weights("models/global_round_0001.pt", weights)
    out = st.get_weights("models/global_round_0001.pt")
    assert torch.allclose(out["a"], weights["a"])
    assert torch.allclose(out["b"], weights["b"])


@pytest.mark.integration
def test_put_and_get_json_roundtrip(s3_bucket):
    st = Storage(bucket=s3_bucket, region="us-east-1")
    payload = {"current_round": 7, "trust_scores": {"bank_01": 0.95}}
    st.put_json("checkpoints/round_0007.json", payload)
    out = st.get_json("checkpoints/round_0007.json")
    assert out == payload


@pytest.mark.integration
def test_latest_round_finds_max_key(s3_bucket):
    st = Storage(bucket=s3_bucket, region="us-east-1")
    for r in [1, 5, 12, 3]:
        st.put_weights(f"models/global_round_{r:04d}.pt", {"a": torch.zeros(2)})
    assert st.latest_round() == 12


@pytest.mark.integration
def test_latest_round_returns_none_when_empty(s3_bucket):
    st = Storage(bucket=s3_bucket, region="us-east-1")
    assert st.latest_round() is None

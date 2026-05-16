import io

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from server.app.auth import hash_password


@pytest.fixture
def app(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        monkeypatch.setenv("ADMIN_PASSWORD_HASH", hash_password("hunter2"))
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        from server.app.main import build_app
        a = build_app(start_round_loop=False)
        yield a, TestClient(a)


@pytest.mark.integration
def test_dataset_upload_writes_to_s3_and_bumps_version(app):
    a, client = app
    client.post("/admin/login", json={"password": "hunter2"})
    csv_bytes = b"transaction_id,is_fraud\n1,0\n2,1\n" * 50
    files = {"file": ("bank_03.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/admin/dataset/bank_03", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bank_id"] == "bank_03"
    assert body["dataset_version"] == 2

    cp = a.state.cp
    assert cp.banks["bank_03"].dataset_version == 2

    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket="fl-test", Key="datasets/bank_03.csv")
    assert obj["Body"].read() == csv_bytes


@pytest.mark.integration
def test_dataset_upload_too_large_rejected(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        monkeypatch.setenv("ADMIN_PASSWORD_HASH", hash_password("hunter2"))
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        monkeypatch.setenv("DATASET_UPLOAD_MAX_BYTES", "1024")
        from server.app.main import build_app
        a = build_app(start_round_loop=False)
        client = TestClient(a)
        client.post("/admin/login", json={"password": "hunter2"})
        files = {"file": ("big.csv", io.BytesIO(b"x" * 2000), "text/csv")}
        r = client.post("/admin/dataset/bank_03", files=files)
        assert r.status_code == 413


@pytest.mark.integration
def test_dataset_upload_requires_cookie(app):
    a, client = app
    files = {"file": ("x.csv", io.BytesIO(b"a"), "text/csv")}
    r = client.post("/admin/dataset/bank_03", files=files)
    assert r.status_code == 401

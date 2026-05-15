import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws


@pytest.mark.integration
def test_app_starts_and_health_ok(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        from server.app.main import build_app
        app = build_app(start_round_loop=False)
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_static_mount_404_when_missing(monkeypatch):
    """If server/app/static/ doesn't exist (Plan 1 default), the mount returns 404
    for unknown paths instead of crashing."""
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")

        from server.app.main import build_app
        app = build_app(start_round_loop=False)
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code in (200, 404)

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from server.app.auth import COOKIE_NAME, hash_password


@pytest.fixture
def app(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("ADMIN_PASSWORD_HASH", hash_password("hunter2"))
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        from server.app.main import build_app
        a = build_app(start_round_loop=False)
        yield TestClient(a)


@pytest.mark.integration
def test_login_with_correct_password_sets_cookie(app):
    r = app.post("/admin/login", json={"password": "hunter2"})
    assert r.status_code == 200
    assert COOKIE_NAME in r.cookies


@pytest.mark.integration
def test_login_with_wrong_password_returns_401(app):
    r = app.post("/admin/login", json={"password": "wrong"})
    assert r.status_code == 401


@pytest.mark.integration
def test_logout_clears_cookie(app):
    app.post("/admin/login", json={"password": "hunter2"})
    r = app.post("/admin/logout")
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "")
    assert "Max-Age=0" in set_cookie or 'max-age=0' in set_cookie.lower()

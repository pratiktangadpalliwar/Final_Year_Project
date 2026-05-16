import bcrypt
import jwt
import pytest
from fastapi import FastAPI, HTTPException, Depends
from fastapi.testclient import TestClient

from server.app.auth import (
    hash_password,
    issue_cookie,
    require_admin,
    verify_password,
)


def test_hash_then_verify():
    h = hash_password("hunter2")
    assert verify_password("hunter2", h)
    assert not verify_password("wrong", h)


def test_hash_different_each_call_but_both_verify():
    a = hash_password("hunter2")
    b = hash_password("hunter2")
    assert a != b
    assert verify_password("hunter2", a)
    assert verify_password("hunter2", b)


def test_issue_cookie_returns_signed_jwt(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    token = issue_cookie(secret="test-secret", ttl_minutes=60)
    decoded = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert decoded["role"] == "admin"
    assert "exp" in decoded


def test_require_admin_rejects_missing_cookie(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "x")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", hash_password("p"))
    app = FastAPI()

    @app.get("/protected")
    def protected(_: None = Depends(require_admin)):
        return {"ok": True}

    client = TestClient(app)
    r = client.get("/protected")
    assert r.status_code == 401


def test_require_admin_accepts_valid_cookie(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "x")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", hash_password("p"))
    app = FastAPI()

    @app.get("/protected")
    def protected(_: None = Depends(require_admin)):
        return {"ok": True}

    client = TestClient(app)
    token = issue_cookie(secret="test-secret", ttl_minutes=60)
    client.cookies.set("fl_admin", token)
    r = client.get("/protected")
    assert r.status_code == 200

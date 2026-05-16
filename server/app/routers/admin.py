"""Admin endpoints. Plan 2 fills Plan 1 stubs."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from server.app.auth import (
    COOKIE_NAME,
    issue_cookie,
    require_admin,
    verify_password,
)
from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.storage import Storage


class LoginIn(BaseModel):
    password: str


def build_router(*, cp: ControlPlane, storage: Storage) -> APIRouter:
    router = APIRouter(prefix="/admin")

    @router.post("/login")
    def login(payload: LoginIn, response: Response):
        s = Settings()
        if not s.admin_password_hash or not s.jwt_secret:
            raise HTTPException(500, "auth not configured")
        if not verify_password(payload.password, s.admin_password_hash):
            raise HTTPException(401, "bad password")
        token = issue_cookie(secret=s.jwt_secret, ttl_minutes=s.jwt_ttl_minutes)
        response.set_cookie(
            COOKIE_NAME, token,
            max_age=s.jwt_ttl_minutes * 60,
            httponly=True, secure=False, samesite="strict",
        )
        return {"ok": True}

    @router.post("/logout")
    def logout(response: Response):
        response.set_cookie(COOKIE_NAME, "", max_age=0, httponly=True, samesite="strict")
        return {"ok": True}

    # The remaining endpoints below this line require admin cookie.
    protected = APIRouter(dependencies=[Depends(require_admin)])

    @protected.post("/pause")
    def pause():
        cp.pause()
        return {"paused": True}

    @protected.post("/resume")
    def resume():
        cp.resume()
        return {"paused": False}

    @protected.post("/reset")
    def reset():
        cp.reset_rounds()
        return {"current_round": 0}

    @protected.post("/fault")
    def fault():
        raise HTTPException(501, "Task 7.4")

    @protected.post("/dataset/{bank_id}")
    def dataset(bank_id: str):
        raise HTTPException(501, "Task 7.5")

    router.include_router(protected)
    return router

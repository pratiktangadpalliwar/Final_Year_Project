"""Admin endpoints. Plan 1 ships only stubs returning 501 so the dashboard
contract is visible. Plan 2 implements pause/resume/fault/dataset/login."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from server.app.auth import require_admin


def build_router() -> APIRouter:
    router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

    @router.post("/login")
    def login():
        raise HTTPException(501, "implemented in Plan 2")

    @router.post("/pause")
    def pause():
        raise HTTPException(501, "implemented in Plan 2")

    @router.post("/resume")
    def resume():
        raise HTTPException(501, "implemented in Plan 2")

    @router.post("/reset")
    def reset():
        raise HTTPException(501, "implemented in Plan 2")

    @router.post("/fault")
    def fault():
        raise HTTPException(501, "implemented in Plan 2")

    @router.post("/dataset/{bank_id}")
    def dataset(bank_id: str):
        raise HTTPException(501, "implemented in Plan 2")

    return router

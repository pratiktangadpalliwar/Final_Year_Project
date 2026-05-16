"""bcrypt + JWT cookie auth for /admin/* and /ws/live.

If ADMIN_PASSWORD_HASH env var is unset, require_admin is a pass-through
(Plan 1 mode — useful for local dev). In production the helm chart sets
both ADMIN_PASSWORD_HASH and JWT_SECRET."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import HTTPException, Request

COOKIE_NAME = "fl_admin"


def hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def verify_password(plaintext: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plaintext.encode(), hashed.encode())
    except ValueError:
        return False


def issue_cookie(*, secret: str, ttl_minutes: int = 480) -> str:
    now = datetime.now(UTC)
    payload = {
        "role": "admin",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def require_admin(request: Request) -> None:
    """FastAPI dependency. Raises 401 if cookie missing or invalid.
    Pass-through if ADMIN_PASSWORD_HASH not configured (Plan 1 dev mode)."""
    from server.app.config import Settings  # late import to avoid cycle in tests
    s = Settings()
    if not s.admin_password_hash:
        return  # auth disabled
    if not s.jwt_secret:
        raise HTTPException(500, "JWT_SECRET not configured")
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, "missing cookie")
    try:
        jwt.decode(token, s.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"invalid cookie: {e}") from e

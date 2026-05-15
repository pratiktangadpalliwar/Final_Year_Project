"""Authentication. Plan 1: pass-through (no cookie required) — the cluster
NetworkPolicy is the security boundary in the demo. Plan 2 replaces this with
a bcrypt+JWT cookie gate for /admin/* and /ws."""
from __future__ import annotations

from fastapi import Request


def require_admin(_: Request) -> None:
    """Plan 1 stub. Plan 2 will raise 401 here when no valid cookie is present."""
    return None

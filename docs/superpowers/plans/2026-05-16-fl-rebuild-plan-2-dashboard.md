# FL Rebuild — Plan 2: Dashboard + Admin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a single-page React dashboard (login + live training view + controls + dataset upload + fault injection) into the fl-server image so the operator can drive the demo from one URL with one password.

**Architecture:** Vite + React + TypeScript bundle built into `server/app/static/` and served by FastAPI's StaticFiles mount. Server gains real `/admin/*` endpoints (filling Plan 1 stubs), bcrypt + JWT cookie auth, S3 multipart dataset upload, rollback wiring (already-built `eval.py` called from `round_loop`), `max_rounds` enforcement, and client-side dataset re-fetch on `dataset_version` bump. WebSocket `/ws/live` continues to broadcast all round events (Plan 1 already does); the dashboard is the consumer.

**Tech Stack:** React 18, Vite 5, TypeScript 5, recharts (line charts), native WebSocket API, FastAPI (existing), bcrypt + PyJWT (existing in server requirements), boto3 multipart upload.

**Predecessors:** Plan 1 (FL core) — `claude/upbeat-archimedes-846e98` branch, merged or layered on.
**Successor:** Plan 3 (AWS infra + deploy.sh).

**Reference design:** `docs/superpowers/specs/2026-05-15-fl-rebuild-design.md` sections 5 (round lifecycle), 6 (dashboard UI/API), 8 (errors), 11 (acceptance #2,#4,#5,#6,#7).

---

## File structure (locked in this plan)

```
server/app/
├── auth.py                          MODIFY — replace stub with bcrypt+JWT cookie gate
├── routers/
│   ├── admin.py                     MODIFY — fill 6 stubs
│   └── ...
├── round_loop.py                    MODIFY — wire rollback (calls eval) + max_rounds check
└── eval.py                          MODIFY — accept S3-loaded validation set

client/app/
├── round_runner.py                  MODIFY — re-fetch dataset on version bump
└── main.py                          MODIFY — pass dataset_loader closure that can refresh

dashboard/                           NEW (entire dir)
├── package.json
├── tsconfig.json
├── vite.config.ts                   outDir = ../server/app/static
├── index.html
├── public/
└── src/
    ├── main.tsx                     entry
    ├── App.tsx                      router + auth guard
    ├── lib/
    │   ├── api.ts                   typed REST client
    │   ├── ws.ts                    typed WS client + reconnect
    │   └── types.ts                 shared TS types (mirror server JSON shapes)
    ├── pages/
    │   ├── Login.tsx                password form
    │   └── Dashboard.tsx            single-page layout
    └── components/
        ├── TopBar.tsx               round# / state / controls / logout
        ├── GlobalMetrics.tsx        5 sparklines (AUC/F1/precision/recall/loss)
        ├── BankGrid.tsx             4×2 grid wrapper
        ├── BankCard.tsx             per-bank card (sparkline + 9-cell numeric grid)
        ├── BankDrillIn.tsx          modal: 16-field metric block + LOCAL/GLOBAL overlay
        ├── ControlPanel.tsx         pause/resume/reset
        ├── FaultPanel.tsx           per-bank fault buttons
        ├── DatasetUpload.tsx        drag-drop CSV
        ├── EventLog.tsx             live ws feed
        └── Sparkline.tsx            shared SVG sparkline primitive (no chart lib)

tests/
├── unit/server/
│   ├── test_auth.py                 NEW
│   └── test_round_loop_rollback.py  NEW
├── integration/
│   ├── test_admin_routes.py         NEW
│   └── test_dataset_upload.py       NEW
└── e2e/                             updated dashboard build smoke
```

**Files explicitly NOT in this plan (deferred to Plan 3):** `infra/`, `k8s/`, `deploy.sh`, `teardown.sh`, ECR push automation, real terraform-provisioned bucket name in production. Plan 2's dashboard runs against the same docker-compose + minio stack as Plan 1.

---

## Phase 7 — Server-side admin + auth + rollback

### Task 7.1: Real `auth.py` — bcrypt + JWT cookie

**Files:**
- Modify: `server/app/auth.py` (replace stub)
- Modify: `server/app/config.py` (add `admin_password_hash` requirement, `jwt_secret`, `jwt_ttl_minutes`)
- Test: `tests/unit/server/test_auth.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/server/test_auth.py`:

```python
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
```

- [ ] **Step 2: Run — fail (functions don't exist)**

Run: `python -m pytest tests/unit/server/test_auth.py -v`
Expected: ImportError.

- [ ] **Step 3: Modify `server/app/config.py`**

Add JWT/auth fields. Replace the existing auth section:

```python
    # --- Auth ---
    admin_password_hash: str | None = None  # bcrypt hash; None = auth disabled
    jwt_secret: str | None = None  # required if admin_password_hash set
    jwt_ttl_minutes: int = 480  # 8h cookie lifetime
    cors_origin: str = "*"
```

(Keep all other fields unchanged.)

- [ ] **Step 4: Replace `server/app/auth.py`**

```python
"""bcrypt + JWT cookie auth for /admin/* and /ws/live.

If ADMIN_PASSWORD_HASH env var is unset, require_admin is a pass-through
(Plan 1 mode — useful for local dev). In production the helm chart sets
both ADMIN_PASSWORD_HASH and JWT_SECRET."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    now = datetime.now(timezone.utc)
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
```

- [ ] **Step 5: Run — pass**

Run: `python -m pytest tests/unit/server/test_auth.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add server/app/auth.py server/app/config.py tests/unit/server/test_auth.py
git commit -m "feat(auth): bcrypt + JWT cookie gate for admin/ws routes"
```

---

### Task 7.2: Real `/admin/login` + `/admin/logout`

**Files:**
- Modify: `server/app/routers/admin.py` (replace login/logout stubs)
- Test: `tests/integration/test_admin_routes.py` (new)

- [ ] **Step 1: Write failing test**

`tests/integration/test_admin_routes.py`:

```python
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
    # max_age 0 in Set-Cookie header → browser drops it
    set_cookie = r.headers.get("set-cookie", "")
    assert "Max-Age=0" in set_cookie or 'max-age=0' in set_cookie.lower()
```

- [ ] **Step 2: Run — fails (login is 501)**

Run: `python -m pytest tests/integration/test_admin_routes.py -v -m integration`
Expected: 401 vs 200 mismatch + 200 vs 501 (login still returns 501).

- [ ] **Step 3: Replace `server/app/routers/admin.py` login/logout**

Replace the file's `login` and `logout` stubs:

```python
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


class LoginIn(BaseModel):
    password: str


def build_router() -> APIRouter:
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
            httponly=True, secure=False, samesite="strict",  # secure=True in Plan 3 (https)
        )
        return {"ok": True}

    @router.post("/logout")
    def logout(response: Response):
        response.set_cookie(COOKIE_NAME, "", max_age=0, httponly=True, samesite="strict")
        return {"ok": True}

    # The remaining endpoints below this line require admin cookie.
    protected = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

    @protected.post("/pause")
    def pause():
        raise HTTPException(501, "implemented in next task")

    @protected.post("/resume")
    def resume():
        raise HTTPException(501, "implemented in next task")

    @protected.post("/reset")
    def reset():
        raise HTTPException(501, "implemented in next task")

    @protected.post("/fault")
    def fault():
        raise HTTPException(501, "implemented in next task")

    @protected.post("/dataset/{bank_id}")
    def dataset(bank_id: str):
        raise HTTPException(501, "implemented in next task")

    router.include_router(protected)
    return router
```

- [ ] **Step 4: Run — pass**

Run: `python -m pytest tests/integration/test_admin_routes.py -v -m integration`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/routers/admin.py tests/integration/test_admin_routes.py
git commit -m "feat(admin): /admin/login + /admin/logout with bcrypt + JWT cookie"
```

---

### Task 7.3: `/admin/pause`, `/admin/resume`, `/admin/reset`

**Files:**
- Modify: `server/app/routers/admin.py` (replace 3 stubs)
- Modify: `server/app/main.py` (pass `cp` + `rm` + `storage` into admin router)
- Test: extend `tests/integration/test_admin_routes.py`

- [ ] **Step 1: Extend the test**

Append to `tests/integration/test_admin_routes.py`:

```python
@pytest.mark.integration
def test_pause_then_resume_changes_cp_state(app):
    app.post("/admin/login", json={"password": "hunter2"})
    r = app.post("/admin/pause")
    assert r.status_code == 200
    status = app.get("/round/status").json()
    assert status["paused"] is True

    r = app.post("/admin/resume")
    assert r.status_code == 200
    status = app.get("/round/status").json()
    assert status["paused"] is False


@pytest.mark.integration
def test_pause_requires_cookie(app):
    # without login → 401
    r = app.post("/admin/pause")
    assert r.status_code == 401


@pytest.mark.integration
def test_reset_zeroes_round_counter(app):
    app.post("/admin/login", json={"password": "hunter2"})
    # bump round
    a = app.app.state.cp
    a.global_.current_round = 42
    r = app.post("/admin/reset")
    assert r.status_code == 200
    assert a.global_.current_round == 0
```

- [ ] **Step 2: Run — fails**

Run: `python -m pytest tests/integration/test_admin_routes.py -v -m integration -k "pause or resume or reset"`
Expected: 501 from stubs.

- [ ] **Step 3: Modify `build_router` signature in `server/app/routers/admin.py`**

Change `build_router()` to accept `*, cp: ControlPlane, storage: Storage`. Replace the 3 protected stubs:

```python
from server.app.control_plane import ControlPlane
from server.app.storage import Storage


def build_router(*, cp: ControlPlane, storage: Storage) -> APIRouter:
    router = APIRouter(prefix="/admin")

    # ... login + logout unchanged ...

    protected = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

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
        # ControlPlane snapshot to S3 happens at next round_loop tick
        return {"current_round": 0}

    @protected.post("/fault")
    def fault():
        raise HTTPException(501, "Task 7.4")

    @protected.post("/dataset/{bank_id}")
    def dataset(bank_id: str):
        raise HTTPException(501, "Task 7.5")

    router.include_router(protected)
    return router
```

- [ ] **Step 4: Update `server/app/main.py` — pass cp + storage to admin.build_router**

Change the line `app.include_router(admin.build_router())` to:

```python
app.include_router(admin.build_router(cp=cp, storage=storage))
```

- [ ] **Step 5: Run — pass**

Run: `python -m pytest tests/integration/test_admin_routes.py -v -m integration`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add server/app/routers/admin.py server/app/main.py tests/integration/test_admin_routes.py
git commit -m "feat(admin): /admin/pause + /admin/resume + /admin/reset (cookie-gated)"
```

---

### Task 7.4: `/admin/fault`

**Files:**
- Modify: `server/app/routers/admin.py`
- Test: extend `tests/integration/test_admin_routes.py`

- [ ] **Step 1: Extend the test**

Append:

```python
@pytest.mark.integration
def test_fault_byzantine_persists_to_cp(app):
    app.post("/admin/login", json={"password": "hunter2"})
    r = app.post("/admin/fault", json={"bank_id": "bank_04", "fault": "byzantine"})
    assert r.status_code == 200
    cp = app.app.state.cp
    assert cp.banks["bank_04"].fault == "byzantine"


@pytest.mark.integration
def test_fault_invalid_value_rejected(app):
    app.post("/admin/login", json={"password": "hunter2"})
    r = app.post("/admin/fault", json={"bank_id": "bank_04", "fault": "boom"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Replace `fault` stub**

Add a Pydantic model + impl:

```python
from typing import Literal
from pydantic import BaseModel


class FaultIn(BaseModel):
    bank_id: str
    fault: Literal["none", "crash", "straggle", "byzantine", "partition"]


# ... inside protected router:
    @protected.post("/fault")
    def fault(payload: FaultIn):
        cp.set_fault(payload.bank_id, payload.fault)
        return {"bank_id": payload.bank_id, "fault": payload.fault}
```

- [ ] **Step 4: Run — pass (2 new tests)**

- [ ] **Step 5: Commit**

```bash
git add server/app/routers/admin.py tests/integration/test_admin_routes.py
git commit -m "feat(admin): /admin/fault with Literal validation"
```

---

### Task 7.5: `/admin/dataset/{bank_id}` — multipart upload to S3

**Files:**
- Modify: `server/app/routers/admin.py` (fill last stub)
- Modify: `server/app/storage.py` (add `put_stream` for chunked upload)
- Test: `tests/integration/test_dataset_upload.py` (new)

- [ ] **Step 1: Write failing test**

`tests/integration/test_dataset_upload.py`:

```python
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
    assert body["dataset_version"] == 2  # bumped from default 1

    cp = a.state.cp
    assert cp.banks["bank_03"].dataset_version == 2

    # S3 has the bytes
    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket="fl-test", Key="datasets/bank_03.csv")
    assert obj["Body"].read() == csv_bytes


@pytest.mark.integration
def test_dataset_upload_too_large_rejected(app, monkeypatch):
    # use a small max for the test
    monkeypatch.setenv("DATASET_UPLOAD_MAX_BYTES", "1024")
    a, client = app
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
```

- [ ] **Step 2: Add `dataset_upload_max_bytes` to `Settings`** (`server/app/config.py`)

Add field:
```python
    dataset_upload_max_bytes: int = 600 * 1024 * 1024  # 600MB cap
```

- [ ] **Step 3: Add `put_stream` to `server/app/storage.py`**

Append to the `Storage` class:

```python
    def put_stream(self, key: str, stream, content_length: int | None = None) -> None:
        """Streams an open file-like (with .read()) to S3 via multipart upload.
        Used by dataset upload — avoids loading the whole CSV in memory."""
        # boto3's upload_fileobj does the multipart logic for us.
        self._s3.upload_fileobj(stream, self.bucket, key)
```

- [ ] **Step 4: Replace `dataset` stub in `admin.py`**

```python
from fastapi import UploadFile

# inside protected router:
    @protected.post("/dataset/{bank_id}")
    async def dataset(bank_id: str, file: UploadFile):
        # body length cap (FastAPI doesn't enforce by default)
        max_bytes = Settings().dataset_upload_max_bytes
        contents = await file.read()  # for files this small (≤600MB) buffer in RAM is acceptable
        if len(contents) > max_bytes:
            raise HTTPException(413, f"dataset > {max_bytes} bytes")
        import io as _io
        storage.put_stream(f"datasets/{bank_id}.csv", _io.BytesIO(contents))
        cp.bump_dataset_version(bank_id)
        return {"bank_id": bank_id, "dataset_version": cp.banks[bank_id].dataset_version}
```

- [ ] **Step 5: Run — pass**

Run: `python -m pytest tests/integration/test_dataset_upload.py -v -m integration`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add server/app/routers/admin.py server/app/storage.py server/app/config.py tests/integration/test_dataset_upload.py
git commit -m "feat(admin): /admin/dataset/{bank_id} multipart upload + version bump"
```

---

### Task 7.6: Wire rollback into `round_loop`

**Files:**
- Modify: `server/app/round_loop.py` (call `eval.evaluate` after aggregation; rollback if metric drop > threshold)
- Modify: `server/app/eval.py` (add `load_validation_set(storage)` helper)
- Test: `tests/integration/test_round_loop_rollback.py` (uses moto)

- [ ] **Step 1: Write failing test**

`tests/integration/test_round_loop_rollback.py`:

```python
import io
import pickle

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import reset_pending
from server.app.routers.metrics import reset_history
from server.app.storage import Storage
from server.app.ws_hub import WsHub
from tests.shared.fakes import fake_post_update, fake_register


def _seed_validation_set(storage):
    """8 rows of bogus features so eval can run."""
    X = torch.randn(8, 19)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    buf = io.BytesIO()
    pickle.dump({"X": X, "y": y}, buf)
    storage.put_bytes("validation/val_set.pkl", buf.getvalue())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rollback_triggers_when_metric_drops_more_than_threshold(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        # tiny threshold so any drop trips rollback
        monkeypatch.setenv("ROLLBACK_THRESHOLD", "0.01")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())
        _seed_validation_set(st)

        rm = RoundManager(min_nodes=3, quorum_pct=0.6)
        cp = ControlPlane()
        hub = WsHub()
        reset_pending(); reset_history()

        # establish a baseline AUC at round >5 (rollback only fires after warm-up)
        rm.current_round = 6
        # Manually push a high baseline
        from server.app.routers.metrics import push_global_metrics
        push_global_metrics({"round": 6, "auc": 0.9, "method": "fedavg"})

        for i in range(3):
            fake_register(rm, f"bank_0{i+1}")
            fake_post_update(f"bank_0{i+1}", round=7, storage=st, perturbation=0.5)  # noisy → AUC will tank

        from server.app.round_loop import run_one_round
        await run_one_round(rm=rm, cp=cp, storage=st, hub=hub, settings=s, target_round=7)

        # Round 7 should have rolled back; latest_round still 6 (no global_round_0007 written)
        assert st.latest_round(prefix="models/global_round_") == 6
```

- [ ] **Step 2: Run — fails (rollback not wired)**

- [ ] **Step 3: Add `load_validation_set` to `server/app/eval.py`**

Append:

```python
import pickle


def load_validation_set(storage) -> tuple[torch.Tensor, np.ndarray]:
    """Reads val_set.pkl from S3. Returns (X tensor, y numpy)."""
    raw = storage.get_bytes("validation/val_set.pkl")
    obj = pickle.loads(raw)  # noqa: S301 — trusted source (we wrote it)
    return obj["X"], obj["y"]
```

- [ ] **Step 4: Modify `server/app/round_loop.py` — wire rollback after DP**

After the existing `new_global = dp.privatize(aggregated)` line and BEFORE `storage.put_weights(...)`, insert:

```python
    # Rollback gate: if eval metric dropped > threshold and we have history, abort.
    try:
        from server.app.eval import evaluate, load_validation_set
        from server.app.routers.metrics import _global_history

        X_val, y_val = load_validation_set(storage)
        candidate_model = FraudDetectionModel()
        candidate_model.set_weights(new_global)
        new_metrics = evaluate(candidate_model, X_val, y_val)

        prev_auc = None
        for row in reversed(_global_history):
            if "auc" in row:
                prev_auc = row["auc"]
                break
        delta = new_metrics.auc - (prev_auc or new_metrics.auc)
        if target_round > 5 and prev_auc is not None and delta < -settings.rollback_threshold:
            cp.global_.state = "idle"
            await hub.broadcast({
                "type": "round_rolled_back",
                "round": target_round,
                "prev_auc": prev_auc,
                "candidate_auc": new_metrics.auc,
            })
            _snapshot_control(cp, storage)
            return
        # otherwise record the new auc into the metric stream
        eval_metrics = {"auc": new_metrics.auc, "f1": new_metrics.f1, "precision": new_metrics.precision,
                        "recall": new_metrics.recall, "accuracy": new_metrics.accuracy,
                        "val_loss": new_metrics.val_loss}
    except Exception as e:
        _LOG.warning("eval/rollback skipped: %s", e)
        eval_metrics = {}
```

Modify the existing `push_global_metrics({...})` block to merge `eval_metrics`:

```python
    push_global_metrics({
        "round": target_round,
        "method": method,
        "n_participants": len(updates),
        "n_suspicious": len(suspicious),
        **eval_metrics,
    })
```

- [ ] **Step 5: Run — pass**

Run: `python -m pytest tests/integration/test_round_loop_rollback.py -v -m integration`
Expected: 1 passed.

- [ ] **Step 6: Verify all other tests still pass**

Run: `python -m pytest tests/unit tests/integration --tb=short`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add server/app/round_loop.py server/app/eval.py tests/integration/test_round_loop_rollback.py
git commit -m "feat(round_loop): wire rollback via eval.evaluate against S3 val_set.pkl"
```

---

### Task 7.7: `max_rounds` enforcement + dataset-version re-fetch in client

**Files:**
- Modify: `server/app/round_loop.py`
- Modify: `client/app/round_runner.py`
- Modify: `client/app/main.py`
- Test: extend `tests/unit/client/test_round_runner.py`, add `tests/integration/test_max_rounds.py`

- [ ] **Step 1: Add `max_rounds` check in `run_round_loop`**

In `server/app/round_loop.py:run_round_loop`, after `target = cp.global_.current_round + 1`, add:

```python
        if target > settings.max_rounds:
            cp.pause()
            await hub.broadcast({"type": "event", "level": "info", "msg": f"max_rounds={settings.max_rounds} reached; auto-pause"})
            await asyncio.sleep(5)
            continue
```

- [ ] **Step 2: Wire dataset re-fetch in `RoundRunner`**

Modify `client/app/round_runner.py`. Change the `dataset_loader` callable signature to accept the version (so `main.py` can decide whether to re-read):

```python
@dataclass
class RoundRunner:
    bank_id: str
    server: _Server
    storage: _Storage
    trainer: _Trainer
    dataset_loader: Callable[[int], tuple[torch.Tensor, torch.Tensor]]  # accepts current dataset_version
    last_round_seen: int = -1
    last_dataset_version: int = 0
    crashed_once: bool = False
    val_frac: float = 0.15
```

In `tick`, replace the `X, y = self.dataset_loader()` line with:

```python
        version = int(ctrl["dataset_version"])
        X, y = self.dataset_loader(version)
        self.last_dataset_version = version
```

- [ ] **Step 3: Update `client/app/main.py` to wrap dataset_loader with cache + re-fetch**

In `client/app/main.py:main`, replace the `runner = RoundRunner(...)` block:

```python
    cache = {"version": 0, "X": None, "y": None}

    def loader(version: int) -> tuple[torch.Tensor, torch.Tensor]:
        if cache["version"] != version or cache["X"] is None:
            _LOG.info("re-loading dataset (version %d)", version)
            assert_dataset_present(s.dataset_path, min_size_bytes=s.dataset_min_bytes)
            X_tr, y_tr, X_v, y_v, _ = preprocess(s.dataset_path, val_frac=0.15)
            cache["X"] = torch.cat([X_tr, X_v], dim=0)
            cache["y"] = torch.cat([y_tr, y_v], dim=0)
            cache["version"] = version
        return cache["X"], cache["y"]

    # warm cache once so first tick doesn't need to load
    loader(1)

    runner = RoundRunner(
        bank_id=s.bank_id,
        server=server,
        storage=storage,
        trainer=trainer,
        dataset_loader=loader,
        last_round_seen=-1,
    )
```

Note: the init container outside (Plan 3 helm chart) re-runs `aws s3 cp` only on pod restart; for live in-pod re-fetch, Plan 3 will add a small sidecar that watches dataset_version and re-downloads. For Plan 2 we accept that swapping a CSV via dashboard requires a pod restart unless the file at `/work/data/bank.csv` was overwritten externally.

- [ ] **Step 4: Update existing round_runner test signature**

In `tests/unit/client/test_round_runner.py`, update each `dataset_loader=lambda: ...` to `dataset_loader=lambda v: ...`.

- [ ] **Step 5: Add a max_rounds integration test (optional but desirable)**

`tests/integration/test_max_rounds.py`:

```python
import asyncio
import boto3
import pytest
from moto import mock_aws

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import reset_pending
from server.app.routers.metrics import reset_history
from server.app.storage import Storage
from server.app.ws_hub import WsHub


@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_rounds_pauses_loop(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        monkeypatch.setenv("MAX_ROUNDS", "2")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())

        rm = RoundManager(); cp = ControlPlane()
        cp.global_.current_round = 2  # already at max
        hub = WsHub()
        reset_pending(); reset_history()

        from server.app.round_loop import run_round_loop
        task = asyncio.create_task(run_round_loop(rm=rm, cp=cp, storage=st, hub=hub, settings=s))
        await asyncio.sleep(2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert cp.global_.paused is True
```

- [ ] **Step 6: Run — pass**

Run: `python -m pytest tests/unit/client/test_round_runner.py tests/integration/test_max_rounds.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add server/app/round_loop.py client/app/round_runner.py client/app/main.py tests/unit/client/test_round_runner.py tests/integration/test_max_rounds.py
git commit -m "feat: max_rounds auto-pause + dataset_version re-fetch hook in round_runner"
```

---

## Phase 8 — Dashboard React app

### Task 8.1: Vite + React scaffold

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/tsconfig.json`
- Create: `dashboard/vite.config.ts`
- Create: `dashboard/index.html`
- Create: `dashboard/src/main.tsx` (one-line bootstrap)
- Create: `dashboard/src/App.tsx` (placeholder "FL Demo" centered text)
- Create: `dashboard/.gitignore` (node_modules, dist)

- [ ] **Step 1: Write `dashboard/package.json`**

```json
{
  "name": "fl-dashboard",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "tsc --noEmit"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.2"
  },
  "devDependencies": {
    "@types/react": "^18.3.10",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.2",
    "typescript": "^5.6.2",
    "vite": "^5.4.8"
  }
}
```

- [ ] **Step 2: Write `dashboard/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Write `dashboard/vite.config.ts`**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "../server/app/static"),
    emptyOutDir: true,
  },
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    proxy: {
      "/admin": "http://localhost:8080",
      "/banks": "http://localhost:8080",
      "/metrics": "http://localhost:8080",
      "/round": "http://localhost:8080",
      "/control": "http://localhost:8080",
      "/health": "http://localhost:8080",
      "/ws":     { target: "ws://localhost:8080", ws: true },
    },
  },
});
```

- [ ] **Step 4: Write `dashboard/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FL Demo</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Write `dashboard/src/main.tsx`**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
```

- [ ] **Step 6: Write `dashboard/src/App.tsx` (placeholder)**

```tsx
export default function App() {
  return (
    <div style={{ padding: 24 }}>
      <h1>FL Demo</h1>
      <p>Dashboard scaffold ready. Plan 2 will wire the real UI.</p>
    </div>
  );
}
```

- [ ] **Step 7: Write `dashboard/src/index.css`**

```css
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #0e1117;
  color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
```

- [ ] **Step 8: Write `dashboard/.gitignore`**

```
node_modules/
dist/
.vite/
```

- [ ] **Step 9: Install + build**

Run from `dashboard/`:

```bash
cd dashboard
npm install
npm run build
```

Expected: `vite build` writes to `../server/app/static/` (which is `server/app/static/`).

- [ ] **Step 10: Verify the bundle is served by fl-server**

Run from repo root:

```bash
python -m pytest tests/integration/test_main_smoke.py -v -m integration
```

The static-mount test should now return 200 (not 404) for `/`.

- [ ] **Step 11: Commit**

```bash
git add dashboard/ server/app/static/
git commit -m "feat(dashboard): vite + react scaffold; bundle into server/app/static"
```

---

### Task 8.2: Typed REST client + WS client

**Files:**
- Create: `dashboard/src/lib/types.ts`
- Create: `dashboard/src/lib/api.ts`
- Create: `dashboard/src/lib/ws.ts`

- [ ] **Step 1: Write `dashboard/src/lib/types.ts`**

```ts
export type Fault = "none" | "crash" | "straggle" | "byzantine" | "partition";
export type RoundState = "idle" | "collecting" | "aggregating" | "stalled";

export interface Bank {
  bank_id: string;
  bank_name: string;
  n_samples: number;
  trust: number;
  suspended: boolean;
  dataset_version: number;
  fault: Fault;
  cumulative_eps: number;
}

export interface RoundStatus {
  round: number;
  state: RoundState;
  paused: boolean;
  active_banks: number;
  quorum_size: number;
}

export interface BankMetricRow {
  round: number;
  train_loss?: number;
  val_loss?: number;
  val_auc?: number;
  val_f1?: number;
  val_precision?: number;
  val_recall?: number;
  val_accuracy?: number;
  weight_norm?: number;
  dp_sigma?: number;
}

export interface GlobalMetricRow {
  round: number;
  method?: string;
  n_participants?: number;
  n_suspicious?: number;
  auc?: number;
  f1?: number;
  precision?: number;
  recall?: number;
  accuracy?: number;
  val_loss?: number;
}

export type WsEvent =
  | { type: "round_started"; round: number; quorum_size: number }
  | { type: "round_completed"; round: number; method: string }
  | { type: "round_stalled"; round: number; received: number; reason?: string }
  | { type: "round_rolled_back"; round: number; prev_auc: number; candidate_auc: number }
  | { type: "bank_update"; bank_id: string; round: number; metrics: Record<string, number> }
  | { type: "event"; level: "info" | "warn" | "error"; msg: string };
```

- [ ] **Step 2: Write `dashboard/src/lib/api.ts`**

```ts
import type { Bank, BankMetricRow, GlobalMetricRow, RoundStatus, Fault } from "./types";

async function http<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    credentials: "include",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${method} ${url} → ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  login:  (password: string) => http<{ ok: boolean }>("POST", "/admin/login", { password }),
  logout: ()                  => http<{ ok: boolean }>("POST", "/admin/logout"),
  pause:  ()                  => http<{ paused: boolean }>("POST", "/admin/pause"),
  resume: ()                  => http<{ paused: boolean }>("POST", "/admin/resume"),
  reset:  ()                  => http<{ current_round: number }>("POST", "/admin/reset"),
  setFault: (bank_id: string, fault: Fault) =>
              http<{ bank_id: string; fault: Fault }>("POST", "/admin/fault", { bank_id, fault }),

  uploadDataset: async (bank_id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`/admin/dataset/${bank_id}`, { method: "POST", body: fd, credentials: "include" });
    if (!r.ok) throw new Error(`upload → ${r.status}`);
    return r.json() as Promise<{ bank_id: string; dataset_version: number }>;
  },

  banks:        () => http<Bank[]>("GET", "/banks"),
  bankHistory:  (bank_id: string, n = 50) =>
                  http<BankMetricRow[]>("GET", `/banks/${bank_id}/history?n=${n}`),
  metrics:      (n = 50) =>
                  http<{ history: GlobalMetricRow[]; cumulative_eps_global: number; current_round: number }>(
                    "GET", `/metrics?n=${n}`,
                  ),
  roundStatus:  () => http<RoundStatus>("GET", "/round/status"),
};
```

- [ ] **Step 3: Write `dashboard/src/lib/ws.ts`**

```ts
import type { WsEvent } from "./types";

export class LiveSocket {
  private ws: WebSocket | null = null;
  private listeners: ((e: WsEvent) => void)[] = [];
  private retryDelayMs = 1000;

  connect() {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`;
    this.ws = new WebSocket(url);
    this.ws.onmessage = (ev) => {
      try {
        const data: WsEvent = JSON.parse(ev.data);
        this.listeners.forEach((fn) => fn(data));
      } catch (e) {
        console.warn("bad ws frame", e);
      }
    };
    this.ws.onclose = () => {
      this.ws = null;
      setTimeout(() => this.connect(), this.retryDelayMs);
    };
    this.ws.onerror = () => this.ws?.close();
  }

  subscribe(fn: (e: WsEvent) => void): () => void {
    this.listeners.push(fn);
    return () => { this.listeners = this.listeners.filter((f) => f !== fn); };
  }
}

export const liveSocket = new LiveSocket();
```

- [ ] **Step 4: Smoke check the build still passes**

Run from `dashboard/`:

```bash
npm run build
```

Expected: `vite build` succeeds, `tsc` finds no type errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/
git commit -m "feat(dashboard): typed REST client + WebSocket client with reconnect"
```

---

### Task 8.3: Login page + auth context

**Files:**
- Create: `dashboard/src/pages/Login.tsx`
- Modify: `dashboard/src/App.tsx` (add router + auth guard)
- Create: `dashboard/src/lib/auth-context.tsx`

- [ ] **Step 1: Write `dashboard/src/lib/auth-context.tsx`**

```tsx
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api";

type AuthState = { loggedIn: boolean; loading: boolean };
const AuthCtx = createContext<{ state: AuthState; refresh: () => Promise<void> }>({
  state: { loggedIn: false, loading: true },
  refresh: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ loggedIn: false, loading: true });

  // Plan 2: trust localStorage flag set on successful login. The cookie is
  // HttpOnly so JS can't read it; cookie validity is enforced server-side on
  // every gated request. If the flag drifts (cookie expired) the next admin
  // call will 401 and the user is re-prompted.
  const refresh = async () => {
    const v = localStorage.getItem("fl_logged_in") === "1";
    setState({ loggedIn: v, loading: false });
  };

  useEffect(() => { refresh(); }, []);

  return <AuthCtx.Provider value={{ state, refresh }}>{children}</AuthCtx.Provider>;
}

export const useAuth = () => useContext(AuthCtx);

export async function performLogin(password: string) {
  await api.login(password);
  localStorage.setItem("fl_logged_in", "1");
}

export async function performLogout() {
  await api.logout();
  localStorage.removeItem("fl_logged_in");
}
```

- [ ] **Step 2: Write `dashboard/src/pages/Login.tsx`**

```tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { performLogin, useAuth } from "../lib/auth-context";

export default function Login() {
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();
  const { refresh } = useAuth();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    try {
      await performLogin(password);
      await refresh();
      nav("/");
    } catch {
      setErr("invalid password");
    }
  };

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh" }}>
      <form onSubmit={submit} style={{ background: "#161b22", padding: 24, borderRadius: 8, minWidth: 320 }}>
        <h2>FL Demo</h2>
        <input
          type="password"
          autoFocus
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="admin password"
          style={{ width: "100%", padding: 10, fontSize: 14, marginBottom: 12,
                   background: "#0e1117", color: "#e6edf3", border: "1px solid #30363d", borderRadius: 6 }}
        />
        <button type="submit" style={{ width: "100%", padding: 10, background: "#1f6feb", color: "#fff",
                                       border: 0, borderRadius: 6, fontSize: 14, cursor: "pointer" }}>
          Sign in
        </button>
        {err && <p style={{ color: "#f85149", marginTop: 12 }}>{err}</p>}
      </form>
    </div>
  );
}
```

- [ ] **Step 3: Replace `dashboard/src/App.tsx`**

```tsx
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./lib/auth-context";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";

function Guard({ children }: { children: JSX.Element }) {
  const { state } = useAuth();
  if (state.loading) return <div style={{ padding: 24 }}>Loading…</div>;
  return state.loggedIn ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Guard><Dashboard /></Guard>} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
```

- [ ] **Step 4: Add a placeholder `Dashboard.tsx`**

`dashboard/src/pages/Dashboard.tsx`:

```tsx
import { performLogout } from "../lib/auth-context";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const nav = useNavigate();
  const logout = async () => {
    await performLogout();
    nav("/login");
  };
  return (
    <div style={{ padding: 24 }}>
      <h1>FL Demo (placeholder)</h1>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

- [ ] **Step 5: Build + smoke**

Run from `dashboard/`:

```bash
npm run build
```

Expected: success.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/
git commit -m "feat(dashboard): login page + auth context + router with guard"
```

---

### Task 8.4: Sparkline primitive + GlobalMetrics

**Files:**
- Create: `dashboard/src/components/Sparkline.tsx`
- Create: `dashboard/src/components/GlobalMetrics.tsx`

- [ ] **Step 1: Write `Sparkline.tsx`** (no chart lib — pure SVG)

```tsx
type Props = { values: number[]; color: string; height?: number };

export default function Sparkline({ values, color, height = 50 }: Props) {
  if (values.length < 2) {
    return <svg width="100%" height={height}><text x="4" y="14" fill="#7d8590" fontSize="10">no data</text></svg>;
  }
  const W = 200, H = height;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const dx = W / (values.length - 1);
  const points = values.map((v, i) => `${i * dx},${H - ((v - min) / span) * (H - 4) - 2}`).join(" L ");
  const path = `M ${points}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height }}>
      <path d={path} stroke={color} strokeWidth={1.5} fill="none" />
      <path d={`${path} L ${W},${H} L 0,${H} Z`} fill={color} fillOpacity={0.15} />
    </svg>
  );
}
```

- [ ] **Step 2: Write `GlobalMetrics.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { GlobalMetricRow } from "../lib/types";
import Sparkline from "./Sparkline";

const SPECS: { key: keyof GlobalMetricRow; label: string; color: string }[] = [
  { key: "auc",        label: "AUC-ROC",   color: "#58a6ff" },
  { key: "f1",         label: "F1",        color: "#a371f7" },
  { key: "precision",  label: "Precision", color: "#3fb950" },
  { key: "recall",     label: "Recall",    color: "#d29922" },
  { key: "val_loss",   label: "Val loss",  color: "#f85149" },
];

export default function GlobalMetrics({ history }: { history: GlobalMetricRow[] }) {
  return (
    <div style={{ padding: "14px 18px", borderBottom: "1px solid #30363d" }}>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
        Global model — last {history.length} rounds
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
        {SPECS.map(({ key, label, color }) => {
          const series = history.map((r) => r[key] ?? null).filter((v): v is number => v != null);
          const current = series.length ? series[series.length - 1].toFixed(3) : "—";
          return (
            <div key={key} style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6, padding: 8 }}>
              <div style={{ fontSize: 10, opacity: 0.7, display: "flex", justifyContent: "space-between" }}>
                <span>{label}</span>
                <b style={{ color }}>{current}</b>
              </div>
              <Sparkline values={series} color={color} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Smoke build**

```bash
npm run build
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/Sparkline.tsx dashboard/src/components/GlobalMetrics.tsx
git commit -m "feat(dashboard): Sparkline primitive + GlobalMetrics 5-strip"
```

---

### Task 8.5: BankCard + BankGrid

**Files:**
- Create: `dashboard/src/components/BankCard.tsx`
- Create: `dashboard/src/components/BankGrid.tsx`

- [ ] **Step 1: Write `BankCard.tsx`**

```tsx
import type { Bank, BankMetricRow, Fault } from "../lib/types";
import Sparkline from "./Sparkline";

const STATUS_COLOR: Record<Fault, string> = {
  none: "#3fb950", crash: "#f85149", straggle: "#d29922",
  byzantine: "#d29922", partition: "#f85149",
};

export default function BankCard({
  bank, history, onClick, onFault, onSwap,
}: {
  bank: Bank; history: BankMetricRow[];
  onClick: () => void;
  onFault: () => void;
  onSwap: () => void;
}) {
  const series = history.map((r) => r.val_auc).filter((v): v is number => v != null);
  const last = history.at(-1) ?? {} as BankMetricRow;
  const border = bank.fault === "none" ? "#30363d" : STATUS_COLOR[bank.fault];
  return (
    <div
      onClick={onClick}
      style={{ background: "#161b22", border: `1px solid ${border}`, borderRadius: 8,
               padding: 10, cursor: "pointer" }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13 }}>{bank.bank_id}</div>
          <div style={{ fontSize: 10, opacity: 0.6 }}>
            trust {bank.trust.toFixed(2)} · ds v{bank.dataset_version} · n={bank.n_samples}
          </div>
        </div>
        <span style={{ background: STATUS_COLOR[bank.fault] + "33", color: STATUS_COLOR[bank.fault],
                       padding: "2px 7px", borderRadius: 10, fontSize: 10 }}>
          {bank.fault === "none" ? "idle" : bank.fault}
        </span>
      </div>
      <Sparkline values={series} color="#58a6ff" height={30} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 4,
                    fontFamily: "monospace", fontSize: 9.5, marginTop: 6 }}>
        <div><span style={{ opacity: 0.6 }}>loss</span> <b>{last.val_loss?.toFixed(3) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>auc</span>  <b>{last.val_auc?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>f1</span>   <b>{last.val_f1?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>prec</span> <b>{last.val_precision?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>rec</span>  <b>{last.val_recall?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>acc</span>  <b>{last.val_accuracy?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>‖w‖</span>  <b>{last.weight_norm?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>σ</span>    <b>{last.dp_sigma?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>r#</span>   <b>{last.round ?? "—"}</b></div>
      </div>
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button onClick={(e) => { e.stopPropagation(); onFault(); }}
                style={btn(border)}>⚠ fault</button>
        <button onClick={(e) => { e.stopPropagation(); onSwap(); }}
                style={btn("#30363d")}>📂 swap</button>
      </div>
    </div>
  );
}

const btn = (border: string): React.CSSProperties => ({
  flex: 1, background: "#21262d", border: `1px solid ${border}`,
  color: "#e6edf3", padding: 3, borderRadius: 4, fontSize: 10, cursor: "pointer",
});
```

- [ ] **Step 2: Write `BankGrid.tsx`**

```tsx
import type { Bank, BankMetricRow, Fault } from "../lib/types";
import BankCard from "./BankCard";

export default function BankGrid({
  banks, histories, onCardClick, onFault, onSwap,
}: {
  banks: Bank[];
  histories: Record<string, BankMetricRow[]>;
  onCardClick: (id: string) => void;
  onFault: (id: string) => void;
  onSwap: (id: string) => void;
}) {
  return (
    <div style={{ padding: "14px 18px" }}>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
        Banks · click card for full local history · drop CSV to swap dataset
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        {banks.map((b) => (
          <BankCard
            key={b.bank_id}
            bank={b}
            history={histories[b.bank_id] ?? []}
            onClick={() => onCardClick(b.bank_id)}
            onFault={() => onFault(b.bank_id)}
            onSwap={() => onSwap(b.bank_id)}
          />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Smoke build**

```bash
npm run build
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/BankCard.tsx dashboard/src/components/BankGrid.tsx
git commit -m "feat(dashboard): BankCard + BankGrid components"
```

---

### Task 8.6: TopBar + ControlPanel + EventLog

**Files:**
- Create: `dashboard/src/components/TopBar.tsx`
- Create: `dashboard/src/components/EventLog.tsx`

- [ ] **Step 1: Write `TopBar.tsx`**

```tsx
import type { RoundStatus } from "../lib/types";
import { api } from "../lib/api";
import { performLogout } from "../lib/auth-context";
import { useNavigate } from "react-router-dom";

export default function TopBar({ status, banks, eps, onChange }: {
  status: RoundStatus; banks: number; eps: number; onChange: () => void;
}) {
  const nav = useNavigate();
  const action = (fn: () => Promise<unknown>) => async () => { await fn(); onChange(); };

  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "12px 18px", borderBottom: "1px solid #30363d", background: "#161b22" }}>
      <div>
        <span style={{ fontWeight: 700, fontSize: 15 }}>FL Demo</span>
        <span style={{ opacity: 0.6, marginLeft: 10, fontSize: 12 }}>
          round&nbsp;<b style={{ color: "#58a6ff" }}>{status.round}</b> ·
          state&nbsp;<b style={{ color: status.state === "stalled" ? "#f85149" : "#3fb950" }}>{status.state}</b> ·
          banks&nbsp;<b>{banks}</b> · ε&nbsp;<b>{eps.toFixed(2)}</b>
        </span>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button style={ctrl} onClick={action(api.resume)}>▶ Resume</button>
        <button style={ctrl} onClick={action(api.pause)}>⏸ Pause</button>
        <button style={ctrl} onClick={action(api.reset)}>↺ Reset</button>
        <button style={{ ...ctrl, borderColor: "#f85149", color: "#f85149" }}
                onClick={async () => { await performLogout(); nav("/login"); }}>Logout</button>
      </div>
    </div>
  );
}

const ctrl: React.CSSProperties = {
  background: "#21262d", border: "1px solid #30363d", color: "#e6edf3",
  padding: "5px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer",
};
```

- [ ] **Step 2: Write `EventLog.tsx`**

```tsx
import type { WsEvent } from "../lib/types";

const COLOR: Record<string, string> = {
  info: "#58a6ff", warn: "#d29922", error: "#f85149", ok: "#3fb950",
};

function describe(e: WsEvent): { color: string; text: string } {
  switch (e.type) {
    case "round_started":   return { color: "info",  text: `round ${e.round} started — quorum ${e.quorum_size} needed` };
    case "round_completed": return { color: "ok",    text: `round ${e.round} published via ${e.method}` };
    case "round_stalled":   return { color: "warn",  text: `round ${e.round} stalled (${e.received} received)` };
    case "round_rolled_back": return { color: "warn", text: `round ${e.round} ROLLED BACK (auc ${e.candidate_auc.toFixed(3)} < ${e.prev_auc.toFixed(3)})` };
    case "bank_update":     return { color: "ok",    text: `${e.bank_id} round ${e.round} auc=${(e.metrics.val_auc ?? 0).toFixed(2)}` };
    case "event":           return { color: e.level, text: e.msg };
  }
}

export default function EventLog({ events }: { events: WsEvent[] }) {
  return (
    <div style={{ padding: "14px 18px", borderTop: "1px solid #30363d" }}>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
        Event log (live, WebSocket)
      </div>
      <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6,
                    padding: 10, fontFamily: "monospace", fontSize: 11, lineHeight: 1.7,
                    maxHeight: 200, overflow: "auto" }}>
        {events.slice(-50).map((e, i) => {
          const { color, text } = describe(e);
          const ts = new Date().toLocaleTimeString();
          return (
            <div key={i}>
              <span style={{ color: COLOR[color] }}>{ts}</span> {text}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Smoke build + commit**

```bash
npm run build
git add dashboard/src/components/TopBar.tsx dashboard/src/components/EventLog.tsx
git commit -m "feat(dashboard): TopBar with global controls + EventLog (WS feed)"
```

---

### Task 8.7: BankDrillIn modal + DatasetUpload + FaultPanel

**Files:**
- Create: `dashboard/src/components/BankDrillIn.tsx`
- Create: `dashboard/src/components/DatasetUpload.tsx`
- Create: `dashboard/src/components/FaultPanel.tsx`

- [ ] **Step 1: Write `BankDrillIn.tsx`**

```tsx
import type { BankMetricRow, GlobalMetricRow } from "../lib/types";
import Sparkline from "./Sparkline";

export default function BankDrillIn({
  bankId, local, global_, onClose,
}: {
  bankId: string; local: BankMetricRow[]; global_: GlobalMetricRow[]; onClose: () => void;
}) {
  const last = local.at(-1) ?? {} as BankMetricRow;
  return (
    <div style={overlay} onClick={onClose}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>{bankId} — local history</div>
            <div style={{ fontSize: 10, opacity: 0.6 }}>{local.length} rounds</div>
          </div>
          <button onClick={onClose} style={closeBtn}>close ✕</button>
        </div>
        <div style={{ fontSize: 10, opacity: 0.7, margin: "6px 0", textTransform: "uppercase" }}>
          LOCAL val_auc vs GLOBAL auc
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <Sparkline values={local.map((r) => r.val_auc ?? 0)} color="#d29922" height={80} />
          <Sparkline values={global_.map((r) => r.auc ?? 0)} color="#3fb950" height={80} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginTop: 12,
                      fontFamily: "monospace", fontSize: 10 }}>
          {(Object.entries(last) as [keyof BankMetricRow, number | undefined][]).map(([k, v]) => (
            <div key={k}>
              <div style={{ opacity: 0.6 }}>{k}</div>
              <b>{v != null ? (typeof v === "number" ? v.toFixed(3) : v) : "—"}</b>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = {
  position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100,
};
const panel: React.CSSProperties = {
  background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
  padding: 16, maxWidth: 800, width: "90%", maxHeight: "80vh", overflow: "auto",
};
const closeBtn: React.CSSProperties = {
  background: "#21262d", border: "1px solid #30363d", color: "#e6edf3",
  padding: "4px 10px", borderRadius: 4, fontSize: 11, cursor: "pointer",
};
```

- [ ] **Step 2: Write `DatasetUpload.tsx`** (file picker dialog)

```tsx
import { useRef, useState } from "react";
import { api } from "../lib/api";

export default function DatasetUpload({
  bankId, onClose,
}: { bankId: string; onClose: (success: boolean) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const upload = async () => {
    const f = inputRef.current?.files?.[0];
    if (!f) { setErr("pick a file first"); return; }
    setBusy(true); setErr(null);
    try {
      await api.uploadDataset(bankId, f);
      onClose(true);
    } catch (e: unknown) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={overlay} onClick={() => onClose(false)}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Swap dataset for {bankId}</h3>
        <input ref={inputRef} type="file" accept=".csv" />
        <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
          <button onClick={upload} disabled={busy} style={primary}>
            {busy ? "uploading…" : "Upload"}
          </button>
          <button onClick={() => onClose(false)} style={secondary}>Cancel</button>
        </div>
        {err && <p style={{ color: "#f85149", marginTop: 12 }}>{err}</p>}
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = { position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 };
const panel: React.CSSProperties = { background: "#161b22", border: "1px solid #30363d",
  borderRadius: 8, padding: 16, minWidth: 340 };
const primary: React.CSSProperties = { background: "#1f6feb", color: "#fff",
  border: 0, borderRadius: 6, padding: "6px 14px", cursor: "pointer" };
const secondary: React.CSSProperties = { background: "#21262d", color: "#e6edf3",
  border: "1px solid #30363d", borderRadius: 6, padding: "6px 14px", cursor: "pointer" };
```

- [ ] **Step 3: Write `FaultPanel.tsx`** (modal with 5 buttons)

```tsx
import { api } from "../lib/api";
import type { Fault } from "../lib/types";

const FAULTS: Fault[] = ["none", "crash", "straggle", "byzantine", "partition"];

export default function FaultPanel({
  bankId, currentFault, onClose,
}: { bankId: string; currentFault: Fault; onClose: () => void }) {
  const set = async (f: Fault) => { await api.setFault(bankId, f); onClose(); };
  return (
    <div style={overlay} onClick={onClose}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Fault for {bankId}</h3>
        <p style={{ fontSize: 11, opacity: 0.7 }}>currently: {currentFault}</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {FAULTS.map((f) => (
            <button key={f} onClick={() => set(f)}
                    style={{ ...btn, ...(currentFault === f ? { borderColor: "#1f6feb" } : {}) }}>
              {f}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = { position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 };
const panel: React.CSSProperties = { background: "#161b22", border: "1px solid #30363d",
  borderRadius: 8, padding: 16, minWidth: 340 };
const btn: React.CSSProperties = { background: "#21262d", color: "#e6edf3",
  border: "1px solid #30363d", borderRadius: 6, padding: "6px 14px", cursor: "pointer", fontSize: 12 };
```

- [ ] **Step 4: Smoke build + commit**

```bash
npm run build
git add dashboard/src/components/BankDrillIn.tsx dashboard/src/components/DatasetUpload.tsx dashboard/src/components/FaultPanel.tsx
git commit -m "feat(dashboard): BankDrillIn modal + DatasetUpload + FaultPanel"
```

---

### Task 8.8: Wire everything in `Dashboard.tsx`

**Files:**
- Modify: `dashboard/src/pages/Dashboard.tsx` (replace placeholder)

- [ ] **Step 1: Replace `Dashboard.tsx`**

```tsx
import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { liveSocket } from "../lib/ws";
import type { Bank, BankMetricRow, GlobalMetricRow, RoundStatus, WsEvent } from "../lib/types";
import TopBar from "../components/TopBar";
import GlobalMetrics from "../components/GlobalMetrics";
import BankGrid from "../components/BankGrid";
import EventLog from "../components/EventLog";
import BankDrillIn from "../components/BankDrillIn";
import DatasetUpload from "../components/DatasetUpload";
import FaultPanel from "../components/FaultPanel";

const POLL_MS = 2000;

export default function Dashboard() {
  const [banks, setBanks] = useState<Bank[]>([]);
  const [status, setStatus] = useState<RoundStatus>({ round: 0, state: "idle", paused: false, active_banks: 0, quorum_size: 0 });
  const [globalHistory, setGlobalHistory] = useState<GlobalMetricRow[]>([]);
  const [bankHistories, setBankHistories] = useState<Record<string, BankMetricRow[]>>({});
  const [eps, setEps] = useState(0);
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [drillIn, setDrillIn] = useState<string | null>(null);
  const [uploadFor, setUploadFor] = useState<string | null>(null);
  const [faultFor, setFaultFor] = useState<string | null>(null);

  const refresh = async () => {
    const [b, s, m] = await Promise.all([api.banks(), api.roundStatus(), api.metrics(50)]);
    setBanks(b); setStatus(s); setGlobalHistory(m.history); setEps(m.cumulative_eps_global);
    const histories: Record<string, BankMetricRow[]> = {};
    await Promise.all(b.map(async (bk) => {
      histories[bk.bank_id] = await api.bankHistory(bk.bank_id, 50);
    }));
    setBankHistories(histories);
  };

  useEffect(() => {
    refresh().catch(console.warn);
    const id = setInterval(() => refresh().catch(console.warn), POLL_MS);
    liveSocket.connect();
    const unsub = liveSocket.subscribe((e) => {
      setEvents((prev) => [...prev, e].slice(-100));
      // event-driven refresh on round transitions for snappier UI
      if (e.type === "round_completed" || e.type === "round_stalled" || e.type === "round_rolled_back") {
        refresh().catch(console.warn);
      }
    });
    return () => { clearInterval(id); unsub(); };
  }, []);

  const currentFault = useMemo(
    () => banks.find((b) => b.bank_id === faultFor)?.fault ?? "none",
    [banks, faultFor],
  );

  return (
    <div>
      <TopBar status={status} banks={banks.length} eps={eps} onChange={refresh} />
      <GlobalMetrics history={globalHistory} />
      <BankGrid
        banks={banks}
        histories={bankHistories}
        onCardClick={setDrillIn}
        onFault={setFaultFor}
        onSwap={setUploadFor}
      />
      <EventLog events={events} />
      {drillIn && (
        <BankDrillIn bankId={drillIn} local={bankHistories[drillIn] ?? []} global_={globalHistory}
                     onClose={() => setDrillIn(null)} />
      )}
      {uploadFor && (
        <DatasetUpload bankId={uploadFor} onClose={(ok) => { setUploadFor(null); if (ok) refresh(); }} />
      )}
      {faultFor && (
        <FaultPanel bankId={faultFor} currentFault={currentFault}
                    onClose={() => { setFaultFor(null); refresh(); }} />
      )}
    </div>
  );
}
```

- [ ] **Step 2: Build + verify**

```bash
cd dashboard && npm run build && cd -
python -m pytest tests/integration/test_main_smoke.py -v -m integration
```

Expected: build succeeds; smoke `/` returns 200 (dashboard is mounted).

- [ ] **Step 3: Manual visual check (optional but helpful)**

Run dev server in one shell:
```bash
cd dashboard && npm run dev
```
Open `http://localhost:5173/login`, log in with the password set via `ADMIN_PASSWORD_HASH`. (Use the docker-compose stack from Plan 1 for the backend.)

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/pages/Dashboard.tsx server/app/static/
git commit -m "feat(dashboard): wire TopBar + GlobalMetrics + BankGrid + EventLog + modals"
```

---

## Phase 9 — Final integration + tests

### Task 9.1: CI workflow update — build dashboard before pytest

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Modify `lint-and-test` job** — insert before `Lint`:

```yaml
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: dashboard/package-lock.json
      - name: Build dashboard
        run: |
          cd dashboard
          npm ci
          npm run build
      - name: Lint dashboard (tsc --noEmit)
        run: |
          cd dashboard
          npm run lint
```

- [ ] **Step 2: Push + verify CI green**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build dashboard before python tests; add tsc lint job"
git push origin claude/plan-2-dashboard
```

Watch GitHub Actions; expect green.

---

### Task 9.2: Final verification

- [ ] **Step 1: Run the full suite**

```bash
python -m pytest tests/unit tests/integration -v --tb=short
ruff check server client tests
cd dashboard && npm run build && npm run lint && cd -
```

Expected: all green.

- [ ] **Step 2: Tag**

```bash
git tag plan-2-complete
git push origin plan-2-complete
```

---

## What's NOT done in Plan 2 (intentionally deferred)

- Terraform infra, helm chart, `deploy.sh`/`teardown.sh` (Plan 3)
- ECR push automation (Plan 3)
- Real validation set generator `dataset/build_val_set.py` (Plan 3 builds the production version; Plan 2 uses an in-test pickle)
- NetworkPolicies + IRSA bindings (Plan 3)
- HTTPS / ACM cert wiring on ALB (Plan 3)
- Cleanup of v1 legacy files at `server/*.py`, `client/*.py`, `helm/`, `terraform/` (Plan 3)
- Switching FastAPI `on_event` → `lifespan` (cosmetic; deferred until either a deprecation breaks)
- Dashboard polish: drag-drop CSV directly on BankCard (Plan 2 uses a modal file-picker — same effect, simpler)
- Multi-user dashboard / roles (single-operator demo)

---

*End of Plan 2.*

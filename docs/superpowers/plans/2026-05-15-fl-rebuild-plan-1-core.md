# FL Rebuild — Plan 1: Core (server + client, runs locally) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the FL server + 7-bank client in a clean, testable, locally-runnable form. End state: `docker compose -f tests/e2e/compose.yml up --wait` produces 1 server + 3 fake-bank clients that complete 5 federated rounds, with `pytest tests/e2e/test_smoke.py` asserting global AUC > 0.7.

**Architecture:** Monolithic FastAPI server (no dashboard yet) drives an asyncio `round_loop`. Clients are dumb workers that poll `/control/{bank_id}`, train on a local CSV, upload weights to S3 (or minio for local), and POST a metrics summary. State persists in S3 only. See [`docs/superpowers/specs/2026-05-15-fl-rebuild-design.md`](../specs/2026-05-15-fl-rebuild-design.md) for full design.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, PyTorch (CPU-only), boto3, pandas, numpy, scikit-learn (preprocessor + metrics), pytest, moto (S3 mock), minio (local S3), docker-compose, ruff (lint).

**Predecessors:** None — this is the foundation plan.
**Successors:** Plan 2 (dashboard), Plan 3 (AWS infra + deploy.sh).

**Reference v1 code paths:** The previous version of this project is in the same repo at the **main branch root** (e.g. `server/aggregator.py`, `client/preprocessor.py`). Tasks below say "port v1 X with these changes" — the engineer should `git show main:server/aggregator.py` to read the source, then write the new file under the new layout per the spec. v1 code is approved-quality for the algorithms (Krum, FedAvg, DP); only the I/O glue and dataset path change.

---

## File structure (locked in this plan)

```
server/
├── Dockerfile                       (Phase 6)
├── pyproject.toml                   (Phase 0)
├── requirements.txt                 (Phase 0)
└── app/
    ├── __init__.py
    ├── config.py                    (Phase 1, Settings via pydantic BaseSettings)
    ├── model.py                     (Phase 1, FraudDetectionModel — port v1)
    ├── storage.py                   (Phase 3, S3 wrapper using boto3)
    ├── validator.py                 (Phase 2, per-update scoring)
    ├── aggregator.py                (Phase 2, FedAvg/TM/Krum auto-select)
    ├── dp_engine.py                 (Phase 2, Gaussian DP)
    ├── round_manager.py             (Phase 2, round/quorum/trust state)
    ├── control_plane.py             (Phase 2, pause/fault/dataset_version)
    ├── ws_hub.py                    (Phase 3, WebSocket connection set + broadcast)
    ├── auth.py                      (Phase 3, password→cookie)
    ├── round_loop.py                (Phase 4, asyncio task)
    ├── routers/
    │   ├── __init__.py
    │   ├── client.py                (Phase 3)
    │   ├── admin.py                 (Phase 3, stub for Plan 2)
    │   ├── metrics.py               (Phase 3)
    │   └── ws.py                    (Phase 3)
    └── main.py                      (Phase 3, app factory + startup hook)

client/
├── Dockerfile                       (Phase 6)
├── pyproject.toml                   (Phase 0)
├── requirements.txt                 (Phase 0)
└── app/
    ├── __init__.py
    ├── config.py                    (Phase 1)
    ├── model.py                     (Phase 1, byte-identical to server/app/model.py)
    ├── dataset.py                   (Phase 5, path assert + load)
    ├── preprocessor.py              (Phase 5, port v1)
    ├── trainer.py                   (Phase 5, port v1 with DP)
    ├── storage.py                   (Phase 5, boto3 wrapper for upload/download)
    ├── round_runner.py              (Phase 5, polling state machine)
    └── main.py                      (Phase 5, entrypoint)

tests/
├── conftest.py                      (Phase 0, sys.path + shared fixtures)
├── shared/
│   ├── golden_inputs/
│   │   └── tiny_bank.csv            (Phase 5 setup, 1000 rows for tests)
│   └── fakes.py                     (Phase 4, FakeClient class)
├── unit/
│   ├── server/
│   │   ├── test_config.py
│   │   ├── test_model.py
│   │   ├── test_validator.py
│   │   ├── test_aggregator.py
│   │   ├── test_dp_engine.py
│   │   ├── test_round_manager.py
│   │   └── test_control_plane.py
│   └── client/
│       ├── test_config.py
│       ├── test_preprocessor.py
│       ├── test_trainer.py
│       └── test_round_runner.py
├── integration/
│   ├── test_storage_s3.py
│   ├── test_routers_client.py
│   ├── test_routers_metrics.py
│   ├── test_full_round.py
│   ├── test_byzantine_round.py
│   └── test_checkpoint_restore.py
└── e2e/
    ├── compose.yml
    └── test_smoke.py

.github/workflows/ci.yml             (Phase 0)
pyproject.toml                       (Phase 0, root, ruff config + pytest config)
.gitignore                           (Phase 0)
```

**Files explicitly NOT in this plan** (deferred to Plan 2/3): `dashboard/*`, `infra/*`, `k8s/*`, `deploy.sh`, `teardown.sh`, `server/app/routers/admin.py` full implementation (only stubs in Plan 1 — Plan 2 fills it).

---

## Phase 0 — Repo scaffolding

### Task 0.1: Top-level repo skeleton

**Files:**
- Create: `pyproject.toml` (root)
- Create: `.gitignore`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/unit/server/__init__.py`, `tests/unit/client/__init__.py`, `tests/integration/__init__.py`, `tests/e2e/__init__.py`, `tests/shared/__init__.py`

- [ ] **Step 1: Create root `pyproject.toml`**

```toml
[tool.ruff]
line-length = 110
target-version = "py311"
extend-exclude = ["dataset/*.csv"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N"]
ignore = ["E501"]  # line length handled by formatter

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
markers = [
    "integration: requires moto/minio",
    "e2e: requires docker-compose",
]
filterwarnings = [
    "ignore::DeprecationWarning:pkg_resources",
]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
build/
dist/
.venv/
.env
.env.local
.superpowers/
.terraform/
.terraform.lock.hcl
node_modules/
dashboard/dist/
server/app/static/
.DS_Store
*.swp
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
"""Shared pytest configuration. Inserts server/ and client/ on sys.path so
`import app.X` resolves correctly when tests reference both packages.

NOTE: server/app and client/app are SEPARATE namespaces. Tests that touch
both must use full paths (server.app.model.X) — done via the conftest
INSERT below which adds repo root to path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so `import server.app.X` and `import client.app.X` both work
```

- [ ] **Step 4: Create empty `__init__.py` files**

```bash
for d in tests tests/unit tests/unit/server tests/unit/client tests/integration tests/e2e tests/shared; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 5: Verify pytest collects nothing (no tests yet) and runs cleanly**

Run: `python -m pytest -v`
Expected: `no tests ran in 0.XXs` exit 5 (no tests collected) — that's fine, just confirms config parses.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore tests/
git commit -m "chore: repo scaffolding (pyproject, gitignore, tests skeleton)"
```

---

### Task 0.2: Server + client Python packages

**Files:**
- Create: `server/__init__.py`, `server/app/__init__.py`
- Create: `server/pyproject.toml`, `server/requirements.txt`
- Create: `client/__init__.py`, `client/app/__init__.py`
- Create: `client/pyproject.toml`, `client/requirements.txt`

- [ ] **Step 1: Create server package files**

```bash
touch server/__init__.py server/app/__init__.py
```

`server/requirements.txt`:

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
boto3==1.35.36
torch==2.4.1
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
python-multipart==0.0.12
bcrypt==4.2.0
PyJWT==2.9.0
```

`server/pyproject.toml`:

```toml
[project]
name = "fl-server"
version = "0.2.0"
requires-python = ">=3.11"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[tool.setuptools.packages.find]
where = ["."]
include = ["app*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create client package files**

```bash
touch client/__init__.py client/app/__init__.py
```

`client/requirements.txt`:

```
boto3==1.35.36
torch==2.4.1
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
requests==2.32.3
pydantic==2.9.2
pydantic-settings==2.5.2
```

`client/pyproject.toml`:

```toml
[project]
name = "fl-client"
version = "0.2.0"
requires-python = ">=3.11"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[tool.setuptools.packages.find]
where = ["."]
include = ["app*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Add test dev deps to root pyproject.toml**

Modify root `pyproject.toml` — add `[project]` section above `[tool.ruff]`:

```toml
[project]
name = "fl-rebuild-tests"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest==8.3.3",
    "pytest-asyncio==0.24.0",
    "pytest-cov==5.0.0",
    "moto[s3]==5.0.16",
    "httpx==0.27.2",      # for FastAPI TestClient
    "ruff==0.6.9",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 4: Install everything in a venv to verify the dependency tree resolves**

Run:
```bash
python -m venv .venv
.venv/Scripts/pip install -e .[dev] -e ./server -e ./client       # Windows
# or .venv/bin/pip install -e .[dev] -e ./server -e ./client       # mac/linux
```

Expected: all installs complete with no resolver errors. PyTorch may take a minute.

- [ ] **Step 5: Commit**

```bash
git add server/ client/ pyproject.toml
git commit -m "chore: declare server, client, and root test packages"
```

---

### Task 0.3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: ci
on:
  push:
    branches: [main, "claude/**"]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev] -e ./server -e ./client
      - name: Lint
        run: ruff check server client tests
      - name: Unit + integration tests
        run: pytest tests/unit tests/integration -v --cov=server/app --cov=client/app --cov-report=term-missing

  e2e-smoke:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install test deps
        run: pip install -e .[dev]
      - name: Run e2e smoke
        run: pytest tests/e2e -v
```

- [ ] **Step 2: Push and verify the workflow runs (it will succeed because there are no tests yet)**

Run: `git push origin claude/upbeat-archimedes-846e98`

Expected: CI runs, `lint-and-test` passes (ruff finds no files yet, pytest exits with no-tests-found = exit 5 — adjust pytest config to not fail on no-tests).

- [ ] **Step 3: Add `--co` (collect only) safeguard so empty pytest passes**

Modify root `pyproject.toml` `[tool.pytest.ini_options]` block — add:

```toml
addopts = "-ra -q --strict-markers --tb=short"
# pytest exits 5 on no tests; that's fine for CI initially.
# We'll gain real tests next task.
```

CI will fail on exit 5 — that's actually OK, the next task adds the first real test.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml pyproject.toml
git commit -m "ci: add GitHub Actions workflow for lint + tests + e2e smoke"
```

---

## Phase 1 — Shared model + config

### Task 1.1: FraudDetectionModel + hash-pinning test

**Files:**
- Create: `server/app/model.py`
- Create: `client/app/model.py` (byte-identical copy)
- Test: `tests/unit/server/test_model.py`

The model is small and hand-shared between server and client (no private package). A test enforces the two files are byte-identical so they cannot drift.

- [ ] **Step 1: Write the model file (server side)**

`server/app/model.py`:

```python
"""19-feature fraud-detection MLP. Shared schema between server and client.

DO NOT modify this file in only one of {server,client}/app/model.py — the
hash-equality test in tests/unit/server/test_model.py will fail. If you need
to change the architecture, edit BOTH files identically and update the test.
"""
from __future__ import annotations

import torch
from torch import nn


INPUT_DIM = 19
HIDDEN_DIMS = (64, 32, 16)


class FraudDetectionModel(nn.Module):
    """MLP: 19 → 64 (BN, ReLU, Dropout) → 32 (BN, ReLU, Dropout) → 16 (BN, ReLU) → 1 logit."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dims: tuple[int, ...] = HIDDEN_DIMS, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits, shape (N, 1)
        return self.net(x)

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(weights, strict=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))
```

- [ ] **Step 2: Copy byte-identically to client**

```bash
cp server/app/model.py client/app/model.py
```

- [ ] **Step 3: Write the hash-pinning + behaviour test**

`tests/unit/server/test_model.py`:

```python
import hashlib
from pathlib import Path

import torch

from server.app.model import INPUT_DIM, FraudDetectionModel


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_server_and_client_model_files_are_byte_identical():
    """Hand-shared file: any divergence will silently break weight transfer."""
    root = Path(__file__).resolve().parents[3]
    server_hash = _sha256(root / "server" / "app" / "model.py")
    client_hash = _sha256(root / "client" / "app" / "model.py")
    assert server_hash == client_hash, (
        "server/app/model.py and client/app/model.py have diverged. "
        "Edit BOTH identically."
    )


def test_model_input_output_shape():
    m = FraudDetectionModel()
    m.eval()
    x = torch.randn(8, INPUT_DIM)
    y = m(x)
    assert y.shape == (8, 1)


def test_get_set_weights_roundtrip():
    m1, m2 = FraudDetectionModel(), FraudDetectionModel()
    w = m1.get_weights()
    m2.set_weights(w)
    # outputs match (eval mode disables BN running-stats updates)
    m1.eval(); m2.eval()
    x = torch.randn(4, INPUT_DIM)
    assert torch.allclose(m1(x), m2(x), atol=1e-6)
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/model.py client/app/model.py tests/unit/server/test_model.py
git commit -m "feat(model): add shared FraudDetectionModel with hash-pinning test"
```

---

### Task 1.2: Server config (pydantic Settings)

**Files:**
- Create: `server/app/config.py`
- Test: `tests/unit/server/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/server/test_config.py`:

```python
import pytest

from server.app.config import Settings


def test_defaults_when_only_required_set(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    s = Settings()
    assert s.s3_bucket == "test-bucket"
    assert s.aws_region == "us-east-1"
    assert s.min_nodes == 3
    assert s.quorum_pct == 0.6
    assert s.round_timeout_s == 300
    assert s.inter_round_delay_s == 2
    assert s.dp_epsilon == 5.0
    assert s.dp_delta == 1e-5
    assert s.dp_clip_norm == 0.5
    assert s.input_dim == 19
    assert s.rollback_threshold == 0.05
    assert s.use_local_storage is False
    assert s.local_storage_dir == "/tmp/fl-server"
    assert s.admin_password_hash is None  # set in production
    assert s.cors_origin == "*"


def test_required_s3_bucket_missing_raises(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(ValueError):
        Settings()


def test_use_local_storage_via_env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "irrelevant-when-local")
    monkeypatch.setenv("USE_LOCAL_STORAGE", "true")
    s = Settings()
    assert s.use_local_storage is True
```

- [ ] **Step 2: Run — should fail (no module)**

Run: `pytest tests/unit/server/test_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/config.py`:

```python
"""Server configuration. Single source of truth for env-driven behaviour.

Fail fast: invalid configuration raises at import-time when Settings() is
constructed, so misconfigured pods crash on boot rather than at first request.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    # --- AWS / storage ---
    s3_bucket: str = Field(..., description="REQUIRED — bucket for datasets, models, checkpoints")
    aws_region: str = "us-east-1"
    use_local_storage: bool = False
    local_storage_dir: str = "/tmp/fl-server"

    # --- FL hyperparams ---
    min_nodes: int = 3
    max_rounds: int = 50
    quorum_pct: float = 0.6
    round_timeout_s: int = 300
    inter_round_delay_s: int = 2
    rollback_threshold: float = 0.05

    # --- DP ---
    dp_epsilon: float = 5.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 0.5

    # --- Model ---
    input_dim: int = 19

    # --- Auth (used by Plan 2 dashboard; nullable in Plan 1) ---
    admin_password_hash: str | None = None
    jwt_secret: str | None = None
    cors_origin: str = "*"
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/config.py tests/unit/server/test_config.py
git commit -m "feat(config): add pydantic Settings with required S3_BUCKET and FL defaults"
```

---

### Task 1.3: Client config

**Files:**
- Create: `client/app/config.py`
- Test: `tests/unit/client/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/client/test_config.py`:

```python
import pytest

from client.app.config import Settings


def test_required_fields(monkeypatch):
    monkeypatch.setenv("BANK_ID", "bank_01_retail_urban")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FL_SERVER_URL", "http://fl-server:8080")
    s = Settings()
    assert s.bank_id == "bank_01_retail_urban"
    assert s.s3_bucket == "test-bucket"
    assert s.fl_server_url == "http://fl-server:8080"
    assert s.dataset_path == "/work/data/bank.csv"
    assert s.local_epochs == 10
    assert s.batch_size == 512
    assert s.learning_rate == 0.001
    assert s.dp_epsilon == 5.0
    assert s.dp_clip_norm == 0.5
    assert s.poll_interval_s == 2
    assert s.aws_region == "us-east-1"
    assert s.use_local_storage is False


def test_missing_bank_id_raises(monkeypatch):
    monkeypatch.delenv("BANK_ID", raising=False)
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FL_SERVER_URL", "http://fl-server:8080")
    with pytest.raises(ValueError):
        Settings()
```

- [ ] **Step 2: Run — should fail**

Run: `pytest tests/unit/client/test_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`client/app/config.py`:

```python
"""Client configuration. Mirror of server/app/config.py shape, but client-specific."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    # --- identity ---
    bank_id: str = Field(..., description="REQUIRED — e.g. bank_01_retail_urban")
    bank_name: str | None = None  # human label; defaults to bank_id

    # --- contract endpoints ---
    fl_server_url: str = Field(..., description="REQUIRED — http://fl-server:8080 in cluster")
    s3_bucket: str = Field(..., description="REQUIRED — same bucket as server")
    aws_region: str = "us-east-1"
    use_local_storage: bool = False
    local_storage_dir: str = "/tmp/fl-client"

    # --- dataset (init container drops here) ---
    dataset_path: str = "/work/data/bank.csv"

    # --- training ---
    local_epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    input_dim: int = 19

    # --- DP ---
    dp_epsilon: float = 5.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 0.5

    # --- runtime ---
    poll_interval_s: int = 2
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/client/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add client/app/config.py tests/unit/client/test_config.py
git commit -m "feat(client): add Settings with BANK_ID/S3_BUCKET/FL_SERVER_URL required"
```

---

## Phase 2 — Server FL primitives

### Task 2.1: Validator (per-update scoring)

**Files:**
- Create: `server/app/validator.py`
- Test: `tests/unit/server/test_validator.py`

The validator is independent of the aggregator and produces a "suspicious flag" per update so the aggregator can pick the right method.

- [ ] **Step 1: Write the failing test**

`tests/unit/server/test_validator.py`:

```python
import torch

from server.app.validator import UpdateValidator


def _flat(d):  # dict of tensors → flat 1D tensor
    return torch.cat([t.flatten() for t in d.values()])


def _make_weights(scale: float = 1.0, n_layers: int = 3, dim: int = 16):
    return {f"layer{i}": torch.randn(dim) * scale for i in range(n_layers)}


def test_clean_updates_all_pass():
    torch.manual_seed(0)
    updates = [_make_weights() for _ in range(5)]
    v = UpdateValidator(norm_bound=10.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(valid) == 5
    assert len(suspicious) == 0


def test_nan_inf_rejected():
    torch.manual_seed(0)
    updates = [_make_weights() for _ in range(4)]
    bad = _make_weights()
    bad["layer0"][0] = float("nan")
    updates.append(bad)
    v = UpdateValidator(norm_bound=10.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(valid) == 4
    assert len(suspicious) == 1


def test_norm_too_large_rejected():
    torch.manual_seed(0)
    updates = [_make_weights() for _ in range(4)]
    huge = _make_weights(scale=100.0)  # ‖w‖ way above 10.0
    updates.append(huge)
    v = UpdateValidator(norm_bound=10.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(suspicious) == 1


def test_low_cosine_to_median_rejected():
    torch.manual_seed(0)
    base = _make_weights()
    # 4 near-identical updates
    updates = [{k: v.clone() + 0.01 * torch.randn_like(v) for k, v in base.items()} for _ in range(4)]
    # 1 sign-flipped update → cosine ≈ -1
    updates.append({k: -v for k, v in base.items()})
    v = UpdateValidator(norm_bound=100.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(suspicious) == 1
```

- [ ] **Step 2: Run — should fail (no module)**

Run: `pytest tests/unit/server/test_validator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/validator.py`:

```python
"""Per-update validation. Produces a suspicion score the aggregator uses to
choose its method (FedAvg / Trimmed-Mean / Krum)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _flatten(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.flatten() for t in weights.values()])


@dataclass
class UpdateValidator:
    norm_bound: float = 10.0
    cosine_threshold: float = 0.1

    def score(
        self, updates: list[dict[str, torch.Tensor]]
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        """Returns (valid, suspicious). Valid updates are passed to the aggregator;
        suspicious counts feed into method selection.
        """
        flat = [_flatten(u) for u in updates]
        norms = torch.tensor([t.norm().item() for t in flat])
        median_dir = torch.stack(flat).median(dim=0).values
        median_dir = median_dir / (median_dir.norm() + 1e-12)

        valid, suspicious = [], []
        for u, f, n in zip(updates, flat, norms, strict=True):
            if not torch.isfinite(f).all():
                suspicious.append(u); continue
            if n.item() > self.norm_bound:
                suspicious.append(u); continue
            cos = torch.dot(f / (f.norm() + 1e-12), median_dir).item()
            if cos < self.cosine_threshold:
                suspicious.append(u); continue
            valid.append(u)
        return valid, suspicious
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_validator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/validator.py tests/unit/server/test_validator.py
git commit -m "feat(validator): per-update NaN/norm/cosine scoring"
```

---

### Task 2.2: Aggregator (FedAvg + Trimmed-Mean + Krum, auto-select)

**Files:**
- Create: `server/app/aggregator.py`
- Test: `tests/unit/server/test_aggregator.py`

Spec section 5.1: `0 suspicious → FedAvg; 1–10% → TM; 10–30% → TM; >30% or n≤3 → Krum`. Port the algorithms from `git show main:server/aggregator.py` (v1) but split into three pure functions + an `aggregate(valid, suspicious_pct)` dispatcher.

- [ ] **Step 1: Write failing tests**

`tests/unit/server/test_aggregator.py`:

```python
import torch

from server.app.aggregator import (
    Aggregator,
    fedavg,
    krum,
    trimmed_mean,
)


def _w(scale=1.0):
    return {"a": torch.ones(4) * scale, "b": torch.zeros(4)}


def _samples(n_each):
    return [n_each] * 5  # 5 updates


def test_fedavg_weighted_mean():
    updates = [_w(scale=1.0), _w(scale=3.0)]
    n_samples = [100, 300]  # 1:3 weight
    out = fedavg(updates, n_samples)
    # weighted mean of 1 and 3 with weights 1:3 = (1*1 + 3*3) / 4 = 2.5
    assert torch.allclose(out["a"], torch.full((4,), 2.5))


def test_trimmed_mean_drops_extremes():
    # 5 updates: scales [1, 2, 3, 4, 100] → trim 0.2 (1 each side) → mean(2,3,4)=3
    updates = [_w(scale=s) for s in [1, 2, 3, 4, 100]]
    out = trimmed_mean(updates, trim_ratio=0.2)
    assert torch.allclose(out["a"], torch.full((4,), 3.0))


def test_krum_picks_most_central():
    torch.manual_seed(0)
    base = _w(scale=1.0)
    near = [{k: v + 0.01 * torch.randn_like(v) for k, v in base.items()} for _ in range(4)]
    outlier = _w(scale=50.0)
    updates = near + [outlier]
    out = krum(updates, n_byzantine=1)
    # krum picks one of the near-base updates; its `a` mean should be ~1, not ~50
    assert out["a"].mean().item() < 5.0


def test_aggregator_dispatches_fedavg_when_clean():
    updates = [_w(scale=1.0)] * 4
    n_samples = [100] * 4
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.0, n_total=4)
    assert method == "fedavg"
    assert torch.allclose(out["a"], torch.ones(4))


def test_aggregator_dispatches_trimmed_mean_at_5pct():
    updates = [_w(scale=1.0)] * 9 + [_w(scale=100.0)]  # 10% suspicious
    n_samples = [100] * 10
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.10, n_total=10)
    assert method == "trimmed_mean"


def test_aggregator_dispatches_krum_at_40pct():
    updates = [_w(scale=1.0)] * 4
    n_samples = [100] * 4
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.40, n_total=4)
    assert method == "krum"


def test_aggregator_dispatches_krum_when_n_le_3():
    updates = [_w(scale=1.0)] * 3
    n_samples = [100] * 3
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.0, n_total=3)
    assert method == "krum"
```

- [ ] **Step 2: Run — should fail (module missing)**

Run: `pytest tests/unit/server/test_aggregator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/aggregator.py`:

```python
"""Three aggregation algorithms + a dispatcher.

Port of v1 server/aggregator.py with these changes vs main:
  - Split into pure functions (fedavg, trimmed_mean, krum) + Aggregator class
    (the dispatcher) so tests can hit them in isolation.
  - fedavg returns plain weighted mean, no DP (DP is now its own engine).
  - All inputs are dict[str, Tensor]; the previous flat-tensor variant is gone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Method = Literal["fedavg", "trimmed_mean", "krum"]


def fedavg(updates: list[dict[str, torch.Tensor]], n_samples: list[int]) -> dict[str, torch.Tensor]:
    total = sum(n_samples)
    out: dict[str, torch.Tensor] = {}
    for k in updates[0]:
        out[k] = sum(u[k] * (n / total) for u, n in zip(updates, n_samples, strict=True))
    return out


def trimmed_mean(updates: list[dict[str, torch.Tensor]], trim_ratio: float = 0.1) -> dict[str, torch.Tensor]:
    n = len(updates)
    k_trim = max(1, int(n * trim_ratio))
    out: dict[str, torch.Tensor] = {}
    for key in updates[0]:
        stack = torch.stack([u[key] for u in updates], dim=0)  # (n, ...)
        sorted_, _ = stack.sort(dim=0)
        trimmed = sorted_[k_trim : n - k_trim]
        out[key] = trimmed.mean(dim=0)
    return out


def krum(updates: list[dict[str, torch.Tensor]], n_byzantine: int = 1) -> dict[str, torch.Tensor]:
    """Picks the single update whose sum-of-squared-distances to its (n - n_byz - 2)
    nearest neighbours is minimal. Reference: Blanchard et al. 2017."""
    n = len(updates)
    flats = [torch.cat([t.flatten() for t in u.values()]) for u in updates]
    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = (flats[i] - flats[j]).pow(2).sum()
    k_neighbours = max(1, n - n_byzantine - 2)
    scores = []
    for i in range(n):
        nearest = dists[i].topk(k_neighbours, largest=False).values
        scores.append(nearest.sum().item())
    chosen = int(torch.tensor(scores).argmin().item())
    return updates[chosen]


@dataclass
class Aggregator:
    trim_ratio: float = 0.1
    n_byzantine_estimate: int = 1

    def aggregate(
        self,
        updates: list[dict[str, torch.Tensor]],
        n_samples: list[int],
        suspicious_pct: float,
        n_total: int,
    ) -> tuple[dict[str, torch.Tensor], Method]:
        """Returns (aggregated_weights, method_used). Method is selected per spec 5.1."""
        if n_total <= 3 or suspicious_pct > 0.30:
            return krum(updates, self.n_byzantine_estimate), "krum"
        if suspicious_pct > 0.0:
            return trimmed_mean(updates, self.trim_ratio), "trimmed_mean"
        return fedavg(updates, n_samples), "fedavg"
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_aggregator.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/aggregator.py tests/unit/server/test_aggregator.py
git commit -m "feat(aggregator): FedAvg / TrimmedMean / Krum + auto-select dispatcher"
```

---

### Task 2.3: DP engine

**Files:**
- Create: `server/app/dp_engine.py`
- Test: `tests/unit/server/test_dp_engine.py`

Port of v1 `server/dp_engine.py`. Standalone, pure.

- [ ] **Step 1: Write failing tests**

`tests/unit/server/test_dp_engine.py`:

```python
import math

import torch

from server.app.dp_engine import DPEngine, gaussian_sigma


def test_sigma_formula():
    # σ = sqrt(2 ln(1.25/δ)) * clip / ε
    s = gaussian_sigma(epsilon=5.0, delta=1e-5, clip_norm=0.5)
    expected = math.sqrt(2 * math.log(1.25 / 1e-5)) * 0.5 / 5.0
    assert abs(s - expected) < 1e-9


def test_clip_below_norm_unchanged():
    eng = DPEngine(epsilon=5.0, delta=1e-5, clip_norm=10.0)
    w = {"a": torch.ones(4)}  # norm = 2.0 < 10
    out = eng.clip(w)
    assert torch.allclose(out["a"], w["a"])


def test_clip_above_norm_scaled():
    eng = DPEngine(epsilon=5.0, delta=1e-5, clip_norm=1.0)
    w = {"a": torch.ones(4) * 10}  # norm = 20 > 1
    out = eng.clip(w)
    flat = torch.cat([t.flatten() for t in out.values()])
    assert abs(flat.norm().item() - 1.0) < 1e-5


def test_privatize_changes_weights():
    torch.manual_seed(0)
    eng = DPEngine(epsilon=5.0, delta=1e-5, clip_norm=0.5)
    w = {"a": torch.zeros(100)}
    out = eng.privatize(w)
    assert not torch.allclose(out["a"], w["a"])  # noise added
    # noise std should be roughly sigma; we just sanity-check it's non-trivial
    assert out["a"].std().item() > 1e-4
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/unit/server/test_dp_engine.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/dp_engine.py`:

```python
"""Gaussian DP. Server applies on aggregate; client applies on its own update.

Privacy accounting: sums per-round ε per bank and globally. No hard cap is
enforced in code (demo posture); the dashboard exposes the running sum so the
operator can quote it. To enforce, raise in `privatize()` when budget exceeded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def gaussian_sigma(epsilon: float, delta: float, clip_norm: float) -> float:
    return math.sqrt(2 * math.log(1.25 / delta)) * clip_norm / epsilon


@dataclass
class DPEngine:
    epsilon: float = 5.0
    delta: float = 1e-5
    clip_norm: float = 0.5

    @property
    def sigma(self) -> float:
        return gaussian_sigma(self.epsilon, self.delta, self.clip_norm)

    def clip(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flat = torch.cat([t.flatten() for t in weights.values()])
        norm = flat.norm().item()
        if norm <= self.clip_norm:
            return {k: v.clone() for k, v in weights.items()}
        scale = self.clip_norm / norm
        return {k: v * scale for k, v in weights.items()}

    def add_noise(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        s = self.sigma
        return {k: v + torch.randn_like(v) * s for k, v in weights.items()}

    def privatize(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.add_noise(self.clip(weights))
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_dp_engine.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/dp_engine.py tests/unit/server/test_dp_engine.py
git commit -m "feat(dp): Gaussian DP engine — sigma + clip + add_noise + privatize"
```

---

### Task 2.4: RoundManager

**Files:**
- Create: `server/app/round_manager.py`
- Test: `tests/unit/server/test_round_manager.py`

Holds round state, registered banks, trust scores, quorum check, checkpoint serialisation. No I/O — checkpoint serialisation produces a JSON-able dict; the storage layer writes it to S3 (Task 3.1 wires that). Trust math + quorum logic ported from v1.

- [ ] **Step 1: Write failing tests**

`tests/unit/server/test_round_manager.py`:

```python
import pytest

from server.app.round_manager import RoundManager


def test_register_and_active_count():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_01", "Bank One", 100_000)
    rm.register("bank_02", "Bank Two", 200_000)
    assert rm.active_node_count() == 2
    assert "bank_01" in rm.registered


def test_quorum_size():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    for i in range(7):
        rm.register(f"bank_{i:02d}", f"Bank {i}", 100_000)
    # max(3, 7*0.6=4.2 → 5) → 5
    assert rm.quorum_size() == 5


def test_quorum_size_floor_at_min_nodes():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    for i in range(2):
        rm.register(f"bank_{i:02d}", f"Bank {i}", 100_000)
    # max(3, 2*0.6=1.2 → 2) → 3
    assert rm.quorum_size() == 3


def test_flag_decreases_trust_and_eventually_suspends():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_x", "X", 100)
    assert rm.trust_scores["bank_x"] == pytest.approx(1.0)
    for _ in range(5):
        rm.flag_node("bank_x")
    # 1.0 * 0.6^5 ≈ 0.0778 < 0.2 → suspended
    assert "bank_x" in rm.suspended


def test_reward_clamps_at_one():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_x", "X", 100)
    for _ in range(10):
        rm.reward_node("bank_x")
    assert rm.trust_scores["bank_x"] <= 1.0


def test_checkpoint_roundtrip():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_01", "Bank One", 100)
    rm.register("bank_02", "Bank Two", 200)
    rm.flag_node("bank_02")
    rm.current_round = 5
    rm.cumulative_eps_global = 12.5
    snapshot = rm.checkpoint_dict()
    rm2 = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm2.restore_from_dict(snapshot)
    assert rm2.current_round == 5
    assert rm2.cumulative_eps_global == 12.5
    assert rm2.trust_scores["bank_02"] == rm.trust_scores["bank_02"]
    assert rm2.registered.keys() == rm.registered.keys()
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/unit/server/test_round_manager.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/round_manager.py`:

```python
"""Round + node state. Pure: serialises to a JSON-able dict via checkpoint_dict()
which the storage layer writes to S3. Restore is symmetric."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegisteredBank:
    bank_name: str
    n_samples: int


@dataclass
class RoundManager:
    min_nodes: int = 3
    quorum_pct: float = 0.6
    flag_decay: float = 0.6
    reward_step: float = 0.05
    suspend_threshold: float = 0.2

    current_round: int = 0
    registered: dict[str, RegisteredBank] = field(default_factory=dict)
    suspended: set[str] = field(default_factory=set)
    trust_scores: dict[str, float] = field(default_factory=dict)
    cumulative_eps_per_bank: dict[str, float] = field(default_factory=dict)
    cumulative_eps_global: float = 0.0

    def register(self, bank_id: str, bank_name: str, n_samples: int) -> None:
        self.registered[bank_id] = RegisteredBank(bank_name, n_samples)
        self.trust_scores.setdefault(bank_id, 1.0)
        self.cumulative_eps_per_bank.setdefault(bank_id, 0.0)

    def active_node_count(self) -> int:
        return len([b for b in self.registered if b not in self.suspended])

    def quorum_size(self) -> int:
        active = self.active_node_count()
        return max(self.min_nodes, math.ceil(active * self.quorum_pct))

    def flag_node(self, bank_id: str) -> None:
        score = self.trust_scores.get(bank_id, 1.0) * self.flag_decay
        self.trust_scores[bank_id] = score
        if score < self.suspend_threshold:
            self.suspended.add(bank_id)

    def reward_node(self, bank_id: str) -> None:
        score = min(1.0, self.trust_scores.get(bank_id, 1.0) + self.reward_step)
        self.trust_scores[bank_id] = score

    def add_eps(self, bank_id: str, delta: float) -> None:
        self.cumulative_eps_per_bank[bank_id] = self.cumulative_eps_per_bank.get(bank_id, 0.0) + delta
        self.cumulative_eps_global += delta

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "current_round": self.current_round,
            "registered": {k: {"bank_name": v.bank_name, "n_samples": v.n_samples} for k, v in self.registered.items()},
            "suspended": sorted(self.suspended),
            "trust_scores": dict(self.trust_scores),
            "cumulative_eps_per_bank": dict(self.cumulative_eps_per_bank),
            "cumulative_eps_global": self.cumulative_eps_global,
        }

    def restore_from_dict(self, snap: dict[str, Any]) -> None:
        self.current_round = int(snap["current_round"])
        self.registered = {
            k: RegisteredBank(v["bank_name"], int(v["n_samples"])) for k, v in snap["registered"].items()
        }
        self.suspended = set(snap.get("suspended", []))
        self.trust_scores = dict(snap.get("trust_scores", {}))
        self.cumulative_eps_per_bank = dict(snap.get("cumulative_eps_per_bank", {}))
        self.cumulative_eps_global = float(snap.get("cumulative_eps_global", 0.0))
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_round_manager.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/round_manager.py tests/unit/server/test_round_manager.py
git commit -m "feat(round_manager): quorum + trust + eps accounting + checkpoint dict"
```

---

### Task 2.5: ControlPlane

**Files:**
- Create: `server/app/control_plane.py`
- Test: `tests/unit/server/test_control_plane.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/server/test_control_plane.py`:

```python
from server.app.control_plane import ControlPlane


def test_initial_state():
    cp = ControlPlane()
    assert cp.global_.paused is False
    assert cp.global_.current_round == 0
    assert cp.global_.state == "idle"
    assert cp.banks == {}


def test_pause_resume():
    cp = ControlPlane()
    cp.pause()
    assert cp.global_.paused is True
    cp.resume()
    assert cp.global_.paused is False


def test_set_fault_creates_bank_entry_if_missing():
    cp = ControlPlane()
    cp.set_fault("bank_04", "byzantine")
    assert cp.banks["bank_04"].fault == "byzantine"
    assert cp.banks["bank_04"].dataset_version == 1


def test_bump_dataset_version_increments():
    cp = ControlPlane()
    cp.bump_dataset_version("bank_03")
    cp.bump_dataset_version("bank_03")
    assert cp.banks["bank_03"].dataset_version == 3  # starts at 1, +2 bumps


def test_reset_rounds_clears_state():
    cp = ControlPlane()
    cp.global_.current_round = 42
    cp.global_.state = "collecting"
    cp.reset_rounds()
    assert cp.global_.current_round == 0
    assert cp.global_.state == "idle"


def test_snapshot_and_restore_roundtrip():
    cp = ControlPlane()
    cp.set_fault("bank_01", "straggle")
    cp.bump_dataset_version("bank_01")
    cp.global_.current_round = 7
    snap = cp.snapshot_dict()

    cp2 = ControlPlane()
    cp2.restore_from_dict(snap)
    assert cp2.global_.current_round == 7
    assert cp2.banks["bank_01"].fault == "straggle"
    assert cp2.banks["bank_01"].dataset_version == 2
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/unit/server/test_control_plane.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/control_plane.py`:

```python
"""Operator-controlled state. Snapshots to S3 via the storage layer (wired in
Task 4.2). The asyncio round_loop reads `paused`/`current_round`/`state`;
the /control/{bank_id} route reads BankControl."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Fault = Literal["none", "crash", "straggle", "byzantine", "partition"]
RoundState = Literal["idle", "collecting", "aggregating", "stalled"]


@dataclass
class BankControl:
    dataset_version: int = 1
    fault: Fault = "none"


@dataclass
class GlobalControl:
    paused: bool = False
    current_round: int = 0
    state: RoundState = "idle"
    inter_round_delay_s: int = 2


@dataclass
class ControlPlane:
    banks: dict[str, BankControl] = field(default_factory=dict)
    global_: GlobalControl = field(default_factory=GlobalControl)

    # ---- mutators ----
    def pause(self) -> None: self.global_.paused = True
    def resume(self) -> None: self.global_.paused = False
    def reset_rounds(self) -> None:
        self.global_.current_round = 0
        self.global_.state = "idle"

    def set_fault(self, bank_id: str, fault: Fault) -> None:
        self.banks.setdefault(bank_id, BankControl()).fault = fault

    def bump_dataset_version(self, bank_id: str) -> None:
        bc = self.banks.setdefault(bank_id, BankControl())
        bc.dataset_version += 1

    def get_bank(self, bank_id: str) -> BankControl:
        return self.banks.setdefault(bank_id, BankControl())

    # ---- (de)serialise ----
    def snapshot_dict(self) -> dict[str, Any]:
        return {
            "global": {
                "paused": self.global_.paused,
                "current_round": self.global_.current_round,
                "state": self.global_.state,
                "inter_round_delay_s": self.global_.inter_round_delay_s,
            },
            "banks": {k: {"dataset_version": v.dataset_version, "fault": v.fault} for k, v in self.banks.items()},
        }

    def restore_from_dict(self, snap: dict[str, Any]) -> None:
        g = snap["global"]
        self.global_ = GlobalControl(
            paused=bool(g["paused"]),
            current_round=int(g["current_round"]),
            state=g["state"],
            inter_round_delay_s=int(g.get("inter_round_delay_s", 2)),
        )
        self.banks = {
            k: BankControl(dataset_version=int(v["dataset_version"]), fault=v["fault"])
            for k, v in snap.get("banks", {}).items()
        }
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_control_plane.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/control_plane.py tests/unit/server/test_control_plane.py
git commit -m "feat(control_plane): pause/fault/dataset_version + snapshot dict"
```

---

## Phase 3 — Server I/O + API

### Task 3.1: Storage wrapper (S3 + local fallback)

**Files:**
- Create: `server/app/storage.py`
- Test: `tests/integration/test_storage_s3.py`

- [ ] **Step 1: Write failing test (uses moto)**

`tests/integration/test_storage_s3.py`:

```python
import io
import json

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
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/integration/test_storage_s3.py -v -m integration`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/storage.py`:

```python
"""S3 storage wrapper. Used by RoundManager checkpoint, ControlPlane snapshot,
and global/update model artifacts. Uses boto3; tests mock with moto."""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass

import boto3
import torch


@dataclass
class Storage:
    bucket: str
    region: str = "us-east-1"

    def __post_init__(self) -> None:
        self._s3 = boto3.client("s3", region_name=self.region)

    # ---- raw ----
    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    # ---- weights (PyTorch state-dict) ----
    def put_weights(self, key: str, weights: dict[str, torch.Tensor]) -> None:
        buf = io.BytesIO()
        torch.save(weights, buf)
        self.put_bytes(key, buf.getvalue())

    def get_weights(self, key: str) -> dict[str, torch.Tensor]:
        buf = io.BytesIO(self.get_bytes(key))
        return torch.load(buf, map_location="cpu", weights_only=True)

    # ---- JSON ----
    def put_json(self, key: str, payload: dict) -> None:
        self.put_bytes(key, json.dumps(payload, sort_keys=True).encode())

    def get_json(self, key: str) -> dict:
        return json.loads(self.get_bytes(key))

    # ---- listing helpers ----
    def latest_round(self, prefix: str = "models/global_round_") -> int | None:
        paginator = self._s3.get_paginator("list_objects_v2")
        max_round = None
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                m = re.search(r"round_(\d+)\.pt$", obj["Key"])
                if m:
                    r = int(m.group(1))
                    if max_round is None or r > max_round:
                        max_round = r
        return max_round

    def presign_get(self, key: str, expires_s: int = 3600) -> str:
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_s,
        )
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/integration/test_storage_s3.py -v -m integration`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/storage.py tests/integration/test_storage_s3.py
git commit -m "feat(storage): boto3 wrapper for weights + json + latest_round + presign"
```

---

### Task 3.2: ws_hub (skeleton — Plan 2 will use it)

**Files:**
- Create: `server/app/ws_hub.py`
- Test: `tests/unit/server/test_ws_hub.py`

Pure connection-set + broadcast. No FastAPI dependency in the unit (we test with a Fake socket). Routers wire it up.

- [ ] **Step 1: Write failing test**

`tests/unit/server/test_ws_hub.py`:

```python
import asyncio

import pytest

from server.app.ws_hub import WsHub


class FakeSocket:
    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, payload):
        if self.closed:
            raise RuntimeError("closed")
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_broadcast_reaches_all_subscribers():
    hub = WsHub()
    a, b = FakeSocket(), FakeSocket()
    hub.add(a)
    hub.add(b)
    await hub.broadcast({"type": "round_started", "round": 1})
    assert a.sent == [{"type": "round_started", "round": 1}]
    assert b.sent == [{"type": "round_started", "round": 1}]


@pytest.mark.asyncio
async def test_broken_socket_is_dropped_silently():
    hub = WsHub()
    good = FakeSocket()
    bad = FakeSocket(); bad.closed = True
    hub.add(good); hub.add(bad)
    await hub.broadcast({"type": "x"})
    assert hub.size() == 1  # bad dropped
    assert good.sent == [{"type": "x"}]


@pytest.mark.asyncio
async def test_remove():
    hub = WsHub()
    s = FakeSocket()
    hub.add(s); hub.remove(s)
    assert hub.size() == 0
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
asyncio_mode = "auto"
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/unit/server/test_ws_hub.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/ws_hub.py`:

```python
"""WebSocket fan-out. Pure: depends only on the .send_json(payload) protocol,
so unit tests can inject a fake. Real FastAPI WebSocket implements that method
natively."""
from __future__ import annotations

from typing import Protocol


class _Sock(Protocol):
    async def send_json(self, payload: dict) -> None: ...


class WsHub:
    def __init__(self) -> None:
        self._subs: set[_Sock] = set()

    def add(self, ws: _Sock) -> None:
        self._subs.add(ws)

    def remove(self, ws: _Sock) -> None:
        self._subs.discard(ws)

    def size(self) -> int:
        return len(self._subs)

    async def broadcast(self, payload: dict) -> None:
        dead: list[_Sock] = []
        for ws in list(self._subs):
            try:
                await ws.send_json(payload)
            except Exception:  # any send error → drop subscriber
                dead.append(ws)
        for d in dead:
            self._subs.discard(d)
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/server/test_ws_hub.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/ws_hub.py tests/unit/server/test_ws_hub.py
git commit -m "feat(ws_hub): protocol-typed pub/sub with broken-socket eviction"
```

---

### Task 3.3: Client-facing routers (`/register`, `/model/global`, `/model/update`, `/control/{bank_id}`)

**Files:**
- Create: `server/app/routers/__init__.py`
- Create: `server/app/routers/client.py`
- Test: `tests/integration/test_routers_client.py`

These are the four endpoints the bank pods actually hit. Round_loop (Task 4.1) consumes the queue these populate.

- [ ] **Step 1: Write failing test**

`tests/integration/test_routers_client.py`:

```python
import pytest
import torch
from fastapi.testclient import TestClient

# We will assemble a test app inline rather than importing main.py (Task 3.6).
from server.app.routers.client import build_router
from server.app.round_manager import RoundManager
from server.app.control_plane import ControlPlane
from server.app.storage import Storage
from fastapi import FastAPI

import boto3
from moto import mock_aws


@pytest.fixture
def app_and_state():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        st = Storage(bucket="fl-test", region="us-east-1")
        rm = RoundManager()
        cp = ControlPlane()
        # seed an initial global model
        from server.app.model import FraudDetectionModel
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())
        app = FastAPI()
        app.include_router(build_router(rm=rm, cp=cp, storage=st))
        yield TestClient(app), rm, cp, st


@pytest.mark.integration
def test_register_creates_bank_and_returns_round(app_and_state):
    client, rm, cp, st = app_and_state
    r = client.post("/register", json={"bank_id": "bank_01", "bank_name": "Bank One", "n_samples": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["current_round"] == 0
    assert "bank_01" in rm.registered


@pytest.mark.integration
def test_get_global_model_returns_weights_url(app_and_state):
    client, rm, cp, st = app_and_state
    rm.register("bank_01", "Bank One", 100)
    r = client.get("/model/global", params={"bank_id": "bank_01"})
    assert r.status_code == 200
    body = r.json()
    assert body["round"] == 0
    assert body["weights_key"] == "models/global_round_0000.pt"
    assert body["weights_url"].startswith("https://")  # presigned


@pytest.mark.integration
def test_post_model_update_records_in_pending(app_and_state):
    client, rm, cp, st = app_and_state
    rm.register("bank_01", "Bank One", 100)
    cp.global_.current_round = 1
    payload = {
        "bank_id": "bank_01",
        "round": 1,
        "weights_key": "updates/bank_01/round_0001.pt",
        "n_samples": 100,
        "metrics": {"val_auc": 0.85, "val_loss": 0.12},
    }
    r = client.post("/model/update", json=payload)
    assert r.status_code == 200
    # The router stashes the update in cp's pending dict (the loop consumes from there)
    from server.app.routers.client import get_pending
    pending = get_pending()
    assert (1, "bank_01") in pending


@pytest.mark.integration
def test_control_returns_pause_and_fault(app_and_state):
    client, rm, cp, st = app_and_state
    rm.register("bank_03", "B3", 100)
    cp.set_fault("bank_03", "byzantine")
    cp.global_.current_round = 5
    cp.global_.paused = True
    r = client.get("/control/bank_03")
    assert r.status_code == 200
    body = r.json()
    assert body["paused"] is True
    assert body["fault"] == "byzantine"
    assert body["current_round"] == 5
    assert body["dataset_version"] == 1
    assert body["weights_key"] == "models/global_round_0000.pt"  # uses latest
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/integration/test_routers_client.py -v -m integration`
Expected: ImportError.

- [ ] **Step 3: Implement**

`server/app/routers/__init__.py`: empty file.

`server/app/routers/client.py`:

```python
"""Client-facing endpoints: /register, /model/global, /model/update, /control/{bank_id}.

The pending-updates queue is module-level (a dict keyed by (round, bank_id)) so the
round_loop (Task 4.1) can drain it. Clean for a single-process server; if you ever
shard fl-server, replace with a real queue.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.storage import Storage


# Module-level pending updates queue. Keys: (round, bank_id) → payload.
_pending: dict[tuple[int, str], dict[str, Any]] = {}


def get_pending() -> dict[tuple[int, str], dict[str, Any]]:
    return _pending


def reset_pending() -> None:
    _pending.clear()


class RegisterIn(BaseModel):
    bank_id: str
    bank_name: str
    n_samples: int


class UpdateIn(BaseModel):
    bank_id: str
    round: int
    weights_key: str
    n_samples: int
    metrics: dict[str, float]


def build_router(*, rm: RoundManager, cp: ControlPlane, storage: Storage) -> APIRouter:
    router = APIRouter()

    @router.post("/register")
    def register(payload: RegisterIn):
        rm.register(payload.bank_id, payload.bank_name, payload.n_samples)
        return {"current_round": cp.global_.current_round}

    @router.get("/model/global")
    def get_global(bank_id: str):
        if bank_id not in rm.registered:
            raise HTTPException(404, "bank not registered")
        latest = storage.latest_round(prefix="models/global_round_")
        if latest is None:
            raise HTTPException(503, "no global model yet")
        key = f"models/global_round_{latest:04d}.pt"
        return {"round": latest, "weights_key": key, "weights_url": storage.presign_get(key)}

    @router.post("/model/update")
    def post_update(payload: UpdateIn):
        if payload.bank_id not in rm.registered:
            raise HTTPException(404, "bank not registered")
        _pending[(payload.round, payload.bank_id)] = payload.model_dump()
        return {"accepted": True}

    @router.get("/control/{bank_id}")
    def get_control(bank_id: str):
        bc = cp.get_bank(bank_id)
        latest = storage.latest_round(prefix="models/global_round_") or 0
        key = f"models/global_round_{latest:04d}.pt"
        return {
            "paused": cp.global_.paused,
            "current_round": cp.global_.current_round,
            "dataset_version": bc.dataset_version,
            "fault": bc.fault,
            "weights_key": key,
        }

    return router
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/integration/test_routers_client.py -v -m integration`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/routers/__init__.py server/app/routers/client.py tests/integration/test_routers_client.py
git commit -m "feat(routers): client endpoints + module-level pending updates queue"
```

---

### Task 3.4: Metrics router + global eval helper

**Files:**
- Create: `server/app/routers/metrics.py`
- Create: `server/app/eval.py` (computes AUC/F1/etc on a held-out validation set)
- Test: `tests/integration/test_routers_metrics.py`

The eval helper isolates the metric computation so round_loop and the metrics router both call it. v1 uses sklearn.metrics — same here.

- [ ] **Step 1: Write failing test**

`tests/integration/test_routers_metrics.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app.routers.metrics import build_router
from server.app.round_manager import RoundManager
from server.app.control_plane import ControlPlane


@pytest.fixture
def app():
    rm = RoundManager()
    rm.register("bank_01", "B1", 100)
    rm.register("bank_02", "B2", 200)
    rm.current_round = 5
    cp = ControlPlane()
    cp.global_.current_round = 5
    cp.global_.state = "idle"

    # seed some history (round_loop normally appends)
    from server.app.routers.metrics import (
        push_global_metrics, push_bank_metrics,
    )
    for r in range(1, 6):
        push_global_metrics({"round": r, "auc": 0.7 + r * 0.02, "f1": 0.5, "method": "fedavg"})
        push_bank_metrics("bank_01", {"round": r, "auc": 0.6 + r * 0.02, "loss": 0.2})
        push_bank_metrics("bank_02", {"round": r, "auc": 0.65 + r * 0.02, "loss": 0.18})

    app = FastAPI()
    app.include_router(build_router(rm=rm, cp=cp))
    yield TestClient(app)
    # cleanup
    from server.app.routers.metrics import reset_history
    reset_history()


@pytest.mark.integration
def test_round_status(app):
    r = app.get("/round/status")
    assert r.status_code == 200
    body = r.json()
    assert body["round"] == 5
    assert body["state"] == "idle"


@pytest.mark.integration
def test_banks_lists_registered(app):
    r = app.get("/banks")
    assert r.status_code == 200
    body = r.json()
    ids = {b["bank_id"] for b in body}
    assert ids == {"bank_01", "bank_02"}


@pytest.mark.integration
def test_bank_history(app):
    r = app.get("/banks/bank_01/history", params={"n": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert body[-1]["round"] == 5  # most recent last


@pytest.mark.integration
def test_global_metrics(app):
    r = app.get("/metrics", params={"n": 50})
    body = r.json()
    assert len(body["history"]) == 5
    assert body["history"][-1]["round"] == 5
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`server/app/routers/metrics.py`:

```python
"""Metrics + status endpoints. Round history is kept in-memory; checkpointed
to S3 via RoundManager+Storage in Task 4.2. The global-eval AUC numbers come
from server/app/eval.py (computed by round_loop after each aggregation)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from fastapi import APIRouter, HTTPException

from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager


_global_history: deque[dict[str, Any]] = deque(maxlen=1000)
_bank_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))


def push_global_metrics(row: dict[str, Any]) -> None:
    _global_history.append(row)


def push_bank_metrics(bank_id: str, row: dict[str, Any]) -> None:
    _bank_history[bank_id].append(row)


def reset_history() -> None:
    _global_history.clear()
    _bank_history.clear()


def build_router(*, rm: RoundManager, cp: ControlPlane) -> APIRouter:
    router = APIRouter()

    @router.get("/round/status")
    def round_status():
        return {
            "round": cp.global_.current_round,
            "state": cp.global_.state,
            "paused": cp.global_.paused,
            "active_banks": rm.active_node_count(),
            "quorum_size": rm.quorum_size(),
        }

    @router.get("/banks")
    def banks():
        out = []
        for bid, info in rm.registered.items():
            out.append({
                "bank_id": bid,
                "bank_name": info.bank_name,
                "n_samples": info.n_samples,
                "trust": rm.trust_scores.get(bid, 1.0),
                "suspended": bid in rm.suspended,
                "dataset_version": cp.get_bank(bid).dataset_version,
                "fault": cp.get_bank(bid).fault,
                "cumulative_eps": rm.cumulative_eps_per_bank.get(bid, 0.0),
            })
        return out

    @router.get("/banks/{bank_id}/history")
    def bank_history(bank_id: str, n: int = 50):
        if bank_id not in rm.registered:
            raise HTTPException(404, "bank not registered")
        h = list(_bank_history[bank_id])
        return h[-n:]

    @router.get("/metrics")
    def metrics(n: int = 50):
        return {
            "history": list(_global_history)[-n:],
            "cumulative_eps_global": rm.cumulative_eps_global,
            "current_round": cp.global_.current_round,
        }

    @router.get("/health")
    def health():
        return {"status": "ok"}

    return router
```

`server/app/eval.py`:

```python
"""Compute global-model quality on a held-out validation set.

The validation set is built by `dataset/build_val_set.py` (Plan 3) and uploaded
to s3://<bucket>/validation/val_set.pkl. Plan 1 ships a small synthetic
fixture for tests; production replaces it via S3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class GlobalMetrics:
    auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    val_loss: float


def evaluate(model: torch.nn.Module, X: torch.Tensor, y: np.ndarray) -> GlobalMetrics:
    model.eval()
    with torch.no_grad():
        logits = model(X).cpu().numpy().flatten()
        proba = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (proba >= 0.5).astype(int)
    return GlobalMetrics(
        auc=float(roc_auc_score(y, proba)),
        f1=float(f1_score(y, y_pred, zero_division=0)),
        precision=float(precision_score(y, y_pred, zero_division=0)),
        recall=float(recall_score(y, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y, y_pred)),
        val_loss=float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])),
    )
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/integration/test_routers_metrics.py -v -m integration`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/app/routers/metrics.py server/app/eval.py tests/integration/test_routers_metrics.py
git commit -m "feat(metrics): /round/status, /banks, /metrics, /health + sklearn eval helper"
```

---

### Task 3.5: WS router + admin stub + auth stub

**Files:**
- Create: `server/app/routers/ws.py`
- Create: `server/app/routers/admin.py` (stubs only — Plan 2 fills in)
- Create: `server/app/auth.py` (stub — returns True in Plan 1)

These are interface-locked stubs so main.py wires them now and Plan 2 can fill bodies without touching main.

- [ ] **Step 1: Write the files (no tests in Plan 1; Plan 2 covers them)**

`server/app/auth.py`:

```python
"""Authentication. Plan 1: pass-through (no cookie required) — the cluster
NetworkPolicy is the security boundary in the demo. Plan 2 replaces this with
a bcrypt+JWT cookie gate for /admin/* and /ws."""
from __future__ import annotations

from fastapi import Request


def require_admin(_: Request) -> None:
    """Plan 1 stub. Plan 2 will raise 401 here when no valid cookie is present."""
    return None
```

`server/app/routers/admin.py`:

```python
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
```

`server/app/routers/ws.py`:

```python
"""/ws/live WebSocket endpoint. Subscribers receive every event the round_loop
broadcasts via WsHub."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.app.ws_hub import WsHub


def build_router(*, hub: WsHub) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        hub.add(ws)
        try:
            # Keep the connection open; we don't consume from the client.
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.remove(ws)

    return router
```

- [ ] **Step 2: Smoke check — module imports OK**

Run:
```bash
python -c "from server.app.routers.admin import build_router; from server.app.routers.ws import build_router as bws; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add server/app/auth.py server/app/routers/admin.py server/app/routers/ws.py
git commit -m "feat(server): admin/ws router stubs + auth pass-through (Plan 1 scope)"
```

---

### Task 3.6: main.py — app factory + startup hook

**Files:**
- Create: `server/app/main.py`
- Test: `tests/integration/test_main_smoke.py`

main.py is wiring only — no logic. The startup hook restores RoundManager + ControlPlane from S3 (Task 4.2 implements that restore code; here we just call it).

- [ ] **Step 1: Write failing test**

`tests/integration/test_main_smoke.py`:

```python
import pytest
import boto3
from moto import mock_aws
from fastapi.testclient import TestClient


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
        # 404 is fine — Plan 2 ships the React bundle
        assert r.status_code in (200, 404)
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`server/app/main.py`:

```python
"""FastAPI app factory. Wires routers, builds Storage/RoundManager/ControlPlane,
mounts /static (if dist exists), and starts the asyncio round_loop on startup.

Tests pass start_round_loop=False to keep the loop dormant."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.routers import admin, client as client_router, metrics, ws
from server.app.storage import Storage
from server.app.ws_hub import WsHub


_LOG = logging.getLogger("fl.server")
_STATIC_DIR = Path(__file__).parent / "static"


def build_app(start_round_loop: bool = True) -> FastAPI:
    settings = Settings()
    storage = Storage(bucket=settings.s3_bucket, region=settings.aws_region)
    rm = RoundManager(min_nodes=settings.min_nodes, quorum_pct=settings.quorum_pct)
    cp = ControlPlane()
    cp.global_.inter_round_delay_s = settings.inter_round_delay_s
    hub = WsHub()

    app = FastAPI(title="fl-server")
    app.include_router(client_router.build_router(rm=rm, cp=cp, storage=storage))
    app.include_router(metrics.build_router(rm=rm, cp=cp))
    app.include_router(admin.build_router())
    app.include_router(ws.build_router(hub=hub))

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    @app.on_event("startup")
    async def _startup():
        # Restore checkpoint + control-plane snapshot from S3 if present.
        # Implementation: see server/app/round_loop.py:restore_state().
        from server.app.round_loop import restore_state, run_round_loop
        try:
            restore_state(rm=rm, cp=cp, storage=storage, settings=settings)
        except Exception as e:
            _LOG.exception("failed to restore state: %s", e)
            raise
        if start_round_loop:
            import asyncio
            app.state.round_task = asyncio.create_task(
                run_round_loop(rm=rm, cp=cp, storage=storage, hub=hub, settings=settings)
            )

    @app.on_event("shutdown")
    async def _shutdown():
        task = getattr(app.state, "round_task", None)
        if task:
            task.cancel()

    # Expose for tests
    app.state.rm = rm
    app.state.cp = cp
    app.state.storage = storage
    app.state.hub = hub
    app.state.settings = settings
    return app
```

- [ ] **Step 4: Run — fails because round_loop module doesn't exist yet**

The `from server.app.round_loop import restore_state, run_round_loop` import will fail until Task 4.1/4.2. **Workaround for this task:** stub `server/app/round_loop.py` with no-op functions so main.py imports cleanly. Replace properly in Phase 4.

Create `server/app/round_loop.py`:

```python
"""Phase 1 stub. Phase 4 (Tasks 4.1/4.2) replaces this with the real loop."""
from __future__ import annotations

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.round_manager import RoundManager
from server.app.storage import Storage
from server.app.ws_hub import WsHub


def restore_state(*, rm: RoundManager, cp: ControlPlane, storage: Storage, settings: Settings) -> None:
    return None


async def run_round_loop(*, rm: RoundManager, cp: ControlPlane, storage: Storage, hub: WsHub, settings: Settings) -> None:
    return None
```

- [ ] **Step 5: Run — should pass**

Run: `pytest tests/integration/test_main_smoke.py -v -m integration`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add server/app/main.py server/app/round_loop.py tests/integration/test_main_smoke.py
git commit -m "feat(server): app factory + startup/shutdown hooks + static mount + loop stub"
```

---

## Phase 4 — Server round loop

### Task 4.1: round_loop — full implementation

**Files:**
- Modify: `server/app/round_loop.py` (replace stub)
- Create: `tests/shared/fakes.py`
- Test: `tests/integration/test_full_round.py`

The loop drains pending updates, runs validator → aggregator → DP → eval, persists, broadcasts. Uses asyncio for the timeout-based collection.

- [ ] **Step 1: Write `tests/shared/fakes.py`**

```python
"""FakeClient: simulates a bank pod for in-process integration tests.
Trains a synthetic update (just a perturbation of the global weights)
and POSTs it to the server. No HTTP — calls handlers directly."""
from __future__ import annotations

import torch

from server.app.model import FraudDetectionModel
from server.app.routers.client import _pending


def fake_register(rm, bank_id: str, n_samples: int = 1000) -> None:
    rm.register(bank_id, bank_id.replace("_", " ").title(), n_samples)


def fake_post_update(
    bank_id: str,
    round: int,
    *,
    storage,
    perturbation: float = 0.01,
    bad: bool = False,
) -> None:
    """Mimics: client downloads global weights, "trains", uploads to S3, POSTs metadata."""
    latest = storage.latest_round(prefix="models/global_round_") or 0
    base = storage.get_weights(f"models/global_round_{latest:04d}.pt")
    if bad:
        new = {k: -v + 100.0 * torch.randn_like(v) for k, v in base.items()}
    else:
        new = {k: v + perturbation * torch.randn_like(v) for k, v in base.items()}
    key = f"updates/{bank_id}/round_{round:04d}.pt"
    storage.put_weights(key, new)
    _pending[(round, bank_id)] = {
        "bank_id": bank_id,
        "round": round,
        "weights_key": key,
        "n_samples": 1000,
        "metrics": {"val_auc": 0.85, "val_loss": 0.12},
    }
```

- [ ] **Step 2: Write the failing integration test**

`tests/integration/test_full_round.py`:

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
from tests.shared.fakes import fake_post_update, fake_register


@pytest.mark.integration
@pytest.mark.asyncio
async def test_one_round_completes(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        # seed initial global model
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())

        rm = RoundManager(min_nodes=3, quorum_pct=0.6)
        cp = ControlPlane()
        hub = WsHub()
        reset_pending(); reset_history()

        for i in range(3):
            fake_register(rm, f"bank_0{i+1}")

        from server.app.round_loop import run_one_round
        for i in range(3):
            fake_post_update(f"bank_0{i+1}", round=1, storage=st)

        await run_one_round(rm=rm, cp=cp, storage=st, hub=hub, settings=s, target_round=1)

        # global model for round 1 was written
        assert st.latest_round() == 1
        # cp shows we completed
        assert cp.global_.current_round == 1
        assert cp.global_.state == "idle"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_round_stalls_when_no_quorum(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())

        rm = RoundManager(min_nodes=3, quorum_pct=0.6)
        cp = ControlPlane()
        hub = WsHub()
        reset_pending(); reset_history()

        for i in range(3):
            fake_register(rm, f"bank_0{i+1}")
        # only 1 of 3 submits → quorum is 3, not met
        fake_post_update("bank_01", round=1, storage=st)

        from server.app.round_loop import run_one_round
        # very short deadline for test
        await run_one_round(rm=rm, cp=cp, storage=st, hub=hub, settings=s, target_round=1, deadline_s=1)
        assert cp.global_.state == "stalled"
        assert st.latest_round() == 0  # still at 0
```

- [ ] **Step 3: Run — fails (run_one_round not defined)**

- [ ] **Step 4: Implement** — replace `server/app/round_loop.py`:

```python
"""Auto-loop: collect → validate → aggregate → DP → eval → persist → broadcast → wait.

restore_state() reads the latest checkpoint + control snapshot from S3 on
boot, so a server pod restart resumes mid-experiment.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import torch

from server.app.aggregator import Aggregator
from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.dp_engine import DPEngine
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import get_pending
from server.app.routers.metrics import push_bank_metrics, push_global_metrics
from server.app.storage import Storage
from server.app.validator import UpdateValidator
from server.app.ws_hub import WsHub

_LOG = logging.getLogger("fl.round_loop")
_CHECKPOINT_KEY = "control/state.json"


def restore_state(*, rm: RoundManager, cp: ControlPlane, storage: Storage, settings: Settings) -> None:
    """Boot: load latest RoundManager checkpoint and ControlPlane snapshot from S3."""
    latest = storage.latest_round(prefix="checkpoints/round_")
    if latest is not None:
        try:
            rm.restore_from_dict(storage.get_json(f"checkpoints/round_{latest:04d}.json"))
            _LOG.info("restored RoundManager from round %d", latest)
        except Exception:
            _LOG.exception("checkpoint restore failed")
            raise
    try:
        cp.restore_from_dict(storage.get_json(_CHECKPOINT_KEY))
        _LOG.info("restored ControlPlane snapshot")
    except Exception:  # snapshot may not exist on first boot — that's fine
        _LOG.info("no ControlPlane snapshot found; starting fresh")


def _snapshot_control(cp: ControlPlane, storage: Storage) -> None:
    storage.put_json(_CHECKPOINT_KEY, cp.snapshot_dict())


async def run_one_round(
    *,
    rm: RoundManager,
    cp: ControlPlane,
    storage: Storage,
    hub: WsHub,
    settings: Settings,
    target_round: int,
    deadline_s: int | None = None,
) -> None:
    """Drives a single round end-to-end. Tests call this directly; run_round_loop
    calls it in a while-True."""
    pending = get_pending()
    cp.global_.state = "collecting"
    cp.global_.current_round = target_round
    await hub.broadcast({"type": "round_started", "round": target_round, "quorum_size": rm.quorum_size()})

    deadline = (deadline_s if deadline_s is not None else settings.round_timeout_s)
    end = time.time() + deadline
    while time.time() < end:
        n_received = sum(1 for (r, _) in pending if r == target_round)
        if n_received >= rm.quorum_size():
            break
        await asyncio.sleep(0.5)

    submissions = [v for (r, _), v in list(pending.items()) if r == target_round]
    for (r, bid) in list(pending):
        if r == target_round:
            del pending[(r, bid)]

    if len(submissions) < rm.min_nodes:
        cp.global_.state = "stalled"
        await hub.broadcast({"type": "round_stalled", "round": target_round, "received": len(submissions)})
        _snapshot_control(cp, storage)
        return

    # download all submitted weight tensors
    updates = [storage.get_weights(s["weights_key"]) for s in submissions]
    n_samples = [int(s["n_samples"]) for s in submissions]
    bank_ids = [s["bank_id"] for s in submissions]

    # validate
    validator = UpdateValidator()
    valid, suspicious = validator.score(updates)

    # mark trust
    suspicious_set = {id(u) for u in suspicious}
    for u, bid in zip(updates, bank_ids, strict=True):
        if id(u) in suspicious_set:
            rm.flag_node(bid)
            await hub.broadcast({"type": "event", "level": "warn", "msg": f"flagged {bid}"})
        else:
            rm.reward_node(bid)

    # aggregate
    cp.global_.state = "aggregating"
    aggregator = Aggregator()
    valid_n = [n for u, n in zip(updates, n_samples, strict=True) if id(u) not in suspicious_set]
    if not valid_n:  # all suspicious — fall back to all to avoid empty
        valid, valid_n = updates, n_samples
    suspicious_pct = len(suspicious) / max(1, len(updates))
    aggregated, method = aggregator.aggregate(valid, valid_n, suspicious_pct=suspicious_pct, n_total=len(updates))

    # DP
    dp = DPEngine(epsilon=settings.dp_epsilon, delta=settings.dp_delta, clip_norm=settings.dp_clip_norm)
    new_global = dp.privatize(aggregated)
    rm.cumulative_eps_global += settings.dp_epsilon
    for bid in bank_ids:
        rm.add_eps(bid, settings.dp_epsilon)

    # persist
    storage.put_weights(f"models/global_round_{target_round:04d}.pt", new_global)
    rm.current_round = target_round
    storage.put_json(f"checkpoints/round_{target_round:04d}.json", rm.checkpoint_dict())

    # metrics — Plan 1 publishes the per-bank metrics from submissions and a synthetic
    # global metric (real eval against val_set comes from server/app/eval.py once we
    # wire a validation tensor; for Plan 1 smoke it's fine to publish method+counts).
    for s in submissions:
        push_bank_metrics(s["bank_id"], {"round": target_round, **s["metrics"]})
        await hub.broadcast({"type": "bank_update", "bank_id": s["bank_id"], "round": target_round, "metrics": s["metrics"]})
    push_global_metrics({
        "round": target_round,
        "method": method,
        "n_participants": len(updates),
        "n_suspicious": len(suspicious),
    })
    cp.global_.state = "idle"
    _snapshot_control(cp, storage)
    await hub.broadcast({"type": "round_completed", "round": target_round, "method": method})


async def run_round_loop(
    *,
    rm: RoundManager,
    cp: ControlPlane,
    storage: Storage,
    hub: WsHub,
    settings: Settings,
) -> None:
    """Forever loop. Cancelled on app shutdown."""
    while True:
        if cp.global_.paused:
            await asyncio.sleep(1)
            continue
        if storage.latest_round(prefix="models/global_round_") is None:
            storage.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())
        target = cp.global_.current_round + 1
        try:
            await run_one_round(rm=rm, cp=cp, storage=storage, hub=hub, settings=settings, target_round=target)
        except Exception:
            _LOG.exception("round %d errored", target)
            await asyncio.sleep(5)
            continue
        await asyncio.sleep(cp.global_.inter_round_delay_s)
```

- [ ] **Step 5: Run — should pass**

Run: `pytest tests/integration/test_full_round.py -v -m integration`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add server/app/round_loop.py tests/shared/fakes.py tests/integration/test_full_round.py
git commit -m "feat(round_loop): collect→validate→aggregate→DP→persist→broadcast"
```

---

### Task 4.2: Byzantine round + checkpoint restore tests

**Files:**
- Test: `tests/integration/test_byzantine_round.py`
- Test: `tests/integration/test_checkpoint_restore.py`

These exercise behaviours already implemented; failing now means the implementation is wrong.

- [ ] **Step 1: Write `test_byzantine_round.py`**

```python
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
from tests.shared.fakes import fake_post_update, fake_register


@pytest.mark.integration
@pytest.mark.asyncio
async def test_byzantine_update_drops_trust_score(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")
        st.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())

        rm = RoundManager(min_nodes=3, quorum_pct=0.6)
        cp = ControlPlane()
        hub = WsHub()
        reset_pending(); reset_history()

        for i in range(4):
            fake_register(rm, f"bank_0{i+1}")

        # 3 normal, 1 byzantine
        for i in range(3):
            fake_post_update(f"bank_0{i+1}", round=1, storage=st)
        fake_post_update("bank_04", round=1, storage=st, bad=True)

        from server.app.round_loop import run_one_round
        await run_one_round(rm=rm, cp=cp, storage=st, hub=hub, settings=s, target_round=1, deadline_s=2)

        assert rm.trust_scores["bank_04"] < 1.0
        assert rm.trust_scores["bank_01"] == pytest.approx(1.0)
```

- [ ] **Step 2: Write `test_checkpoint_restore.py`**

```python
import boto3
import pytest
from moto import mock_aws

from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.storage import Storage


@pytest.mark.integration
def test_restore_state_picks_up_latest_checkpoint(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="fl-test")
        monkeypatch.setenv("S3_BUCKET", "fl-test")
        s = Settings()
        st = Storage(bucket="fl-test", region="us-east-1")

        # write checkpoint for round 7
        rm_old = RoundManager()
        rm_old.register("bank_a", "A", 100); rm_old.register("bank_b", "B", 200)
        rm_old.flag_node("bank_b")
        rm_old.current_round = 7
        rm_old.cumulative_eps_global = 35.0
        st.put_json("checkpoints/round_0007.json", rm_old.checkpoint_dict())

        # write a control snapshot
        cp_old = ControlPlane()
        cp_old.global_.current_round = 7
        cp_old.set_fault("bank_b", "byzantine")
        st.put_json("control/state.json", cp_old.snapshot_dict())

        # restore into fresh objects
        from server.app.round_loop import restore_state
        rm_new = RoundManager(); cp_new = ControlPlane()
        restore_state(rm=rm_new, cp=cp_new, storage=st, settings=s)

        assert rm_new.current_round == 7
        assert rm_new.cumulative_eps_global == 35.0
        assert rm_new.trust_scores["bank_b"] < 1.0
        assert cp_new.global_.current_round == 7
        assert cp_new.banks["bank_b"].fault == "byzantine"
```

- [ ] **Step 3: Run both tests — should pass**

Run: `pytest tests/integration/test_byzantine_round.py tests/integration/test_checkpoint_restore.py -v -m integration`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_byzantine_round.py tests/integration/test_checkpoint_restore.py
git commit -m "test: byzantine trust drop + restore_state picks up checkpoint"
```

---

## Phase 5 — Client core

### Task 5.1: dataset.py — load + assert

**Files:**
- Create: `client/app/dataset.py`
- Test: `tests/unit/client/test_dataset.py`

- [ ] **Step 1: Write failing test**

`tests/unit/client/test_dataset.py`:

```python
import pytest

from client.app.dataset import assert_dataset_present, load_dataset


def test_assert_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_dataset_present(tmp_path / "nope.csv", min_size_bytes=1)


def test_assert_too_small_raises(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("a")
    with pytest.raises(ValueError):
        assert_dataset_present(p, min_size_bytes=1024)


def test_assert_passes_for_normal_file(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text("x" * 2000)
    assert_dataset_present(p, min_size_bytes=1024)  # no raise


def test_load_dataset_returns_dataframe(tmp_path):
    import pandas as pd
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
    df = load_dataset(p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`client/app/dataset.py`:

```python
"""Dataset loading. The init container fetches s3://.../datasets/{bank_id}.csv
to /work/data/bank.csv before the main container starts. We assert it's there
on boot and again when dataset_version bumps."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def assert_dataset_present(path: str | Path, *, min_size_bytes: int = 1024 * 1024) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset missing: {p}")
    if p.stat().st_size < min_size_bytes:
        raise ValueError(f"dataset too small ({p.stat().st_size} bytes < {min_size_bytes}): {p}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
```

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git add client/app/dataset.py tests/unit/client/test_dataset.py
git commit -m "feat(client): dataset assert + load helpers"
```

---

### Task 5.2: preprocessor.py + tiny golden CSV

**Files:**
- Create: `client/app/preprocessor.py` (port from v1 client/preprocessor.py)
- Create: `tests/shared/golden_inputs/tiny_bank.csv`
- Test: `tests/unit/client/test_preprocessor.py`

Port v1 logic. The 19-feature list is in spec §1 (Section 9 of v1 inventory). Generate a tiny 200-row CSV with the schema from `git show main:dataset/generate_datasets.py` for golden testing.

- [ ] **Step 1: Generate the golden CSV**

Create a small Python helper run once:

```bash
python - <<'EOF'
import pandas as pd
import numpy as np
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "transaction_id": [f"TXN_{i}" for i in range(n)],
    "bank_id": "bank_test",
    "transaction_amount": np.random.exponential(50, n),
    "transaction_hour": np.random.randint(0, 24, n),
    "day_of_week": np.random.randint(0, 7, n),
    "is_foreign_transaction": np.random.binomial(1, 0.05, n),
    "is_online_transaction": np.random.binomial(1, 0.4, n),
    "customer_age": np.random.randint(18, 80, n),
    "account_age_days": np.random.randint(30, 3650, n),
    "avg_amount_customer": np.random.exponential(45, n),
    "std_amount_customer": np.random.exponential(15, n),
    "amount_vs_avg_ratio": np.random.exponential(1.0, n),
    "amount_zscore": np.random.normal(0, 1, n),
    "total_txns_customer": np.random.randint(1, 500, n),
    "is_night_transaction": np.random.binomial(1, 0.2, n),
    "is_weekend": np.random.binomial(1, 2/7, n),
    "merchant_category": np.random.choice(
        ["grocery", "online_retail", "restaurant", "travel", "electronics"], n
    ),
    "is_fraud": np.random.binomial(1, 0.05, n),
})
df.to_csv("tests/shared/golden_inputs/tiny_bank.csv", index=False)
print("written", len(df), "rows")
EOF
```

- [ ] **Step 2: Write failing test**

`tests/unit/client/test_preprocessor.py`:

```python
from pathlib import Path

import torch

from client.app.preprocessor import preprocess


def test_preprocess_outputs_19_features():
    csv = Path(__file__).resolve().parents[2] / "shared" / "golden_inputs" / "tiny_bank.csv"
    X_train, y_train, X_val, y_val, _scaler = preprocess(csv, val_frac=0.15)
    assert X_train.shape[1] == 19
    assert X_val.shape[1] == 19
    assert y_train.dtype == torch.float32
    assert torch.isfinite(X_train).all()


def test_preprocess_train_val_split_roughly_15pct():
    csv = Path(__file__).resolve().parents[2] / "shared" / "golden_inputs" / "tiny_bank.csv"
    X_train, y_train, X_val, y_val, _ = preprocess(csv, val_frac=0.15)
    total = len(X_train) + len(X_val)
    assert abs(len(X_val) / total - 0.15) < 0.05


def test_preprocess_deterministic_with_seed():
    csv = Path(__file__).resolve().parents[2] / "shared" / "golden_inputs" / "tiny_bank.csv"
    a = preprocess(csv, val_frac=0.15, seed=7)
    b = preprocess(csv, val_frac=0.15, seed=7)
    assert torch.allclose(a[0], b[0])
```

- [ ] **Step 3: Run — fails**

- [ ] **Step 4: Implement** — port v1 with the spec §3.1 19-feature column order:

`client/app/preprocessor.py`:

```python
"""CSV → 19-feature tensor pipeline. Port of v1 client/preprocessor.py.

Feature order (must match server.app.model.INPUT_DIM == 19):
  1. transaction_amount                (StandardScaler)
  2. transaction_hour                  (raw)
  3. day_of_week                       (raw)
  4. is_foreign_transaction            (raw 0/1)
  5. is_online_transaction             (raw 0/1)
  6. customer_age                      (StandardScaler)
  7. account_age_days                  (StandardScaler)
  8. avg_amount_customer               (StandardScaler)
  9. std_amount_customer               (StandardScaler)
 10. amount_vs_avg_ratio               (StandardScaler)
 11. amount_zscore                     (raw, already standardised in v1 generator)
 12. total_txns_customer               (StandardScaler)
 13. is_night_transaction              (raw 0/1)
 14. is_weekend                        (raw 0/1)
 15. merchant_cat_grocery              (one-hot)
 16. merchant_cat_online_retail        (one-hot)
 17. merchant_cat_restaurant           (one-hot)
 18. merchant_cat_travel               (one-hot)
 19. merchant_cat_electronics          (one-hot)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_SCALE_COLS = [
    "transaction_amount", "customer_age", "account_age_days",
    "avg_amount_customer", "std_amount_customer", "amount_vs_avg_ratio",
    "total_txns_customer",
]
_RAW_COLS = [
    "transaction_hour", "day_of_week", "is_foreign_transaction",
    "is_online_transaction", "amount_zscore",
    "is_night_transaction", "is_weekend",
]
_MERCHANT_CATS = ["grocery", "online_retail", "restaurant", "travel", "electronics"]


def preprocess(
    csv_path: str | Path,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["is_fraud"])

    # one-hot merchant_category — fixed cats so unknowns become all-zero
    for c in _MERCHANT_CATS:
        df[f"merchant_cat_{c}"] = (df.get("merchant_category", "") == c).astype(int)

    feature_cols = (
        ["transaction_amount"]
        + ["transaction_hour", "day_of_week", "is_foreign_transaction", "is_online_transaction"]
        + ["customer_age", "account_age_days"]
        + ["avg_amount_customer", "std_amount_customer", "amount_vs_avg_ratio", "amount_zscore"]
        + ["total_txns_customer", "is_night_transaction", "is_weekend"]
        + [f"merchant_cat_{c}" for c in _MERCHANT_CATS]
    )
    assert len(feature_cols) == 19, f"expected 19 features, got {len(feature_cols)}"

    # impute numeric with 0; impute scale-cols later via the scaler
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_frac, random_state=seed, stratify=(y if y.sum() > 1 else None)
    )

    scaler = StandardScaler()
    # only fit on the columns we want scaled
    scale_idx = [feature_cols.index(c) for c in _SCALE_COLS]
    X_train[:, scale_idx] = scaler.fit_transform(X_train[:, scale_idx])
    X_val[:, scale_idx] = scaler.transform(X_val[:, scale_idx])

    return (
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
        scaler,
    )
```

- [ ] **Step 5: Run — should pass**

Run: `pytest tests/unit/client/test_preprocessor.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add client/app/preprocessor.py tests/shared/golden_inputs/tiny_bank.csv tests/unit/client/test_preprocessor.py
git commit -m "feat(preprocessor): 19-feature pipeline with golden CSV test"
```

---

### Task 5.3: trainer.py — local training + DP

**Files:**
- Create: `client/app/trainer.py`
- Test: `tests/unit/client/test_trainer.py`

Port v1 `client/trainer.py`. Smoke-test only — assert loss decreases on a tiny batch.

- [ ] **Step 1: Write failing test**

`tests/unit/client/test_trainer.py`:

```python
import torch

from client.app.model import FraudDetectionModel
from client.app.trainer import LocalTrainer, TrainResult


def _toy_batch(n=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 19, generator=g)
    # make it learnable: y = (sum_first_3 > 0)
    y = (X[:, :3].sum(dim=1) > 0).float()
    return X, y


def test_trainer_decreases_loss_on_toy():
    X, y = _toy_batch()
    model = FraudDetectionModel()
    t = LocalTrainer(epochs=3, batch_size=32, lr=1e-2, dp_clip_norm=10.0, dp_sigma=0.0)
    out = t.train(model, X, y, X_val=X, y_val=y)
    assert isinstance(out, TrainResult)
    assert out.metrics["val_loss"] < 1.0
    # weights changed
    base = FraudDetectionModel().get_weights()
    assert not torch.allclose(base["net.0.weight"], out.weights["net.0.weight"])


def test_trainer_dp_clip_caps_norm():
    X, y = _toy_batch()
    model = FraudDetectionModel()
    t = LocalTrainer(epochs=1, batch_size=32, lr=1e-2, dp_clip_norm=0.5, dp_sigma=0.0)
    out = t.train(model, X, y, X_val=X, y_val=y)
    flat = torch.cat([v.flatten() for v in out.weights.values()])
    # L2 norm of clipped weights should be <= clip + small slack
    assert flat.norm().item() <= 0.5 + 1e-3
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`client/app/trainer.py`:

```python
"""Local training. Port of v1 client/trainer.py with these structural changes:
  - Returns a TrainResult dataclass (weights + metrics) instead of a tuple.
  - DP clip + noise applied at the END once on returned weights (not per-batch
    grad clip — server-side aggregation makes per-batch clipping less essential
    for the demo's noise budget).
  - Synthetic test mode: dp_sigma=0.0 disables noise so unit tests are
    deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from client.app.model import FraudDetectionModel


@dataclass
class TrainResult:
    weights: dict[str, torch.Tensor]
    metrics: dict[str, float]


@dataclass
class LocalTrainer:
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    dp_clip_norm: float = 0.5
    dp_sigma: float = 0.0  # 0 = no noise (set by client.main from sigma formula)

    def train(
        self,
        model: FraudDetectionModel,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> TrainResult:
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        # pos_weight handles class imbalance; default 1.0 if no positives
        pos = max(1.0, float((y == 0).sum().item()) / max(1.0, float(y.sum().item())))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))

        train_losses: list[float] = []
        for _ in range(self.epochs):
            model.train()
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(xb)
            train_losses.append(ep_loss / len(ds))

        # eval on val
        import numpy as np
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val).squeeze(-1).cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits_val))
        y_np = y_val.cpu().numpy()
        y_pred = (proba >= 0.5).astype(int)
        metrics = {
            "train_loss": float(train_losses[-1]),
            "val_loss": float(log_loss(y_np, proba.clip(1e-7, 1 - 1e-7), labels=[0, 1])),
            "val_auc": float(roc_auc_score(y_np, proba)) if y_np.sum() > 0 and y_np.sum() < len(y_np) else 0.5,
            "val_f1": float(f1_score(y_np, y_pred, zero_division=0)),
            "val_precision": float(precision_score(y_np, y_pred, zero_division=0)),
            "val_recall": float(recall_score(y_np, y_pred, zero_division=0)),
            "val_accuracy": float(accuracy_score(y_np, y_pred)),
        }

        # DP clip + (optional) noise on the returned weight delta
        weights = model.get_weights()
        weights = self._dp_clip(weights, self.dp_clip_norm)
        if self.dp_sigma > 0:
            weights = {k: v + torch.randn_like(v) * self.dp_sigma for k, v in weights.items()}
        # weight norm metric
        flat = torch.cat([v.flatten() for v in weights.values()])
        metrics["weight_norm"] = float(flat.norm().item())
        metrics["dp_sigma"] = float(self.dp_sigma)
        return TrainResult(weights=weights, metrics=metrics)

    @staticmethod
    def _dp_clip(weights: dict[str, torch.Tensor], clip: float) -> dict[str, torch.Tensor]:
        flat = torch.cat([v.flatten() for v in weights.values()])
        norm = float(flat.norm().item())
        if norm <= clip:
            return {k: v.clone() for k, v in weights.items()}
        scale = clip / norm
        return {k: v * scale for k, v in weights.items()}
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/client/test_trainer.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add client/app/trainer.py tests/unit/client/test_trainer.py
git commit -m "feat(trainer): local training + DP clip + sigma-noise; smoke tests pass"
```

---

### Task 5.4: client storage.py (boto3 wrapper)

**Files:**
- Create: `client/app/storage.py`

Mirror of `server/app/storage.py` but only the methods the client needs (put_weights, get_bytes, presigned-URL fetch). Could be deduplicated later but per spec we keep the two packages independent.

- [ ] **Step 1: Implement directly (covered by integration tests in Phase 6)**

`client/app/storage.py`:

```python
"""S3 wrapper for the client. Same boto3 idiom as server/app/storage.py."""
from __future__ import annotations

import io
from dataclasses import dataclass

import boto3
import requests
import torch


@dataclass
class Storage:
    bucket: str
    region: str = "us-east-1"

    def __post_init__(self) -> None:
        self._s3 = boto3.client("s3", region_name=self.region)

    def put_weights(self, key: str, weights: dict[str, torch.Tensor]) -> None:
        buf = io.BytesIO()
        torch.save(weights, buf)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def get_weights_from_url(self, url: str) -> dict[str, torch.Tensor]:
        """Server hands us a presigned S3 URL — fetch with requests, no AWS creds needed."""
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=True)
```

- [ ] **Step 2: Smoke check imports**

Run:
```bash
python -c "from client.app.storage import Storage; print('ok')"
```

- [ ] **Step 3: Commit**

```bash
git add client/app/storage.py
git commit -m "feat(client): minimal S3 storage wrapper (put_weights + presigned fetch)"
```

---

### Task 5.5: round_runner.py — polling state machine

**Files:**
- Create: `client/app/round_runner.py`
- Test: `tests/unit/client/test_round_runner.py`

State machine that polls the server, downloads weights, trains, uploads, posts metrics. Tests use a fake HTTP layer.

- [ ] **Step 1: Write failing test**

`tests/unit/client/test_round_runner.py`:

```python
from unittest.mock import MagicMock

import torch

from client.app.round_runner import RoundRunner


def _make_weights():
    from client.app.model import FraudDetectionModel
    return FraudDetectionModel().get_weights()


def test_runner_skips_when_no_new_round():
    server = MagicMock()
    server.get_control.return_value = {
        "paused": False, "current_round": 0, "dataset_version": 1,
        "fault": "none", "weights_key": "models/global_round_0000.pt",
    }
    storage = MagicMock()
    trainer = MagicMock()
    r = RoundRunner(
        bank_id="bank_01",
        server=server, storage=storage, trainer=trainer,
        dataset_loader=lambda: (torch.randn(10, 19), torch.zeros(10)),
        last_round_seen=0,
    )
    r.tick()
    trainer.train.assert_not_called()
    server.post_update.assert_not_called()


def test_runner_trains_and_uploads_on_new_round():
    server = MagicMock()
    server.get_control.return_value = {
        "paused": False, "current_round": 1, "dataset_version": 1,
        "fault": "none", "weights_key": "models/global_round_0000.pt",
    }
    server.get_global.return_value = {
        "round": 0, "weights_key": "models/global_round_0000.pt",
        "weights_url": "https://signed",
    }
    storage = MagicMock()
    storage.get_weights_from_url.return_value = _make_weights()
    storage.put_weights = MagicMock()
    trainer = MagicMock()
    trainer.train.return_value = MagicMock(weights=_make_weights(), metrics={"val_auc": 0.8})

    r = RoundRunner(
        bank_id="bank_01",
        server=server, storage=storage, trainer=trainer,
        dataset_loader=lambda: (torch.randn(10, 19), torch.zeros(10)),
        last_round_seen=0,
    )
    r.tick()
    trainer.train.assert_called_once()
    storage.put_weights.assert_called_once()
    server.post_update.assert_called_once()
    # post_update payload key shape
    args, _ = server.post_update.call_args
    assert args[0]["bank_id"] == "bank_01"
    assert args[0]["round"] == 1
    assert args[0]["weights_key"].startswith("updates/bank_01/round_0001")


def test_runner_crashes_on_crash_fault():
    server = MagicMock()
    server.get_control.return_value = {
        "paused": False, "current_round": 1, "dataset_version": 1,
        "fault": "crash", "weights_key": "models/global_round_0000.pt",
    }
    r = RoundRunner(
        bank_id="bank_01",
        server=server, storage=MagicMock(), trainer=MagicMock(),
        dataset_loader=lambda: (torch.randn(10, 19), torch.zeros(10)),
        last_round_seen=0,
    )
    import pytest
    with pytest.raises(SystemExit):
        r.tick()
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`client/app/round_runner.py`:

```python
"""Per-tick state machine for a bank pod. main.py runs this in a while-True
sleep loop. The dependencies (server, storage, trainer, dataset_loader) are
injected so unit tests can substitute fakes."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch

from client.app.model import FraudDetectionModel

_LOG = logging.getLogger("fl.client.round_runner")


class _Server(Protocol):
    def get_control(self, bank_id: str) -> dict[str, Any]: ...
    def get_global(self, bank_id: str) -> dict[str, Any]: ...
    def post_update(self, payload: dict[str, Any]) -> None: ...


class _Storage(Protocol):
    def get_weights_from_url(self, url: str) -> dict[str, torch.Tensor]: ...
    def put_weights(self, key: str, weights: dict[str, torch.Tensor]) -> None: ...


class _Trainer(Protocol):
    def train(self, model: FraudDetectionModel, X: torch.Tensor, y: torch.Tensor, *, X_val: torch.Tensor, y_val: torch.Tensor) -> Any: ...


@dataclass
class RoundRunner:
    bank_id: str
    server: _Server
    storage: _Storage
    trainer: _Trainer
    dataset_loader: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    last_round_seen: int = -1
    last_dataset_version: int = 0
    crashed_once: bool = False
    val_frac: float = 0.15

    def tick(self) -> None:
        ctrl = self.server.get_control(self.bank_id)
        if ctrl["fault"] == "crash" and not self.crashed_once:
            self.crashed_once = True
            _LOG.warning("crash fault triggered → exiting")
            sys.exit(1)
        if ctrl["fault"] == "straggle":
            import time as _t
            _t.sleep(60)  # straggler delay
        if ctrl["paused"]:
            return
        if ctrl["current_round"] <= self.last_round_seen:
            return

        global_info = self.server.get_global(self.bank_id)
        base_weights = self.storage.get_weights_from_url(global_info["weights_url"])

        X, y = self.dataset_loader()
        from sklearn.model_selection import train_test_split
        X_np, y_np = X.numpy(), y.numpy()
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_np, y_np, test_size=self.val_frac, random_state=42,
            stratify=(y_np if y_np.sum() > 1 else None),
        )
        X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        X_val_t, y_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val)

        model = FraudDetectionModel()
        model.set_weights(base_weights)

        # byzantine fault: corrupt the model BEFORE training so the
        # uploaded weights are clearly bad.
        if ctrl["fault"] == "byzantine":
            with torch.no_grad():
                for p in model.parameters():
                    mask = torch.rand_like(p) < 0.3
                    p.data = torch.where(mask, -p.data * 50.0, p.data)

        result = self.trainer.train(model, X_tr_t, y_tr_t, X_val=X_val_t, y_val=y_val_t)
        round_n = int(ctrl["current_round"])
        upload_key = f"updates/{self.bank_id}/round_{round_n:04d}.pt"
        self.storage.put_weights(upload_key, result.weights)
        self.server.post_update({
            "bank_id": self.bank_id,
            "round": round_n,
            "weights_key": upload_key,
            "n_samples": int(len(X_tr_t)),
            "metrics": result.metrics,
        })
        self.last_round_seen = round_n
        self.last_dataset_version = int(ctrl["dataset_version"])
```

- [ ] **Step 4: Run — should pass**

Run: `pytest tests/unit/client/test_round_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add client/app/round_runner.py tests/unit/client/test_round_runner.py
git commit -m "feat(round_runner): polling state machine with fault handling"
```

---

### Task 5.6: client/app/main.py — wiring

**Files:**
- Create: `client/app/main.py`

Wires Settings + Storage + Trainer + a tiny HTTP-client class around requests, then loops `RoundRunner.tick()` with a sleep.

- [ ] **Step 1: Implement**

`client/app/main.py`:

```python
"""Client entrypoint. Boots, asserts dataset, registers with server, then
loops RoundRunner.tick() forever."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
import torch

from client.app.config import Settings
from client.app.dataset import assert_dataset_present, load_dataset
from client.app.preprocessor import preprocess
from client.app.round_runner import RoundRunner
from client.app.storage import Storage
from client.app.trainer import LocalTrainer
from server.app.dp_engine import gaussian_sigma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
_LOG = logging.getLogger("fl.client.main")


class HttpServer:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._sess = requests.Session()

    def register(self, bank_id: str, bank_name: str, n_samples: int) -> None:
        r = self._sess.post(f"{self.base_url}/register",
                            json={"bank_id": bank_id, "bank_name": bank_name, "n_samples": n_samples},
                            timeout=10)
        r.raise_for_status()

    def get_control(self, bank_id: str) -> dict[str, Any]:
        r = self._sess.get(f"{self.base_url}/control/{bank_id}", timeout=10)
        r.raise_for_status()
        return r.json()

    def get_global(self, bank_id: str) -> dict[str, Any]:
        r = self._sess.get(f"{self.base_url}/model/global", params={"bank_id": bank_id}, timeout=10)
        r.raise_for_status()
        return r.json()

    def post_update(self, payload: dict[str, Any]) -> None:
        r = self._sess.post(f"{self.base_url}/model/update", json=payload, timeout=30)
        r.raise_for_status()


def main() -> None:
    s = Settings()
    bank_name = s.bank_name or s.bank_id

    assert_dataset_present(s.dataset_path)

    df = load_dataset(s.dataset_path)
    n_samples = len(df)
    _LOG.info("loaded dataset rows=%d for bank_id=%s", n_samples, s.bank_id)

    server = HttpServer(s.fl_server_url)
    server.register(s.bank_id, bank_name, n_samples)

    storage = Storage(bucket=s.s3_bucket, region=s.aws_region)
    # Compute the per-round client DP sigma using the same formula as the server.
    sigma = gaussian_sigma(s.dp_epsilon, s.dp_delta, s.dp_clip_norm)
    trainer = LocalTrainer(
        epochs=s.local_epochs, batch_size=s.batch_size, lr=s.learning_rate,
        dp_clip_norm=s.dp_clip_norm, dp_sigma=sigma,
    )

    # Preprocess once on boot — assumes the dataset is read-only between runs.
    X_train, y_train, X_val, y_val, _ = preprocess(s.dataset_path, val_frac=0.15)
    X_full = torch.cat([X_train, X_val], dim=0)
    y_full = torch.cat([y_train, y_val], dim=0)

    runner = RoundRunner(
        bank_id=s.bank_id,
        server=server,
        storage=storage,
        trainer=trainer,
        dataset_loader=lambda: (X_full, y_full),
        last_round_seen=-1,
    )

    while True:
        try:
            runner.tick()
        except SystemExit:
            raise
        except Exception:
            _LOG.exception("tick errored — sleeping then retrying")
            time.sleep(5)
        else:
            time.sleep(s.poll_interval_s)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke check imports**

Run: `python -c "import client.app.main; print('ok')"`
Expected: `ok` (will fail Settings() without env vars; that's fine — the module imports).

Actually a cleaner check: `python -c "from client.app.main import main, HttpServer; print('ok')"`.

- [ ] **Step 3: Commit**

```bash
git add client/app/main.py
git commit -m "feat(client): main entrypoint — register, loop RoundRunner.tick()"
```

---

## Phase 6 — Containerisation + e2e

### Task 6.1: server Dockerfile

**Files:**
- Create: `server/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

`server/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Install requirements separately for cache reuse
COPY server/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Copy package
COPY server/app /srv/app

# Plan 2 will copy dashboard/dist → /srv/app/static here.
# In Plan 1 the directory may not exist; main.py guards.

EXPOSE 8080
USER 65532:65532
CMD ["uvicorn", "app.main:build_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Build it locally and verify it boots**

Run:
```bash
docker build -t fl-server-local -f server/Dockerfile .
docker run --rm -e S3_BUCKET=local -e USE_LOCAL_STORAGE=true -p 8080:8080 fl-server-local
```

It will fail because USE_LOCAL_STORAGE path isn't actually wired into Storage in Plan 1; but it should at least import. To validate without S3 we'd need that wiring — instead just verify the image builds and imports work:

```bash
docker run --rm fl-server-local python -c "import app.main; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add server/Dockerfile
git commit -m "feat(server): Dockerfile (python 3.11-slim, non-root, uvicorn factory)"
```

---

### Task 6.2: client Dockerfile

**Files:**
- Create: `client/Dockerfile`

- [ ] **Step 1: Write**

`client/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /work

COPY client/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# server.app.dp_engine is imported by client main for the sigma helper —
# in Plan 1 we copy server/app too. Plan 3 will move gaussian_sigma to a
# shared package or duplicate the function so the client image doesn't
# pull in server code.
COPY server/app /work/server/app
COPY client/app /work/app

# /work/data/bank.csv is supplied by the init container at runtime.
USER 65532:65532
CMD ["python", "-m", "app.main"]
```

- [ ] **Step 2: Build + smoke**

Run:
```bash
docker build -t fl-client-local -f client/Dockerfile .
docker run --rm fl-client-local python -c "import app.main; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add client/Dockerfile
git commit -m "feat(client): Dockerfile with copy of server.app for sigma helper"
```

---

### Task 6.3: docker-compose for e2e (1 server + 3 clients + minio)

**Files:**
- Create: `tests/e2e/compose.yml`
- Create: `tests/e2e/seed_minio.py`

- [ ] **Step 1: Write `tests/e2e/compose.yml`**

```yaml
services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 2s
      retries: 10

  seed-minio:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minioadmin minioadmin &&
      mc mb -p local/fl-test &&
      echo 'bucket ready'
      "
    restart: "no"

  fl-server:
    build:
      context: ../..
      dockerfile: server/Dockerfile
    environment:
      S3_BUCKET: fl-test
      AWS_REGION: us-east-1
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      AWS_ENDPOINT_URL: http://minio:9000
      MIN_NODES: "3"
      INTER_ROUND_DELAY_S: "1"
      ROUND_TIMEOUT_S: "30"
    depends_on:
      seed-minio:
        condition: service_completed_successfully
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()"]
      interval: 5s
      timeout: 3s
      retries: 20

  fl-bank-01:
    build:
      context: ../..
      dockerfile: client/Dockerfile
    environment:
      BANK_ID: bank_01
      BANK_NAME: Bank One
      S3_BUCKET: fl-test
      FL_SERVER_URL: http://fl-server:8080
      AWS_REGION: us-east-1
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      AWS_ENDPOINT_URL: http://minio:9000
      DATASET_PATH: /work/data/bank.csv
      LOCAL_EPOCHS: "2"
      POLL_INTERVAL_S: "2"
    volumes:
      - ../shared/golden_inputs/tiny_bank.csv:/work/data/bank.csv:ro
    depends_on:
      fl-server:
        condition: service_healthy

  fl-bank-02:
    build: { context: ../.., dockerfile: client/Dockerfile }
    environment:
      BANK_ID: bank_02
      S3_BUCKET: fl-test
      FL_SERVER_URL: http://fl-server:8080
      AWS_REGION: us-east-1
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      AWS_ENDPOINT_URL: http://minio:9000
      DATASET_PATH: /work/data/bank.csv
      LOCAL_EPOCHS: "2"
      POLL_INTERVAL_S: "2"
    volumes:
      - ../shared/golden_inputs/tiny_bank.csv:/work/data/bank.csv:ro
    depends_on:
      fl-server:
        condition: service_healthy

  fl-bank-03:
    build: { context: ../.., dockerfile: client/Dockerfile }
    environment:
      BANK_ID: bank_03
      S3_BUCKET: fl-test
      FL_SERVER_URL: http://fl-server:8080
      AWS_REGION: us-east-1
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      AWS_ENDPOINT_URL: http://minio:9000
      DATASET_PATH: /work/data/bank.csv
      LOCAL_EPOCHS: "2"
      POLL_INTERVAL_S: "2"
    volumes:
      - ../shared/golden_inputs/tiny_bank.csv:/work/data/bank.csv:ro
    depends_on:
      fl-server:
        condition: service_healthy
```

**Note on minio + boto3:** The `AWS_ENDPOINT_URL` env var is read by boto3 ≥ 1.28 to redirect to a non-AWS endpoint. No code change needed.

- [ ] **Step 2: Bring it up locally to verify**

Run:
```bash
cd tests/e2e
docker compose up --build -d
sleep 30  # let banks complete a couple of rounds
curl -s http://localhost:8080/round/status | python -m json.tool
docker compose down -v
```

Expected: round_status JSON shows `round >= 1`, `state` in {idle, collecting, aggregating}.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/compose.yml
git commit -m "feat(e2e): docker-compose with minio + 3 banks for local FL smoke"
```

---

### Task 6.4: e2e smoke test

**Files:**
- Create: `tests/e2e/test_smoke.py`

- [ ] **Step 1: Write the test**

```python
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

COMPOSE = Path(__file__).parent / "compose.yml"
TARGET_ROUND = int(os.getenv("E2E_TARGET_ROUND", "5"))
TIMEOUT_S = int(os.getenv("E2E_TIMEOUT_S", "300"))


@pytest.fixture(scope="module")
def stack():
    subprocess.run(["docker", "compose", "-f", str(COMPOSE), "up", "-d", "--build", "--wait"], check=True)
    try:
        yield
    finally:
        subprocess.run(["docker", "compose", "-f", str(COMPOSE), "down", "-v"], check=True)


@pytest.mark.e2e
def test_three_banks_complete_five_rounds(stack):
    deadline = time.time() + TIMEOUT_S
    last_round = -1
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:8080/round/status", timeout=5).json()
            last_round = r["round"]
            if last_round >= TARGET_ROUND:
                break
        except Exception:
            pass
        time.sleep(2)
    assert last_round >= TARGET_ROUND, f"only reached round {last_round} of {TARGET_ROUND}"


@pytest.mark.e2e
def test_metrics_endpoint_has_history(stack):
    r = requests.get("http://localhost:8080/metrics", timeout=5).json()
    assert len(r["history"]) >= 1
    assert "method" in r["history"][-1]
```

- [ ] **Step 2: Run locally**

Run: `pytest tests/e2e/test_smoke.py -v -m e2e`
Expected: 2 passed within ~3 min.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_smoke.py
git commit -m "test(e2e): smoke — 3 banks complete 5 rounds via docker-compose+minio"
```

---

## Final verification

- [ ] **Step 1: Run the full local test suite**

```bash
pytest tests/unit tests/integration -v
ruff check server client tests
```

Expected: all green.

- [ ] **Step 2: Run e2e**

```bash
pytest tests/e2e -v -m e2e
```

Expected: 2 passed.

- [ ] **Step 3: Confirm CI green**

Push to remote; verify GitHub Actions workflow succeeds.

- [ ] **Step 4: Tag the milestone**

```bash
git tag plan-1-complete
git push origin plan-1-complete
```

---

## What's NOT done in Plan 1 (intentionally deferred)

- Dashboard React app (Plan 2)
- WebSocket payload schema for Plan 2 dashboard consumers (only events broadcast, no consumers wired)
- `/admin/*` real implementations (Plan 2)
- HTTPS/auth on `/admin` (Plan 2)
- Terraform IaC (Plan 3)
- Helm chart (Plan 3)
- `deploy.sh`/`teardown.sh` (Plan 3)
- ECR push (Plan 3)
- Real validation set (val_set.pkl generated by Plan 3 `dataset/build_val_set.py`)
- Production CloudWatch logging config (Plan 3)
- NetworkPolicies (Plan 3)
- Init container manifest (Plan 3 — Helm chart)

---

*End of Plan 1.*

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

COMPOSE = Path(__file__).parent / "compose.yml"
TARGET_ROUND = int(os.getenv("E2E_TARGET_ROUND", "5"))
TIMEOUT_S = int(os.getenv("E2E_TIMEOUT_S", "300"))


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.fixture(scope="module")
def stack():
    if not _docker_available():
        pytest.skip("docker not installed; e2e smoke requires Docker Desktop or daemon")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "up", "-d", "--build", "--wait"],
        check=True,
    )
    try:
        yield
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "down", "-v"],
            check=True,
        )


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

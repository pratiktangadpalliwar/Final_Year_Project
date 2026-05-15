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
        r = self._sess.post(
            f"{self.base_url}/register",
            json={"bank_id": bank_id, "bank_name": bank_name, "n_samples": n_samples},
            timeout=10,
        )
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
    sigma = gaussian_sigma(s.dp_epsilon, s.dp_delta, s.dp_clip_norm)
    trainer = LocalTrainer(
        epochs=s.local_epochs, batch_size=s.batch_size, lr=s.learning_rate,
        dp_clip_norm=s.dp_clip_norm, dp_sigma=sigma,
    )

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

"""Per-tick state machine for a bank pod. main.py runs this in a while-True
sleep loop. The dependencies (server, storage, trainer, dataset_loader) are
injected so unit tests can substitute fakes."""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

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
    def train(
        self, model: FraudDetectionModel, X: torch.Tensor, y: torch.Tensor,
        *, X_val: torch.Tensor, y_val: torch.Tensor,
    ) -> Any: ...


@dataclass
class RoundRunner:
    bank_id: str
    server: _Server
    storage: _Storage
    trainer: _Trainer
    dataset_loader: Callable[[int], tuple[torch.Tensor, torch.Tensor]]  # arg = dataset_version
    last_round_seen: int = -1
    # last_dataset_version: tracked for Plan 2 (operator drops new CSV → server bumps
    # version → runner re-fetches dataset before next round). Plan 1 captures the
    # dataset once at boot in client.app.main, so a bump won't trigger anything yet.
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
            time.sleep(60)
        if ctrl["paused"]:
            return
        if ctrl["current_round"] <= self.last_round_seen:
            return

        global_info = self.server.get_global(self.bank_id)
        base_weights = self.storage.get_weights_from_url(global_info["weights_url"])

        version = int(ctrl["dataset_version"])
        X, y = self.dataset_loader(version)
        self.last_dataset_version = version
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
        # last_dataset_version already updated at top of tick after loader call

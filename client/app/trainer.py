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

        weights = model.get_weights()
        weights = self._dp_clip(weights, self.dp_clip_norm)
        if self.dp_sigma > 0:
            weights = {
                k: (v + torch.randn_like(v) * self.dp_sigma if v.is_floating_point() else v.clone())
                for k, v in weights.items()
            }
        flat = torch.cat([v.flatten() for v in weights.values() if v.is_floating_point()])
        metrics["weight_norm"] = float(flat.norm().item())
        metrics["dp_sigma"] = float(self.dp_sigma)
        return TrainResult(weights=weights, metrics=metrics)

    @staticmethod
    def _dp_clip(weights: dict[str, torch.Tensor], clip: float) -> dict[str, torch.Tensor]:
        flat = torch.cat([v.flatten() for v in weights.values() if v.is_floating_point()])
        norm = float(flat.norm().item())
        if norm <= clip:
            return {k: v.clone() for k, v in weights.items()}
        scale = clip / norm
        return {k: (v * scale if v.is_floating_point() else v.clone()) for k, v in weights.items()}

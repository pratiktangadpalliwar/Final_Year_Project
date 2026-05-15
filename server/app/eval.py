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

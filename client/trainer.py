"""
client/trainer.py
-----------------
PyTorch local trainer for fraud detection.
Handles class imbalance, DP noise injection, and metric reporting.
"""

import os
import time
import logging
import numpy as np
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics  import roc_auc_score, f1_score, precision_score, recall_score

from model import FraudDetectionModel
from utils import setup_logging

logger = setup_logging("fl-trainer")

# ── Training config from environment ──────────────────────────────
EPOCHS        = int(os.environ.get("LOCAL_EPOCHS",    "10"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",      "512"))
LR            = float(os.environ.get("LEARNING_RATE", "0.001"))
DP_EPSILON    = float(os.environ.get("DP_EPSILON",    "5.0"))
DP_DELTA      = float(os.environ.get("DP_DELTA",      "1e-5"))
DP_CLIP_NORM  = float(os.environ.get("DP_CLIP_NORM",  "0.5"))
INPUT_DIM     = int(os.environ.get("INPUT_DIM",       "19"))
HIDDEN_DIMS   = [64, 32, 16]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


class LocalTrainer:
    """
    Trains a local PyTorch model on bank data.

    Key features:
    - Weighted sampling to handle fraud class imbalance
    - BCEWithLogitsLoss + class weights for imbalanced labels
    - Gradient clipping (DP preparation)
    - Differential Privacy noise on final weights
    - Validation metrics: AUC-ROC, F1, Precision, Recall
    """

    def __init__(self, bank_id: str, n_samples: int, current_round: int):
        self.bank_id       = bank_id
        self.n_samples     = n_samples
        self.current_round = current_round
        self.device        = torch.device(DEVICE)
        logger.info(f"Trainer initialised | Device: {DEVICE} | "
                    f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")

    # ══════════════════════════════════════════════════════════════
    # MAIN TRAIN METHOD
    # ══════════════════════════════════════════════════════════════
    def train(
        self,
        X_train        : np.ndarray,
        y_train        : np.ndarray,
        X_val          : np.ndarray,
        y_val          : np.ndarray,
        global_weights : Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Optional[List[torch.Tensor]], dict]:
        """
        Perform local training and return (weights, metrics).
        """
        try:
            _train_start = time.time()
            # ── Build model ───────────────────────────────────────
            model = FraudDetectionModel(
                input_dim   = X_train.shape[1],
                hidden_dims = HIDDEN_DIMS,
            ).to(self.device)

            if global_weights:
                model.set_weights([w.to(self.device) for w in global_weights])
                logger.info("Loaded global model weights")

            # ── Data loaders ──────────────────────────────────────
            train_loader = self._make_loader(X_train, y_train, shuffle=True)
            val_loader   = self._make_loader(X_val,   y_val,   shuffle=False)

            # ── Loss (weighted for imbalance) ─────────────────────
            pos_weight = torch.tensor(
                [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                dtype=torch.float32
            ).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # ── Optimiser ─────────────────────────────────────────
            optimiser = torch.optim.Adam(model.parameters(), lr=LR,
                                         weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, patience=3, factor=0.5, verbose=False
            )

            # ── Training loop ─────────────────────────────────────
            best_val_loss = float("inf")
            best_weights  = None

            for epoch in range(1, EPOCHS + 1):
                train_loss = self._train_epoch(model, train_loader,
                                               criterion, optimiser)
                val_loss, val_metrics = self._validate(model, val_loader,
                                                       criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights  = model.get_weights()

                if epoch % 2 == 0 or epoch == EPOCHS:
                    logger.info(
                        f"  Epoch {epoch:3d}/{EPOCHS} | "
                        f"Train loss: {train_loss:.4f} | "
                        f"Val loss: {val_loss:.4f} | "
                        f"AUC: {val_metrics.get('auc', 0):.4f}"
                    )

            # ── Apply Differential Privacy ────────────────────────
            if best_weights:
                best_weights = self._apply_dp(best_weights)

            metrics = {
                "loss"           : float(best_val_loss),
                "accuracy"       : float(val_metrics.get("accuracy",  0)),
                "auc"            : float(val_metrics.get("auc",       0)),
                "f1"             : float(val_metrics.get("f1",        0)),
                "precision"      : float(val_metrics.get("precision", 0)),
                "recall"         : float(val_metrics.get("recall",    0)),
                "training_time_s": round(time.time() - _train_start, 2),
                "epochs"         : EPOCHS,
                "n_samples"      : self.n_samples,
            }

            return best_weights, metrics

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            return None, {}

    # ══════════════════════════════════════════════════════════════
    # EPOCH HELPERS
    # ══════════════════════════════════════════════════════════════
    def _train_epoch(self, model, loader, criterion, optimiser) -> float:
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimiser.zero_grad()
            logits = model(X_batch).squeeze()
            loss   = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping (required for DP)
            nn.utils.clip_grad_norm_(model.parameters(), DP_CLIP_NORM)
            optimiser.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _validate(self, model, loader, criterion) -> Tuple[float, dict]:
        model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits  = model(X_batch).squeeze()
                loss    = criterion(logits, y_batch)
                total_loss += loss.item()
                probs   = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist()  if probs.ndim > 0 else [float(probs)])
                all_labels.extend(y_batch.cpu().numpy().tolist())

        avg_loss   = total_loss / max(len(loader), 1)
        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds      = (all_probs >= 0.5).astype(int)

        metrics = {}
        try:
            metrics["accuracy"] = float((preds == all_labels.astype(int)).mean())
            if len(np.unique(all_labels)) > 1:
                metrics["auc"]       = float(roc_auc_score(all_labels, all_probs))
                metrics["f1"]        = float(f1_score(all_labels, preds, zero_division=0))
                metrics["precision"] = float(precision_score(all_labels, preds, zero_division=0))
                metrics["recall"]    = float(recall_score(all_labels, preds, zero_division=0))
        except Exception:
            pass

        return avg_loss, metrics

    # ══════════════════════════════════════════════════════════════
    # DATA LOADER (weighted sampling for imbalance)
    # ══════════════════════════════════════════════════════════════
    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds  = TensorDataset(X_t, y_t)

        if shuffle and (y == 1).sum() > 0:
            # Over-sample minority class
            n_legit = (y == 0).sum()
            n_fraud = (y == 1).sum()
            weights = np.where(y == 1,
                               n_legit / n_fraud,   # up-weight fraud
                               1.0)
            sampler = WeightedRandomSampler(
                weights     = torch.tensor(weights, dtype=torch.float64),
                num_samples = len(y),
                replacement = True,
            )
            return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=DEVICE == "cuda")

        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

    # ══════════════════════════════════════════════════════════════
    # DIFFERENTIAL PRIVACY
    # ══════════════════════════════════════════════════════════════
    def _apply_dp(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """Clip and add Gaussian noise to model weights."""
        sigma = (np.sqrt(2 * np.log(1.25 / float(DP_DELTA)))
                 * DP_CLIP_NORM / DP_EPSILON)

        dp_weights = []
        for w in weights:
            w_np   = w.cpu().numpy().copy()
            norm   = np.linalg.norm(w_np)
            if norm > DP_CLIP_NORM:
                w_np = w_np * (DP_CLIP_NORM / norm)
            noise  = np.random.normal(0, sigma, w_np.shape)
            dp_weights.append(torch.tensor(w_np + noise, dtype=torch.float32))

        logger.info(f"DP applied: ε={DP_EPSILON}, δ={DP_DELTA}, σ={sigma:.4f}")
        return dp_weights

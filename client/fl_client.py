"""
client/fl_client.py
-------------------
Handles all communication between the bank node and FL server.
Orchestrates the full training pipeline:
  register → download model → train → upload update → notify
"""

import os
import io
import time
import logging
import requests
from pathlib import Path
from typing  import Optional

import torch

from preprocessor import Preprocessor
from trainer      import LocalTrainer
from utils        import setup_logging

logger     = setup_logging("fl-client")
STORAGE_DIR= Path(os.environ.get("LOCAL_STORAGE_DIR", "/tmp/fl-client"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class FLClient:
    """
    Federated Learning client for a bank node.

    Lifecycle per CSV drop:
    1. register()           — announce to server
    2. get_global_model()   — download latest weights
    3. train()              — local training with PyTorch
    4. upload_update()      — push weights to S3
    5. submit_update()      — notify server via REST
    """

    def __init__(self, bank_id: str, bank_name: str, server_url: str):
        self.bank_id    = bank_id
        self.bank_name  = bank_name
        self.server_url = server_url.rstrip("/")
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Storage (S3 or local)
        from storage_client import ClientStorage
        self.storage = ClientStorage()

        self.current_round   : int   = 0
        self.model_version   : str   = "v0.0"
        self.n_samples       : int   = 0
        self.registered      : bool  = False

    # ══════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════════
    def run_training_pipeline(self, csv_path: Path) -> bool:
        """
        Execute full FL training pipeline for a given CSV.
        Returns True on success.
        """
        try:
            # ── 1. Preprocess data ────────────────────────────────
            logger.info("Step 1/5 — Preprocessing data...")
            preprocessor = Preprocessor()
            X_train, y_train, X_val, y_val, n_samples = \
                preprocessor.prepare(csv_path)

            if X_train is None or n_samples == 0:
                logger.error("Preprocessing failed")
                return False

            self.n_samples = n_samples
            logger.info(f"  Samples: {n_samples:,} | "
                        f"Fraud rate: {y_train.mean()*100:.3f}%")

            # ── 2. Register with server ───────────────────────────
            logger.info("Step 2/5 — Registering with FL server...")
            if not self._register():
                return False

            # ── 3. Download global model ──────────────────────────
            logger.info("Step 3/5 — Downloading global model...")
            global_weights = self._get_global_model()

            # ── 4. Train locally ──────────────────────────────────
            logger.info("Step 4/5 — Training locally...")
            trainer = LocalTrainer(
                bank_id      = self.bank_id,
                n_samples    = n_samples,
                current_round= self.current_round,
            )
            local_weights, metrics = trainer.train(
                X_train        = X_train,
                y_train        = y_train,
                X_val          = X_val,
                y_val          = y_val,
                global_weights = global_weights,
            )

            if local_weights is None:
                logger.error("Training failed")
                return False

            logger.info(f"  Loss: {metrics['loss']:.4f} | "
                        f"AUC: {metrics.get('auc', 0):.4f} | "
                        f"F1: {metrics.get('f1', 0):.4f}")

            # ── 5. Upload update and notify server ────────────────
            logger.info("Step 5/5 — Uploading update to server...")
            weights_key = self._upload_update(local_weights)
            if not weights_key:
                return False

            success = self._submit_update(weights_key, metrics)
            return success

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return False

    # ══════════════════════════════════════════════════════════════
    # REST CALLS
    # ══════════════════════════════════════════════════════════════
    def _register(self) -> bool:
        for attempt in range(5):
            try:
                resp = self.session.post(
                    f"{self.server_url}/register",
                    json={
                        "bank_id"  : self.bank_id,
                        "bank_name": self.bank_name,
                        "n_samples": self.n_samples,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data                = resp.json()
                    self.current_round  = data.get("current_round", 0)
                    self.registered     = True
                    logger.info(f"Registered | Round: {self.current_round} | "
                                f"Trust: {data.get('trust_score', 1.0):.2f}")
                    return True
                logger.warning(f"Register failed (HTTP {resp.status_code}): {resp.text}")
            except requests.exceptions.ConnectionError:
                wait = 10 * (attempt + 1)
                logger.warning(f"Server unreachable — retrying in {wait}s "
                               f"(attempt {attempt+1}/5)")
                time.sleep(wait)

        logger.error("Failed to register after 5 attempts")
        return False

    def _get_global_model(self) -> Optional[list]:
        try:
            resp = self.session.get(
                f"{self.server_url}/model/global",
                params  = {"bank_id": self.bank_id},
                timeout = 60,
            )
            if resp.status_code == 404:
                logger.info("No global model yet — starting from scratch")
                return None

            if resp.status_code != 200:
                logger.warning(f"Model download failed: {resp.text}")
                return None

            data          = resp.json()
            self.current_round = data.get("round", 0)
            weights_url   = data.get("weights_url", "")

            if weights_url.startswith("local://"):
                # Local storage mode
                path = Path(weights_url.replace("local://", ""))
                weights = torch.load(path, map_location="cpu")
            else:
                # Download from presigned S3 URL
                model_resp = requests.get(weights_url, timeout=120)
                model_resp.raise_for_status()
                buf     = io.BytesIO(model_resp.content)
                weights = torch.load(buf, map_location="cpu")

            logger.info(f"Downloaded global model (round {self.current_round})")
            return weights

        except Exception as e:
            logger.warning(f"Could not download global model: {e} — using random init")
            return None

    def _upload_update(self, weights: list) -> Optional[str]:
        try:
            key = (f"updates/{self.bank_id}/"
                   f"round_{self.current_round:04d}.pt")
            success = self.storage.upload_weights(weights, key)
            if success:
                logger.info(f"Weights uploaded: {key}")
                return key
            return None
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None

    def _submit_update(self, weights_key: str, metrics: dict) -> bool:
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.server_url}/model/update",
                    json={
                        "bank_id"      : self.bank_id,
                        "round"        : self.current_round,
                        "weights_key"  : weights_key,
                        "n_samples"    : self.n_samples,
                        "local_metrics": metrics,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(f"Update accepted | "
                                f"{data['updates_received']}/{data['nodes_total']} nodes | "
                                f"Quorum: {data['quorum_reached']}")
                    return True
                if resp.status_code == 409:
                    logger.warning("Stale round — will retry next round")
                    return False
                logger.warning(f"Submit failed (HTTP {resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"Submit error: {e}")
                time.sleep(5 * (attempt + 1))

        return False

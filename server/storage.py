"""
server/storage.py
-----------------
Storage abstraction supporting AWS S3 (production) and
local filesystem (development / docker-compose testing).
"""

import os
import io
import json
import logging
import pickle
from datetime  import datetime, timedelta
from pathlib   import Path
from typing    import Optional, List

import torch
import numpy as np

logger = logging.getLogger("fl-server")

USE_LOCAL = os.environ.get("USE_LOCAL_STORAGE", "false").lower() == "true"


# ══════════════════════════════════════════════════════════════════
# S3 STORAGE
# ══════════════════════════════════════════════════════════════════
class S3Storage:
    """
    Handles all model weight and checkpoint I/O via AWS S3.
    Falls back to local filesystem when USE_LOCAL_STORAGE=true.
    """

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.bucket    = cfg["s3_bucket"]
        self.use_local = cfg.get("use_local_storage", False)
        self.local_dir = Path(os.environ.get("LOCAL_STORAGE_DIR", "/tmp/fl-storage"))
        self.local_dir.mkdir(parents=True, exist_ok=True)

        if not self.use_local:
            import boto3
            self.s3     = boto3.client("s3", region_name=cfg["aws_region"])
            self.s3_res = boto3.resource("s3", region_name=cfg["aws_region"])

    # ── Upload ────────────────────────────────────────────────────
    def upload_weights(self, weights: list, key: str) -> bool:
        buf = io.BytesIO()
        torch.save(weights, buf)
        buf.seek(0)

        if self.use_local:
            path = self.local_dir / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.read())
            return True

        try:
            self.s3.upload_fileobj(buf, self.bucket, key)
            return True
        except Exception as e:
            logger.error(f"S3 upload failed ({key}): {e}")
            return False

    # ── Download ──────────────────────────────────────────────────
    def download_weights(self, key: str) -> Optional[list]:
        if self.use_local:
            path = self.local_dir / key
            if not path.exists():
                return None
            return torch.load(path, map_location="cpu")

        try:
            buf = io.BytesIO()
            self.s3.download_fileobj(self.bucket, key, buf)
            buf.seek(0)
            return torch.load(buf, map_location="cpu")
        except Exception as e:
            logger.error(f"S3 download failed ({key}): {e}")
            return None

    @staticmethod
    def load_weights_static(key: str) -> Optional[list]:
        """Static loader used by aggregator."""
        storage = S3Storage.__new__(S3Storage)
        storage.use_local = USE_LOCAL
        storage.local_dir = Path(os.environ.get("LOCAL_STORAGE_DIR", "/tmp/fl-storage"))
        if not USE_LOCAL:
            import boto3
            storage.s3     = boto3.client("s3")
            storage.bucket = os.environ.get("S3_BUCKET", "fl-models")
        return storage.download_weights(key)

    # ── Presigned URL (for clients to download model) ─────────────
    def get_presigned_url(self, key: str, expiry: int = 3600) -> str:
        if self.use_local:
            return f"local://{self.local_dir / key}"

        try:
            return self.s3.generate_presigned_url(
                "get_object",
                Params     = {"Bucket": self.bucket, "Key": key},
                ExpiresIn  = expiry,
            )
        except Exception as e:
            logger.error(f"Presigned URL failed: {e}")
            return ""

    # ── Latest model key ──────────────────────────────────────────
    def get_latest_model_key(self) -> Optional[str]:
        if self.use_local:
            models = sorted(self.local_dir.glob("models/global_round_*.pt"))
            return str(models[-1].relative_to(self.local_dir)) if models else None

        try:
            response = self.s3.list_objects_v2(
                Bucket = self.bucket, Prefix = "models/global_round_"
            )
            objects = response.get("Contents", [])
            if not objects:
                return None
            latest = max(objects, key=lambda x: x["LastModified"])
            return latest["Key"]
        except Exception as e:
            logger.error(f"List objects failed: {e}")
            return None

    # ── Checkpointing ─────────────────────────────────────────────
    def save_checkpoint(self, weights: list, round_num: int,
                        agg_info: dict) -> bool:
        key = f"models/global_round_{round_num:04d}.pt"
        return self.upload_weights(weights, key)

    def load_latest_checkpoint(self) -> Optional[dict]:
        key = self.get_latest_model_key()
        if not key:
            return None

        weights = self.download_weights(key)
        if weights is None:
            return None

        round_num = int(key.split("_round_")[1].split(".")[0])
        return {
            "round"  : round_num,
            "weights": weights,
            "version": f"v{round_num}.0",
            "key"    : key,
        }

    def restore_checkpoint(self, round_num: int) -> bool:
        """Promote a past checkpoint as the current global model."""
        key = f"models/global_round_{round_num:04d}.pt"
        weights = self.download_weights(key)
        if weights is None:
            return False
        return self.upload_weights(weights, "models/global_current.pt")

    # ── Validation set (for anti-corruption checks) ───────────────
    def load_validation_set(self) -> Optional[dict]:
        """Load server-side held-out validation set if available."""
        key = "validation/val_set.pkl"
        if self.use_local:
            path = self.local_dir / key
            if not path.exists():
                return None
            with open(path, "rb") as f:
                return pickle.load(f)

        try:
            buf = io.BytesIO()
            self.s3.download_fileobj(self.bucket, key, buf)
            buf.seek(0)
            return pickle.load(buf)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════
# DYNAMODB STATE
# ══════════════════════════════════════════════════════════════════
class DynamoDBState:
    """Persists FL round state to DynamoDB (or local JSON in dev mode)."""

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.table_name= cfg["dynamodb_table"]
        self.use_local = cfg.get("use_local_storage", False)
        self.local_dir = Path(os.environ.get("LOCAL_STORAGE_DIR", "/tmp/fl-storage"))
        self.local_dir.mkdir(parents=True, exist_ok=True)

        if not self.use_local:
            import boto3
            dynamodb    = boto3.resource("dynamodb", region_name=cfg["aws_region"])
            self.table  = dynamodb.Table(self.table_name)

    def save_round_state(self, state: dict):
        round_num = state.get("current_round", 0)

        if self.use_local:
            path = self.local_dir / f"state/round_{round_num:04d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            return

        try:
            self.table.put_item(Item={
                "round_id" : str(round_num),
                "state"    : json.dumps(state, default=str),
                "updated_at": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            logger.error(f"DynamoDB save failed: {e}")

    def load_round_state(self, round_num: int) -> Optional[dict]:
        if self.use_local:
            path = self.local_dir / f"state/round_{round_num:04d}.json"
            if not path.exists():
                return None
            with open(path) as f:
                return json.load(f)

        try:
            response = self.table.get_item(Key={"round_id": str(round_num)})
            item     = response.get("Item")
            return json.loads(item["state"]) if item else None
        except Exception as e:
            logger.error(f"DynamoDB load failed: {e}")
            return None

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
        # match round_NNNN with any extension (.pt for weights, .json for checkpoints)
        paginator = self._s3.get_paginator("list_objects_v2")
        max_round = None
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                m = re.search(r"round_(\d+)\.\w+$", obj["Key"])
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

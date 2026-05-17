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

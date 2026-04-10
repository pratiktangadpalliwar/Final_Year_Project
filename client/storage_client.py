"""client/storage_client.py — S3/local storage for bank node."""

import os, io, logging
from pathlib import Path
import torch

logger    = logging.getLogger("fl-client")
USE_LOCAL = os.environ.get("USE_LOCAL_STORAGE", "false").lower() == "true"


class ClientStorage:
    def __init__(self):
        self.use_local = USE_LOCAL
        self.local_dir = Path(os.environ.get("LOCAL_STORAGE_DIR", "/tmp/fl-client"))
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.bucket    = os.environ.get("S3_BUCKET", "fl-models")

        if not self.use_local:
            import boto3
            self.s3 = boto3.client("s3",
                                   region_name=os.environ.get("AWS_REGION","us-east-1"))

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
            logger.error(f"Upload failed: {e}")
            return False

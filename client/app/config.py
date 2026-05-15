"""Client configuration. Mirror of server/app/config.py shape, but client-specific."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    # --- identity ---
    bank_id: str = Field(..., description="REQUIRED — e.g. bank_01_retail_urban")
    bank_name: str | None = None  # human label; defaults to bank_id

    # --- contract endpoints ---
    fl_server_url: str = Field(..., description="REQUIRED — http://fl-server:8080 in cluster")
    s3_bucket: str = Field(..., description="REQUIRED — same bucket as server")
    aws_region: str = "us-east-1"
    use_local_storage: bool = False
    local_storage_dir: str = "/tmp/fl-client"

    # --- dataset (init container drops here) ---
    dataset_path: str = "/work/data/bank.csv"

    # --- training ---
    local_epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    input_dim: int = 19

    # --- DP ---
    dp_epsilon: float = 5.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 0.5

    # --- runtime ---
    poll_interval_s: int = 2

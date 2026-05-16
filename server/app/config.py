"""Server configuration. Single source of truth for env-driven behaviour.

Fail fast: invalid configuration raises at import-time when Settings() is
constructed, so misconfigured pods crash on boot rather than at first request.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    # --- AWS / storage ---
    s3_bucket: str = Field(..., description="REQUIRED - bucket for datasets, models, checkpoints")
    aws_region: str = "us-east-1"
    use_local_storage: bool = False
    local_storage_dir: str = "/tmp/fl-server"
    dataset_upload_max_bytes: int = 600 * 1024 * 1024  # 600MB cap for /admin/dataset upload

    # --- FL hyperparams ---
    min_nodes: int = 3
    max_rounds: int = 50
    quorum_pct: float = 0.6
    round_timeout_s: int = 300
    inter_round_delay_s: int = 2
    rollback_threshold: float = 0.05

    # --- DP ---
    dp_epsilon: float = 5.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 0.5

    # --- Model ---
    input_dim: int = 19

    # --- Auth ---
    admin_password_hash: str | None = None  # bcrypt hash; None = auth disabled
    jwt_secret: str | None = None  # required if admin_password_hash set
    jwt_ttl_minutes: int = 480  # 8h cookie lifetime
    cors_origin: str = "*"

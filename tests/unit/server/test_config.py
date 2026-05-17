import pytest

from server.app.config import Settings


def test_defaults_when_only_required_set(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    s = Settings()
    assert s.s3_bucket == "test-bucket"
    assert s.aws_region == "us-east-1"
    assert s.min_nodes == 3
    assert s.quorum_pct == 0.6
    assert s.round_timeout_s == 300
    assert s.inter_round_delay_s == 2
    assert s.dp_epsilon == 5.0
    assert s.dp_delta == 1e-5
    assert s.dp_clip_norm == 0.5
    assert s.input_dim == 19
    assert s.rollback_threshold == 0.05
    assert s.use_local_storage is False
    assert s.local_storage_dir == "/tmp/fl-server"
    assert s.admin_password_hash is None  # set in production
    assert s.cors_origin == "*"


def test_required_s3_bucket_missing_raises(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(ValueError):
        Settings()


def test_use_local_storage_via_env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "irrelevant-when-local")
    monkeypatch.setenv("USE_LOCAL_STORAGE", "true")
    s = Settings()
    assert s.use_local_storage is True

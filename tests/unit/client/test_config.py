import pytest

from client.app.config import Settings


def test_required_fields(monkeypatch):
    monkeypatch.setenv("BANK_ID", "bank_01_retail_urban")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FL_SERVER_URL", "http://fl-server:8080")
    s = Settings()
    assert s.bank_id == "bank_01_retail_urban"
    assert s.s3_bucket == "test-bucket"
    assert s.fl_server_url == "http://fl-server:8080"
    assert s.dataset_path == "/work/data/bank.csv"
    assert s.local_epochs == 10
    assert s.batch_size == 512
    assert s.learning_rate == 0.001
    assert s.dp_epsilon == 5.0
    assert s.dp_clip_norm == 0.5
    assert s.poll_interval_s == 2
    assert s.aws_region == "us-east-1"
    assert s.use_local_storage is False


def test_missing_bank_id_raises(monkeypatch):
    monkeypatch.delenv("BANK_ID", raising=False)
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FL_SERVER_URL", "http://fl-server:8080")
    with pytest.raises(ValueError):
        Settings()

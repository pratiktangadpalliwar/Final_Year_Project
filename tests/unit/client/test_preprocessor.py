from pathlib import Path

import torch

from client.app.preprocessor import preprocess


_CSV = Path(__file__).resolve().parents[2] / "shared" / "golden_inputs" / "tiny_bank.csv"


def test_preprocess_outputs_19_features():
    X_train, y_train, X_val, y_val, _scaler = preprocess(_CSV, val_frac=0.15)
    assert X_train.shape[1] == 19
    assert X_val.shape[1] == 19
    assert y_train.dtype == torch.float32
    assert torch.isfinite(X_train).all()


def test_preprocess_train_val_split_roughly_15pct():
    X_train, y_train, X_val, y_val, _ = preprocess(_CSV, val_frac=0.15)
    total = len(X_train) + len(X_val)
    assert abs(len(X_val) / total - 0.15) < 0.05


def test_preprocess_deterministic_with_seed():
    a = preprocess(_CSV, val_frac=0.15, seed=7)
    b = preprocess(_CSV, val_frac=0.15, seed=7)
    assert torch.allclose(a[0], b[0])

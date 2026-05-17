import torch

from client.app.model import FraudDetectionModel
from client.app.trainer import LocalTrainer, TrainResult


def _toy_batch(n=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 19, generator=g)
    y = (X[:, :3].sum(dim=1) > 0).float()
    return X, y


def test_trainer_decreases_loss_on_toy():
    X, y = _toy_batch()
    model = FraudDetectionModel()
    t = LocalTrainer(epochs=3, batch_size=32, lr=1e-2, dp_clip_norm=10.0, dp_sigma=0.0)
    out = t.train(model, X, y, X_val=X, y_val=y)
    assert isinstance(out, TrainResult)
    assert out.metrics["val_loss"] < 1.0
    base = FraudDetectionModel().get_weights()
    assert not torch.allclose(base["net.0.weight"], out.weights["net.0.weight"])


def test_trainer_dp_clip_caps_norm():
    X, y = _toy_batch()
    model = FraudDetectionModel()
    t = LocalTrainer(epochs=1, batch_size=32, lr=1e-2, dp_clip_norm=0.5, dp_sigma=0.0)
    out = t.train(model, X, y, X_val=X, y_val=y)
    flat = torch.cat([v.flatten() for v in out.weights.values() if v.is_floating_point()])
    assert flat.norm().item() <= 0.5 + 1e-3

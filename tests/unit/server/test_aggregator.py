import torch

from server.app.aggregator import (
    Aggregator,
    fedavg,
    krum,
    trimmed_mean,
)


def _w(scale=1.0):
    return {"a": torch.ones(4) * scale, "b": torch.zeros(4)}


def _samples(n_each):
    return [n_each] * 5  # 5 updates


def test_fedavg_weighted_mean():
    updates = [_w(scale=1.0), _w(scale=3.0)]
    n_samples = [100, 300]  # 1:3 weight
    out = fedavg(updates, n_samples)
    # weighted mean of 1 and 3 with weights 1:3 = (1*1 + 3*3) / 4 = 2.5
    assert torch.allclose(out["a"], torch.full((4,), 2.5))


def test_trimmed_mean_drops_extremes():
    # 5 updates: scales [1, 2, 3, 4, 100] → trim 0.2 (1 each side) → mean(2,3,4)=3
    updates = [_w(scale=s) for s in [1, 2, 3, 4, 100]]
    out = trimmed_mean(updates, trim_ratio=0.2)
    assert torch.allclose(out["a"], torch.full((4,), 3.0))


def test_krum_picks_most_central():
    torch.manual_seed(0)
    base = _w(scale=1.0)
    near = [{k: v + 0.01 * torch.randn_like(v) for k, v in base.items()} for _ in range(4)]
    outlier = _w(scale=50.0)
    updates = near + [outlier]
    out = krum(updates, n_byzantine=1)
    # krum picks one of the near-base updates; its `a` mean should be ~1, not ~50
    assert out["a"].mean().item() < 5.0


def test_aggregator_dispatches_fedavg_when_clean():
    updates = [_w(scale=1.0)] * 4
    n_samples = [100] * 4
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.0, n_total=4)
    assert method == "fedavg"
    assert torch.allclose(out["a"], torch.ones(4))


def test_aggregator_dispatches_trimmed_mean_at_5pct():
    updates = [_w(scale=1.0)] * 9 + [_w(scale=100.0)]  # 10% suspicious
    n_samples = [100] * 10
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.10, n_total=10)
    assert method == "trimmed_mean"


def test_aggregator_dispatches_krum_at_40pct():
    updates = [_w(scale=1.0)] * 4
    n_samples = [100] * 4
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.40, n_total=4)
    assert method == "krum"


def test_aggregator_dispatches_krum_when_n_le_3():
    updates = [_w(scale=1.0)] * 3
    n_samples = [100] * 3
    a = Aggregator()
    out, method = a.aggregate(updates, n_samples, suspicious_pct=0.0, n_total=3)
    assert method == "krum"

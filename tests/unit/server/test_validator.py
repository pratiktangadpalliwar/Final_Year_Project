import torch

from server.app.validator import UpdateValidator


def _flat(d):  # dict of tensors → flat 1D tensor
    return torch.cat([t.flatten() for t in d.values()])


def _make_weights(scale: float = 1.0, n_layers: int = 3, dim: int = 16):
    return {f"layer{i}": torch.randn(dim) * scale for i in range(n_layers)}


def test_clean_updates_all_pass():
    torch.manual_seed(0)
    updates = [_make_weights() for _ in range(5)]
    v = UpdateValidator(norm_bound=10.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(valid) == 5
    assert len(suspicious) == 0


def test_nan_inf_rejected():
    torch.manual_seed(0)
    updates = [_make_weights() for _ in range(4)]
    bad = _make_weights()
    bad["layer0"][0] = float("nan")
    updates.append(bad)
    v = UpdateValidator(norm_bound=10.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(valid) == 4
    assert len(suspicious) == 1


def test_norm_too_large_rejected():
    torch.manual_seed(0)
    updates = [_make_weights() for _ in range(4)]
    huge = _make_weights(scale=100.0)  # ‖w‖ way above 10.0
    updates.append(huge)
    v = UpdateValidator(norm_bound=10.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(suspicious) == 1


def test_low_cosine_to_median_rejected():
    torch.manual_seed(0)
    base = _make_weights()
    # 4 near-identical updates
    updates = [{k: v.clone() + 0.01 * torch.randn_like(v) for k, v in base.items()} for _ in range(4)]
    # 1 sign-flipped update → cosine ≈ -1
    updates.append({k: -v for k, v in base.items()})
    v = UpdateValidator(norm_bound=100.0, cosine_threshold=0.1)
    valid, suspicious = v.score(updates)
    assert len(suspicious) == 1

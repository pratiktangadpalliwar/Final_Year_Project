import math

import torch

from server.app.dp_engine import DPEngine, gaussian_sigma


def test_sigma_formula():
    # σ = sqrt(2 ln(1.25/δ)) * clip / ε
    s = gaussian_sigma(epsilon=5.0, delta=1e-5, clip_norm=0.5)
    expected = math.sqrt(2 * math.log(1.25 / 1e-5)) * 0.5 / 5.0
    assert abs(s - expected) < 1e-9


def test_clip_below_norm_unchanged():
    eng = DPEngine(epsilon=5.0, delta=1e-5, clip_norm=10.0)
    w = {"a": torch.ones(4)}  # norm = 2.0 < 10
    out = eng.clip(w)
    assert torch.allclose(out["a"], w["a"])


def test_clip_above_norm_scaled():
    eng = DPEngine(epsilon=5.0, delta=1e-5, clip_norm=1.0)
    w = {"a": torch.ones(4) * 10}  # norm = 20 > 1
    out = eng.clip(w)
    flat = torch.cat([t.flatten() for t in out.values()])
    assert abs(flat.norm().item() - 1.0) < 1e-5


def test_privatize_changes_weights():
    torch.manual_seed(0)
    eng = DPEngine(epsilon=5.0, delta=1e-5, clip_norm=0.5)
    w = {"a": torch.zeros(100)}
    out = eng.privatize(w)
    assert not torch.allclose(out["a"], w["a"])  # noise added
    # noise std should be roughly sigma; we just sanity-check it's non-trivial
    assert out["a"].std().item() > 1e-4

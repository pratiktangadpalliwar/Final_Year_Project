"""FakeClient: simulates a bank pod for in-process integration tests.
Trains a synthetic update (just a perturbation of the global weights)
and POSTs it to the server. No HTTP — calls handlers directly."""
from __future__ import annotations

import torch

from server.app.routers.client import _pending


def fake_register(rm, bank_id: str, n_samples: int = 1000) -> None:
    rm.register(bank_id, bank_id.replace("_", " ").title(), n_samples)


def fake_post_update(
    bank_id: str,
    round: int,
    *,
    storage,
    perturbation: float = 0.01,
    bad: bool = False,
) -> None:
    """Mimics: client downloads global weights, "trains", uploads to S3, POSTs metadata."""
    latest = storage.latest_round(prefix="models/global_round_") or 0
    base = storage.get_weights(f"models/global_round_{latest:04d}.pt")
    new = {}
    for k, v in base.items():
        if not v.is_floating_point():
            new[k] = v.clone()  # int buffers (e.g. BN num_batches_tracked) — pass through
            continue
        if bad:
            new[k] = -v + 100.0 * torch.randn_like(v)
        else:
            new[k] = v + perturbation * torch.randn_like(v)
    key = f"updates/{bank_id}/round_{round:04d}.pt"
    storage.put_weights(key, new)
    _pending[(round, bank_id)] = {
        "bank_id": bank_id,
        "round": round,
        "weights_key": key,
        "n_samples": 1000,
        "metrics": {"val_auc": 0.85, "val_loss": 0.12},
    }

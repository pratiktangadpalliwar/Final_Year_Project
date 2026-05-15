"""Per-update validation. Produces a suspicion score the aggregator uses to
choose its method (FedAvg / Trimmed-Mean / Krum)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _flatten(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    # int buffers (e.g. BN num_batches_tracked) excluded — float-only for norm/cosine
    return torch.cat([t.flatten().float() for t in weights.values() if t.is_floating_point()])


@dataclass
class UpdateValidator:
    norm_bound: float = 10.0
    cosine_threshold: float = 0.1

    def score(
        self, updates: list[dict[str, torch.Tensor]]
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        """Returns (valid, suspicious). Valid updates are passed to the aggregator;
        suspicious counts feed into method selection.
        """
        flat = [_flatten(u) for u in updates]
        norms = torch.tensor([t.norm().item() for t in flat])
        median_dir = torch.stack(flat).median(dim=0).values
        median_dir = median_dir / (median_dir.norm() + 1e-12)

        valid, suspicious = [], []
        for u, f, n in zip(updates, flat, norms, strict=True):
            if not torch.isfinite(f).all():
                suspicious.append(u); continue
            if n.item() > self.norm_bound:
                suspicious.append(u); continue
            cos = torch.dot(f / (f.norm() + 1e-12), median_dir).item()
            if cos < self.cosine_threshold:
                suspicious.append(u); continue
            valid.append(u)
        return valid, suspicious

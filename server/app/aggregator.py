"""Three aggregation algorithms + a dispatcher.

Port of v1 server/aggregator.py with these changes vs main:
  - Split into pure functions (fedavg, trimmed_mean, krum) + Aggregator class
    (the dispatcher) so tests can hit them in isolation.
  - fedavg returns plain weighted mean, no DP (DP is now its own engine).
  - All inputs are dict[str, Tensor]; the previous flat-tensor variant is gone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Method = Literal["fedavg", "trimmed_mean", "krum"]


def fedavg(updates: list[dict[str, torch.Tensor]], n_samples: list[int]) -> dict[str, torch.Tensor]:
    total = sum(n_samples)
    out: dict[str, torch.Tensor] = {}
    for k in updates[0]:
        out[k] = sum(u[k] * (n / total) for u, n in zip(updates, n_samples, strict=True))
    return out


def trimmed_mean(updates: list[dict[str, torch.Tensor]], trim_ratio: float = 0.1) -> dict[str, torch.Tensor]:
    n = len(updates)
    k_trim = max(1, int(n * trim_ratio))
    out: dict[str, torch.Tensor] = {}
    for key in updates[0]:
        stack = torch.stack([u[key] for u in updates], dim=0)  # (n, ...)
        sorted_, _ = stack.sort(dim=0)
        trimmed = sorted_[k_trim : n - k_trim]
        out[key] = trimmed.mean(dim=0)
    return out


def krum(updates: list[dict[str, torch.Tensor]], n_byzantine: int = 1) -> dict[str, torch.Tensor]:
    """Picks the single update whose sum-of-squared-distances to its (n - n_byz - 2)
    nearest neighbours is minimal. Reference: Blanchard et al. 2017."""
    n = len(updates)
    flats = [torch.cat([t.flatten() for t in u.values()]) for u in updates]
    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = (flats[i] - flats[j]).pow(2).sum()
    k_neighbours = max(1, n - n_byzantine - 2)
    scores = []
    for i in range(n):
        nearest = dists[i].topk(k_neighbours, largest=False).values
        scores.append(nearest.sum().item())
    chosen = int(torch.tensor(scores).argmin().item())
    return updates[chosen]


@dataclass
class Aggregator:
    trim_ratio: float = 0.1
    n_byzantine_estimate: int = 1

    def aggregate(
        self,
        updates: list[dict[str, torch.Tensor]],
        n_samples: list[int],
        suspicious_pct: float,
        n_total: int,
    ) -> tuple[dict[str, torch.Tensor], Method]:
        """Returns (aggregated_weights, method_used). Method is selected per spec 5.1."""
        if n_total <= 3 or suspicious_pct > 0.30:
            return krum(updates, self.n_byzantine_estimate), "krum"
        if suspicious_pct > 0.0:
            return trimmed_mean(updates, self.trim_ratio), "trimmed_mean"
        return fedavg(updates, n_samples), "fedavg"

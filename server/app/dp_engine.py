"""Gaussian DP. Server applies on aggregate; client applies on its own update.

Privacy accounting: sums per-round ε per bank and globally. No hard cap is
enforced in code (demo posture); the dashboard exposes the running sum so the
operator can quote it. To enforce, raise in `privatize()` when budget exceeded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def gaussian_sigma(epsilon: float, delta: float, clip_norm: float) -> float:
    return math.sqrt(2 * math.log(1.25 / delta)) * clip_norm / epsilon


@dataclass
class DPEngine:
    epsilon: float = 5.0
    delta: float = 1e-5
    clip_norm: float = 0.5

    @property
    def sigma(self) -> float:
        return gaussian_sigma(self.epsilon, self.delta, self.clip_norm)

    def clip(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flat = torch.cat([t.flatten() for t in weights.values()])
        norm = flat.norm().item()
        if norm <= self.clip_norm:
            return {k: v.clone() for k, v in weights.items()}
        scale = self.clip_norm / norm
        return {k: v * scale for k, v in weights.items()}

    def add_noise(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        s = self.sigma
        return {k: v + torch.randn_like(v) * s for k, v in weights.items()}

    def privatize(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.add_noise(self.clip(weights))

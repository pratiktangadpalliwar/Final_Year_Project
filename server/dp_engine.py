"""
server/dp_engine.py  —  Differential Privacy (Gaussian mechanism)
"""
import numpy as np
import torch


class DifferentialPrivacy:
    def __init__(self, epsilon: float = 5.0, delta: float = 1e-5,
                 clip_norm: float = 0.5):
        self.epsilon   = epsilon
        self.delta     = delta
        self.clip_norm = clip_norm
        self.sigma     = np.sqrt(2 * np.log(1.25 / delta)) * clip_norm / epsilon

    def add_noise_flat(self, flat: np.ndarray) -> np.ndarray:
        norm    = np.linalg.norm(flat)
        clipped = flat * (self.clip_norm / norm) if norm > self.clip_norm else flat
        noise   = np.random.normal(0, self.sigma, clipped.shape)
        return clipped + noise

    def privacy_spent(self, n_rounds: int) -> dict:
        return {
            "rounds"      : n_rounds,
            "epsilon"     : self.epsilon,
            "composed_eps": round(float(self.epsilon * np.sqrt(n_rounds)), 4),
            "delta"       : self.delta,
        }

"""
shared/model.py
---------------
PyTorch FraudDetectionModel — used by both server (validation)
and client (local training). Keep this file identical in both
server/ and client/ directories.
"""

import torch
import torch.nn as nn
from typing import List


class FraudDetectionModel(nn.Module):
    """
    Multilayer perceptron for binary fraud classification.

    Architecture: Input → [64 → BN → ReLU → Drop] →
                          [32 → BN → ReLU → Drop] →
                          [16 → BN → ReLU] → 1 (logit)

    Uses BatchNorm for training stability and Dropout for
    regularisation — important for imbalanced fraud datasets.
    """

    def __init__(self, input_dim: int = 19,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network    = nn.Sequential(*layers)
        self.input_dim  = input_dim
        self.hidden_dims= hidden_dims

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probability (0–1)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze()

    def get_weights(self) -> List[torch.Tensor]:
        """Return list of parameter tensors (detached copies)."""
        return [p.data.clone() for p in self.parameters()]

    def set_weights(self, weights: List[torch.Tensor]):
        """Load weight tensors into model parameters."""
        for p, w in zip(self.parameters(), weights):
            p.data.copy_(w)

    @staticmethod
    def from_weights(weights: List[torch.Tensor],
                     input_dim: int = 19,
                     hidden_dims: List[int] = None) -> "FraudDetectionModel":
        """Create a model and immediately load weights."""
        model = FraudDetectionModel(input_dim, hidden_dims)
        model.set_weights(weights)
        return model

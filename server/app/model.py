"""19-feature fraud-detection MLP. Shared schema between server and client.

DO NOT modify this file in only one of {server,client}/app/model.py — the
hash-equality test in tests/unit/server/test_model.py will fail. If you need
to change the architecture, edit BOTH files identically and update the test.
"""
from __future__ import annotations

import torch
from torch import nn

INPUT_DIM = 19
HIDDEN_DIMS = (64, 32, 16)


class FraudDetectionModel(nn.Module):
    """MLP: 19 → 64 (BN, ReLU, Dropout) → 32 (BN, ReLU, Dropout) → 16 (BN, ReLU) → 1 logit."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dims: tuple[int, ...] = HIDDEN_DIMS, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits, shape (N, 1)
        return self.net(x)

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(weights, strict=True)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))

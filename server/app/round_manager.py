"""Round + node state. Pure: serialises to a JSON-able dict via checkpoint_dict()
which the storage layer writes to S3. Restore is symmetric."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegisteredBank:
    bank_name: str
    n_samples: int


@dataclass
class RoundManager:
    min_nodes: int = 3
    quorum_pct: float = 0.6
    flag_decay: float = 0.6
    reward_step: float = 0.05
    suspend_threshold: float = 0.2

    current_round: int = 0
    registered: dict[str, RegisteredBank] = field(default_factory=dict)
    suspended: set[str] = field(default_factory=set)
    trust_scores: dict[str, float] = field(default_factory=dict)
    cumulative_eps_per_bank: dict[str, float] = field(default_factory=dict)
    cumulative_eps_global: float = 0.0

    def register(self, bank_id: str, bank_name: str, n_samples: int) -> None:
        self.registered[bank_id] = RegisteredBank(bank_name, n_samples)
        self.trust_scores.setdefault(bank_id, 1.0)
        self.cumulative_eps_per_bank.setdefault(bank_id, 0.0)

    def active_node_count(self) -> int:
        return len([b for b in self.registered if b not in self.suspended])

    def quorum_size(self) -> int:
        active = self.active_node_count()
        return max(self.min_nodes, math.ceil(active * self.quorum_pct))

    def flag_node(self, bank_id: str) -> None:
        score = self.trust_scores.get(bank_id, 1.0) * self.flag_decay
        self.trust_scores[bank_id] = score
        if score < self.suspend_threshold:
            self.suspended.add(bank_id)

    def reward_node(self, bank_id: str) -> None:
        score = min(1.0, self.trust_scores.get(bank_id, 1.0) + self.reward_step)
        self.trust_scores[bank_id] = score

    def add_eps(self, bank_id: str, delta: float) -> None:
        self.cumulative_eps_per_bank[bank_id] = self.cumulative_eps_per_bank.get(bank_id, 0.0) + delta
        self.cumulative_eps_global += delta

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "current_round": self.current_round,
            "registered": {k: {"bank_name": v.bank_name, "n_samples": v.n_samples} for k, v in self.registered.items()},
            "suspended": sorted(self.suspended),
            "trust_scores": dict(self.trust_scores),
            "cumulative_eps_per_bank": dict(self.cumulative_eps_per_bank),
            "cumulative_eps_global": self.cumulative_eps_global,
        }

    def restore_from_dict(self, snap: dict[str, Any]) -> None:
        self.current_round = int(snap["current_round"])
        self.registered = {
            k: RegisteredBank(v["bank_name"], int(v["n_samples"])) for k, v in snap["registered"].items()
        }
        self.suspended = set(snap.get("suspended", []))
        self.trust_scores = dict(snap.get("trust_scores", {}))
        self.cumulative_eps_per_bank = dict(snap.get("cumulative_eps_per_bank", {}))
        self.cumulative_eps_global = float(snap.get("cumulative_eps_global", 0.0))

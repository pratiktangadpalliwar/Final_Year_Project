"""Operator-controlled state. Snapshots to S3 via the storage layer (wired in
Task 4.2). The asyncio round_loop reads `paused`/`current_round`/`state`;
the /control/{bank_id} route reads BankControl."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Fault = Literal["none", "crash", "straggle", "byzantine", "partition"]
RoundState = Literal["idle", "collecting", "aggregating", "stalled"]


@dataclass
class BankControl:
    dataset_version: int = 1
    fault: Fault = "none"


@dataclass
class GlobalControl:
    paused: bool = False
    current_round: int = 0
    state: RoundState = "idle"
    inter_round_delay_s: int = 2


@dataclass
class ControlPlane:
    banks: dict[str, BankControl] = field(default_factory=dict)
    global_: GlobalControl = field(default_factory=GlobalControl)

    # ---- mutators ----
    def pause(self) -> None: self.global_.paused = True
    def resume(self) -> None: self.global_.paused = False
    def reset_rounds(self) -> None:
        self.global_.current_round = 0
        self.global_.state = "idle"

    def set_fault(self, bank_id: str, fault: Fault) -> None:
        self.banks.setdefault(bank_id, BankControl()).fault = fault

    def bump_dataset_version(self, bank_id: str) -> None:
        bc = self.banks.setdefault(bank_id, BankControl())
        bc.dataset_version += 1

    def get_bank(self, bank_id: str) -> BankControl:
        return self.banks.setdefault(bank_id, BankControl())

    # ---- (de)serialise ----
    def snapshot_dict(self) -> dict[str, Any]:
        return {
            "global": {
                "paused": self.global_.paused,
                "current_round": self.global_.current_round,
                "state": self.global_.state,
                "inter_round_delay_s": self.global_.inter_round_delay_s,
            },
            "banks": {k: {"dataset_version": v.dataset_version, "fault": v.fault} for k, v in self.banks.items()},
        }

    def restore_from_dict(self, snap: dict[str, Any]) -> None:
        g = snap["global"]
        self.global_ = GlobalControl(
            paused=bool(g["paused"]),
            current_round=int(g["current_round"]),
            state=g["state"],
            inter_round_delay_s=int(g.get("inter_round_delay_s", 2)),
        )
        self.banks = {
            k: BankControl(dataset_version=int(v["dataset_version"]), fault=v["fault"])
            for k, v in snap.get("banks", {}).items()
        }

"""
server/round_manager.py
-----------------------
Manages FL training rounds, node registry, trust scoring,
checkpointing, rollback, and fault tolerance logic.
"""

import os
import copy
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing  import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("fl-server")


class RoundManager:
    """
    Orchestrates FL rounds with full fault tolerance:
    - Node registration & lifecycle
    - Quorum management (configurable % of active nodes)
    - Trust scoring & Byzantine node suspension
    - Automatic rollback on model degradation
    - Round timeouts (handles stragglers)
    - Checkpoint history
    """

    def __init__(self, cfg: dict, storage, aggregator):
        self.cfg         = cfg
        self.storage     = storage
        self.aggregator  = aggregator

        # Round state
        self.current_round   = 0
        self.model_version   = "v0.0"
        self.global_weights  = None     # current global model weights (list of tensors)
        self.checkpoint_history: List[dict] = []

        # Node registry
        self.registered_nodes: Dict[str, dict] = {}
        self.suspended_nodes : set = set()
        self.trust_scores    : Dict[str, float] = {}

        # Per-round update tracking
        self.pending_updates : Dict[str, dict] = {}
        self._lock           = threading.Lock()

        # Metrics
        self.metrics_history : List[dict] = []
        self.last_global_loss: float = float("inf")

        # Config
        self.min_nodes_per_round   = cfg["min_nodes"]
        self.max_rounds            = cfg["max_rounds"]
        self.quorum_pct            = cfg["quorum_pct"]
        self.rollback_threshold    = cfg["rollback_threshold"]
        self.round_timeout_seconds = int(os.environ.get("ROUND_TIMEOUT", "300"))

        # Straggler timeout thread
        self._timeout_thread: Optional[threading.Thread] = None
        self._round_start_time: Optional[datetime] = None

        # Callback wired by app.py to avoid circular import
        self._aggregation_callback = None
        # Byzantine tracking for rounds-to-recovery metric
        self._byzantine_start_round: Optional[int] = None

    def set_aggregation_callback(self, fn):
        """Register aggregation trigger — avoids circular import with app.py."""
        self._aggregation_callback = fn

    # ══════════════════════════════════════════════════════════════
    # INITIALISATION
    # ══════════════════════════════════════════════════════════════
    def initialise(self):
        """Load latest checkpoint from S3 if available, else start fresh."""
        checkpoint = self.storage.load_latest_checkpoint()
        if checkpoint:
            self.current_round  = checkpoint["round"] + 1
            self.global_weights = checkpoint["weights"]
            self.model_version  = checkpoint["version"]
            logger.info(f"Resumed from checkpoint: round {self.current_round}")
        else:
            self.current_round  = 0
            self.global_weights = None
            self.model_version  = "v0.0"
            logger.info("Starting fresh — no checkpoint found")

        self._start_round_timer()

    # ══════════════════════════════════════════════════════════════
    # NODE MANAGEMENT
    # ══════════════════════════════════════════════════════════════
    def register_node(self, bank_id: str, bank_name: str,
                      n_samples: int) -> dict:
        with self._lock:
            if bank_id not in self.trust_scores:
                self.trust_scores[bank_id] = 1.0   # fresh trust

            self.registered_nodes[bank_id] = {
                "bank_id"      : bank_id,
                "bank_name"    : bank_name,
                "n_samples"    : n_samples,
                "registered_at": datetime.utcnow().isoformat(),
                "last_seen"    : datetime.utcnow().isoformat(),
                "rounds_participated": 0,
                "rounds_skipped"     : 0,
            }

            # Re-admit previously suspended node if trust recovered
            if bank_id in self.suspended_nodes:
                if self.trust_scores[bank_id] >= 0.3:
                    self.suspended_nodes.discard(bank_id)
                    logger.info(f"Node {bank_id} re-admitted after suspension")

        return {
            "registered"    : True,
            "current_round" : self.current_round,
            "trust_score"   : self.trust_scores[bank_id],
            "suspended"     : bank_id in self.suspended_nodes,
        }

    def is_registered(self, bank_id: str) -> bool:
        return bank_id in self.registered_nodes

    def is_suspended(self, bank_id: str) -> bool:
        return bank_id in self.suspended_nodes

    @property
    def active_nodes(self) -> int:
        return len([n for n in self.registered_nodes
                    if n not in self.suspended_nodes])

    @property
    def updates_received(self) -> int:
        return len(self.pending_updates)

    # ══════════════════════════════════════════════════════════════
    # TRUST & REPUTATION
    # ══════════════════════════════════════════════════════════════
    def flag_node(self, bank_id: str, reason: str):
        """Reduce trust score; suspend if below threshold."""
        with self._lock:
            prev = self.trust_scores.get(bank_id, 1.0)
            self.trust_scores[bank_id] = round(prev * 0.6, 4)

            logger.warning(f"Node {bank_id} flagged: {reason} | "
                           f"Trust: {prev:.2f} → {self.trust_scores[bank_id]:.2f}")

            if self.trust_scores[bank_id] < 0.2:
                self.suspended_nodes.add(bank_id)
                logger.warning(f"Node {bank_id} SUSPENDED (trust={self.trust_scores[bank_id]:.2f})")

    def reward_node(self, bank_id: str):
        """Slightly recover trust after a clean round."""
        with self._lock:
            prev = self.trust_scores.get(bank_id, 1.0)
            self.trust_scores[bank_id] = min(1.0, round(prev + 0.05, 4))

    # ══════════════════════════════════════════════════════════════
    # UPDATE ACCEPTANCE
    # ══════════════════════════════════════════════════════════════
    def accept_update(self, bank_id: str, weights_key: str,
                      n_samples: int, metrics: dict) -> bool:
        with self._lock:
            if bank_id in self.pending_updates:
                return False   # already submitted this round

            self.pending_updates[bank_id] = {
                "bank_id"    : bank_id,
                "weights_key": weights_key,
                "n_samples"  : n_samples,
                "metrics"    : metrics,
                "received_at": datetime.utcnow().isoformat(),
                "trust"      : self.trust_scores.get(bank_id, 1.0),
            }

            if bank_id in self.registered_nodes:
                self.registered_nodes[bank_id]["rounds_participated"] += 1
                self.registered_nodes[bank_id]["last_seen"] = datetime.utcnow().isoformat()

        return True

    def quorum_reached(self) -> bool:
        """True when enough nodes have submitted updates."""
        if self.active_nodes < self.min_nodes_per_round:
            return False
        required = max(
            self.min_nodes_per_round,
            int(self.active_nodes * self.quorum_pct)
        )
        return len(self.pending_updates) >= required

    def get_pending_updates(self) -> List[dict]:
        with self._lock:
            return list(self.pending_updates.values())

    # ══════════════════════════════════════════════════════════════
    # MODEL VALIDATION (anti-corruption)
    # ══════════════════════════════════════════════════════════════
    def validate_new_model(self, new_weights: list) -> Tuple[bool, str]:
        """
        Validate aggregated model before accepting it.

        Checks:
        1. Weight norms are within reasonable range
        2. No NaN/Inf values
        3. Accuracy hasn't dropped more than rollback_threshold
        """
        for i, w in enumerate(new_weights):
            if torch.isnan(w).any() or torch.isinf(w).any():
                return False, f"Layer {i} contains NaN/Inf values"

            norm = w.norm().item()
            if norm > 1e6:
                return False, f"Layer {i} norm too large: {norm:.2f}"

        # If we have a validation set check loss improvement
        val_result = self.storage.load_validation_set()
        if val_result and self.global_weights is not None:
            old_loss = self._compute_val_loss(self.global_weights, val_result)
            new_loss = self._compute_val_loss(new_weights,         val_result)

            if new_loss > old_loss + self.rollback_threshold:
                return False, (f"Validation loss degraded: "
                               f"{old_loss:.4f} → {new_loss:.4f}")

        return True, "ok"

    def _compute_val_loss(self, weights: list, val_data: dict) -> float:
        """Quick validation loss computation using current model architecture."""
        try:
            from model import FraudDetectionModel
            import torch.nn as nn

            model = FraudDetectionModel(
                input_dim   = self.cfg["input_dim"],
                hidden_dims = self.cfg["hidden_dims"],
            )
            # Load weights
            state = {k: v for k, v in zip(model.state_dict().keys(), weights)}
            model.load_state_dict(state, strict=False)
            model.eval()

            X = torch.tensor(val_data["X"], dtype=torch.float32)
            y = torch.tensor(val_data["y"], dtype=torch.float32)

            with torch.no_grad():
                logits = model(X).squeeze()
                loss   = nn.BCEWithLogitsLoss()(logits, y)
            return loss.item()
        except Exception:
            return 0.0   # skip validation if model not yet loaded

    # ══════════════════════════════════════════════════════════════
    # ROUND ADVANCEMENT
    # ══════════════════════════════════════════════════════════════
    def advance_round(self, new_weights: list, agg_info: dict):
        """Move to next round after successful aggregation."""
        with self._lock:
            old_round = self.current_round

            # Save checkpoint
            self.checkpoint_history.append({
                "round"   : old_round,
                "weights" : copy.deepcopy(new_weights),
                "version" : self.model_version,
                "agg_info": agg_info,
                "saved_at": datetime.utcnow().isoformat(),
            })
            # Keep last 5 checkpoints in memory
            if len(self.checkpoint_history) > 5:
                self.checkpoint_history.pop(0)

            # Update global model
            self.global_weights = new_weights
            self.current_round  = old_round + 1
            self.model_version  = f"v{self.current_round}.0"

            # Reward participating nodes
            for bank_id in self.pending_updates:
                self.reward_node(bank_id)

            # ── Aggregate client metrics for thesis ───────────────
            updates_list = list(self.pending_updates.values())
            client_mets  = [u.get("metrics", {}) for u in updates_list]

            _m: dict = {}
            for _key in ["loss", "accuracy", "auc", "f1", "precision", "recall"]:
                vals = [float(m[_key]) for m in client_mets
                        if _key in m and isinstance(m.get(_key), (int, float))]
                if vals:
                    _m[f"avg_{_key}"] = round(float(np.mean(vals)), 4)

            _t = [float(m["training_time_s"]) for m in client_mets
                  if m.get("training_time_s", 0) > 0]
            if _t:
                _m["avg_training_time_s"]   = round(float(np.mean(_t)), 2)
                _m["total_training_time_s"] = round(float(np.sum(_t)),  2)

            if new_weights:
                _np = sum(int(t.numel()) for t in new_weights)
                _m["model_params"]        = _np
                _m["comm_overhead_bytes"] = _np * 4 * len(updates_list)
                _m["comm_overhead_mb"]    = round(_m["comm_overhead_bytes"] / 1e6, 3)

            _n_rej  = len(agg_info.get("rejected", []))
            _n_tot  = len(updates_list) + _n_rej
            _m["n_rejected"]               = _n_rej
            _m["n_participated"]           = len(updates_list)
            _m["byzantine_detection_rate"] = round(_n_rej / max(_n_tot, 1), 4)
            _m["aggregation_method"]       = agg_info.get("method", "fedavg")
            _m["suspicious_ratio"]         = agg_info.get("suspicious_ratio", 0.0)

            _method = agg_info.get("method", "fedavg")
            if _method in ("krum", "trimmed_mean", "median"):
                if self._byzantine_start_round is None:
                    self._byzantine_start_round = old_round
                _m["in_byzantine_mode"] = True
            else:
                if self._byzantine_start_round is not None:
                    _m["rounds_to_recovery"]    = old_round - self._byzantine_start_round
                    self._byzantine_start_round = None
                _m["in_byzantine_mode"] = False

            if self.metrics_history:
                _prev = self.metrics_history[-1].get("metrics", {})
                _pl   = _prev.get("avg_loss")
                _cl   = _m.get("avg_loss")
                if _pl is not None and _cl is not None:
                    _m["loss_delta"]       = round(float(_pl) - float(_cl), 4)
                    _m["convergence_rate"] = round(
                        _m["loss_delta"] / max(float(_pl), 1e-8), 4)

            # Record round metrics
            self.metrics_history.append({
                "round"             : old_round,
                "nodes_participated": len(updates_list),
                "nodes_total"       : self.active_nodes,
                "rejected_nodes"    : agg_info.get("rejected", []),
                "metrics"           : _m,
                "timestamp"         : datetime.utcnow().isoformat(),
            })

            # Clear updates for next round
            self.pending_updates.clear()

            logger.info(f"Advanced to round {self.current_round} | "
                        f"Method: {agg_info.get('method')} | "
                        f"Rejected: {agg_info.get('rejected', [])}")

        self._start_round_timer()

    # ══════════════════════════════════════════════════════════════
    # ROLLBACK
    # ══════════════════════════════════════════════════════════════
    def rollback(self, target_round: Optional[int] = None) -> bool:
        """
        Roll back to a previous checkpoint.
        If target_round is None, rolls back one round.
        """
        with self._lock:
            if not self.checkpoint_history:
                logger.error("No checkpoints available for rollback")
                return False

            if target_round is not None:
                candidates = [c for c in self.checkpoint_history
                              if c["round"] == target_round]
                if not candidates:
                    logger.error(f"No checkpoint for round {target_round}")
                    return False
                checkpoint = candidates[-1]
            else:
                checkpoint = self.checkpoint_history[-1]

            self.global_weights = checkpoint["weights"]
            self.current_round  = checkpoint["round"]
            self.model_version  = checkpoint["version"]
            self.pending_updates.clear()

            # Also restore from S3
            self.storage.restore_checkpoint(checkpoint["round"])

            logger.warning(f"Rolled back to round {self.current_round}")

        self._start_round_timer()
        return True

    # ══════════════════════════════════════════════════════════════
    # STRAGGLER TIMEOUT
    # ══════════════════════════════════════════════════════════════
    def _start_round_timer(self):
        """Start timeout thread — force aggregation if stragglers miss deadline."""
        self._round_start_time = datetime.utcnow()
        if self._timeout_thread and self._timeout_thread.is_alive():
            return

        self._timeout_thread = threading.Thread(
            target = self._timeout_watchdog,
            daemon = True,
        )
        self._timeout_thread.start()

    def _timeout_watchdog(self):
        """
        Watchdog: if round hasn't completed after timeout,
        force aggregation with however many updates are available.
        """
        while True:
            time.sleep(30)   # check every 30 seconds
            if not self._round_start_time:
                continue

            elapsed = (datetime.utcnow() - self._round_start_time).seconds
            if elapsed < self.round_timeout_seconds:
                continue

            with self._lock:
                n_updates = len(self.pending_updates)

            if n_updates >= self.min_nodes_per_round:
                logger.warning(f"Round timeout — forcing aggregation with "
                               f"{n_updates} updates (stragglers skipped)")

                # Mark nodes that didn't submit as stragglers
                submitted = set(self.pending_updates.keys())
                for node in list(self.registered_nodes.keys()):
                    if node not in submitted and node not in self.suspended_nodes:
                        self.registered_nodes[node]["rounds_skipped"] = \
                            self.registered_nodes[node].get("rounds_skipped", 0) + 1
                        logger.warning(f"Straggler: {node} missed round {self.current_round}")

                if self._aggregation_callback:
                    self._aggregation_callback()
            else:
                logger.warning(f"Round timeout — only {n_updates} updates, "
                               f"need {self.min_nodes_per_round} — waiting")
            break

    # ══════════════════════════════════════════════════════════════
    # STATUS
    # ══════════════════════════════════════════════════════════════
    def get_status(self) -> dict:
        return {
            "current_round"    : self.current_round,
            "model_version"    : self.model_version,
            "active_nodes"     : self.active_nodes,
            "registered_nodes" : list(self.registered_nodes.keys()),
            "suspended_nodes"  : list(self.suspended_nodes),
            "updates_received" : len(self.pending_updates),
            "quorum_required"  : int(self.active_nodes * self.quorum_pct),
            "quorum_reached"   : self.quorum_reached(),
            "trust_scores"     : self.trust_scores,
            "max_rounds"       : self.max_rounds,
            "round_elapsed_s"   : (
                (datetime.utcnow() - self._round_start_time).seconds
                if self._round_start_time else 0
            ),
            "last_round_metrics": (
                self.metrics_history[-1] if self.metrics_history else {}
            ),
        }

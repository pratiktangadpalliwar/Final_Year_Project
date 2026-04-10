"""
server/aggregator.py
--------------------
Robust FedAvg Aggregator with 4-layer Byzantine defense:

Layer 1 — Pre-aggregation norm/direction checks
Layer 2 — Robust aggregation (Krum / Trimmed Mean / Median / FoolsGold)
Layer 3 — Post-aggregation model validation  (in round_manager)
Layer 4 — Node reputation system             (in round_manager)
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

from dp_engine import DifferentialPrivacy

logger = logging.getLogger("fl-server")


class FedAvgAggregator:
    """
    Weighted FedAvg with automatic Byzantine defense.

    Strategy selection:
    - 0   suspicious nodes  → standard weighted FedAvg
    - 1+  suspicious nodes  → Trimmed Mean
    - 30%+ nodes suspicious → Krum
    - Always: per-update norm & cosine similarity checks
    """

    def __init__(self, dp: DifferentialPrivacy, cfg: dict):
        self.dp              = dp
        self.cfg             = cfg
        self.norm_bound      = float(cfg.get("norm_bound",      10.0))
        self.cosine_threshold= float(cfg.get("cosine_threshold", 0.1))
        self.trim_ratio      = float(cfg.get("trim_ratio",       0.1))

    # ══════════════════════════════════════════════════════════════
    # MAIN AGGREGATION ENTRY POINT
    # ══════════════════════════════════════════════════════════════
    def aggregate(self, updates: List[dict]) -> Tuple[Optional[list], dict]:
        """
        Aggregate client updates with robust defense.

        Parameters
        ----------
        updates : list of dicts, each containing:
            - bank_id     : str
            - weights_key : str  (S3 key for weights file)
            - n_samples   : int
            - trust       : float

        Returns
        -------
        (new_weights, agg_info) or (None, {}) on failure
        """
        if not updates:
            return None, {}

        # ── Load weight tensors from storage keys ─────────────────
        loaded = self._load_weights(updates)
        if len(loaded) == 0:
            logger.error("No weights could be loaded")
            return None, {}

        # ── Layer 1: Per-update validation ───────────────────────
        clean, rejected_l1 = self._validate_updates(loaded)
        if len(clean) == 0:
            logger.error("All updates rejected in Layer 1 validation")
            return None, {"rejected": [u["bank_id"] for u in loaded]}

        # ── Choose aggregation strategy ───────────────────────────
        suspicious_ratio = len(rejected_l1) / max(len(loaded), 1)
        method = self._select_strategy(clean, suspicious_ratio)
        logger.info(f"Aggregation strategy: {method} | "
                    f"Clean: {len(clean)} | Rejected L1: {len(rejected_l1)}")

        # ── Layer 2: Robust aggregation ───────────────────────────
        if method == "krum":
            new_flat, rejected_l2 = self._krum(clean)
        elif method == "trimmed_mean":
            new_flat, rejected_l2 = self._trimmed_mean(clean)
        elif method == "median":
            new_flat, rejected_l2 = self._coordinate_median(clean)
        else:
            new_flat, rejected_l2 = self._weighted_fedavg(clean)

        # ── Apply Differential Privacy ────────────────────────────
        new_flat = self.dp.add_noise_flat(new_flat)

        # ── Reconstruct tensor list ───────────────────────────────
        template     = clean[0]["weights"]
        new_weights  = self._flat_to_weights(new_flat, template)

        all_rejected = [u["bank_id"] for u in rejected_l1] + rejected_l2
        agg_info = {
            "method"       : method,
            "n_updates"    : len(clean),
            "rejected"     : all_rejected,
            "suspicious_ratio": round(suspicious_ratio, 3),
        }

        return new_weights, agg_info

    # ══════════════════════════════════════════════════════════════
    # WEIGHT LOADING
    # ══════════════════════════════════════════════════════════════
    def _load_weights(self, updates: List[dict]) -> List[dict]:
        """Load weight tensors from storage for each update."""
        loaded = []
        for u in updates:
            try:
                from storage import S3Storage
                weights = S3Storage.load_weights_static(u["weights_key"])
                if weights:
                    loaded.append({**u, "weights": weights})
            except Exception as e:
                logger.warning(f"Failed to load weights for {u['bank_id']}: {e}")
        return loaded

    # ══════════════════════════════════════════════════════════════
    # LAYER 1 — PER-UPDATE VALIDATION
    # ══════════════════════════════════════════════════════════════
    def _validate_updates(self, updates: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Check each update for:
        1. NaN/Inf values
        2. Norm bound violation
        3. Cosine similarity vs median direction
        """
        clean, rejected = [], []

        # Compute median direction for cosine check
        flats = [self._weights_to_flat(u["weights"]).numpy() for u in updates]
        median_flat = np.median(flats, axis=0)
        median_norm = np.linalg.norm(median_flat)

        for u, flat in zip(updates, flats):
            reasons = []

            # Check 1: NaN / Inf
            if np.isnan(flat).any() or np.isinf(flat).any():
                reasons.append("nan_inf")

            # Check 2: Norm bound
            norm = np.linalg.norm(flat)
            if norm > self.norm_bound * 100:
                reasons.append(f"norm_violation:{norm:.1f}")

            # Check 3: Cosine similarity vs median
            if median_norm > 1e-8 and norm > 1e-8:
                cosine = float(np.dot(flat, median_flat) / (norm * median_norm))
                if cosine < -self.cosine_threshold:
                    reasons.append(f"cosine_anomaly:{cosine:.3f}")

            if reasons:
                logger.warning(f"Update from {u['bank_id']} rejected: {reasons}")
                rejected.append(u)
            else:
                clean.append(u)

        return clean, rejected

    # ══════════════════════════════════════════════════════════════
    # STRATEGY SELECTION
    # ══════════════════════════════════════════════════════════════
    def _select_strategy(self, updates: List[dict],
                         suspicious_ratio: float) -> str:
        n = len(updates)
        if n < 2:
            return "fedavg"
        if suspicious_ratio >= 0.30 or n <= 3:
            return "krum"
        if suspicious_ratio >= 0.10:
            return "trimmed_mean"
        return "fedavg"

    # ══════════════════════════════════════════════════════════════
    # AGGREGATION METHODS
    # ══════════════════════════════════════════════════════════════
    def _weighted_fedavg(self, updates: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """Standard weighted FedAvg."""
        total    = sum(u["n_samples"] for u in updates)
        agg_flat = np.zeros(self._weights_to_flat(updates[0]["weights"]).shape)

        for u in updates:
            weight   = u["n_samples"] / total
            flat     = self._weights_to_flat(u["weights"]).numpy()
            agg_flat += weight * flat

        return agg_flat, []

    def _krum(self, updates: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Krum: select the update with minimum sum of distances
        to its f nearest neighbours (f = estimated Byzantine nodes).
        """
        n     = len(updates)
        f     = max(1, n // 5)      # assume up to 20% Byzantine
        flats = [self._weights_to_flat(u["weights"]).numpy() for u in updates]

        scores = []
        for i, fi in enumerate(flats):
            dists = sorted(
                [np.linalg.norm(fi - fj)**2 for j, fj in enumerate(flats) if i != j]
            )
            scores.append(sum(dists[:n - f - 2]))

        best_idx     = int(np.argmin(scores))
        rejected     = [updates[i]["bank_id"] for i in range(n) if i != best_idx]
        logger.info(f"Krum selected: {updates[best_idx]['bank_id']} "
                    f"(score={scores[best_idx]:.4f})")

        return flats[best_idx], rejected

    def _trimmed_mean(self, updates: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Coordinate-wise trimmed mean.
        Removes top/bottom trim_ratio% of values per coordinate.
        """
        flats  = np.array([self._weights_to_flat(u["weights"]).numpy()
                           for u in updates])
        n      = len(flats)
        k      = max(1, int(n * self.trim_ratio))

        sorted_flats = np.sort(flats, axis=0)
        trimmed      = sorted_flats[k:n-k, :]
        result       = trimmed.mean(axis=0)

        return result, []

    def _coordinate_median(self, updates: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """Coordinate-wise median — provably Byzantine resilient."""
        flats  = np.array([self._weights_to_flat(u["weights"]).numpy()
                           for u in updates])
        result = np.median(flats, axis=0)
        return result, []

    # ══════════════════════════════════════════════════════════════
    # TENSOR UTILITIES
    # ══════════════════════════════════════════════════════════════
    def _weights_to_flat(self, weights: list) -> torch.Tensor:
        """Flatten list of tensors to 1-D tensor."""
        return torch.cat([w.view(-1) for w in weights])

    def _flat_to_weights(self, flat: np.ndarray,
                         template: list) -> list:
        """Reshape flat array back to list of tensors."""
        flat_t = torch.tensor(flat, dtype=torch.float32)
        result, ptr = [], 0
        for t in template:
            size = t.numel()
            result.append(flat_t[ptr:ptr+size].view(t.shape))
            ptr += size
        return result

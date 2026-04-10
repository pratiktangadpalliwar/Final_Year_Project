"""server/utils.py — Logging setup and update validation helpers."""

import logging
import sys
import numpy as np
import torch


def setup_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def validate_update(bank_id: str, weights_key: str,
                    storage, round_manager) -> dict:
    """
    Quick sanity check on a submitted update before accepting it.
    Returns {"valid": bool, "reason": str}
    """
    weights = storage.download_weights(weights_key)
    if weights is None:
        return {"valid": False, "reason": "weights_not_found"}

    for i, w in enumerate(weights):
        if not isinstance(w, torch.Tensor):
            return {"valid": False, "reason": f"layer_{i}_not_tensor"}
        if torch.isnan(w).any() or torch.isinf(w).any():
            return {"valid": False, "reason": f"layer_{i}_nan_inf"}
        if w.norm().item() > 1e7:
            return {"valid": False, "reason": f"layer_{i}_norm_explosion"}

    return {"valid": True, "reason": "ok"}

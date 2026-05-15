"""CSV → 19-feature tensor pipeline. Port of v1 client/preprocessor.py.

Feature order (must match server.app.model.INPUT_DIM == 19):
  1. transaction_amount                (StandardScaler)
  2. transaction_hour                  (raw)
  3. day_of_week                       (raw)
  4. is_foreign_transaction            (raw 0/1)
  5. is_online_transaction             (raw 0/1)
  6. customer_age                      (StandardScaler)
  7. account_age_days                  (StandardScaler)
  8. avg_amount_customer               (StandardScaler)
  9. std_amount_customer               (StandardScaler)
 10. amount_vs_avg_ratio               (StandardScaler)
 11. amount_zscore                     (raw, already standardised in v1 generator)
 12. total_txns_customer               (StandardScaler)
 13. is_night_transaction              (raw 0/1)
 14. is_weekend                        (raw 0/1)
 15. merchant_cat_grocery              (one-hot)
 16. merchant_cat_online_retail        (one-hot)
 17. merchant_cat_restaurant           (one-hot)
 18. merchant_cat_travel               (one-hot)
 19. merchant_cat_electronics          (one-hot)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_SCALE_COLS = [
    "transaction_amount", "customer_age", "account_age_days",
    "avg_amount_customer", "std_amount_customer", "amount_vs_avg_ratio",
    "total_txns_customer",
]
_MERCHANT_CATS = ["grocery", "online_retail", "restaurant", "travel", "electronics"]


def preprocess(
    csv_path: str | Path,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["is_fraud"])

    for c in _MERCHANT_CATS:
        df[f"merchant_cat_{c}"] = (df.get("merchant_category", "") == c).astype(int)

    feature_cols = (
        ["transaction_amount"]
        + ["transaction_hour", "day_of_week", "is_foreign_transaction", "is_online_transaction"]
        + ["customer_age", "account_age_days"]
        + ["avg_amount_customer", "std_amount_customer", "amount_vs_avg_ratio", "amount_zscore"]
        + ["total_txns_customer", "is_night_transaction", "is_weekend"]
        + [f"merchant_cat_{c}" for c in _MERCHANT_CATS]
    )
    assert len(feature_cols) == 19, f"expected 19 features, got {len(feature_cols)}"

    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_frac, random_state=seed, stratify=(y if y.sum() > 1 else None)
    )

    scaler = StandardScaler()
    scale_idx = [feature_cols.index(c) for c in _SCALE_COLS]
    X_train[:, scale_idx] = scaler.fit_transform(X_train[:, scale_idx])
    X_val[:, scale_idx] = scaler.transform(X_val[:, scale_idx])

    return (
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
        scaler,
    )

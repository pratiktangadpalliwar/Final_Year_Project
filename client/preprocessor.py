"""
client/preprocessor.py
-----------------------
Prepares the bank CSV for PyTorch training.
Handles feature engineering, encoding, scaling, and train/val split.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing  import Tuple, Optional
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler

from utils import setup_logging

logger = setup_logging("preprocessor")

# Features used for training (must match across all banks)
FEATURE_COLS = [
    "transaction_amount",
    "transaction_hour",
    "day_of_week",
    "is_foreign_transaction",
    "is_online_transaction",
    "customer_age",
    "account_age_days",
    "avg_amount_customer",
    "std_amount_customer",
    "amount_vs_avg_ratio",
    "amount_zscore",
    "total_txns_customer",
    "is_night_transaction",
    "is_weekend",
    # Encoded categoricals
    "merchant_cat_grocery",
    "merchant_cat_online_retail",
    "merchant_cat_restaurant",
    "merchant_cat_travel",
    "merchant_cat_electronics",
]

TARGET_COL = "is_fraud"


class Preprocessor:
    """
    Transforms raw bank CSV into train/val numpy arrays.
    Automatically handles:
    - Missing columns (fills with 0)
    - Merchant category one-hot encoding
    - Temporal feature extraction
    - StandardScaler normalization
    - Stratified train/val split
    """

    def prepare(
        self,
        csv_path: Path,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Load and prepare CSV for training.

        Returns
        -------
        X_train, y_train, X_val, y_val, n_samples
        Returns (None, None, None, None, 0) on failure.
        """
        try:
            logger.info(f"Loading: {csv_path}")
            df = pd.read_csv(csv_path, low_memory=False)
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

            df = self._clean(df)
            df = self._engineer_features(df)
            df = self._encode_categoricals(df)

            X, y = self._extract_arrays(df)
            if X is None:
                return None, None, None, None, 0

            X = self._scale(X)

            # Stratified split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size    = val_size,
                stratify     = y,
                random_state = random_state,
            )

            logger.info(
                f"Split → Train: {len(y_train):,} | Val: {len(y_val):,} | "
                f"Fraud rate: {y_train.mean()*100:.3f}%"
            )
            return X_train, y_train, X_val, y_val, len(y_train)

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            return None, None, None, None, 0

    # ── Cleaning ──────────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop metadata columns not used in training
        drop_cols = ["transaction_id", "bank_id", "bank_name", "bank_type",
                     "customer_id", "timestamp", "fraud_type", "fault_scenario"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Fill missing values
        df = df.fillna(0)

        # Remove extreme outliers in amount
        if "transaction_amount" in df.columns:
            q99 = df["transaction_amount"].quantile(0.999)
            df  = df[df["transaction_amount"] <= q99]

        return df

    # ── Feature engineering ───────────────────────────────────────
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure all numeric features exist
        if "transaction_hour" not in df.columns and "timestamp" in df.columns:
            df["transaction_hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        if "day_of_week" not in df.columns and "timestamp" in df.columns:
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

        if "is_night_transaction" not in df.columns and "transaction_hour" in df.columns:
            df["is_night_transaction"] = df["transaction_hour"].apply(
                lambda h: 1 if h >= 22 or h <= 5 else 0
            )

        if "is_weekend" not in df.columns and "day_of_week" in df.columns:
            df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

        # Compute amount ratio if missing
        if "amount_vs_avg_ratio" not in df.columns:
            avg = df["transaction_amount"].mean()
            df["amount_vs_avg_ratio"] = (df["transaction_amount"] / max(avg, 0.01)).round(4)

        if "amount_zscore" not in df.columns:
            mu  = df["transaction_amount"].mean()
            std = df["transaction_amount"].std()
            df["amount_zscore"] = ((df["transaction_amount"] - mu) / max(std, 0.01)).round(4)

        if "avg_amount_customer" not in df.columns:
            df["avg_amount_customer"] = df["transaction_amount"].mean()
        if "std_amount_customer" not in df.columns:
            df["std_amount_customer"] = df["transaction_amount"].std()
        if "total_txns_customer" not in df.columns:
            df["total_txns_customer"] = len(df)

        return df

    # ── Categorical encoding ──────────────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "merchant_category" in df.columns:
            for cat in ["grocery", "online_retail", "restaurant",
                        "travel", "electronics"]:
                df[f"merchant_cat_{cat}"] = (
                    df["merchant_category"] == cat
                ).astype(int)
            df = df.drop(columns=["merchant_category"])

        if "merchant_category_code" in df.columns:
            df = df.drop(columns=["merchant_category_code"])

        return df

    # ── Extract feature matrix ────────────────────────────────────
    def _extract_arrays(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        if TARGET_COL not in df.columns:
            logger.error(f"Target column '{TARGET_COL}' not found")
            return None, None

        # Use only available feature columns
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing   = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            logger.warning(f"Missing features (will use 0): {missing}")
            for c in missing:
                df[c] = 0.0

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)

        return X, y

    # ── Scaling ───────────────────────────────────────────────────
    def _scale(self, X: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(X).astype(np.float32)

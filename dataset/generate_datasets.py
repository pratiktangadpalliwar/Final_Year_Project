"""
generate_datasets.py
--------------------
Generates 7 realistic bank transaction datasets for federated learning.

Each bank has:
- 500,000 transactions
- Unique customer profiles & fraud typologies
- Non-IID data distributions (realistic heterogeneity)
- Fault scenario metadata embedded in a separate config CSV

Output: data/bank_XX_<name>.csv  (one per bank)
        data/dataset_metadata.csv (summary stats)
        data/fault_scenarios.csv  (FL fault injection reference)

Usage:
    python generate_datasets.py

No external data needed — all generated from statistical profiles.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from bank_profiles import BANK_PROFILES, FAULT_SCENARIOS

# ── Config ────────────────────────────────────────────────────────
OUTPUT_DIR   = "data"
RANDOM_SEED  = 42
START_DATE   = datetime(2023, 1, 1)
END_DATE     = datetime(2024, 12, 31)

MERCHANT_CATEGORY_CODES = {
    "grocery":          5411,
    "online_retail":    5965,
    "restaurant":       5812,
    "gas_station":      5541,
    "entertainment":    7996,
    "pharmacy":         5912,
    "travel":           4722,
    "electronics":      5732,
    "jewelry":          5944,
    "business_services":7389,
    "office_supplies":  5943,
    "other":            9999,
    "social_engineering": 9998,
    "synthetic_identity": 9997,
}

FRAUD_AMOUNT_MULTIPLIERS = {
    "card_not_present":     (1.5, 3.0),
    "account_takeover":     (2.0, 5.0),
    "identity_theft":       (1.8, 4.0),
    "card_present_stolen":  (1.2, 2.5),
    "friendly_fraud":       (1.0, 2.0),
    "insider":              (3.0, 8.0),
    "wire_fraud":           (5.0, 15.0),
    "invoice_fraud":        (4.0, 12.0),
    "synthetic_identity":   (2.0, 5.0),
    "social_engineering":   (1.5, 4.0),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def rng(seed):
    return np.random.RandomState(seed)


def generate_customer_ids(n_customers, bank_id, rs):
    """Generate unique customer IDs for a bank."""
    return [f"{bank_id[:8].upper()}_{i:06d}" for i in rs.permutation(n_customers)]


def sample_amounts(profile, n, rs, is_fraud=False, fraud_type=None):
    """Sample transaction amounts based on bank profile."""
    amt = profile["amount"]

    if is_fraud and fraud_type and fraud_type in FRAUD_AMOUNT_MULTIPLIERS:
        lo, hi = FRAUD_AMOUNT_MULTIPLIERS[fraud_type]
        multiplier = rs.uniform(lo, hi, n)
    else:
        multiplier = np.ones(n)

    size_draw = rs.random(n)
    amounts   = np.zeros(n)

    small_mask  = size_draw < amt["small_pct"]
    large_mask  = size_draw > (1 - amt["large_pct"])
    medium_mask = ~small_mask & ~large_mask

    amounts[small_mask]  = np.abs(rs.normal(amt["small_mean"],
                                             amt["small_std"],
                                             small_mask.sum()))
    amounts[medium_mask] = np.abs(rs.normal(amt["medium_mean"],
                                             amt["medium_std"],
                                             medium_mask.sum()))
    amounts[large_mask]  = np.abs(rs.normal(amt["large_mean"],
                                             amt["large_std"],
                                             large_mask.sum()))

    amounts = np.clip(amounts * multiplier, 0.50, 500_000)
    return np.round(amounts, 2)


def sample_hours(weights, n, rs):
    """Sample transaction hours from a weighted distribution."""
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return rs.choice(24, size=n, p=w)


def sample_merchant_categories(categories, n, rs):
    """Sample merchant categories."""
    cats  = list(categories.keys())
    probs = np.array(list(categories.values()), dtype=float)
    probs /= probs.sum()
    return rs.choice(cats, size=n, p=probs)


def generate_timestamps(n, start, end, hours, rs):
    """Generate realistic timestamps within date range."""
    total_seconds = int((end - start).total_seconds())
    base_offsets  = rs.randint(0, total_seconds, n)
    timestamps    = [start + timedelta(seconds=int(s)) for s in base_offsets]
    # Replace hours with the sampled fraud/legit hour distribution
    timestamps = [
        t.replace(hour=int(h), minute=rs.randint(0, 60),
                  second=rs.randint(0, 60))
        for t, h in zip(timestamps, hours)
    ]
    return timestamps


def compute_behavioral_features(df, customer_col="customer_id",
                                 amount_col="transaction_amount",
                                 ts_col="timestamp"):
    """
    Add derived behavioural features that are powerful fraud signals.
    Computed per-customer from transaction history.
    """
    df = df.sort_values([customer_col, ts_col]).reset_index(drop=True)

    # Per-customer rolling stats
    df["avg_amount_customer"]   = df.groupby(customer_col)[amount_col].transform("mean").round(2)
    df["std_amount_customer"]   = df.groupby(customer_col)[amount_col].transform("std").fillna(0).round(2)
    df["amount_vs_avg_ratio"]   = (df[amount_col] / df["avg_amount_customer"].clip(lower=0.01)).round(4)

    # Transaction count features
    df["total_txns_customer"]   = df.groupby(customer_col)[amount_col].transform("count")

    # Hour-based features
    df["is_night_transaction"]  = df["transaction_hour"].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)
    df["is_weekend"]            = pd.to_datetime(df[ts_col]).dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)

    # Amount anomaly score (z-score per customer)
    df["amount_zscore"]         = ((df[amount_col] - df["avg_amount_customer"])
                                   / df["std_amount_customer"].clip(lower=0.01)).round(4)

    return df


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_bank_dataset(bank_id, profile, global_seed):
    """Generate a single bank's transaction dataset."""
    rs          = rng(global_seed)
    n_total     = profile["n_transactions"]
    fraud_rate  = profile["fraud_rate"]
    n_fraud     = int(n_total * fraud_rate)
    n_legit     = n_total - n_fraud

    print(f"  Generating {n_total:,} transactions  "
          f"({n_fraud:,} fraud @ {fraud_rate*100:.3f}%)")

    cust_profile = profile["customer"]
    n_customers  = cust_profile["n_customers"]
    customer_ids = generate_customer_ids(n_customers, bank_id, rs)

    # ── Customer metadata ─────────────────────────────────────────
    customer_ages = np.clip(
        rs.normal(cust_profile["age_mean"], cust_profile["age_std"], n_customers),
        cust_profile["age_min"], cust_profile["age_max"]
    ).astype(int)

    customer_account_ages = np.clip(
        rs.normal(cust_profile["account_age_mean_days"],
                  cust_profile["account_age_std_days"], n_customers),
        30, 5000
    ).astype(int)

    cust_age_map    = dict(zip(customer_ids, customer_ages))
    cust_acage_map  = dict(zip(customer_ids, customer_account_ages))

    # ── LEGITIMATE TRANSACTIONS ───────────────────────────────────
    legit_customers = rs.choice(customer_ids, size=n_legit, replace=True)
    legit_amounts   = sample_amounts(profile, n_legit, rs, is_fraud=False)
    legit_hours     = sample_hours(profile["hour_weights_legit"], n_legit, rs)
    legit_merchants = sample_merchant_categories(profile["merchant_categories"],
                                                  n_legit, rs)
    legit_foreign   = rs.binomial(1, profile["foreign_transaction_rate_legit"], n_legit)
    legit_online    = rs.binomial(1, profile["online_transaction_rate_legit"],  n_legit)
    legit_ts        = generate_timestamps(n_legit, START_DATE, END_DATE,
                                          legit_hours, rs)

    # ── FRAUDULENT TRANSACTIONS ───────────────────────────────────
    fraud_types      = list(profile["fraud_types"].keys())
    fraud_type_probs = np.array(list(profile["fraud_types"].values()), dtype=float)
    fraud_type_probs /= fraud_type_probs.sum()
    fraud_type_labels = rs.choice(fraud_types, size=n_fraud, p=fraud_type_probs)

    fraud_customers  = rs.choice(customer_ids, size=n_fraud, replace=True)
    fraud_amounts    = np.array([
        sample_amounts(profile, 1, rs, is_fraud=True, fraud_type=ft)[0]
        for ft in fraud_type_labels
    ])
    fraud_hours      = sample_hours(profile["hour_weights_fraud"], n_fraud, rs)
    fraud_merchants  = sample_merchant_categories(profile["merchant_categories"],
                                                   n_fraud, rs)
    fraud_foreign    = rs.binomial(1, profile["foreign_transaction_rate_fraud"], n_fraud)
    fraud_online     = rs.binomial(1, profile["online_transaction_rate_fraud"],  n_fraud)
    fraud_ts         = generate_timestamps(n_fraud, START_DATE, END_DATE,
                                           fraud_hours, rs)

    # ── ASSEMBLE DATAFRAME ────────────────────────────────────────
    legit_df = pd.DataFrame({
        "transaction_id":       [f"TXN_{bank_id[:4].upper()}_{i:08d}"
                                 for i in range(n_legit)],
        "customer_id":          legit_customers,
        "timestamp":            legit_ts,
        "transaction_amount":   legit_amounts,
        "merchant_category":    legit_merchants,
        "merchant_category_code": [MERCHANT_CATEGORY_CODES.get(m, 9999)
                                    for m in legit_merchants],
        "transaction_hour":     legit_hours,
        "is_foreign_transaction": legit_foreign,
        "is_online_transaction":  legit_online,
        "fraud_type":           "none",
        "is_fraud":             0,
    })

    fraud_df = pd.DataFrame({
        "transaction_id":       [f"TXN_{bank_id[:4].upper()}_{i+n_legit:08d}"
                                 for i in range(n_fraud)],
        "customer_id":          fraud_customers,
        "timestamp":            fraud_ts,
        "transaction_amount":   fraud_amounts,
        "merchant_category":    fraud_merchants,
        "merchant_category_code": [MERCHANT_CATEGORY_CODES.get(m, 9999)
                                    for m in fraud_merchants],
        "transaction_hour":     fraud_hours,
        "is_foreign_transaction": fraud_foreign,
        "is_online_transaction":  fraud_online,
        "fraud_type":           fraud_type_labels,
        "is_fraud":             1,
    })

    df = pd.concat([legit_df, fraud_df], ignore_index=True)

    # ── Add customer demographic features ─────────────────────────
    df["customer_age"]         = df["customer_id"].map(cust_age_map)
    df["account_age_days"]     = df["customer_id"].map(cust_acage_map)

    # ── Add day_of_week ───────────────────────────────────────────
    df["day_of_week"]          = pd.to_datetime(df["timestamp"]).dt.dayofweek

    # ── Shuffle (mix fraud and legit) ─────────────────────────────
    df = df.sample(frac=1, random_state=global_seed).reset_index(drop=True)

    # ── Compute behavioural features ──────────────────────────────
    df = compute_behavioral_features(df)

    # ── Add bank metadata columns ─────────────────────────────────
    df["bank_id"]              = bank_id
    df["bank_name"]            = profile["name"]
    df["bank_type"]            = profile["type"]
    df["fault_scenario"]       = profile["fault_scenario"]

    # ── Final column order ────────────────────────────────────────
    cols = [
        "transaction_id", "bank_id", "bank_name", "bank_type",
        "customer_id", "timestamp", "day_of_week",
        "transaction_amount", "transaction_hour",
        "merchant_category", "merchant_category_code",
        "is_foreign_transaction", "is_online_transaction",
        "customer_age", "account_age_days",
        "avg_amount_customer", "std_amount_customer",
        "amount_vs_avg_ratio", "amount_zscore",
        "total_txns_customer", "is_night_transaction", "is_weekend",
        "fraud_type", "is_fraud",
        "fault_scenario",
    ]
    df = df[cols]

    return df


# ═══════════════════════════════════════════════════════════════════
# METADATA & FAULT SCENARIO EXPORTS
# ═══════════════════════════════════════════════════════════════════

def export_metadata(summary_rows):
    """Export dataset summary statistics."""
    meta_df = pd.DataFrame(summary_rows)
    path    = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")
    meta_df.to_csv(path, index=False)
    print(f"\n  Metadata saved → {path}")


def export_fault_scenarios():
    """Export fault scenario reference for FL simulation."""
    rows = []
    for scenario_id, info in FAULT_SCENARIOS.items():
        rows.append({
            "scenario_id":   scenario_id,
            "description":   info["description"],
            "behavior":      info["behavior"],
            "fl_handling":   info["fl_handling"],
            "affected_bank": info["affected_bank"],
        })

        # Add fault round info from bank profile
        for bank_id, profile in BANK_PROFILES.items():
            if profile["fault_scenario"] == scenario_id and scenario_id != "none":
                rows[-1]["fault_round"]       = profile.get("fault_round", "N/A")
                rows[-1]["fault_rejoin_round"] = profile.get("fault_rejoin_round", "N/A")

    fault_df = pd.DataFrame(rows)
    path     = os.path.join(OUTPUT_DIR, "fault_scenarios.csv")
    fault_df.to_csv(path, index=False)
    print(f"  Fault scenarios saved → {path}")


def export_feature_guide():
    """Export a feature description reference CSV."""
    features = [
        ("transaction_id",          "Unique transaction identifier"),
        ("bank_id",                 "Bank identifier key"),
        ("bank_name",               "Bank full name"),
        ("bank_type",               "Bank category type"),
        ("customer_id",             "Anonymised customer identifier"),
        ("timestamp",               "Transaction datetime"),
        ("day_of_week",             "0=Monday … 6=Sunday"),
        ("transaction_amount",      "Transaction value in USD"),
        ("transaction_hour",        "Hour of transaction (0-23)"),
        ("merchant_category",       "Merchant type label"),
        ("merchant_category_code",  "ISO MCC code"),
        ("is_foreign_transaction",  "1 if transaction outside home country"),
        ("is_online_transaction",   "1 if card-not-present / online"),
        ("customer_age",            "Customer age in years"),
        ("account_age_days",        "Days since account opened"),
        ("avg_amount_customer",     "Customer historical average transaction"),
        ("std_amount_customer",     "Customer historical std dev of transactions"),
        ("amount_vs_avg_ratio",     "This amount / customer average (>2 = anomalous)"),
        ("amount_zscore",           "Z-score of amount vs customer history"),
        ("total_txns_customer",     "Total number of transactions by this customer"),
        ("is_night_transaction",    "1 if transaction between 22:00-05:00"),
        ("is_weekend",              "1 if Saturday or Sunday"),
        ("fraud_type",              "Type of fraud (none if legitimate)"),
        ("is_fraud",                "TARGET: 0=Legitimate  1=Fraud"),
        ("fault_scenario",          "FL fault scenario assigned to this bank node"),
    ]
    guide_df = pd.DataFrame(features, columns=["feature", "description"])
    path     = os.path.join(OUTPUT_DIR, "feature_guide.csv")
    guide_df.to_csv(path, index=False)
    print(f"  Feature guide saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FL DATASET GENERATOR")
    print("  7 Banks | 500,000 transactions each | 3.5M total")
    print("=" * 60)

    total_start  = time.time()
    summary_rows = []

    for i, (bank_id, profile) in enumerate(BANK_PROFILES.items(), 1):
        print(f"\n[{i}/7] {profile['name']}  ({profile['type']})")
        print(f"       Fault scenario: {profile['fault_scenario']}")
        t0 = time.time()

        df = generate_bank_dataset(
            bank_id     = bank_id,
            profile     = profile,
            global_seed = RANDOM_SEED + i * 100,
        )

        # Save CSV
        filename = f"{bank_id}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)

        elapsed = time.time() - t0
        actual_fraud  = df["is_fraud"].sum()
        actual_rate   = df["is_fraud"].mean() * 100

        print(f"  ✅  Saved → {filepath}")
        print(f"       Rows: {len(df):,}  |  Fraud: {actual_fraud:,} ({actual_rate:.3f}%)  |  [{elapsed:.1f}s]")

        summary_rows.append({
            "bank_id":            bank_id,
            "bank_name":          profile["name"],
            "bank_type":          profile["type"],
            "total_transactions": len(df),
            "fraud_count":        int(actual_fraud),
            "fraud_rate_pct":     round(actual_rate, 4),
            "n_customers":        profile["customer"]["n_customers"],
            "n_features":         len(df.columns) - 1,  # excl. is_fraud
            "fault_scenario":     profile["fault_scenario"],
            "fault_round":        profile.get("fault_round", "N/A"),
            "file":               filename,
        })

    # ── Export supporting files ────────────────────────────────────
    export_metadata(summary_rows)
    export_fault_scenarios()
    export_feature_guide()

    total_elapsed = time.time() - total_start
    total_rows    = sum(r["total_transactions"] for r in summary_rows)
    total_fraud   = sum(r["fraud_count"] for r in summary_rows)

    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total transactions : {total_rows:,}")
    print(f"  Total fraud cases  : {total_fraud:,}")
    print(f"  Overall fraud rate : {total_fraud/total_rows*100:.4f}%")
    print(f"  Time elapsed       : {total_elapsed:.1f}s")
    print(f"  Output directory   : {os.path.abspath(OUTPUT_DIR)}/")
    print()
    print("  Files generated:")
    for r in summary_rows:
        print(f"    {r['file']:<45} {r['total_transactions']:>8,} rows")
    print(f"    {'dataset_metadata.csv':<45} summary")
    print(f"    {'fault_scenarios.csv':<45} FL fault reference")
    print(f"    {'feature_guide.csv':<45} feature descriptions")
    print("=" * 60)


if __name__ == "__main__":
    main()

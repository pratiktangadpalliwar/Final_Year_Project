"""
bank_profiles.py
----------------
Defines the 7 bank profiles with realistic characteristics.
Each bank has different customer demographics, transaction patterns,
merchant distributions, and fraud typologies — creating genuine
non-IID data for federated learning.
"""

BANK_PROFILES = {

    "bank_01_retail_urban": {
        "name": "MetroRetail Bank",
        "type": "Large Retail (Urban)",
        "n_transactions": 500_000,
        "fraud_rate": 0.0035,        # 0.35% — high volume card-not-present
        "description": "High-volume urban retail bank. Large customer base, "
                       "many online purchases. Fraud concentrated in CNP and "
                       "account takeover.",

        # Customer demographics
        "customer": {
            "n_customers": 80_000,
            "age_mean": 38,
            "age_std": 12,
            "age_min": 18,
            "age_max": 80,
            "account_age_mean_days": 900,   # avg account age
            "account_age_std_days": 600,
        },

        # Transaction amount distributions (legitimate)
        "amount": {
            "small_pct": 0.55,       # < $50
            "medium_pct": 0.35,      # $50–$500
            "large_pct": 0.10,       # > $500
            "small_mean": 22,   "small_std": 14,
            "medium_mean": 140, "medium_std": 90,
            "large_mean": 850,  "large_std": 450,
        },

        # Merchant category distribution (legitimate)
        "merchant_categories": {
            "grocery":          0.28,
            "online_retail":    0.22,
            "restaurant":       0.15,
            "gas_station":      0.10,
            "entertainment":    0.08,
            "pharmacy":         0.07,
            "travel":           0.05,
            "electronics":      0.03,
            "jewelry":          0.01,
            "other":            0.01,
        },

        # Time patterns (hour weights — fraud spikes late night)
        "hour_weights_legit":  [0.5,0.3,0.2,0.2,0.3,0.5,1.0,2.0,2.5,2.8,
                                 2.8,2.5,2.2,2.2,2.2,2.5,2.8,2.8,2.5,2.2,
                                 2.0,1.8,1.2,0.8],
        "hour_weights_fraud":  [2.0,2.5,2.8,2.8,2.0,1.0,0.5,0.3,0.3,0.5,
                                 0.8,1.0,1.2,1.0,0.8,0.8,1.0,1.2,1.5,1.8,
                                 2.0,2.2,2.5,2.2],

        # Fraud typology weights
        "fraud_types": {
            "card_not_present":     0.40,
            "account_takeover":     0.25,
            "identity_theft":       0.15,
            "card_present_stolen":  0.10,
            "friendly_fraud":       0.07,
            "insider":              0.03,
        },

        # Geographic
        "foreign_transaction_rate_legit": 0.08,
        "foreign_transaction_rate_fraud": 0.35,
        "online_transaction_rate_legit":  0.30,
        "online_transaction_rate_fraud":  0.65,

        # Fault tolerance role (for FL simulation metadata)
        "fault_scenario": "node_crash",
        "fault_round": 25,           # crashes at round 25
    },

    "bank_02_corporate": {
        "name": "CorpFinance Bank",
        "type": "Corporate / Business",
        "n_transactions": 500_000,
        "fraud_rate": 0.0018,        # 0.18% — wire fraud, large anomalies
        "description": "Business banking focused. Large transaction amounts "
                       "common. Fraud patterns: wire fraud, invoice fraud, "
                       "large-amount anomalies.",

        "customer": {
            "n_customers": 15_000,   # fewer but larger accounts
            "age_mean": 44,
            "age_std": 10,
            "age_min": 25,
            "age_max": 72,
            "account_age_mean_days": 1800,
            "account_age_std_days": 900,
        },

        "amount": {
            "small_pct": 0.20,
            "medium_pct": 0.40,
            "large_pct": 0.40,       # much higher large transaction rate
            "small_mean": 35,   "small_std": 18,
            "medium_mean": 280, "medium_std": 150,
            "large_mean": 4500, "large_std": 3000,
        },

        "merchant_categories": {
            "business_services":    0.30,
            "travel":               0.20,
            "online_retail":        0.15,
            "restaurant":           0.12,
            "electronics":          0.08,
            "office_supplies":      0.07,
            "gas_station":          0.04,
            "entertainment":        0.03,
            "grocery":              0.01,
            "other":                0.00,
        },

        "hour_weights_legit":  [0.1,0.1,0.1,0.1,0.2,0.5,1.5,2.8,3.5,3.8,
                                 3.5,2.8,2.0,2.5,3.0,3.2,2.8,2.0,1.2,0.8,
                                 0.5,0.3,0.2,0.1],
        "hour_weights_fraud":  [0.5,0.8,1.0,1.2,1.0,0.8,0.5,0.8,1.5,2.0,
                                 2.2,2.0,1.8,2.0,2.2,2.0,1.8,1.5,1.2,1.0,
                                 0.8,0.8,0.6,0.5],

        "fraud_types": {
            "wire_fraud":           0.35,
            "invoice_fraud":        0.25,
            "account_takeover":     0.20,
            "card_not_present":     0.12,
            "insider":              0.08,
            "identity_theft":       0.00,
            "card_present_stolen":  0.00,
            "friendly_fraud":       0.00,
        },

        "foreign_transaction_rate_legit": 0.18,
        "foreign_transaction_rate_fraud": 0.42,
        "online_transaction_rate_legit":  0.45,
        "online_transaction_rate_fraud":  0.72,

        "fault_scenario": "straggler",
        "fault_round": 15,           # slow from round 15 onwards
    },

    "bank_03_regional_rural": {
        "name": "Heartland Community Bank",
        "type": "Regional / Rural",
        "n_transactions": 500_000,
        "fraud_rate": 0.0012,        # 0.12% — low fraud, identity theft
        "description": "Small regional bank serving rural communities. "
                       "Lower transaction volumes per customer, local merchants. "
                       "Fraud mainly identity theft and check fraud.",

        "customer": {
            "n_customers": 25_000,
            "age_mean": 48,
            "age_std": 15,
            "age_min": 18,
            "age_max": 85,
            "account_age_mean_days": 2500,   # long-standing accounts
            "account_age_std_days": 1200,
        },

        "amount": {
            "small_pct": 0.65,
            "medium_pct": 0.30,
            "large_pct": 0.05,
            "small_mean": 18,   "small_std": 11,
            "medium_mean": 120, "medium_std": 70,
            "large_mean": 650,  "large_std": 280,
        },

        "merchant_categories": {
            "grocery":          0.35,
            "gas_station":      0.18,
            "restaurant":       0.15,
            "pharmacy":         0.12,
            "online_retail":    0.08,
            "entertainment":    0.05,
            "travel":           0.03,
            "electronics":      0.02,
            "business_services":0.01,
            "other":            0.01,
        },

        "hour_weights_legit":  [0.2,0.1,0.1,0.1,0.3,0.8,1.8,2.5,2.8,2.5,
                                 2.2,2.0,2.2,2.0,2.0,2.2,2.5,2.8,2.5,2.0,
                                 1.5,1.0,0.6,0.3],
        "hour_weights_fraud":  [1.5,1.8,2.0,1.8,1.2,0.8,0.5,0.5,0.8,1.0,
                                 1.2,1.0,1.0,1.0,1.0,1.0,1.2,1.5,1.8,2.0,
                                 2.0,1.8,1.5,1.5],

        "fraud_types": {
            "identity_theft":       0.40,
            "card_present_stolen":  0.30,
            "friendly_fraud":       0.15,
            "card_not_present":     0.10,
            "account_takeover":     0.05,
            "insider":              0.00,
            "wire_fraud":           0.00,
            "invoice_fraud":        0.00,
        },

        "foreign_transaction_rate_legit": 0.02,
        "foreign_transaction_rate_fraud": 0.20,
        "online_transaction_rate_legit":  0.12,
        "online_transaction_rate_fraud":  0.38,

        "fault_scenario": "network_partition",
        "fault_round": 30,
    },

    "bank_04_neobank_digital": {
        "name": "NeoVault Digital Bank",
        "type": "Digital / Neobank",
        "n_transactions": 500_000,
        "fraud_rate": 0.0045,        # 0.45% — highest fraud, online-only
        "description": "Digital-only neobank. All transactions online. "
                       "Younger customer base, crypto-linked, high account "
                       "takeover and synthetic identity fraud.",

        "customer": {
            "n_customers": 60_000,
            "age_mean": 29,
            "age_std": 8,
            "age_min": 18,
            "age_max": 55,
            "account_age_mean_days": 400,    # newer accounts
            "account_age_std_days": 280,
        },

        "amount": {
            "small_pct": 0.45,
            "medium_pct": 0.40,
            "large_pct": 0.15,
            "small_mean": 28,   "small_std": 18,
            "medium_mean": 165, "medium_std": 110,
            "large_mean": 1200, "large_std": 800,
        },

        "merchant_categories": {
            "online_retail":    0.40,
            "entertainment":    0.18,
            "restaurant":       0.12,
            "electronics":      0.10,
            "travel":           0.08,
            "grocery":          0.06,
            "business_services":0.03,
            "gas_station":      0.02,
            "pharmacy":         0.01,
            "other":            0.00,
        },

        "hour_weights_legit":  [1.0,0.8,0.6,0.5,0.5,0.6,0.8,1.2,1.5,1.8,
                                 2.0,2.2,2.2,2.0,1.8,1.8,2.0,2.2,2.5,2.8,
                                 2.8,2.5,2.0,1.5],
        "hour_weights_fraud":  [2.5,2.8,3.0,3.0,2.5,1.5,0.8,0.5,0.5,0.8,
                                 1.0,1.2,1.2,1.0,0.8,0.8,1.0,1.5,2.0,2.2,
                                 2.5,2.8,3.0,2.8],

        "fraud_types": {
            "account_takeover":     0.35,
            "card_not_present":     0.30,
            "synthetic_identity":   0.20,
            "identity_theft":       0.10,
            "friendly_fraud":       0.05,
            "insider":              0.00,
            "wire_fraud":           0.00,
            "invoice_fraud":        0.00,
            "card_present_stolen":  0.00,
        },

        "foreign_transaction_rate_legit": 0.22,
        "foreign_transaction_rate_fraud": 0.55,
        "online_transaction_rate_legit":  0.95,   # almost all online
        "online_transaction_rate_fraud":  0.98,

        "fault_scenario": "byzantine",
        "fault_round": 20,           # sends corrupted updates from round 20
    },

    "bank_05_international": {
        "name": "GlobalTrade Bank",
        "type": "International / Cross-Border",
        "n_transactions": 500_000,
        "fraud_rate": 0.0028,        # 0.28% — cross-border fraud
        "description": "International bank with heavy cross-border activity. "
                       "Multi-currency transactions. Fraud: foreign card use, "
                       "currency manipulation, travel fraud.",

        "customer": {
            "n_customers": 35_000,
            "age_mean": 40,
            "age_std": 12,
            "age_min": 21,
            "age_max": 75,
            "account_age_mean_days": 1200,
            "account_age_std_days": 700,
        },

        "amount": {
            "small_pct": 0.30,
            "medium_pct": 0.45,
            "large_pct": 0.25,
            "small_mean": 30,   "small_std": 18,
            "medium_mean": 220, "medium_std": 130,
            "large_mean": 2200, "large_std": 1500,
        },

        "merchant_categories": {
            "travel":           0.30,
            "online_retail":    0.20,
            "restaurant":       0.15,
            "electronics":      0.12,
            "entertainment":    0.08,
            "grocery":          0.06,
            "business_services":0.05,
            "gas_station":      0.03,
            "jewelry":          0.01,
            "other":            0.00,
        },

        "hour_weights_legit":  [0.8,0.6,0.4,0.4,0.5,0.8,1.2,1.8,2.2,2.5,
                                 2.5,2.2,2.0,2.0,2.2,2.5,2.8,2.8,2.5,2.2,
                                 2.0,1.8,1.5,1.0],
        "hour_weights_fraud":  [1.8,2.0,2.2,2.0,1.5,1.0,0.8,1.0,1.5,2.0,
                                 2.2,2.0,1.8,1.8,2.0,2.2,2.0,1.8,1.8,2.0,
                                 2.2,2.0,1.8,1.8],

        "fraud_types": {
            "card_not_present":     0.30,
            "identity_theft":       0.25,
            "account_takeover":     0.20,
            "wire_fraud":           0.15,
            "card_present_stolen":  0.07,
            "friendly_fraud":       0.03,
            "insider":              0.00,
            "invoice_fraud":        0.00,
            "synthetic_identity":   0.00,
        },

        "foreign_transaction_rate_legit": 0.45,   # high legitimate foreign use
        "foreign_transaction_rate_fraud": 0.70,
        "online_transaction_rate_legit":  0.50,
        "online_transaction_rate_fraud":  0.78,

        "fault_scenario": "dropout_rejoin",
        "fault_round": 18,           # drops out at 18, rejoins at 28
        "fault_rejoin_round": 28,
    },

    "bank_06_credit_union": {
        "name": "SafeHarbor Credit Union",
        "type": "Savings / Credit Union",
        "n_transactions": 500_000,
        "fraud_rate": 0.0010,        # 0.10% — lowest fraud, elderly customers
        "description": "Conservative credit union with older, loyal members. "
                       "Low transaction frequency. Fraud mainly social "
                       "engineering targeting elderly customers.",

        "customer": {
            "n_customers": 20_000,
            "age_mean": 58,
            "age_std": 14,
            "age_min": 22,
            "age_max": 90,
            "account_age_mean_days": 3500,   # very long-standing accounts
            "account_age_std_days": 1500,
        },

        "amount": {
            "small_pct": 0.60,
            "medium_pct": 0.32,
            "large_pct": 0.08,
            "small_mean": 25,   "small_std": 15,
            "medium_mean": 130, "medium_std": 75,
            "large_mean": 800,  "large_std": 400,
        },

        "merchant_categories": {
            "grocery":          0.32,
            "pharmacy":         0.18,
            "gas_station":      0.15,
            "restaurant":       0.12,
            "online_retail":    0.08,
            "entertainment":    0.06,
            "travel":           0.05,
            "electronics":      0.02,
            "business_services":0.01,
            "other":            0.01,
        },

        "hour_weights_legit":  [0.1,0.1,0.1,0.1,0.2,0.5,1.2,2.0,2.8,3.0,
                                 2.8,2.5,2.2,2.5,2.8,2.5,2.2,2.0,1.8,1.5,
                                 1.0,0.6,0.3,0.1],
        "hour_weights_fraud":  [0.8,1.0,0.8,0.5,0.3,0.3,0.5,1.0,1.5,2.0,
                                 2.5,2.8,2.5,2.2,2.0,1.8,1.5,1.2,1.0,0.8,
                                 0.8,0.8,0.8,0.8],

        "fraud_types": {
            "social_engineering":   0.35,
            "identity_theft":       0.30,
            "card_present_stolen":  0.20,
            "friendly_fraud":       0.10,
            "card_not_present":     0.05,
            "account_takeover":     0.00,
            "insider":              0.00,
            "wire_fraud":           0.00,
            "invoice_fraud":        0.00,
            "synthetic_identity":   0.00,
        },

        "foreign_transaction_rate_legit": 0.03,
        "foreign_transaction_rate_fraud": 0.15,
        "online_transaction_rate_legit":  0.08,
        "online_transaction_rate_fraud":  0.22,

        "fault_scenario": "none",    # stable node — baseline reference
        "fault_round": None,
    },

    "bank_07_investment_premium": {
        "name": "PinnacleWealth Bank",
        "type": "Investment / Premium",
        "n_transactions": 500_000,
        "fraud_rate": 0.0022,        # 0.22% — high-value insider fraud
        "description": "Premium private banking for high-net-worth clients. "
                       "Very high transaction amounts. Fraud: insider trading, "
                       "high-value wire fraud, sophisticated account takeover.",

        "customer": {
            "n_customers": 8_000,    # small exclusive base
            "age_mean": 52,
            "age_std": 11,
            "age_min": 28,
            "age_max": 80,
            "account_age_mean_days": 2200,
            "account_age_std_days": 1000,
        },

        "amount": {
            "small_pct": 0.10,
            "medium_pct": 0.30,
            "large_pct": 0.60,       # majority are large transactions
            "small_mean": 45,   "small_std": 25,
            "medium_mean": 380, "medium_std": 180,
            "large_mean": 12000,"large_std": 8000,
        },

        "merchant_categories": {
            "travel":           0.25,
            "jewelry":          0.15,
            "business_services":0.15,
            "restaurant":       0.12,
            "electronics":      0.10,
            "online_retail":    0.10,
            "entertainment":    0.08,
            "grocery":          0.03,
            "gas_station":      0.01,
            "other":            0.01,
        },

        "hour_weights_legit":  [0.2,0.1,0.1,0.1,0.1,0.3,0.8,1.5,2.5,3.2,
                                 3.5,3.0,2.5,2.8,3.0,2.8,2.5,2.2,1.8,1.2,
                                 0.8,0.5,0.3,0.2],
        "hour_weights_fraud":  [0.8,1.0,1.2,1.0,0.8,0.5,0.5,0.8,1.5,2.2,
                                 2.5,2.5,2.2,2.5,2.8,2.5,2.2,2.0,1.8,1.5,
                                 1.2,1.0,0.8,0.8],

        "fraud_types": {
            "insider":              0.30,
            "wire_fraud":           0.28,
            "account_takeover":     0.22,
            "invoice_fraud":        0.12,
            "identity_theft":       0.05,
            "card_not_present":     0.03,
            "card_present_stolen":  0.00,
            "friendly_fraud":       0.00,
            "synthetic_identity":   0.00,
            "social_engineering":   0.00,
        },

        "foreign_transaction_rate_legit": 0.30,
        "foreign_transaction_rate_fraud": 0.50,
        "online_transaction_rate_legit":  0.40,
        "online_transaction_rate_fraud":  0.68,

        "fault_scenario": "none",    # stable — second baseline reference
        "fault_round": None,
    },
}


# ── Fault scenario metadata (for FL simulation reference) ──────────
FAULT_SCENARIOS = {
    "node_crash": {
        "description": "Node stops sending updates mid-training",
        "behavior": "Client drops out permanently at fault_round",
        "fl_handling": "Server continues with remaining clients",
        "affected_bank": "bank_01_retail_urban",
    },
    "straggler": {
        "description": "Node is slow — sends updates 2-3 rounds late",
        "behavior": "Updates arrive delayed, server must wait or skip",
        "fl_handling": "Asynchronous aggregation or timeout policy",
        "affected_bank": "bank_02_corporate",
    },
    "network_partition": {
        "description": "Node temporarily unreachable for N rounds",
        "behavior": "Bank 3 unreachable rounds 30-38, then recovers",
        "fl_handling": "Skip in aggregation, re-sync on recovery",
        "affected_bank": "bank_03_regional_rural",
    },
    "byzantine": {
        "description": "Node sends corrupted/adversarial model updates",
        "behavior": "Bank 4 sends scaled-up noise as gradients from round 20",
        "fl_handling": "Robust aggregation (Krum / Trimmed Mean)",
        "affected_bank": "bank_04_neobank_digital",
    },
    "dropout_rejoin": {
        "description": "Node drops out then rejoins with stale model",
        "behavior": "Bank 5 absent rounds 18-28, rejoins with old weights",
        "fl_handling": "Model re-sync on rejoin",
        "affected_bank": "bank_05_international",
    },
    "none": {
        "description": "Stable node — no fault injected",
        "behavior": "Normal FL participation all rounds",
        "fl_handling": "N/A",
        "affected_bank": "bank_06_credit_union / bank_07_investment_premium",
    },
}

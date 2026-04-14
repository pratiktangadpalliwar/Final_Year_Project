"""
client/watcher.py
-----------------
Zero-command CSV file watcher.

The bank operator drops a CSV into /data/input/.
This watcher detects it, validates it, and triggers
the full FL training pipeline automatically.

Runs as the Docker container's main process (CMD in Dockerfile).
"""

import os
import sys
import time
import shutil
import logging
import hashlib
from pathlib import Path
from datetime import datetime

from fl_client import FLClient
from utils     import setup_logging

logger    = setup_logging("bank-watcher")
INPUT_DIR = Path(os.environ.get("INPUT_DIR",     "/data/input"))
DONE_DIR  = Path(os.environ.get("DONE_DIR",      "/data/processed"))
ERROR_DIR = Path(os.environ.get("ERROR_DIR",     "/data/error"))
LOCK_FILE = Path(os.environ.get("LOCK_FILE",     "/tmp/fl_training.lock"))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

# ── Transient vs permanent error classification ────────────────────
# Failure strings that indicate transient issues — safe to retry.
# Permanent issues (bad CSV schema, missing columns) go to ERROR_DIR.
TRANSIENT_PATTERNS = (
    "AccessDenied", "Access Denied", "ExpiredToken", "Throttling",
    "ServiceUnavailable", "InternalError", "RequestTimeout",
    "Connection", "timed out", "Max retries", "Temporary failure",
    "503", "504", "502",
)


def is_transient_error(msg: str) -> bool:
    if not msg:
        return False
    return any(p.lower() in msg.lower() for p in TRANSIENT_PATTERNS)


def recover_error_files():
    """On startup, move retryable files from ERROR_DIR back to INPUT_DIR.
    Rationale: pod restart usually means config/infra fix has shipped,
    so we want to retry anything that was stuck."""
    recovered = 0
    for csv_path in ERROR_DIR.glob("*.csv"):
        try:
            dest = INPUT_DIR / csv_path.name
            if dest.exists():
                csv_path.unlink()
                continue
            shutil.move(str(csv_path), str(dest))
            recovered += 1
        except Exception as e:
            # Can't import logger at module-scope — print is fine on startup
            print(f"[watcher] recover_error_files: {csv_path.name} -> {e}")
    if recovered:
        print(f"[watcher] Recovered {recovered} file(s) from error/ to input/")


def is_valid_csv(path: Path) -> tuple:
    """Basic CSV validation before kicking off training."""
    if path.suffix.lower() != ".csv":
        return False, "not_a_csv"
    if path.stat().st_size < 1024:
        return False, "file_too_small"

    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=5)
        required = {"transaction_amount", "is_fraud"}
        missing  = required - set(df.columns)
        if missing:
            return False, f"missing_columns:{missing}"
    except Exception as e:
        return False, f"parse_error:{e}"

    return True, "ok"


def get_file_hash(path: Path) -> str:
    """SHA256 hash to detect duplicate submissions."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def process_csv(csv_path: Path, client: FLClient) -> str:
    """
    Full pipeline triggered by a new CSV drop:
    1. Validate
    2. Register with FL server
    3. Download global model
    4. Train locally
    5. Upload update
    6. Move CSV to processed/

    Returns one of: "done" | "transient" | "permanent".
    Caller uses this to decide whether to cache the file hash (skip retries)
    or leave it for retry.
    """
    logger.info(f"New CSV detected: {csv_path.name} ({csv_path.stat().st_size/1e6:.1f} MB)")

    # ── Validate ──────────────────────────────────────────────────
    valid, reason = is_valid_csv(csv_path)
    if not valid:
        logger.error(f"Invalid CSV ({reason}): {csv_path.name}")
        shutil.move(str(csv_path), str(ERROR_DIR / csv_path.name))
        return "permanent"

    # ── Prevent concurrent training ───────────────────────────────
    if LOCK_FILE.exists():
        logger.warning("Training already in progress — queuing file")
        time.sleep(5)
        if LOCK_FILE.exists():
            logger.error("Lock still held — skipping this CSV")
            return "transient"

    LOCK_FILE.touch()
    logger.info("Training lock acquired")

    outcome: str = "permanent"
    try:
        success, last_error = client.run_training_pipeline(csv_path)

        if success:
            archive_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{csv_path.name}"
            shutil.move(str(csv_path), str(DONE_DIR / archive_name))
            logger.info(f"Training complete. CSV archived to: {DONE_DIR / archive_name}")
            outcome = "done"
        elif is_transient_error(last_error or ""):
            logger.warning(f"Training hit transient error ({last_error!r}) — "
                           f"leaving CSV in input/ for retry on next poll cycle.")
            outcome = "transient"
        else:
            shutil.move(str(csv_path), str(ERROR_DIR / csv_path.name))
            logger.error(f"Training failed (permanent): {last_error!r}. "
                         f"CSV moved to error dir.")
            outcome = "permanent"

    except Exception as e:
        emsg = str(e)
        logger.error(f"Unhandled error during training: {e}", exc_info=True)
        if is_transient_error(emsg):
            logger.warning("Unhandled error looks transient — leaving CSV for retry.")
            outcome = "transient"
        else:
            shutil.move(str(csv_path), str(ERROR_DIR / csv_path.name))
            outcome = "permanent"
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
        logger.info("Training lock released")

    return outcome


def watch():
    """Main watch loop — polls INPUT_DIR for new CSV files."""
    bank_id   = os.environ.get("BANK_ID",   "bank_unknown")
    bank_name = os.environ.get("BANK_NAME", "Unknown Bank")
    server_url= os.environ.get("FL_SERVER_URL", "http://fl-server:8080")
    poll_secs = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))

    logger.info("=" * 55)
    logger.info(f"  FL Bank Node Started")
    logger.info(f"  Bank ID   : {bank_id}")
    logger.info(f"  Bank Name : {bank_name}")
    logger.info(f"  Server    : {server_url}")
    logger.info(f"  Watching  : {INPUT_DIR}")
    logger.info(f"  Poll      : every {poll_secs}s")
    logger.info("=" * 55)

    # Retry any file stuck in error/ from a previous pod lifetime.
    recover_error_files()

    logger.info("Waiting for CSV file drop in /data/input/ ...")

    client       = FLClient(bank_id=bank_id, bank_name=bank_name,
                            server_url=server_url)
    # Cache hashes only for files we've successfully finished or permanently
    # rejected — transient failures stay retryable.
    seen_hashes  = set()

    while True:
        try:
            csv_files = sorted(INPUT_DIR.glob("*.csv"))

            for csv_path in csv_files:
                file_hash = get_file_hash(csv_path)
                if file_hash in seen_hashes:
                    continue

                logger.info(f"Processing: {csv_path.name}")
                outcome = process_csv(csv_path, client)
                if outcome in ("done", "permanent"):
                    seen_hashes.add(file_hash)

        except KeyboardInterrupt:
            logger.info("Watcher stopped by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Watcher error: {e}", exc_info=True)

        time.sleep(poll_secs)


if __name__ == "__main__":
    watch()

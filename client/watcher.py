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


def process_csv(csv_path: Path, client: FLClient):
    """
    Full pipeline triggered by a new CSV drop:
    1. Validate
    2. Register with FL server
    3. Download global model
    4. Train locally
    5. Upload update
    6. Move CSV to processed/
    """
    logger.info(f"New CSV detected: {csv_path.name} ({csv_path.stat().st_size/1e6:.1f} MB)")

    # ── Validate ──────────────────────────────────────────────────
    valid, reason = is_valid_csv(csv_path)
    if not valid:
        logger.error(f"Invalid CSV ({reason}): {csv_path.name}")
        shutil.move(str(csv_path), str(ERROR_DIR / csv_path.name))
        return

    # ── Prevent concurrent training ───────────────────────────────
    if LOCK_FILE.exists():
        logger.warning("Training already in progress — queuing file")
        time.sleep(5)
        if LOCK_FILE.exists():
            logger.error("Lock still held — skipping this CSV")
            return

    LOCK_FILE.touch()
    logger.info("Training lock acquired")

    try:
        success = client.run_training_pipeline(csv_path)

        if success:
            archive_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{csv_path.name}"
            shutil.move(str(csv_path), str(DONE_DIR / archive_name))
            logger.info(f"Training complete. CSV archived to: {DONE_DIR / archive_name}")
        else:
            shutil.move(str(csv_path), str(ERROR_DIR / csv_path.name))
            logger.error(f"Training failed. CSV moved to error dir.")

    except Exception as e:
        logger.error(f"Unhandled error during training: {e}", exc_info=True)
        shutil.move(str(csv_path), str(ERROR_DIR / csv_path.name))
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
        logger.info("Training lock released")


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
    logger.info("Waiting for CSV file drop in /data/input/ ...")

    client       = FLClient(bank_id=bank_id, bank_name=bank_name,
                            server_url=server_url)
    seen_hashes  = set()

    while True:
        try:
            csv_files = sorted(INPUT_DIR.glob("*.csv"))

            for csv_path in csv_files:
                file_hash = get_file_hash(csv_path)
                if file_hash in seen_hashes:
                    continue

                seen_hashes.add(file_hash)
                logger.info(f"Processing: {csv_path.name}")
                process_csv(csv_path, client)

        except KeyboardInterrupt:
            logger.info("Watcher stopped by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Watcher error: {e}", exc_info=True)

        time.sleep(poll_secs)


if __name__ == "__main__":
    watch()

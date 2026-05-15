"""Dataset loading. The init container fetches s3://.../datasets/{bank_id}.csv
to /work/data/bank.csv before the main container starts. We assert it's there
on boot and again when dataset_version bumps."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def assert_dataset_present(path: str | Path, *, min_size_bytes: int = 1024 * 1024) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset missing: {p}")
    if p.stat().st_size < min_size_bytes:
        raise ValueError(f"dataset too small ({p.stat().st_size} bytes < {min_size_bytes}): {p}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

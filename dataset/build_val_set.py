"""Build the held-out global validation set.

Reads all bank CSVs, runs the same preprocessor as the client,
stratified-samples a fraction of each, concatenates, pickles {X, y} to disk.

Usage:
    python dataset/build_val_set.py \\
        --inputs dataset/bank_*.csv \\
        --frac 0.05 \\
        --out /tmp/val_set.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Allow `python dataset/build_val_set.py` from repo root without installing the
# client package — append repo root to sys.path so `client.app.X` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Same feature pipeline as the client preprocessor.
from client.app.preprocessor import preprocess  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="bank CSV paths (glob expands ok)")
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--out", type=str, default="/tmp/val_set.pkl")
    args = p.parse_args()

    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []

    for path in args.inputs:
        x_tr, y_tr, x_v, y_v, _ = preprocess(path, val_frac=0.15)
        x_full = torch.cat([x_tr, x_v], dim=0).numpy()
        y_full = torch.cat([y_tr, y_v], dim=0).numpy()
        if y_full.sum() > 1 and y_full.sum() < len(y_full):
            x_s, _, y_s, _ = train_test_split(
                x_full, y_full, train_size=args.frac, stratify=y_full, random_state=42,
            )
        else:
            n = max(1, int(len(y_full) * args.frac))
            x_s, y_s = x_full[:n], y_full[:n]
        xs.append(torch.from_numpy(x_s))
        ys.append(y_s)

    X = torch.cat(xs, dim=0)
    y = np.concatenate(ys, axis=0)
    print(f"validation set: {len(X)} rows, {int(y.sum())} positives ({y.mean()*100:.2f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"X": X, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

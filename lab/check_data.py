#!/usr/bin/env python3
"""训练前检查路径、字段、时间轴、缺失值、切分后的可用样本数。"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_series, split_summary, split_ranges, data_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", help="Omit to list available columns first.")
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--sha256", help="Optional instructor-provided whole-file SHA-256.")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"File not found: {args.csv}. Run commands from the repository root.")
    if args.sha256:
        if len(args.sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.sha256):
            parser.error("--sha256 must be a 64-character hexadecimal checksum.")
        hasher = hashlib.sha256()
        with args.csv.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != args.sha256.lower():
            parser.error("File checksum differs from the instructor-provided value.")
        print("Whole-file SHA-256: matched.")
    columns = list(pd.read_csv(args.csv, nrows=0, encoding="utf-8-sig").columns)
    print(f"CSV: {args.csv.resolve()}\nColumns: {columns}")
    if args.target is None:
        print("Now rerun with --target followed by the actual node column name.")
        return
    series = load_series(args.csv, args.target, args.time_column)
    raw = series.to_numpy(dtype=np.float64)
    print(f"Target: {args.target}; interval: 5 minutes")
    print(f"Range: {series.index[0]} -> {series.index[-1]}")
    print(f"Aligned time points: {len(raw)}; missing: {np.isnan(raw).sum()}")
    print(pd.DataFrame(split_summary(series)).T.to_string())
    if np.isnan(raw).mean() > 0.2:
        print("WARNING: over 20% missing. Confirm the sampling interval and missing-data policy.")
    train_stop = split_ranges(len(raw))["train"][1]
    if np.nanstd(raw[:train_stop]) < 1e-12:
        print("WARNING: constant training series; ARIMA will ask you to use persistence or another node.")
    print(f"Selected-series fingerprint: {data_fingerprint(series)}")
    print("PASS: CSV satisfies the baseline input protocol. This does not verify its provenance or units.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

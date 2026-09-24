#!/usr/bin/env python3
"""生成明确标记的人工调试数据，不是课程无线实测数据，不用于报告真实模型效果。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import FREQUENCY, write_json


def make_frame(points: int = 2016, seed: int = 42) -> pd.DataFrame:
    if points < 150:
        raise ValueError("Use at least 150 points for the classroom demo.")
    rng = np.random.default_rng(seed)
    time = np.arange(points)
    noise = np.zeros(points)
    for i in range(1, points):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 12)
    base = 350 + 170 * np.sin(2 * np.pi * time / 288 - 1) + 45 * np.sin(2 * np.pi * time / 72)
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=points, freq=FREQUENCY),
                         "cell_A": np.maximum(base + noise, 0).round(3),
                         "cell_B": np.maximum(0.7 * base + rng.normal(0, 18, points), 0).round(3)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/demo_traffic.csv"))
    parser.add_argument("--points", type=int, default=2016)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".metadata.json").exists():
        parser.error("Output already exists; use a new --out path.")
    frame = make_frame(args.points, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    write_json(args.out.with_suffix(".metadata.json"), {
        "source": "SYNTHETIC; generated for code smoke tests only",
        "real_wireless_measurements": False, "units": "arbitrary demo units",
        "seed": args.seed, "points": args.points, "frequency": FREQUENCY})
    print(f"SYNTHETIC demo saved to {args.out}. This is NOT the course dataset.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

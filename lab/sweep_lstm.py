#!/usr/bin/env python3
"""课堂用的小规模 LSTM 调参：一次只改一个旋钮，按验证集 MSE 排名。

四组设置共用同一条序列、同一种 60/20/20 切分、同一套预处理和同一个随机种子。
隐藏维和学习率不要同时改，否则看不出是哪一个在起作用。

排名只用 train_lstm.py 写下来的验证集误差（标准化对数空间的 MSE）。
测试集 MAE 会抄进汇总表，供选定模型之后对照，不参与选模型。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import ensure_new_output


# 每一组和上一组相比只动一个数。why 会写进汇总表，课堂上直接念。
PRESETS = [
    {"name": "h64_lr1e-4", "hidden_size": 64, "lr": 1e-4,
     "why": "课堂默认：隐藏维 64，学习率 1e-4"},
    {"name": "h32_lr1e-4", "hidden_size": 32, "lr": 1e-4,
     "why": "只把隐藏维减半"},
    {"name": "h128_lr1e-4", "hidden_size": 128, "lr": 1e-4,
     "why": "只把隐藏维加倍"},
    {"name": "h64_lr1e-3", "hidden_size": 64, "lr": 1e-3,
     "why": "只把学习率放大 10 倍"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/cell30_lstm_sweep"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive.")
    if args.epochs < 30:
        print("WARNING: fewer than 30 epochs only shows which run descends faster. "
              "Do not treat that ranking as a fully trained comparison.")
    try:
        ensure_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)

    trainer = Path(__file__).with_name("train_lstm.py")
    rows = []
    for preset in PRESETS:
        out_dir = args.out / preset["name"]
        command = [
            sys.executable, str(trainer),
            "--csv", str(args.csv),
            "--target", args.target,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--hidden-size", str(preset["hidden_size"]),
            "--lr", str(preset["lr"]),
            "--batch-size", "64",
            "--weight-decay", "1e-4",
            "--seed", str(args.seed),
            "--device", args.device,
            "--out", str(out_dir),
        ]
        print("\n" + " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        row = {**preset, "epochs_budget": args.epochs, "returncode": completed.returncode,
               "out": str(out_dir)}
        metrics_path = out_dir / "metrics.json"
        if completed.returncode == 0 and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update({
                "status": "ok",
                "selected_epoch": metrics.get("selected_epoch"),
                "validation_mse": metrics.get("selected_validation_mse"),
                "test_mae": metrics["lstm"]["mae"],
                "test_rmse": metrics["lstm"]["rmse"],
                "persistence_test_mae": metrics["persistence"]["mae"],
            })
        else:
            row.update({"status": "failed", "selected_epoch": None, "validation_mse": None,
                        "test_mae": None, "test_rmse": None, "persistence_test_mae": None})
            print(f"  preset failed with return code {completed.returncode}", flush=True)
        rows.append(row)

    table = pd.DataFrame(rows)
    ok = table[table["status"] == "ok"].copy()
    if ok.empty:
        table.to_csv(args.out / "sweep_summary.csv", index=False)
        raise SystemExit("ERROR: every LSTM preset failed. See the log above.")
    best = float(ok["validation_mse"].min())
    table["selected_by_validation"] = (
        (table["status"] == "ok") & (table["validation_mse"] == best))
    # 验证误差相同就保留预设列表里的第一个，不用测试集打破平局。
    winners = table.index[table["selected_by_validation"]].tolist()
    if len(winners) > 1:
        table.loc[winners[1:], "selected_by_validation"] = False
    table["_ok"] = table["status"].eq("ok")
    table = table.sort_values(by=["_ok", "validation_mse"], ascending=[False, True],
                              na_position="last").drop(columns="_ok")
    table.to_csv(args.out / "sweep_summary.csv", index=False)
    shown = table[["name", "hidden_size", "lr", "selected_epoch", "validation_mse",
                   "test_mae", "selected_by_validation", "why"]]
    print("\n" + shown.to_string(index=False))
    chosen = table.loc[table["selected_by_validation"], "name"].iloc[0]
    test_leader = ok.loc[ok["test_mae"].idxmin(), "name"]
    print(f"\nSelected by validation MSE: {chosen}")
    if test_leader != chosen:
        print(f"Test MAE would have picked {test_leader} instead. "
              "That comparison is reported, not used for selection.")
    print("Test MAE is recorded so the chosen model can be discussed afterwards.")
    print(f"Summary: {args.out.resolve() / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()

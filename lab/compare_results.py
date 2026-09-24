#!/usr/bin/env python3
"""严格对齐测试集，再输出对比表和图片；不会静默取两个结果的交集。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 服务器没有桌面也可以保存图片。
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import calculate_metrics, ensure_new_output


def load_predictions(directory: Path, model: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(directory / "test_predictions.csv")
    required = {"timestamp", "actual", model, "persistence"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Invalid predictions file: {directory}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("Prediction timestamps must be present and unique.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Prediction timestamps are not sorted.")
    if not np.isfinite(frame[["actual", model, "persistence"]].to_numpy(dtype=float)).all():
        raise ValueError("Prediction file contains NaN or infinite values.")
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    return frame, config


def align_predictions(lstm: pd.DataFrame, arima: pd.DataFrame,
                      lstm_config: dict, arima_config: dict) -> pd.DataFrame:
    for key in ("target", "time_column", "data_sha256"):
        if key not in lstm_config or key not in arima_config or lstm_config[key] != arima_config[key]:
            raise ValueError(f"Runs do not use the same {key}; rerun both models on the same data.")
    if not lstm["timestamp"].equals(arima["timestamp"]):
        raise ValueError("Test timestamps differ. Do not compare metrics on different test samples.")
    for key in ("actual", "persistence"):
        if not np.allclose(lstm[key], arima[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Runs disagree on {key} values.")
    result = lstm.copy()
    result["arima"] = arima["arima"].to_numpy()
    return result[["timestamp", "actual", "persistence", "arima", "lstm"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm-dir", type=Path, required=True)
    parser.add_argument("--arima-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--plot-points", type=int, default=288,
                        help="Plot the earliest N scored test points; metrics use ALL scored points.")
    args = parser.parse_args()
    if args.plot_points <= 0:
        parser.error("--plot-points must be positive.")
    ensure_new_output(args.out)
    lstm, lc = load_predictions(args.lstm_dir, "lstm")
    arima, ac = load_predictions(args.arima_dir, "arima")
    combined = align_predictions(lstm, arima, lc, ac)
    rows = [{"model": model, **calculate_metrics(combined["actual"].to_numpy(),
                                                combined[model].to_numpy())}
            for model in ("persistence", "arima", "lstm")]
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out / "predictions_aligned.csv", index=False)
    table.to_csv(args.out / "metrics_comparison.csv", index=False)
    subset = combined.iloc[:args.plot_points].set_index("timestamp")
    # 重建时间网格，让缺失导致的计分空档在图中断开，避免画出误导的连线。
    dense_index = pd.date_range(subset.index[0], subset.index[-1], freq="5min")
    subset = subset.reindex(dense_index)
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for column in ("actual", "persistence", "arima", "lstm"):
        ax.plot(subset.index, subset[column], label=column, linewidth=1.2)
    ax.set_title(f"Rolling one-step forecasts | {lc['target']} | first {min(args.plot_points, len(combined))} scored points")
    ax.set_xlabel("Target timestamp")
    ax.set_ylabel("Traffic (original CSV units)")
    ax.legend(ncol=4)
    ax.grid(True, alpha=0.25)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig(args.out / "prediction_comparison.png", dpi=200)
    plt.close(fig)
    history_path = args.lstm_dir / "history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(history["epoch"], history["train_mse"], label="Training (online epoch average)")
        ax.plot(history["epoch"], history["validation_mse"], label="Validation (fixed checkpoint)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (standardized log space)")
        ax.set_title("LSTM learning curves")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.savefig(args.out / "lstm_learning_curve.png", dpi=200)
        plt.close(fig)
    print(table.to_string(index=False))
    print("Metrics use every scored test point; the forecast plot shows only the chosen prefix.")
    print("ARIMA retains a longer observed-history state; this is NOT a matched-12-input ablation.")
    print(f"Outputs saved locally to: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

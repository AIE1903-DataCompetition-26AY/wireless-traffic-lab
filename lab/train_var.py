#!/usr/bin/env python3
"""30 个小区的联合 VAR：训练段估计交叉滞后，验证集选滞后阶，测试集滚动一步。

每个通道先做 log1p，再各自标准化，统计量只用训练段。参数拟合之后不再更新。
往前滚动时，各通道已经观测到的值会一起进入状态，所以通道之间的滞后系数会参与预测。
计分时间戳与单通道 LSTM 相同：每个 split 内要有完整的 12 个点。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from common import (FREQUENCY, LOOKBACK, ChannelScaler, calculate_metrics, ensure_new_output,
                    environment_versions, load_wide, split_ranges, valid_row_targets,
                    wide_fingerprint, write_json)


def fit_var(train: np.ndarray, lag: int):
    if lag < 1:
        raise ValueError("VAR lag must be at least 1.")
    if train.shape[0] <= lag + train.shape[1]:
        raise ValueError("Too few training rows for this VAR lag.")
    model = VAR(np.asarray(train, dtype=np.float64))
    fitted = model.fit(maxlags=lag, ic=None, trend="c")
    coefs = np.asarray(fitted.coefs, dtype=np.float64)
    intercept = np.asarray(fitted.intercept, dtype=np.float64)
    if coefs.shape != (lag, train.shape[1], train.shape[1]) or intercept.shape != (train.shape[1],):
        raise ValueError("Unexpected VAR parameter shape.")
    check = one_step(coefs, intercept, train[-lag:])
    reference = np.asarray(fitted.forecast(train[-lag:], steps=1), dtype=np.float64).reshape(-1)
    if not np.allclose(check, reference, rtol=1e-6, atol=1e-6):
        raise ValueError("VAR one-step formula does not match statsmodels.")
    return fitted, coefs, intercept


def one_step(coefs: np.ndarray, intercept: np.ndarray, history: np.ndarray) -> np.ndarray:
    """history 的最后一行是最近一个已观测点。coefs[k] 乘的是滞后 k+1。"""
    lag = coefs.shape[0]
    if history.shape[0] < lag:
        raise ValueError("VAR history is shorter than the lag.")
    forecast = np.array(intercept, dtype=np.float64, copy=True)
    for k in range(lag):
        forecast += coefs[k] @ history[-1 - k]
    return forecast


def roll(coefs: np.ndarray, intercept: np.ndarray, history: np.ndarray,
         future: np.ndarray) -> np.ndarray:
    """预测 future 的每一行时，只使用该行之前的观测。预测不用来更新状态。"""
    state = np.asarray(history, dtype=np.float64)
    incoming = np.asarray(future, dtype=np.float64)
    forecasts = np.empty_like(incoming)
    for t in range(len(incoming)):
        forecasts[t] = one_step(coefs, intercept, state)
        state = np.vstack((state, incoming[t]))
    return forecasts


def long_frame(index: pd.DatetimeIndex, columns: list[str], actual: np.ndarray,
               forecast: np.ndarray, persistence: np.ndarray, name: str) -> pd.DataFrame:
    pieces = []
    for i, cell in enumerate(columns):
        pieces.append(pd.DataFrame({
            "timestamp": index, "cell": cell, "actual": actual[:, i],
            name: forecast[:, i], "persistence": persistence[:, i]}))
    return pd.concat(pieces, ignore_index=True)


def macro_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    per_channel = [calculate_metrics(actual[:, i], forecast[:, i])["mae"] for i in range(actual.shape[1])]
    flat = calculate_metrics(actual.reshape(-1), forecast.reshape(-1))
    return {"macro_mae": float(np.mean(per_channel)), "pooled_mae": flat["mae"],
            "pooled_rmse": flat["rmse"], "count_per_channel": flat["count"] // actual.shape[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--out", type=Path, default=Path("runs/cell30_var"))
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    lags = sorted(set(args.lags))
    if any(lag < 1 for lag in lags):
        parser.error("VAR lags must be positive.")
    try:
        ensure_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))

    wide = load_wide(args.csv, args.time_column)
    raw = wide.to_numpy(dtype=np.float64)
    columns = list(wide.columns)
    splits = split_ranges(len(wide))
    targets = {name: valid_row_targets(raw, start, stop) for name, (start, stop) in splits.items()}
    train_stop = splits["train"][1]
    val_start, val_stop = splits["validation"]
    test_start, test_stop = splits["test"]
    scaler = ChannelScaler.fit(raw[:train_stop])
    normalized = scaler.transform(raw).astype(np.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_config.json", {
        "csv": str(args.csv), "time_column": args.time_column, "out": str(args.out),
        "lags": lags, "model": "var", "channels": columns,
        "data_sha256": wide_fingerprint(wide), "environment": environment_versions()})

    records, best = [], None
    for lag in lags:
        parameters = columns.__len__() * (1 + columns.__len__() * lag)
        print(f"Fitting VAR({lag}) on {len(columns)} channels, {parameters} coefficients ...", flush=True)
        record = {"lag": lag, "coefficients": parameters}
        try:
            _, coefs, intercept = fit_var(normalized[:train_stop], lag)
            val_z = roll(coefs, intercept, normalized[:val_start], normalized[val_start:val_stop])
            local = targets["validation"] - val_start
            val_mse = float(np.mean((val_z[local] - normalized[targets["validation"]]) ** 2))
            val_raw = scaler.inverse(val_z[local])
            raw_scores = macro_metrics(raw[targets["validation"]], val_raw)
            record.update({"status": "ok", "validation_mse": val_mse,
                           "validation_macro_mae": raw_scores["macro_mae"], "error": ""})
            print(f"  validation_mse={val_mse:.6f} validation_macro_mae={raw_scores['macro_mae']:.1f}",
                  flush=True)
            if best is None or val_mse < best["validation_mse"]:
                best = {"lag": lag, "coefs": coefs, "intercept": intercept,
                        "validation_mse": val_mse, "coefficients": parameters}
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            record.update({"status": "failed", "validation_mse": None,
                           "validation_macro_mae": None, "error": str(exc)})
            print(f"  candidate failed: {exc}", flush=True)
        records.append(record)
        pd.DataFrame(records).to_csv(args.out / "validation_search.csv", index=False)
    if best is None:
        raise SystemExit("ERROR: every VAR lag failed. See validation_search.csv.")

    observed = normalized[:test_start]
    test_z = roll(best["coefs"], best["intercept"], observed, normalized[test_start:test_stop])
    test_index = targets["test"]
    local = test_index - test_start
    forecast = scaler.inverse(test_z[local])
    actual = raw[test_index]
    persistence = raw[test_index - 1]
    long_frame(wide.index[test_index], columns, actual, forecast, persistence, "var").to_csv(
        args.out / "test_predictions.csv", index=False)
    scores = macro_metrics(actual, forecast)
    baseline = macro_metrics(actual, persistence)
    per_cell = []
    for i, cell in enumerate(columns):
        per_cell.append({"cell": cell,
                         **{f"var_{key}": value for key, value in calculate_metrics(actual[:, i], forecast[:, i]).items()},
                         **{f"persistence_{key}": value for key, value in calculate_metrics(actual[:, i], persistence[:, i]).items()}})
    pd.DataFrame(per_cell).to_csv(args.out / "per_cell_metrics.csv", index=False)
    write_json(args.out / "metrics.json", {
        "selected_lag": best["lag"], "coefficients": best["coefficients"],
        "selected_validation_mse": best["validation_mse"],
        "selection_rule": "validation standardized-log MSE, macro over channels",
        "var": scores, "persistence": baseline})
    write_json(args.out / "var_parameters.json", {
        "lag": best["lag"], "channels": columns,
        "intercept": best["intercept"].tolist(), "coefs": best["coefs"].tolist(),
        "scaler_mean": list(scaler.mean), "scaler_scale": list(scaler.scale),
        "note": "Parameters only. Forecasting also needs the last `lag` observed rows."})
    print(f"Selected VAR({best['lag']}) with {best['coefficients']} coefficients.")
    print(f"VAR         macro MAE={scores['macro_mae']:.1f} pooled MAE={scores['pooled_mae']:.1f}")
    print(f"Persistence macro MAE={baseline['macro_mae']:.1f}")
    print(f"Outputs saved locally to: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

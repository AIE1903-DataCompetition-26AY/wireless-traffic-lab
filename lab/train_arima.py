#!/usr/bin/env python3
"""教学用单节点 ARIMA：训练集拟合参数，验证集选阶，测试集滚动一步预测。

输入接口与 train_lstm.py 相同。ARIMA 使用更长的历史滤波状态，不限定为
12 点输入；两者的验证/测试计分时间戳相同。没有未来插值、全数据拟合或
测试集选阶；extend 只更新状态，不重新估计参数。默认 CPU，无需 PyTorch。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from common import (LOOKBACK, FREQUENCY, Scaler, load_series, split_ranges,
                    valid_target_indices, calculate_metrics, write_json,
                    ensure_new_output, data_fingerprint, environment_versions,
                    split_summary)


def fit_on_train(train: np.ndarray, order: tuple[int, int, int], maxiter: int = 200):
    """只允许把训练段传进来；返回拟合结果和没有被隐藏的 warning 文本。"""
    observed = np.asarray(train)[np.isfinite(train)]
    if observed.size < max(30, 5 * (order[0] + order[2] + order[1] + 1)):
        raise ValueError("Too few observed training points for this ARIMA order; "
                         "use more data or a lower order.")
    if float(np.std(observed)) < 1e-12:
        raise ValueError("Training series is constant. Use persistence or choose another node.")
    # d=0 时估计常数；d>0 时不额外加漂移。第一次实验不增加更多趋势选项。
    trend = "c" if order[1] == 0 else "n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ARIMA(np.asarray(train, dtype=np.float64), order=order, trend=trend,
                      enforce_stationarity=True, enforce_invertibility=True,
                      missing="none")
        fitted = model.fit(method="statespace", method_kwargs={"maxiter": maxiter})
    messages = list(dict.fromkeys(str(item.message) for item in caught))
    if not np.isfinite(fitted.params).all():
        raise ValueError("ARIMA produced non-finite parameters.")
    if not fitted.mle_retvals.get("converged", False):
        raise ValueError("ARIMA optimization did not converge. Try a lower order or "
                         "increase --maxiter. " + " | ".join(messages))
    return fitted, messages


def rolling_one_step(fitted: Any, observations: np.ndarray,
                     engine: str = "step") -> tuple[np.ndarray, Any]:
    """在时间 t 的预测仅依赖截至 t-1 的观测；参数全程保持固定。

    step: 显式的“预测 -> 看到真实值 -> 更新状态”，便于教学。
    fast: 同样参数下批量执行单向 Kalman filter，并读取一步先验预测。
          不是一次 forecast(len(test))，也不读取使用未来值的 smoothed 输出。
          tests/test_arima.py 验证它与 step 等价且不受当前/未来值影响。
    """
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("Observations must be 1D with finite numbers or NaN.")
    if not len(values):
        return np.empty(0), fitted
    if engine == "fast":
        updated = fitted.extend(values)
        predictions = np.asarray(updated.filter_results.forecasts[0], dtype=np.float64)
    elif engine == "step":
        updated = fitted
        predictions = np.empty(len(values), dtype=np.float64)
        for t, actual in enumerate(values):
            # 第一步：还没看见 actual，先预测它。
            predictions[t] = float(np.asarray(updated.forecast(steps=1)).reshape(-1)[0])
            # 第二步：时间推进，actual 成为已观测历史。NaN 不被填成 0。
            # extend 不重新拟合参数；它只进行状态预测/观测更新。
            updated = updated.extend(np.array([actual], dtype=np.float64))
    else:
        raise ValueError("Unknown engine; choose step or fast.")
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite ARIMA predictions; inspect data or lower the order.")
    return predictions, updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True, help="One node/grid traffic column.")
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--out", type=Path, default=Path("runs/traffic_arima"))
    parser.add_argument("--order", type=int, nargs=3, metavar=("P", "D", "Q"), default=(1, 1, 1),
                        help="Fixed ARIMA order, default: 1 1 1. Ignored when --search is used.")
    parser.add_argument("--search", action="store_true",
                        help="Select an order on validation only; search ranges below.")
    parser.add_argument("--p-values", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d-values", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--q-values", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--engine", choices=["step", "fast"], default="step")
    parser.add_argument("--mape-threshold", type=float, default=None)
    args = parser.parse_args()
    orders = (list(product(sorted(set(args.p_values)), sorted(set(args.d_values)),
                           sorted(set(args.q_values)))) if args.search else [tuple(args.order)])
    if any(min(order) < 0 for order in orders) or args.maxiter <= 0:
        parser.error("ARIMA orders must be nonnegative; --maxiter must be positive.")
    if any(order[1] > 2 for order in orders):
        parser.error("This classroom script supports d=0, 1, or 2 only.")
    if args.mape_threshold is not None and (not np.isfinite(args.mape_threshold) or args.mape_threshold < 0):
        parser.error("MAPE threshold must be finite and nonnegative.")
    ensure_new_output(args.out)

    # 1. 读数据 -> 先划分时间 -> 仅用训练集拟合 log1p + 标准化。
    series = load_series(args.csv, args.target, args.time_column)
    raw = series.to_numpy(dtype=np.float64)
    splits = split_ranges(len(raw))
    targets = {name: valid_target_indices(raw, start, stop) for name, (start, stop) in splits.items()}
    train_stop = splits["train"][1]
    val_start, val_stop = splits["validation"]
    test_start, test_stop = splits["test"]
    scaler = Scaler.fit(raw[:train_stop])
    normalized = scaler.transform(raw).astype(np.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": "arima", "data_sha256": data_fingerprint(series),
        "environment": environment_versions()})
    write_json(args.out / "split_summary.json", {
        "target": args.target, "total_time_points": len(raw),
        "missing_values": int(np.isnan(raw).sum()), "lookback_for_scoring_mask": LOOKBACK,
        "horizon": 1, "interval": FREQUENCY, "scaler": asdict(scaler),
        "scoring_policy": "same valid target timestamps as LSTM",
        "history_policy": "continuous observed-history state, not restricted to 12 points",
        "splits": split_summary(series)})

    # 2. 每个候选只拟合训练集；验证集预测误差决定选哪个阶数。
    # 固定 --order 时同样记录验证结果，但不自动更换阶数。
    records, best = [], None
    for order in orders:
        print(f"Fitting ARIMA{order} on training split ...", flush=True)
        record: dict[str, Any] = {"p": order[0], "d": order[1], "q": order[2]}
        try:
            fitted, fit_warnings = fit_on_train(normalized[:train_stop], order, args.maxiter)
            val_z, val_state = rolling_one_step(fitted, normalized[val_start:val_stop], args.engine)
            local_indices = targets["validation"] - val_start
            val_mse = float(np.mean((val_z[local_indices] - normalized[targets["validation"]]) ** 2))
            if not np.isfinite(val_mse):
                raise ValueError("Non-finite validation MSE.")
            val_raw = scaler.inverse(val_z[local_indices])
            raw_metrics = calculate_metrics(raw[targets["validation"]], val_raw, args.mape_threshold)
            record.update({"status": "ok", "validation_mse": val_mse,
                           "validation_mae": raw_metrics["mae"], "validation_rmse": raw_metrics["rmse"],
                           "fit_warnings": " | ".join(fit_warnings), "error": ""})
            for message in fit_warnings:
                print(f"  fit warning: {message}", flush=True)
            print(f"  validation_mse={val_mse:.6f} validation_mae={raw_metrics['mae']:.6f}", flush=True)
            # 相同时保留预定遍历顺序中的第一个，不用测试集打破平局。
            if best is None or val_mse < best["validation_mse"]:
                best = {"order": order, "fitted": fitted, "val_state": val_state,
                        "validation_mse": val_mse, "warnings": fit_warnings}
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            record.update({"status": "failed", "error": str(exc)})
            print(f"  candidate failed: {exc}", flush=True)
        records.append(record)
        pd.DataFrame(records).to_csv(args.out / "validation_search.csv", index=False)
    if best is None:
        raise ValueError("No ARIMA candidate succeeded. See validation_search.csv. "
                         "Try --order 1 0 0, a lower order, more data, or higher --maxiter.")

    # 3. 阶数、参数、scaler 到此都已固定。测试真实值只在其到达后更新状态。
    test_z, final_state = rolling_one_step(best["val_state"], normalized[test_start:test_stop], args.engine)
    test_indices = targets["test"]
    truth = raw[test_indices]
    forecast = scaler.inverse(test_z[test_indices - test_start])
    persistence = raw[test_indices - 1]
    pd.DataFrame({"timestamp": series.index[test_indices], "actual": truth,
                  "arima": forecast, "persistence": persistence}).to_csv(
                      args.out / "test_predictions.csv", index=False)
    scores = calculate_metrics(truth, forecast, args.mape_threshold)
    baseline = calculate_metrics(truth, persistence, args.mape_threshold)
    write_json(args.out / "metrics.json", {
        "target": args.target, "selected_order": best["order"],
        "selected_validation_mse": best["validation_mse"],
        "selection_rule": "validation standardized-log MSE" if args.search else "user-fixed order",
        "units": "original CSV traffic units", "parameters_refit_after_training": False,
        "test_protocol": "rolling one-step, continuous observed-history state; same scoring mask as LSTM",
        "arima": scores, "persistence": baseline})
    fitted = best["fitted"]
    # JSON 仅保存参数和配置，不使用 pickle，也不声称是可独立恢复的完整状态文件。
    write_json(args.out / "arima_parameters.json", {
        "order": best["order"], "trend": "c" if best["order"][1] == 0 else "n",
        "parameter_names": fitted.param_names, "parameters": np.asarray(fitted.params).tolist(),
        "scaler": asdict(scaler), "target": args.target, "interval": FREQUENCY,
        "training_end_timestamp": str(series.index[train_stop - 1]), "warnings": best["warnings"],
        "note": "Parameters only; a complete input history is also needed to reconstruct filter state."})
    (args.out / "model_summary.txt").write_text(fitted.summary().as_text(), encoding="utf-8")
    # 与原 LSTM 示例一致：末尾 12 点完整时才输出下一个时间点。
    if np.isfinite(raw[-LOOKBACK:]).all():
        next_z = np.asarray(final_state.forecast(steps=1), dtype=np.float64).reshape(-1)
        write_json(args.out / "next_forecast.json", {
            "forecast_timestamp": str(series.index[-1] + pd.Timedelta(FREQUENCY)),
            "prediction": float(scaler.inverse(next_z)[0]), "target": args.target,
            "history": "all observed data through the last CSV timestamp; training-fitted parameters"})
    print(f"Selected ARIMA{best['order']}; training-fitted parameters remain fixed.")
    print(f"ARIMA       MAE={scores['mae']:.6f} RMSE={scores['rmse']:.6f} n={scores['count']}")
    print(f"Persistence MAE={baseline['mae']:.6f} RMSE={baseline['rmse']:.6f}")
    print(f"Outputs saved locally to: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

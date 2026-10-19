#!/usr/bin/env python3
"""把调参之后的预测画出来，按片段说明哪里不行。

输入是 train_arima.py / train_lstm.py 已经写好的目录，本脚本不重新搜索阶数，
也不按测试集更换已经选定的模型。另外用同一条序列补两个没有参数可训的参照：
昨天同一时刻，以及只用训练段估计的“每天这个钟点”的均值。

滚动一步和图里的开环多步是两件不同的事，输出里会分开写。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import (LOOKBACK, FREQUENCY, Scaler, calculate_metrics, ensure_new_output,
                    load_series, split_ranges)
from train_arima import fit_on_train
from train_lstm import TrafficLSTM, forecast_next


def configure_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    font_file = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_file.is_file():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_file))
        plt.rcParams["font.family"] = "Microsoft YaHei"
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction_run(directory: Path, column: str) -> tuple[pd.DataFrame, dict, dict]:
    frame = pd.read_csv(directory / "test_predictions.csv")
    required = {"timestamp", "actual", column, "persistence"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"{directory} is missing a '{column}' prediction file.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Timestamps in {directory} must be unique and sorted.")
    values = frame[["actual", column, "persistence"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Predictions in {directory} contain NaN or infinity.")
    return frame, read_json(directory / "run_config.json"), read_json(directory / "metrics.json")


def discover_lstm_dirs(explicit: list[Path], root: Path | None) -> list[Path]:
    found = list(explicit)
    if root is not None:
        if not root.is_dir():
            raise ValueError(f"LSTM root does not exist: {root}")
        found.extend(path for path in sorted(root.iterdir())
                     if path.is_dir() and (path / "test_predictions.csv").is_file())
    unique, seen = [], set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("Pass --lstm-dir or --lstm-root with at least one finished LSTM run.")
    return unique


def align_runs(arima: pd.DataFrame, arima_config: dict,
               lstm_runs: list[tuple[str, pd.DataFrame, dict]]) -> pd.DataFrame:
    combined = arima[["timestamp", "actual", "persistence", "arima"]].copy()
    for name, frame, config in lstm_runs:
        for key in ("target", "time_column", "data_sha256"):
            if arima_config.get(key) != config.get(key):
                raise ValueError(f"{name} does not match the ARIMA run on {key}.")
        if not frame["timestamp"].equals(combined["timestamp"]):
            raise ValueError(f"{name} does not use the same test timestamps as ARIMA.")
        for key in ("actual", "persistence"):
            if not np.allclose(frame[key], combined[key], rtol=1e-10, atol=1e-10):
                raise ValueError(f"{name} disagrees with ARIMA on {key}.")
        if name in combined.columns:
            raise ValueError(f"Duplicate forecast name: {name}")
        combined[name] = frame["lstm"].to_numpy()
    return combined


def slot_of(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 12 + index.minute // 5).to_numpy()


def add_reference_forecasts(frame: pd.DataFrame, series: pd.Series, train_stop: int) -> pd.DataFrame:
    """昨天同一时刻，以及训练段上每个 5 分钟钟点的均值。两者都不看验证/测试。"""
    positions = pd.Series(np.arange(len(series)), index=series.index)
    located = positions.loc[pd.DatetimeIndex(frame["timestamp"])].to_numpy()
    raw = series.to_numpy(dtype=np.float64)
    if np.any(located < 288):
        raise ValueError("Seasonal lag 288 falls before the start of this series.")
    result = frame.copy()
    result["seasonal_naive"] = raw[located - 288]
    slots = slot_of(series.index[:train_stop])
    lookup = np.full(288, np.nan)
    for slot in range(288):
        observed = raw[:train_stop][slots == slot]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            lookup[slot] = float(observed.mean())
    test_slots = slot_of(pd.DatetimeIndex(frame["timestamp"]))
    if np.isnan(lookup[test_slots]).any():
        raise ValueError("Training split does not cover every 5-minute clock slot.")
    result["slot_mean"] = lookup[test_slots]
    return result


def metric_row(name: str, role: str, actual: np.ndarray, forecast: np.ndarray,
               persistence_mae: float, selected: bool) -> dict:
    scores = calculate_metrics(actual, forecast)
    return {"model": name, "role": role, **scores,
            "bias": float(np.mean(forecast - actual)),
            "skill_vs_persistence": float(1.0 - scores["mae"] / persistence_mae),
            "selected_by_validation": selected}


def quartile_labels(jump: pd.Series) -> pd.Series:
    labels = ["Q1 最平缓", "Q2", "Q3", "Q4 跳变最大"]
    try:
        return pd.qcut(jump, 4, labels=labels)
    except ValueError:
        return pd.qcut(jump, 4, duplicates="drop").astype(str)


def save_line_plot(path: Path, frame: pd.DataFrame, columns: list[tuple[str, str, str]],
                   title: str, vline: pd.Timestamp | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
    for column, label, style in columns:
        ax.plot(frame["timestamp"], frame[column] / 1e6, label=label, linewidth=1.35, **_style(style))
    if vline is not None:
        ax.axvline(vline, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("时间")
    ax.set_ylabel("流量（原始单位 / 10^6）")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _style(name: str) -> dict:
    styles = {
        "actual": {"color": "#111111", "linestyle": "-"},
        "persistence": {"color": "#8a8a8a", "linestyle": "--"},
        "seasonal": {"color": "#2ca02c", "linestyle": "--"},
        "slot": {"color": "#9467bd", "linestyle": ":"},
        "arima": {"color": "#1f77b4", "linestyle": "-"},
        "lstm": {"color": "#d62728", "linestyle": "-"},
    }
    return styles[name]


def save_grouped_bars(path: Path, labels: list[str], series: list[tuple[str, np.ndarray, str]],
                      title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for i, (name, values, color) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width=width,
               label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def slice_around(frame: pd.DataFrame, position: int, before: int, after: int) -> pd.DataFrame:
    start = max(0, position - before)
    stop = min(len(frame), position + after + 1)
    return frame.iloc[start:stop].copy()


def load_selected_lstm(directory: Path) -> tuple[TrafficLSTM, Scaler]:
    saved = torch.load(directory / "best_lstm.pt", map_location="cpu", weights_only=True)
    model = TrafficLSTM(int(saved["hidden_size"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, Scaler(**saved["scaler"])


def arima_open_forecast(raw: np.ndarray, train_stop: int, origin: int, steps: int,
                        order: tuple[int, int, int]) -> np.ndarray:
    """参数只在训练段估计；origin 及之前的观测只用来更新状态，再往前预测 steps。"""
    scaler = Scaler.fit(raw[:train_stop])
    normalized = scaler.transform(raw).astype(np.float64)
    fitted, _ = fit_on_train(normalized[:train_stop], order)
    observed = normalized[train_stop:origin + 1]
    state = fitted.extend(observed) if len(observed) else fitted
    future = np.asarray(state.forecast(steps=steps), dtype=np.float64).reshape(-1)
    if future.size != steps or not np.isfinite(future).all():
        raise ValueError("ARIMA open-loop forecast was not a finite vector of the requested length.")
    return scaler.inverse(future)


def lstm_open_forecast(model: TrafficLSTM, scaler: Scaler, raw: np.ndarray,
                       origin: int, steps: int) -> np.ndarray:
    window = list(np.asarray(raw[origin - LOOKBACK + 1:origin + 1], dtype=np.float64))
    forecasts = []
    for _ in range(steps):
        nxt = forecast_next(model, scaler, np.asarray(window[-LOOKBACK:], dtype=np.float64))
        forecasts.append(nxt)
        window.append(nxt)
    return np.asarray(forecasts, dtype=np.float64)


def open_loop_frame(series: pd.Series, raw: np.ndarray, origin: int, steps: int,
                    slot_lookup: np.ndarray, arima_forecast: np.ndarray | None,
                    lstm_forecast: np.ndarray | None) -> pd.DataFrame:
    future = np.arange(origin + 1, origin + 1 + steps)
    frame = pd.DataFrame({
        "timestamp": series.index[future],
        "actual": raw[future],
        "persistence": np.repeat(raw[origin], steps),
        "seasonal_naive": raw[future - 288],
        "slot_mean": slot_lookup[slot_of(series.index[future])],
    })
    if arima_forecast is not None:
        frame["arima"] = arima_forecast
    if lstm_forecast is not None:
        frame["lstm"] = lstm_forecast
    return frame


def fmt(value: float) -> str:
    return f"{value:,.0f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def point_sentence(row: pd.Series, lstm_name: str) -> str:
    return (f"{row['timestamp']} 的真实值是 {fmt(row['actual'])}。"
            f"持久性预测 {fmt(row['persistence'])}（绝对误差 {fmt(row['err_persistence'])}），"
            f"ARIMA 预测 {fmt(row['arima'])}（绝对误差 {fmt(row['err_arima'])}），"
            f"LSTM `{lstm_name}` 预测 {fmt(row[lstm_name])}（绝对误差 {fmt(row['err_lstm'])}）。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", default="traffic")
    parser.add_argument("--arima-dir", type=Path, required=True)
    parser.add_argument("--lstm-dir", type=Path, action="append", default=[])
    parser.add_argument("--lstm-root", type=Path, default=None,
                        help="Every child directory that already contains test_predictions.csv.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--short-horizon", type=int, default=12, help="Open-loop steps, default 1 hour.")
    parser.add_argument("--long-horizon", type=int, default=36, help="Open-loop steps, default 3 hours.")
    args = parser.parse_args()
    if args.short_horizon <= 0 or args.long_horizon <= 0:
        parser.error("Horizons must be positive.")
    try:
        ensure_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    configure_font()

    series = load_series(args.csv, args.target)
    raw = series.to_numpy(dtype=np.float64)
    splits = split_ranges(len(raw))
    train_stop = splits["train"][1]
    test_start = splits["test"][0]
    arima_frame, arima_config, arima_metrics = load_prediction_run(args.arima_dir, "arima")
    if arima_config.get("target") != args.target:
        raise ValueError("ARIMA run target does not match --target.")
    lstm_dirs = discover_lstm_dirs(args.lstm_dir, args.lstm_root)
    lstm_loaded = []
    for directory in lstm_dirs:
        frame, config, metrics = load_prediction_run(directory, "lstm")
        lstm_loaded.append((directory.name, directory, frame, config, metrics))
    combined = align_runs(arima_frame, arima_config,
                          [(name, frame, config) for name, _, frame, config, _ in lstm_loaded])
    combined = add_reference_forecasts(combined, series, train_stop)
    actual = combined["actual"].to_numpy()
    persistence_mae = calculate_metrics(actual, combined["persistence"].to_numpy())["mae"]

    best_val = min(item[4]["selected_validation_mse"] for item in lstm_loaded)
    selected_matches = [item for item in lstm_loaded
                        if item[4]["selected_validation_mse"] == best_val]
    selected_name, selected_dir, _, _, selected_metrics = selected_matches[0]
    order = tuple(int(v) for v in arima_metrics["selected_order"])

    rows = [
        metric_row("persistence", "baseline", actual, combined["persistence"].to_numpy(),
                   persistence_mae, False),
        metric_row("seasonal_naive", "baseline", actual, combined["seasonal_naive"].to_numpy(),
                   persistence_mae, False),
        metric_row("slot_mean", "baseline", actual, combined["slot_mean"].to_numpy(),
                   persistence_mae, False),
        metric_row(f"arima{order}", "validation-selected" if arima_config.get("search") else "fixed-order",
                   actual, combined["arima"].to_numpy(), persistence_mae, True),
    ]
    for name, _, _, _, metrics in lstm_loaded:
        rows.append(metric_row(
            name, "lstm", actual, combined[name].to_numpy(), persistence_mae,
            name == selected_name))
    overall = pd.DataFrame(rows)

    work = combined.copy().reset_index(drop=True)
    work["err_persistence"] = (work["persistence"] - work["actual"]).abs()
    work["err_arima"] = (work["arima"] - work["actual"]).abs()
    work["err_lstm"] = (work[selected_name] - work["actual"]).abs()
    work["err_seasonal"] = (work["seasonal_naive"] - work["actual"]).abs()
    work["hour"] = work["timestamp"].dt.hour
    work["quartile"] = quartile_labels(work["err_persistence"])
    by_hour = (work.groupby("hour")[["err_persistence", "err_arima", "err_lstm"]]
               .mean().reset_index())
    by_jump = (work.groupby("quartile")[["err_persistence", "err_arima", "err_lstm"]]
               .mean().reset_index())
    spike_pos = int(work["err_persistence"].idxmax())
    after_pos = min(spike_pos + 1, len(work) - 1)
    calm = work[work["err_persistence"] <= work["err_persistence"].median()]
    calm_pos = int((calm["err_lstm"] - calm["err_persistence"]).idxmax())
    major_day = work["timestamp"].dt.date.value_counts().idxmax()
    day = work[work["timestamp"].dt.date == major_day]

    args.out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out / "metrics_overall.csv", index=False)
    by_hour.to_csv(args.out / "error_by_hour.csv", index=False)
    by_jump.to_csv(args.out / "error_by_jump_quartile.csv", index=False)
    work.to_csv(args.out / "predictions_with_references.csv", index=False)
    search_path = args.arima_dir / "validation_search.csv"
    if search_path.is_file():
        search = pd.read_csv(search_path)
        search.to_csv(args.out / "arima_order_comparison.csv", index=False)
    else:
        search = None
    lstm_table = pd.DataFrame([{
        "name": name, "hidden_size": config.get("hidden_size"), "lr": config.get("lr"),
        "selected_epoch": metrics.get("selected_epoch"),
        "validation_mse": metrics.get("selected_validation_mse"),
        "test_mae": metrics["lstm"]["mae"], "test_rmse": metrics["lstm"]["rmse"],
        "selected_by_validation": name == selected_name,
    } for name, _, _, config, metrics in lstm_loaded]).sort_values("validation_mse")
    lstm_table.to_csv(args.out / "lstm_comparison.csv", index=False)

    rolling_columns = [
        ("actual", "真实值", "actual"),
        ("persistence", "持久性（上一时刻）", "persistence"),
        ("arima", f"ARIMA{order}", "arima"),
        (selected_name, f"LSTM {selected_name}", "lstm"),
    ]
    save_line_plot(args.out / "02_test_overview.png", work, rolling_columns,
                   f"滚动一步预测 | 本地测试段全部 {len(work)} 个计分点 | {args.target}")
    save_line_plot(args.out / "05_case_spike.png", slice_around(work, spike_pos, 8, 8),
                   rolling_columns, "案例：跳变发生的那一个 5 分钟", work.loc[spike_pos, "timestamp"])
    save_line_plot(args.out / "06_case_after_spike.png", slice_around(work, after_pos, 8, 8),
                   rolling_columns, "案例：跳变之后的下一步，照抄上一点会带偏",
                   work.loc[after_pos, "timestamp"])
    save_line_plot(args.out / "07_case_calm.png", slice_around(work, calm_pos, 18, 18),
                   rolling_columns, "案例：变化不大的时候，灵活模型和持久性相比",
                   work.loc[calm_pos, "timestamp"])
    save_line_plot(args.out / "08_one_day.png", day, rolling_columns,
                   f"滚动一步 | 计分点最多的一天 {major_day}")

    save_grouped_bars(
        args.out / "03_mae_by_hour.png",
        [str(int(h)) for h in by_hour["hour"]],
        [("持久性", by_hour["err_persistence"].to_numpy() / 1e6, "#8a8a8a"),
         (f"ARIMA{order}", by_hour["err_arima"].to_numpy() / 1e6, "#1f77b4"),
         (f"LSTM {selected_name}", by_hour["err_lstm"].to_numpy() / 1e6, "#d62728")],
        "按小时的平均绝对误差 | 测试段里同一小时可能来自相邻两天",
        "MAE（原始单位 / 10^6）")
    save_grouped_bars(
        args.out / "04_mae_by_jump.png",
        [str(v) for v in by_jump["quartile"]],
        [("持久性", by_jump["err_persistence"].to_numpy() / 1e6, "#8a8a8a"),
         (f"ARIMA{order}", by_jump["err_arima"].to_numpy() / 1e6, "#1f77b4"),
         (f"LSTM {selected_name}", by_jump["err_lstm"].to_numpy() / 1e6, "#d62728")],
        "按真实跳变大小分组的 MAE | 跳变 = |真实值 − 上一时刻|",
        "MAE（原始单位 / 10^6）")

    train_hour = (pd.DataFrame({"hour": series.index[:train_stop].hour, "traffic": raw[:train_stop]})
                  .groupby("hour")["traffic"].mean())
    day_hour = day.groupby(day["timestamp"].dt.hour)["actual"].mean()
    fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
    ax.plot(train_hour.index, train_hour.to_numpy() / 1e6, label="训练段按小时平均", color="#1f77b4")
    ax.plot(day_hour.index, day_hour.reindex(range(24)).to_numpy() / 1e6,
            label=f"测试中 {major_day} 的实际小时平均", color="#111111")
    ax.set_xlabel("小时")
    ax.set_ylabel("流量（原始单位 / 10^6）")
    ax.set_title("一天里的起伏：训练段的平均形状，和测试里点数最多的那一天")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(args.out / "01_daily_profile.png", dpi=140)
    plt.close(fig)

    if search is not None and (search["status"] == "ok").any():
        ok_search = search[search["status"] == "ok"].copy()
        ok_search["label"] = ok_search.apply(lambda r: f"({int(r.p)},{int(r.d)},{int(r.q)})", axis=1)
        ok_search = ok_search.sort_values("validation_mse", ascending=False)
        colors = ["#d62728" if tuple(int(v) for v in (r.p, r.d, r.q)) == order else "#1f77b4"
                  for r in ok_search.itertuples()]
        fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
        ax.barh(ok_search["label"], ok_search["validation_mae"] / 1e6, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("验证集 MAE（原始单位 / 10^6）")
        ax.set_ylabel("ARIMA 阶数 (p, d, q)")
        ax.set_title("ARIMA 选阶看验证集 | 红色是按标准化空间 MSE 选中的阶 | 越短越好")
        ax.grid(True, axis="x", alpha=0.25)
        fig.savefig(args.out / "12_arima_orders.png", dpi=140)
        plt.close(fig)

    if len(lstm_table) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
        labels = lstm_table["name"].tolist()
        x = np.arange(len(labels))
        colors = ["#d62728" if flag else "#1f77b4"
                  for flag in lstm_table["selected_by_validation"]]
        axes[0].bar(x, lstm_table["validation_mse"], color=colors)
        axes[0].set_title("用来选模型的验证 MSE")
        axes[0].set_ylabel("MSE（标准化对数空间）")
        axes[1].bar(x, lstm_table["test_mae"] / 1e6, color=colors)
        axes[1].set_title("选定之后才看的测试 MAE")
        axes[1].set_ylabel("MAE（原始单位 / 10^6）")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(args.out / "13_lstm_sweep.png", dpi=140)
        plt.close(fig)

    slot_lookup = np.full(288, np.nan)
    train_slots = slot_of(series.index[:train_stop])
    for slot in range(288):
        observed = raw[:train_stop][train_slots == slot]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            slot_lookup[slot] = float(observed.mean())
    model, scaler = load_selected_lstm(selected_dir)
    morning_candidates = np.flatnonzero(
        (series.index.date == major_day) & (series.index.hour == 7) & (series.index.minute == 0))
    origins = [
        ("test_start", test_start - 1, args.long_horizon,
         "从本地测试段开始前的最后一个观测点，开环预测接下来 3 小时"),
        ("before_spike", int(series.index.get_loc(work.loc[spike_pos, "timestamp"])) - 1,
         args.short_horizon, "最大跳变发生前的最后一个观测点，开环预测接下来 1 小时"),
    ]
    if morning_candidates.size and int(morning_candidates[0]) > 288:
        origins.append(("morning_ramp", int(morning_candidates[0]) - 1, args.long_horizon,
                        f"{major_day} 早上 7 点之前，开环预测接下来 3 小时的爬坡"))

    open_rows = []
    open_notes = []
    for fig_index, (key, origin, steps, title) in enumerate(origins, start=9):
        if origin < LOOKBACK or origin + steps >= len(raw):
            continue
        try:
            arima_future = arima_open_forecast(raw, train_stop, origin, steps, order)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            arima_future = None
            open_notes.append(f"{key} 的 ARIMA 开环没有得到可用预测：{exc}")
        try:
            lstm_future = lstm_open_forecast(model, scaler, raw, origin, steps)
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            lstm_future = None
            open_notes.append(f"{key} 的 LSTM 开环没有得到可用预测：{exc}")
        frame = open_loop_frame(series, raw, origin, steps, slot_lookup, arima_future, lstm_future)
        columns = [
            ("actual", "真实值", "actual"),
            ("persistence", "持久性（停在最后观测）", "persistence"),
            ("seasonal_naive", "昨天同一时刻", "seasonal"),
            ("slot_mean", "训练段同时刻均值", "slot"),
        ]
        if "arima" in frame:
            columns.append(("arima", f"ARIMA{order} 开环", "arima"))
        if "lstm" in frame:
            columns.append(("lstm", f"LSTM {selected_name} 开环", "lstm"))
        save_line_plot(args.out / f"{fig_index:02d}_open_loop_{key}.png", frame, columns, title)
        scored = []
        for column, label, _ in columns:
            if column == "actual":
                continue
            scores = calculate_metrics(frame["actual"].to_numpy(), frame[column].to_numpy())
            open_rows.append({"origin": key, "origin_timestamp": str(series.index[origin]),
                              "horizon_steps": steps, "model": column, "label": label, **scores})
            scored.append((label, scores["mae"]))
        winner = min(scored, key=lambda item: item[1])
        actual_std = float(np.std(frame["actual"]))
        flat = []
        for column, label, _ in columns:
            if column == "actual" or actual_std <= 0:
                continue
            pred_std = float(np.std(frame[column]))
            if pred_std < 0.25 * actual_std:
                flat.append(f"{label}几乎拉成水平线")
        flat_sentence = ("其中" + "、".join(flat) + "。" ) if flat else ""
        copied_spike = ""
        if frame["seasonal_naive"].max() > frame["actual"].max() * 1.4:
            copied_spike = "昨天同一时刻在这段里冲得比今天更高，那是把昨天的突发搬了过来。"
        open_notes.append(
            f"{title}。这 {steps} 个点里 MAE 最小的是「{winner[0]}」，为 {fmt(winner[1])}。"
            + "；".join(f"{label} {fmt(mae)}" for label, mae in scored)
            + "。" + flat_sentence + copied_spike)
    open_table = pd.DataFrame(open_rows)
    if not open_table.empty:
        open_table.to_csv(args.out / "open_loop_metrics.csv", index=False)

    q4_share = float(work.loc[work["quartile"].astype(str).str.startswith("Q4"), "err_persistence"].sum()
                     / work["err_persistence"].sum())
    calm_lstm_worse = float((work.loc[work["quartile"].astype(str).str.startswith("Q1"), "err_lstm"]
                             > work.loc[work["quartile"].astype(str).str.startswith("Q1"), "err_persistence"]).mean())
    test_leader = lstm_table.loc[lstm_table["test_mae"].idxmin(), "name"]
    val_vs_test = (f"若改用测试集 MAE 来排名，会选中 `{test_leader}`。"
                   if test_leader != selected_name else
                   "这一次验证集 MSE 和测试集 MAE 选中的是同一组 LSTM。这是结果碰巧一致，不是可以改用测试集选模型的理由。")
    spike = work.loc[spike_pos]
    after = work.loc[after_pos]
    calm_row = work.loc[calm_pos]
    reading = build_reading(
        target=args.target, order=order, selected_name=selected_name,
        arima_rule=str(arima_metrics.get("selection_rule")),
        overall=overall, lstm_table=lstm_table, by_jump=by_jump,
        search=search, spike=spike, after=after, calm_row=calm_row,
        major_day=major_day, q4_share=q4_share, calm_lstm_worse=calm_lstm_worse,
        val_vs_test=val_vs_test, open_notes=open_notes, n_test=len(work))
    (args.out / "读图说明.md").write_text(reading, encoding="utf-8")
    print(overall[["model", "mae", "rmse", "skill_vs_persistence", "selected_by_validation"]]
          .to_string(index=False))
    print(f"Selected LSTM by validation MSE: {selected_name}")
    print(f"Reading notes: {(args.out / '读图说明.md').resolve()}")


def build_reading(target: str, order: tuple[int, int, int], selected_name: str, arima_rule: str,
                  overall: pd.DataFrame, lstm_table: pd.DataFrame, by_jump: pd.DataFrame,
                  search: pd.DataFrame | None, spike: pd.Series, after: pd.Series,
                  calm_row: pd.Series, major_day, q4_share: float, calm_lstm_worse: float,
                  val_vs_test: str, open_notes: list[str], n_test: int) -> str:
    overall_rows = [[row.model, f"{row.mae:,.0f}", f"{row.rmse:,.0f}",
                     f"{row.bias:,.0f}", f"{row.skill_vs_persistence:.1%}",
                     "是" if row.selected_by_validation else ""]
                    for row in overall.itertuples()]
    lstm_rows = [[row.name, str(row.hidden_size), f"{row.lr:g}", str(row.selected_epoch),
                  f"{row.validation_mse:.4f}", f"{row.test_mae:,.0f}",
                  "是" if row.selected_by_validation else ""]
                 for row in lstm_table.itertuples()]
    jump_rows = [[str(row.quartile), f"{row.err_persistence:,.0f}", f"{row.err_arima:,.0f}",
                  f"{row.err_lstm:,.0f}"] for row in by_jump.itertuples()]
    if search is not None:
        ok = search[search["status"] == "ok"].sort_values("validation_mse")
        search_rows = [[f"({int(row.p)},{int(row.d)},{int(row.q)})", f"{row.validation_mse:.4f}",
                        f"{row.validation_mae:,.0f}",
                        "是" if (int(row.p), int(row.d), int(row.q)) == order else ""]
                       for row in ok.itertuples()]
        mse_best = ok.loc[ok["validation_mse"].idxmin()]
        mae_best = ok.loc[ok["validation_mae"].idxmin()]
        same_winner = (int(mse_best.p), int(mse_best.d), int(mse_best.q)) == (
            int(mae_best.p), int(mae_best.d), int(mae_best.q))
        rank_note = ("验证 MSE 和验证 MAE 的第一名是同一阶。紧挨着的几阶非常接近，两种指标的第二名可以对调。"
                     if same_winner else
                     "验证 MSE 的第一名和验证 MAE 的第一名不是同一阶。脚本只按 MSE 选定，不用 MAE 改口。")
        search_block = ("ARIMA 候选都在验证集上比较。下表按标准化空间的验证 MSE 从小到大排列。"
                        f"{rank_note}\n\n" + md_table(
            ["阶数 (p,d,q)", "验证 MSE（标准化空间）", "验证 MAE（原始单位）", "脚本选中"], search_rows))
    else:
        search_block = "这次 ARIMA 目录没有 `validation_search.csv`，所以没有阶数对比图。"
    open_block = "\n\n".join(f"- {note}" for note in open_notes) if open_notes else "- 这次没有生成开环片段。"
    return f"""# 读图说明：{target} 这一次跑完以后怎么讲

这是脚本根据本次输出写的，换节点或换参数后要重新运行 `analyze_forecasts.py`，不要把下面的数字抄到另一次实验上。

选模型的规则：ARIMA 用 `{arima_rule}`。LSTM 用验证集标准化对数空间 MSE 最低的一组，本次是 `{selected_name}`。{val_vs_test}

## 课堂上按这个顺序讲

时间不够就讲 02、05、06、07、09。调参那一张是 12 和 13。其余的图用来回答学生追问。

### 01 一天里的起伏

`01_daily_profile.png`。先不看模型。流量有明显的夜间低谷和傍晚高峰，所以“每天这个钟点”是有信息的。它回答不了下一个 5 分钟，因为下一个 5 分钟几乎还停在刚才的水平上。

### 02 不要只念总分

`02_test_overview.png` 把本地测试段 {n_test} 个滚动一步预测画在一起。曲线贴得很紧时，MAE 的差距只占总流量的一小段，肉眼会觉得“大家都还行”。下表是全部计分点，不是图上的某一段。

{md_table(["模型", "MAE", "RMSE", "预测减真实（平均）", "相对持久性的技能", "是否选中"], overall_rows)}

技能 = 1 − 模型 MAE / 持久性 MAE。负数表示比“照抄上一时刻”更差。昨天同一时刻、训练段同时刻均值在这一列通常不好看：它们是给更长的预测准备的参照，不是用来赢滚动一步的。

### 12 和 13 调参对比

{search_block}

LSTM 各组只改了隐藏维或学习率。红色（或“是否选中”）对应验证集，不是测试集。

{md_table(["名称", "隐藏维", "学习率", "选中的轮数", "验证 MSE", "测试 MAE", "是否选中"], lstm_rows)}

{("`12_arima_orders.png`、`13_lstm_sweep.png`。讲的时候先挡住右图或测试 MAE 那一列，让学生只用验证列排序，再揭开测试列。"
  if len(lstm_rows) > 1 else
  "`12_arima_orders.png`。这次只传了一组 LSTM，没有生成 13 号对比图。")}

### 03 和 04 误差并不均匀

`03_mae_by_hour.png` 按小时拆 MAE。夜间总量小，绝对误差通常也小；上午和晚上的爬坡、突发会把某些小时抬起来。测试段从切分边界开始，同一个小时可能包含相邻的两天，不要把它说成“纯粹的周五画像”。

`04_mae_by_jump.png` 按真实跳变四分位拆 MAE。持久性的绝对误差有 {q4_share:.0%} 落在跳变最大的那一组点上。分组平均：

{md_table(["组别", "持久性 MAE", "ARIMA MAE", "LSTM MAE"], jump_rows)}

最平缓的一组里，LSTM 差于持久性的计分点占 {calm_lstm_worse:.0%}。平均 MAE 把“平缓时添乱”和“跳变时少错一点”加在了一起。

### 05 跳变发生的那个点

`05_case_spike.png`。竖线是持久性误差最大的计分点。

{point_sentence(spike, selected_name)}

这一步之前序列还在另一水平上。滚动一步的模型主要看见最近 5 到 60 分钟，看不到“5 分钟后会突然冲高”的前兆。三个预测会一起偏低或一起偏高。把隐藏维从 32 改到 128 不会补上一个数据里不存在的前兆。

可以问：如果任务改成预测一个区间，而不是一个点，这个时间戳应该落在区间的什么位置？

### 06 跳变的下一步

`06_case_after_spike.png`。竖线是上一张图那个点的下一个 5 分钟。

{point_sentence(after, selected_name)}

持久性把刚才的尖峰原样抄过来。ARIMA 和 LSTM 这一步没有完全跟着尖峰走，所以误差会小一截。这是它们在平均分上没有被持久性拉开的主要来源之一，不是它们提前看见了突发。

### 07 平缓的一段

`07_case_calm.png`。竖线标的是：真实变化不大、但选中的 LSTM 比持久性差得最多的那个点。

{point_sentence(calm_row, selected_name)}

流量几乎没动时，最好的预测就是上一时刻。网络如果在对数空间里学了一个平滑后的水平，反变换回原始单位后会在高位流量上偏一截。学生可以用预测减真实的符号，看它是整段偏高、整段偏低，还是只在拐点慢半拍。

### 08 拿一天出来看

`08_one_day.png` 是计分点最多的一天（{major_day}）的滚动一步结果。用它检查 05 和 07 是不是孤立的点：尖峰是一根针，平缓段是长长的贴合或长长的偏离。

### 09 到 11 开环：滚动一步赢了，不代表能交空白的未来

下面每段都假装原点之后的真实值还没到。持久性变成一条水平线；LSTM 把自己的预测填回最近 12 点再往下推；ARIMA 用训练段拟合的参数，只把原点之前的观测拿来更新状态，然后一次预测多步。这不是比赛提交文件。

{open_block}

对应图片是 `09_open_loop_test_start.png`、`10_open_loop_before_spike.png`、`11_open_loop_morning_ramp.png`（某一段没有画出来时，上面的列表里就不会有它）。滚动一步里很差的钟点参照，可能只在爬坡这种开环片段里翻盘，反过来也一样。对照每一段的 MAE，不要把两种协议合成一张榜。

## 从图里直接能交给学生的问题

1. 换一个小区重跑。本次节点只是训练表里按列顺序碰到的第一个可用列，不是指定的容易序列。
2. 只改一个 LSTM 旋钮（隐藏维、学习率，或你先改代码再改的 lookback）。用验证集保留一个，测试集只报告一次。
3. 把评估从滚动一步改成开环 12 步、36 步。问持久性什么时候不再是第一名。
4. 针对 05 那个时间戳写一段话：点预测失败，是因为模型容量不够，还是因为输入里没有前兆？
5. 官方测试文件是训练结束之后的空白时间戳。哪些列到那个时候还允许使用，以数据目录里的说明为准，不能把本地回测的真值填回去再滚动。
"""


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

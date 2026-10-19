#!/usr/bin/env python3
"""比较单通道 ARIMA/LSTM 和联合 VAR/LSTM，并挑出具体小区、具体时刻。

联合模型按各自的验证集误差选定。测试集分数在选定之后才汇总。
单通道模型只在它们训练过的那一列上比较，不把没训练的 29 列算进它们的平均分。
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

from analyze_forecasts import configure_font
from common import (LOOKBACK, ChannelScaler, calculate_metrics, ensure_new_output,
                    load_wide, split_ranges)
from train_lstm import TrafficLSTM, forecast_next
from train_lstm_multi import JointLSTM
from train_var import one_step


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_long(directory: Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / "test_predictions.csv")
    required = {"timestamp", "cell", "actual", column, "persistence"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{directory} is missing the '{column}' joint predictions.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame


def discover(explicit: list[Path], root: Path | None) -> list[Path]:
    found = list(explicit)
    if root is not None:
        found.extend(path for path in sorted(root.iterdir())
                     if path.is_dir() and (path / "test_predictions.csv").is_file())
    unique, seen = [], set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("Pass --lstm-dir or --lstm-root.")
    return unique


def fmt(value: float) -> str:
    return f"{value:,.0f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([head, rule, *["| " + " | ".join(row) + " |" for row in rows]])


def save_lines(path: Path, frame: pd.DataFrame, columns: list[tuple[str, str, str]],
               title: str, vline: pd.Timestamp | None = None) -> None:
    colors = {"actual": "#111111", "persistence": "#8a8a8a", "arima": "#1f77b4",
              "lstm": "#d62728", "var": "#ff7f0e", "joint": "#6a3d9a"}
    styles = {"persistence": "--", "actual": "-", "arima": "-", "lstm": "-", "var": "-", "joint": "-"}
    fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
    for column, label, key in columns:
        ax.plot(frame["timestamp"], frame[column] / 1e6, label=label, color=colors[key],
                linestyle=styles[key], linewidth=1.35)
    if vline is not None:
        ax.axvline(vline, color="0.35", linestyle=":", linewidth=1)
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


def channel_mae(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("cell").apply(
        lambda g: float(np.mean(np.abs(g[column] - g["actual"]))))
    return grouped.sort_index()


def var_open_loop(raw: np.ndarray, origin: int, steps: int, params: dict) -> np.ndarray:
    coefs = np.asarray(params["coefs"], dtype=np.float64)
    intercept = np.asarray(params["intercept"], dtype=np.float64)
    scaler = ChannelScaler(tuple(params["scaler_mean"]), tuple(params["scaler_scale"]))
    latent = scaler.transform(raw).astype(np.float64)
    lag = coefs.shape[0]
    state = latent[origin - lag + 1:origin + 1]
    forecasts = []
    for _ in range(steps):
        nxt = one_step(coefs, intercept, state)
        forecasts.append(nxt)
        state = np.vstack((state, nxt))
    return scaler.inverse(np.asarray(forecasts))


def joint_lstm_open_loop(raw: np.ndarray, origin: int, steps: int, directory: Path) -> np.ndarray:
    saved = torch.load(directory / "best_lstm_multi.pt", map_location="cpu", weights_only=True)
    model = JointLSTM(len(saved["channels"]), int(saved["hidden_size"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    scaler = ChannelScaler(tuple(saved["scaler_mean"]), tuple(saved["scaler_scale"]))
    window = scaler.transform(raw).astype(np.float32)[origin - LOOKBACK + 1:origin + 1]
    forecasts = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = model(torch.from_numpy(window[-LOOKBACK:]).unsqueeze(0)).numpy()[0]
            forecasts.append(nxt)
            window = np.vstack((window, nxt))
    return scaler.inverse(np.asarray(forecasts))


def uni_lstm_open_loop(raw_cell: np.ndarray, origin: int, steps: int, directory: Path) -> np.ndarray:
    saved = torch.load(directory / "best_lstm.pt", map_location="cpu", weights_only=True)
    model = TrafficLSTM(int(saved["hidden_size"]))
    model.load_state_dict(saved["state_dict"])
    from common import Scaler
    scaler = Scaler(**saved["scaler"])
    forecasts = []
    window = list(np.asarray(raw_cell[origin - LOOKBACK + 1:origin + 1], dtype=np.float64))
    for _ in range(steps):
        nxt = forecast_next(model, scaler, np.asarray(window[-LOOKBACK:], dtype=np.float64))
        forecasts.append(nxt)
        window.append(nxt)
    return np.asarray(forecasts, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--var-dir", type=Path, required=True)
    parser.add_argument("--lstm-dir", type=Path, action="append", default=[])
    parser.add_argument("--lstm-root", type=Path, default=None)
    parser.add_argument("--uni-arima-dir", type=Path, required=True)
    parser.add_argument("--uni-lstm-dir", type=Path, required=True)
    parser.add_argument("--focus-cell", default="Cell_001")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=36)
    args = parser.parse_args()
    try:
        ensure_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    configure_font()
    wide = load_wide(args.csv)
    if args.focus_cell not in wide.columns:
        raise ValueError(f"{args.focus_cell} is not a channel. Available: {list(wide.columns)}")
    raw = wide.to_numpy(dtype=np.float64)
    columns = list(wide.columns)
    focus = columns.index(args.focus_cell)

    var = load_long(args.var_dir, "var")
    var_metrics = read_json(args.var_dir / "metrics.json")
    var_params = read_json(args.var_dir / "var_parameters.json")
    search = pd.read_csv(args.var_dir / "validation_search.csv")
    lstm_dirs = discover(args.lstm_dir, args.lstm_root)
    lstm_runs = []
    for directory in lstm_dirs:
        frame = load_long(directory, "lstm")
        metrics = read_json(directory / "metrics.json")
        config = read_json(directory / "run_config.json")
        lstm_runs.append((directory.name, directory, frame, metrics, config))
    if not var["timestamp"].equals(lstm_runs[0][2]["timestamp"]) or not var["cell"].equals(lstm_runs[0][2]["cell"]):
        raise ValueError("VAR and joint LSTM do not share test rows.")
    for _, _, frame, _, config in lstm_runs[1:]:
        if config.get("data_sha256") != lstm_runs[0][4].get("data_sha256"):
            raise ValueError("Joint LSTM runs were not trained on the same wide table.")
        if not frame["timestamp"].equals(var["timestamp"]):
            raise ValueError("Joint LSTM test timestamps differ.")
    best_val = min(item[3]["selected_validation_mse"] for item in lstm_runs)
    selected = [item for item in lstm_runs if item[3]["selected_validation_mse"] == best_val][0]
    selected_name, selected_dir, selected_frame, selected_metrics, _ = selected

    merged = var[["timestamp", "cell", "actual", "persistence", "var"]].copy()
    merged["lstm"] = selected_frame["lstm"].to_numpy()
    focus_rows = merged[merged["cell"] == args.focus_cell].sort_values("timestamp")
    uni_arima = pd.read_csv(args.uni_arima_dir / "test_predictions.csv", parse_dates=["timestamp"])
    uni_lstm = pd.read_csv(args.uni_lstm_dir / "test_predictions.csv", parse_dates=["timestamp"])
    if not focus_rows["timestamp"].reset_index(drop=True).equals(uni_arima["timestamp"]):
        raise ValueError(f"{args.focus_cell} timestamps do not match the univariate test file.")
    if not np.allclose(focus_rows["actual"], uni_arima["actual"], rtol=1e-8, atol=1e-6):
        raise ValueError(f"{args.focus_cell} values do not match the univariate series.")
    focus_rows = focus_rows.copy()
    focus_rows["arima"] = uni_arima["arima"].to_numpy()
    focus_rows["uni_lstm"] = uni_lstm["lstm"].to_numpy()

    def cell_table(column: str) -> pd.Series:
        return channel_mae(merged, column)

    mae = pd.DataFrame({
        "persistence": cell_table("persistence"),
        "var": cell_table("var"),
        "lstm_multi": cell_table("lstm"),
    })
    mae["var_skill"] = 1 - mae["var"] / mae["persistence"]
    mae["lstm_skill"] = 1 - mae["lstm_multi"] / mae["persistence"]
    args.out.mkdir(parents=True, exist_ok=True)
    mae.reset_index().rename(columns={"index": "cell"}).to_csv(args.out / "per_cell_mae.csv", index=False)
    search.to_csv(args.out / "var_lag_comparison.csv", index=False)
    lstm_table = pd.DataFrame([{
        "name": name, "hidden_size": config.get("hidden_size"), "lr": config.get("lr"),
        "selected_epoch": metrics.get("selected_epoch"),
        "validation_mse": metrics.get("selected_validation_mse"),
        "test_macro_mae": metrics["lstm"]["macro_mae"],
        "selected_by_validation": name == selected_name,
    } for name, _, _, metrics, config in lstm_runs]).sort_values("validation_mse")
    lstm_table.to_csv(args.out / "lstm_multi_comparison.csv", index=False)

    focus_scores = {name: calculate_metrics(focus_rows["actual"].to_numpy(), focus_rows[column].to_numpy())
                    for name, column in (("persistence", "persistence"), ("arima", "arima"),
                                         ("uni_lstm", "uni_lstm"), ("var", "var"), ("lstm_multi", "lstm"))}
    pd.DataFrame([{"model": name, **scores} for name, scores in focus_scores.items()]).to_csv(
        args.out / "focus_cell_metrics.csv", index=False)

    ok = search[search["status"] == "ok"].sort_values("validation_mse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    labels = [f"VAR({int(row.lag)})" for row in ok.itertuples()]
    colors = ["#d62728" if int(row.lag) == int(var_metrics["selected_lag"]) else "#1f77b4"
              for row in ok.itertuples()]
    ax.barh(labels, ok["validation_mse"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("验证集 MSE（按通道标准化后的对数空间）")
    ax.set_title("联合 VAR 选滞后 | 红色是验证集选中的阶 | 越短越好")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(args.out / "01_var_lags.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    names = lstm_table["name"].tolist()
    x = np.arange(len(names))
    colors = ["#d62728" if flag else "#1f77b4" for flag in lstm_table["selected_by_validation"]]
    axes[0].bar(x, lstm_table["validation_mse"], color=colors)
    axes[0].set_title("用来选联合 LSTM 的验证 MSE")
    axes[0].set_ylabel("MSE（各通道标准化后）")
    axes[1].bar(x, lstm_table["test_macro_mae"] / 1e6, color=colors)
    axes[1].set_title("选定之后的 30 列平均测试 MAE")
    axes[1].set_ylabel("MAE（原始单位 / 10^6）")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(args.out / "02_lstm_multi_sweep.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 4.6), constrained_layout=True)
    x = np.arange(len(mae))
    width = 0.25
    ax.bar(x - width, mae["persistence"] / 1e6, width, label="持久性", color="#8a8a8a")
    ax.bar(x, mae["var"] / 1e6, width, label=f"VAR({int(var_metrics['selected_lag'])})", color="#ff7f0e")
    ax.bar(x + width, mae["lstm_multi"] / 1e6, width, label=f"联合 LSTM {selected_name}", color="#6a3d9a")
    ax.set_xticks(x)
    ax.set_xticklabels(mae.index, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("测试 MAE（原始单位 / 10^6）")
    ax.set_title("30 个小区各自的滚动一步 MAE | 不要只看 30 列平均")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(args.out / "03_per_cell_mae.png", dpi=140)
    plt.close(fig)

    day = focus_rows["timestamp"].dt.date.value_counts().idxmax()
    day_rows = focus_rows[focus_rows["timestamp"].dt.date == day]
    focus_columns = [
        ("actual", "真实值", "actual"),
        ("persistence", "持久性", "persistence"),
        ("arima", "单通道 ARIMA", "arima"),
        ("uni_lstm", "单通道 LSTM", "lstm"),
        ("var", f"VAR({int(var_metrics['selected_lag'])})", "var"),
        ("lstm", f"联合 LSTM {selected_name}", "joint"),
    ]
    save_lines(args.out / "04_focus_day.png", day_rows, focus_columns,
               f"{args.focus_cell} 滚动一步 | {day}")
    spike_pos = int((focus_rows["actual"] - focus_rows["persistence"]).abs().idxmax())
    spike = focus_rows.loc[spike_pos]
    loc = focus_rows.index.get_loc(spike_pos)
    zoom = focus_rows.iloc[max(0, loc - 8):loc + 9]
    save_lines(args.out / "05_focus_spike.png", zoom, focus_columns,
               f"{args.focus_cell} 跳变最大的 5 分钟", spike["timestamp"])

    at_spike = merged[merged["timestamp"] == spike["timestamp"]].copy()
    at_spike["jump"] = at_spike["actual"] - at_spike["persistence"]
    fig, ax = plt.subplots(figsize=(12.5, 4.4), constrained_layout=True)
    colors = ["#d62728" if cell == args.focus_cell else "#1f77b4" for cell in at_spike["cell"]]
    ax.bar(np.arange(len(at_spike)), at_spike["jump"] / 1e6, color=colors)
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xticks(np.arange(len(at_spike)))
    ax.set_xticklabels(at_spike["cell"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("这一步的变化（原始单位 / 10^6）")
    ax.set_title(f"{spike['timestamp']} 各小区相对上一时刻的变化 | 红色是 {args.focus_cell}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(args.out / "06_cross_section_at_spike.png", dpi=140)
    plt.close(fig)

    hurt = mae["lstm_skill"].idxmin()
    helped = mae["var_skill"].idxmax()
    for filename, cell, title in (
            ("07_joint_lstm_hurts.png", hurt, f"联合 LSTM 比持久性差最多的小区 {hurt}"),
            ("08_var_helps.png", helped, f"VAR 比持久性好最多的小区 {helped}")):
        cell_rows = merged[merged["cell"] == cell].sort_values("timestamp")
        cell_day = cell_rows[cell_rows["timestamp"].dt.date == day]
        save_lines(args.out / filename, cell_day,
                   [("actual", "真实值", "actual"), ("persistence", "持久性", "persistence"),
                    ("var", f"VAR({int(var_metrics['selected_lag'])})", "var"),
                    ("lstm", f"联合 LSTM {selected_name}", "joint")],
                   f"{title} | {day}")

    test_start = split_ranges(len(wide))["test"][0]
    morning = np.flatnonzero((wide.index.date == day) & (wide.index.hour == 7) & (wide.index.minute == 0))
    origin = int(morning[0]) - 1 if morning.size else test_start - 1
    steps = args.horizon
    var_future = var_open_loop(raw, origin, steps, var_params)
    lstm_future = joint_lstm_open_loop(raw, origin, steps, selected_dir)
    uni_future = uni_lstm_open_loop(raw[:, focus], origin, steps, args.uni_lstm_dir)
    future_index = wide.index[origin + 1:origin + 1 + steps]
    open_frame = pd.DataFrame({
        "timestamp": future_index,
        "actual": raw[origin + 1:origin + 1 + steps, focus],
        "persistence": np.repeat(raw[origin, focus], steps),
        "uni_lstm": uni_future,
        "var": var_future[:, focus],
        "lstm": lstm_future[:, focus],
    })
    save_lines(args.out / "09_open_loop_morning.png", open_frame, [
        ("actual", "真实值", "actual"),
        ("persistence", "持久性（停在最后观测）", "persistence"),
        ("uni_lstm", "单通道 LSTM 开环", "lstm"),
        ("var", "联合 VAR 开环", "var"),
        ("lstm", "联合 LSTM 开环", "joint"),
    ], f"{args.focus_cell} 从 {wide.index[origin]} 起开环 {steps} 步")
    open_rows = []
    for column, label in (("persistence", "持久性"), ("uni_lstm", "单通道 LSTM"),
                          ("var", "联合 VAR"), ("lstm", "联合 LSTM")):
        scores = calculate_metrics(open_frame["actual"].to_numpy(), open_frame[column].to_numpy())
        open_rows.append({"model": label, **scores})
    pd.DataFrame(open_rows).to_csv(args.out / "open_loop_focus.csv", index=False)

    train_stop = split_ranges(len(wide))["train"][1]
    diffs = np.diff(raw[:train_stop], axis=0)
    focus_diff = diffs[:, focus]
    others = [float(np.corrcoef(focus_diff, diffs[:, i])[0, 1]) if i != focus else 1.0
              for i in range(diffs.shape[1])]
    neighbor = int(np.argsort(others)[-2])
    neighbor_jump = float(at_spike.loc[at_spike["cell"] == columns[neighbor], "jump"].iloc[0])
    focus_jump = float(spike["actual"] - spike["persistence"])
    other_jumps = at_spike.loc[at_spike["cell"] != args.focus_cell, "jump"].abs()
    var_wins = int((mae["var_skill"] > 0).sum())
    lstm_wins = int((mae["lstm_skill"] > 0).sum())
    test_leader = lstm_table.loc[lstm_table["test_macro_mae"].idxmin(), "name"]
    same_pick = test_leader == selected_name
    reading = build_reading(
        focus=args.focus_cell, var_metrics=var_metrics, search=search, lstm_table=lstm_table,
        mae=mae, focus_scores=focus_scores, spike=spike, day=day, columns=columns,
        neighbor=columns[neighbor], neighbor_corr=others[neighbor], neighbor_jump=neighbor_jump,
        focus_jump=focus_jump, other_max=float(other_jumps.max()), var_wins=var_wins,
        lstm_wins=lstm_wins, hurt=hurt, helped=helped, open_rows=open_rows,
        same_pick=same_pick, selected_name=selected_name, origin=str(wide.index[origin]))
    (args.out / "读图说明.md").write_text(reading, encoding="utf-8")
    print(pd.DataFrame([{"model": k, "mae": v["mae"]} for k, v in focus_scores.items()]).to_string(index=False))
    print(f"VAR beats persistence on {var_wins}/{len(mae)} cells")
    print(f"Joint LSTM beats persistence on {lstm_wins}/{len(mae)} cells")
    print(f"Notes: {(args.out / '读图说明.md').resolve()}")


def build_reading(focus, var_metrics, search, lstm_table, mae, focus_scores, spike, day,
                  columns, neighbor, neighbor_corr, neighbor_jump, focus_jump, other_max,
                  var_wins, lstm_wins, hurt, helped, open_rows, same_pick, selected_name,
                  origin) -> str:
    ok = search[search["status"] == "ok"].sort_values("validation_mse")
    var_rows = [[f"VAR({int(row.lag)})", str(int(row.coefficients)),
                 f"{row.validation_mse:.4f}", fmt(row.validation_macro_mae),
                 "是" if int(row.lag) == int(var_metrics["selected_lag"]) else ""]
                for row in ok.itertuples()]
    lstm_rows = [[row.name, str(row.hidden_size), f"{row.lr:g}", str(row.selected_epoch),
                  f"{row.validation_mse:.4f}", fmt(row.test_macro_mae),
                  "是" if row.selected_by_validation else ""]
                 for row in lstm_table.itertuples()]
    focus_rows = [[name, fmt(scores["mae"]), fmt(scores["rmse"])]
                  for name, scores in focus_scores.items()]
    open_table = [[row["model"], fmt(row["mae"])] for row in open_rows]
    macro = {name: float(mae[name].mean()) for name in ("persistence", "var", "lstm_multi")}
    pick = ("验证集和 30 列平均测试 MAE 选中同一组联合 LSTM。" if same_pick else
            "若按 30 列平均测试 MAE 排序，会选到另一组。报告仍采用验证集选中的那组。")
    return f"""# 读图说明：30 个小区联合预测

单通道 ARIMA、单通道 LSTM 保持原来的训练，不重新选。联合 VAR 用验证集标准化空间的 MSE 选滞后，选中 VAR({int(var_metrics['selected_lag'])})。联合 LSTM 用同一指标选超参，选中 `{selected_name}`。{pick}

30 列平均测试 MAE：持久性 {fmt(macro['persistence'])}，VAR {fmt(macro['var'])}，联合 LSTM {fmt(macro['lstm_multi'])}。VAR 在 {var_wins}/30 个小区上好于持久性，联合 LSTM 在 {lstm_wins}/30 个小区上好于持久性。平均分把这些小区加在一起，不能代替下面的单列曲线。

## 调参

VAR 的滞后每增加 1，交叉系数大约增加 30×30 个。验证集随滞后变差，所以更长的滞后没有被留下。

{md_table(["模型", "系数个数", "验证 MSE", "验证宏平均 MAE", "是否选中"], var_rows)}

联合 LSTM 一次只改隐藏维或学习率。左图 `02_lstm_multi_sweep.png` 才用来选。

{md_table(["名称", "隐藏维", "学习率", "选中轮数", "验证 MSE", "测试宏平均 MAE", "是否选中"], lstm_rows)}

这个验证 MSE 是 30 个通道一起算的，不能和单通道 LSTM 的 0.07 直接比大小。比较用下面的原始单位 MAE。

## {focus} 上和单通道模型比

{md_table(["模型", "MAE", "RMSE"], focus_rows)}

`04_focus_day.png` 是 {day} 的滚动一步。`05_focus_spike.png` 竖线在持久性误差最大的点：{spike['timestamp']}，真实值 {fmt(spike['actual'])}，上一时刻 {fmt(spike['persistence'])}。单通道 ARIMA {fmt(spike['arima'])}，单通道 LSTM {fmt(spike['uni_lstm'])}，VAR {fmt(spike['var'])}，联合 LSTM {fmt(spike['lstm'])}。

## 其他小区在这一步有没有先动

`06_cross_section_at_spike.png` 画的是这一分钟正在被预测的变化，不是模型已经看见的上一分钟。{focus} 变化了 {fmt(focus_jump)}。同一分钟里其他小区变化绝对值最大的是 {fmt(other_max)}，所以并不是“别人完全没动”，而是有小区在同一时刻一起跳。这个同时刻的值不能当作输入。训练段上和 {focus} 逐点变化相关性最高的是 {neighbor}，相关系数只有 {neighbor_corr:.2f}，它在这一步的变化是 {fmt(neighbor_jump)}。交叉项要用上邻区，邻区必须在更早的时刻就动过。

## 平均分盖住的两列

`07_joint_lstm_hurts.png` 是联合 LSTM 相对持久性最差的小区 {hurt}。技能 {mae.loc[hurt, 'lstm_skill']:.1%}，VAR 技能 {mae.loc[hurt, 'var_skill']:.1%}。

`08_var_helps.png` 是 VAR 相对持久性最好的小区 {helped}。技能 {mae.loc[helped, 'var_skill']:.1%}，联合 LSTM 技能 {mae.loc[helped, 'lstm_skill']:.1%}。

## 开环

从 {origin} 起，后面 {len(open_rows) and open_rows[0]['count']} 步不再读入真实值。图 `09_open_loop_morning.png` 只画 {focus}。联合模型把 30 列的预测一起填回去，单通道 LSTM 只填自己这一列。

{md_table(["模型", f"{focus} 的开环 MAE"], open_table)}

滚动一步上联合与否，和这段没有最新观测时的排名，是两件事。
"""


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

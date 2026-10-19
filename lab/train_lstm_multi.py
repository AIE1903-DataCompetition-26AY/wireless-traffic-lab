#!/usr/bin/env python3
"""30 个小区的联合 LSTM：一次读入全部通道的最近 12 个点，同时预测下一时刻的 30 列。

切分、按通道 log1p 标准化、AdamW、早停和计分时间戳与单通道 LSTM 相同。
输入维和输出维都是通道数。隐藏状态在每个窗口开始时为 0。
损失是全部通道在标准化空间里的均方误差，因此流量大的小区不会单独主导训练。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import (FREQUENCY, LOOKBACK, ChannelScaler, calculate_metrics, ensure_new_output,
                    environment_versions, load_wide, split_ranges, valid_row_targets,
                    wide_fingerprint, write_json)


class WideWindowDataset(Dataset):
    def __init__(self, normalized: np.ndarray, start: int, stop: int):
        self.values = normalized
        self.targets = valid_row_targets(normalized, start, stop)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = int(self.targets[index])
        x = self.values[j - LOOKBACK:j].copy()
        y = self.values[j].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


class JointLSTM(nn.Module):
    def __init__(self, channels: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, channels)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(xb)
        return self.head(sequence[:, -1, :])


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    device = next(model.parameters()).device
    squared_error, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        squared_error += float(loss.detach().cpu()) * yb.numel()
        count += yb.numel()
    return squared_error / count


@torch.no_grad()
def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    squared_error, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        squared_error += float(nn.functional.mse_loss(model(xb), yb, reduction="sum").cpu())
        count += yb.numel()
    value = squared_error / count
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite validation MSE.")
    return value


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(xb.to(device)).cpu().numpy() for xb, _ in loader], axis=0)


def macro_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    per_channel = [calculate_metrics(actual[:, i], forecast[:, i])["mae"] for i in range(actual.shape[1])]
    flat = calculate_metrics(actual.reshape(-1), forecast.reshape(-1))
    return {"macro_mae": float(np.mean(per_channel)), "pooled_mae": flat["mae"],
            "pooled_rmse": flat["rmse"], "count_per_channel": int(actual.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size, args.patience) <= 0:
        parser.error("Epochs, batch size, hidden size and patience must be positive.")
    if not np.isfinite([args.lr, args.weight_decay]).all() or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1.")
    try:
        ensure_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    wide = load_wide(args.csv, args.time_column)
    raw = wide.to_numpy(dtype=np.float64)
    columns = list(wide.columns)
    splits = split_ranges(len(wide))
    scaler = ChannelScaler.fit(raw[:splits["train"][1]])
    normalized = scaler.transform(raw)
    datasets = {name: WideWindowDataset(normalized, start, stop)
                for name, (start, stop) in splits.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(data, batch_size=args.batch_size, shuffle=(name == "train"),
                               generator=generator if name == "train" else None,
                               num_workers=0, drop_last=False)
               for name, data in datasets.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": "lstm_multi", "channels": columns, "channel_count": len(columns),
        "data_sha256": wide_fingerprint(wide), "environment": environment_versions()})

    model = JointLSTM(len(columns), args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best, bad_epochs, history = float("inf"), 0, []
    checkpoint = args.out / "best_lstm_multi.pt"
    for epoch in range(1, args.epochs + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        train_mse = train_one_epoch(model, loaders["train"], optimizer)
        val_mse = evaluate_mse(model, loaders["validation"])
        scheduler.step(val_mse)
        if val_mse < best:
            best, bad_epochs = val_mse, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "scaler_mean": list(scaler.mean), "scaler_scale": list(scaler.scale),
                        "channels": columns, "hidden_size": args.hidden_size,
                        "lookback": LOOKBACK, "epoch": epoch, "validation_mse": best,
                        "seed": args.seed}, checkpoint)
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse,
                        "learning_rate": lr_used})
        print(f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")
        if bad_epochs >= args.patience:
            break
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    scaler = ChannelScaler(tuple(saved["scaler_mean"]), tuple(saved["scaler_scale"]))
    test_index = datasets["test"].targets
    actual = raw[test_index]
    forecast = scaler.inverse(predict(model, loaders["test"]))
    persistence = raw[test_index - 1]
    pieces = [pd.DataFrame({"timestamp": wide.index[test_index], "cell": cell,
                            "actual": actual[:, i], "lstm": forecast[:, i],
                            "persistence": persistence[:, i]})
              for i, cell in enumerate(columns)]
    pd.concat(pieces, ignore_index=True).to_csv(args.out / "test_predictions.csv", index=False)
    per_cell = []
    for i, cell in enumerate(columns):
        per_cell.append({"cell": cell,
                         **{f"lstm_{key}": value for key, value in calculate_metrics(actual[:, i], forecast[:, i]).items()},
                         **{f"persistence_{key}": value for key, value in calculate_metrics(actual[:, i], persistence[:, i]).items()}})
    pd.DataFrame(per_cell).to_csv(args.out / "per_cell_metrics.csv", index=False)
    scores = macro_metrics(actual, forecast)
    baseline = macro_metrics(actual, persistence)
    write_json(args.out / "metrics.json", {
        "target_channels": columns, "selected_epoch": saved["epoch"],
        "selected_validation_mse": saved["validation_mse"],
        "units": "original CSV traffic units",
        "test_protocol": "rolling observed-history 12-to-30",
        "lstm": scores, "persistence": baseline, "interval": FREQUENCY})
    print(f"Joint LSTM  macro MAE={scores['macro_mae']:.1f} pooled MAE={scores['pooled_mae']:.1f}")
    print(f"Persistence macro MAE={baseline['macro_mae']:.1f}")
    print(f"Outputs saved locally to: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, FloatingPointError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

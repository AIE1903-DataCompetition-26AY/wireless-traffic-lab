#!/usr/bin/env python3
"""Classroom wireless-traffic forecasting: one node, 12 observed steps -> 1.

Based on the instructor-supplied LSTM classroom example, with the model and
training settings preserved. Not a verified reproduction of a research paper.
See docs/METHODS.md and docs/CHANGELOG.md.

Expected local CSV: a `timestamp` column and a nonnegative traffic column.
No network access, cloud logging, or data upload is used by this script.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import (LOOKBACK, FREQUENCY, Scaler, load_series, split_ranges,
                    valid_target_indices, calculate_metrics, write_json,
                    ensure_new_output, data_fingerprint, environment_versions)


# 3. All 12 inputs AND the next target must be inside one time split.
class WindowDataset(Dataset):
    def __init__(self, normalized: np.ndarray, start: int, stop: int):
        self.values = normalized
        self.targets = valid_target_indices(normalized, start, stop)
        if len(self.targets) == 0:
            raise ValueError("A split has no complete 12-to-1 windows; inspect missing data.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = int(self.targets[index])
        # Input [12,1], label [1]. DataLoader adds the batch dimension.
        x = self.values[j - LOOKBACK:j, None].copy()
        y = self.values[j:j + 1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


# 4. A one-layer LSTM + linear prediction head. No hidden state is passed across windows.
class TrafficLSTM(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(xb)  # default h0=c0=0 for this batch
        return self.head(sequence[:, -1, :])


# 5. One pass over training batches = one epoch.
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> float:
    model.train()
    device = next(model.parameters()).device
    squared_error, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = nn.functional.mse_loss(prediction, yb)
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
        prediction = model(xb)
        squared_error += float(nn.functional.mse_loss(prediction, yb, reduction="sum").cpu())
        count += yb.numel()
    value = squared_error / count
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite validation MSE.")
    return value


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(xb.to(device)).cpu().numpy().ravel()
                           for xb, _ in loader])


def save_checkpoint(model: nn.Module, scaler: Scaler, path: Path,
                    metadata: dict[str, Any]) -> None:
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "scaler": asdict(scaler), **metadata}, path)


@torch.no_grad()
def forecast_next(model: nn.Module, scaler: Scaler, last12: np.ndarray) -> float:
    last12 = np.asarray(last12, dtype=np.float64)
    if last12.shape != (LOOKBACK,) or not np.isfinite(last12).all() or np.any(last12 < 0):
        raise ValueError("Next-step inference requires exactly 12 valid raw traffic values.")
    device = next(model.parameters()).device
    xb = torch.tensor(scaler.transform(last12)).reshape(1, LOOKBACK, 1).to(device)
    model.eval()
    return float(scaler.inverse(model(xb).cpu().numpy()).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True, help="One selected cell/grid traffic column.")
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--out", type=Path, default=Path("runs/traffic_lstm"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mape-threshold", type=float, default=None,
                        help="Optional nonnegative threshold fixed before inspecting test results.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size, args.patience) <= 0:
        parser.error("Epochs, batch size, hidden size and patience must be positive.")
    if not np.isfinite([args.lr, args.weight_decay]).all() or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative.")
    if args.mape_threshold is not None and (not np.isfinite(args.mape_threshold) or args.mape_threshold < 0):
        parser.error("MAPE threshold must be nonnegative.")
    try:
        ensure_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))

    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")
    device = torch.device("cuda" if (args.device == "cuda" or
                          args.device == "auto" and torch.cuda.is_available()) else "cpu")

    series = load_series(args.csv, args.target, args.time_column)
    raw = series.to_numpy(dtype=np.float64)
    splits = split_ranges(len(raw))
    scaler = Scaler.fit(raw[:splits["train"][1]])  # ONLY training time points
    normalized = scaler.transform(raw)
    datasets = {name: WindowDataset(normalized, start, stop)
                for name, (start, stop) in splits.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(data, batch_size=args.batch_size, shuffle=(name == "train"),
                               generator=generator if name == "train" else None,
                               num_workers=0, drop_last=False)
               for name, data in datasets.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": "lstm", "data_sha256": data_fingerprint(series),
        "environment": environment_versions()})
    summary = {name: {"start_index_0based": start, "stop_index_exclusive": stop,
                      "start_timestamp": str(series.index[start]),
                      "end_timestamp": str(series.index[stop - 1]),
                      "measurements": stop - start, "valid_windows": len(datasets[name]),
                      "skipped_windows": stop - start - LOOKBACK - len(datasets[name])}
               for name, (start, stop) in splits.items()}
    write_json(args.out / "split_summary.json", {
        "target": args.target, "total_time_points": len(raw), "missing_values": int(np.isnan(raw).sum()),
        "lookback": LOOKBACK, "horizon": 1, "interval": FREQUENCY,
        "boundary_policy": "all inputs and target within their split", "scaler": asdict(scaler),
        "splits": summary})

    model = TrafficLSTM(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best, bad_epochs, history = float("inf"), 0, []
    checkpoint = args.out / "best_lstm.pt"
    for epoch in range(1, args.epochs + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        train_mse = train_one_epoch(model, loaders["train"], optimizer)
        val_mse = evaluate_mse(model, loaders["validation"])
        scheduler.step(val_mse)
        if val_mse < best:
            best, bad_epochs = val_mse, 0
            save_checkpoint(model, scaler, checkpoint, {
                "hidden_size": args.hidden_size, "lookback": LOOKBACK, "horizon": 1,
                "target": args.target, "interval": FREQUENCY, "epoch": epoch,
                "validation_mse": best, "seed": args.seed,
                "training_end_timestamp": str(series.index[splits["train"][1] - 1])})
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse,
                        "learning_rate": lr_used})
        print(f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")
        if bad_epochs >= args.patience:
            break
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)

    # The test targets have not been used by training, scheduler, or checkpoint selection.
    # Load ONLY checkpoints created by this script (never an untrusted checkpoint).
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    scaler = Scaler(**saved["scaler"])
    test_indices = datasets["test"].targets
    truth = raw[test_indices]
    forecast = scaler.inverse(predict(model, loaders["test"]))
    persistence = raw[test_indices - 1]
    predictions = pd.DataFrame({"timestamp": series.index[test_indices], "actual": truth,
                                "lstm": forecast, "persistence": persistence})
    predictions.to_csv(args.out / "test_predictions.csv", index=False)
    write_json(args.out / "metrics.json", {
        "target": args.target, "selected_epoch": saved["epoch"],
        "selected_validation_mse": saved["validation_mse"],
        "units": "original CSV traffic units", "test_protocol": "rolling observed-history 12-to-1",
        "lstm": calculate_metrics(truth, forecast, args.mape_threshold),
        "persistence": calculate_metrics(truth, persistence, args.mape_threshold)})
    if np.isfinite(raw[-LOOKBACK:]).all():
        write_json(args.out / "next_forecast.json", {
            "forecast_timestamp": str(series.index[-1] + pd.Timedelta(FREQUENCY)),
            "prediction": forecast_next(model, scaler, raw[-LOOKBACK:]), "target": args.target})
    scores = calculate_metrics(truth, forecast, args.mape_threshold)
    baseline = calculate_metrics(truth, persistence, args.mape_threshold)
    print(f"LSTM        MAE={scores['mae']:.6f} RMSE={scores['rmse']:.6f} n={scores['count']}")
    print(f"Persistence MAE={baseline['mae']:.6f} RMSE={baseline['rmse']:.6f}")
    print(f"Outputs saved locally to: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, FloatingPointError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

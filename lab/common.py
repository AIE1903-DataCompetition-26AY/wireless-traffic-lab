"""两种模型共用的数据协议；本模块不依赖 PyTorch。

固定任务：一个节点，5 分钟等间隔；时间轴按 60%/20%/20% 切分。
实际 CSV 单位保持不变。缺失值不填零，也不使用未来数据插值。
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOOKBACK = 12
FREQUENCY = "5min"
MAX_TIME_POINTS = 5_000_000  # 防止错误时间戳导致重建一个过大的稠密时间轴。


@dataclass(frozen=True)
class Scaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, raw_train: np.ndarray) -> "Scaler":
        observed = np.asarray(raw_train, dtype=np.float64)
        observed = observed[np.isfinite(observed)]
        if observed.size < 2 or np.any(observed < 0):
            raise ValueError("Need at least two nonnegative observed training values.")
        logged = np.log1p(observed)
        return cls(float(logged.mean()), max(float(logged.std(ddof=0)), 1e-8))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        # float32 与原始 LSTM 示例一致；NaN 保持为 NaN。
        return ((np.log1p(raw) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        with np.errstate(over="raise", invalid="raise"):
            raw = np.expm1(np.asarray(normalized, dtype=np.float64) * self.scale + self.mean)
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite forecast after inverse transform.")
        return np.maximum(raw, 0.0)


def load_series(csv: Path, target: str, time_column: str = "timestamp") -> pd.Series:
    """读取已聚合的宽表，只加载时间列与指定节点列。"""
    if target == time_column:
        raise ValueError("Target and time column must be different.")
    if not csv.is_file():
        raise ValueError(f"CSV not found: {csv}. Check --csv and your working directory.")
    header = pd.read_csv(csv, nrows=0, encoding="utf-8-sig")
    missing = {time_column, target}.difference(header.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}. "
                         f"Available columns: {list(header.columns)}")
    frame = pd.read_csv(csv, usecols=[time_column, target], encoding="utf-8-sig")
    if pd.api.types.is_numeric_dtype(frame[time_column]):
        raise ValueError("Use readable datetime strings, not an ambiguous numeric timestamp. "
                         "Convert Unix timestamps with the correct unit before running.")
    times = pd.to_datetime(frame[time_column], errors="raise")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Timestamps must be present and unique. Resolve duplicates first.")
    if not pd.api.types.is_datetime64_any_dtype(times.dtype):
        raise ValueError("Use one consistent timezone; mixed timezones are unsupported.")
    values = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=np.float64)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Traffic must be nonnegative and finite; missing values may be NaN.")
    series = pd.Series(values, index=pd.DatetimeIndex(times), name=target).sort_index()
    if len(series) < 2:
        raise ValueError("CSV contains too few observations.")
    step = pd.Timedelta(FREQUENCY)
    if np.any((series.index - series.index[0]).asi8 % step.value != 0):
        raise ValueError("Timestamps are not aligned on a common 5-minute grid.")
    grid_length = int((series.index[-1] - series.index[0]) // step) + 1
    if grid_length > MAX_TIME_POINTS:
        raise ValueError("Reindexed timeline would exceed 5,000,000 points. Check timestamps "
                         "or ask the instructor to provide a smaller teaching subset.")
    index = pd.date_range(series.index[0], series.index[-1], freq=FREQUENCY)
    return series.reindex(index)


def split_ranges(length: int) -> dict[str, tuple[int, int]]:
    n_train, n_val = math.floor(0.6 * length), math.floor(0.2 * length)
    splits = {"train": (0, n_train), "validation": (n_train, n_train + n_val),
              "test": (n_train + n_val, length)}
    if any(stop - start <= LOOKBACK for start, stop in splits.values()):
        raise ValueError("Each time split must contain at least 13 measurements.")
    return splits


def valid_target_indices(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    """仅计分：同一 split 内，12 个输入点 + 1 个目标点全部有效的目标。"""
    if not 0 <= start < stop <= len(values):
        raise ValueError("Invalid split boundaries.")
    candidates = np.arange(start + LOOKBACK, stop, dtype=np.int64)
    missing_prefix = np.concatenate(([0], np.cumsum(~np.isfinite(values))))
    targets = candidates[missing_prefix[candidates + 1] - missing_prefix[candidates - LOOKBACK] == 0]
    if targets.size == 0:
        raise ValueError("A split has no complete 12-to-1 windows; inspect missing data.")
    return targets


def calculate_metrics(truth: np.ndarray, forecast: np.ndarray,
                      mape_threshold: float | None = None) -> dict[str, Any]:
    truth, forecast = np.asarray(truth, dtype=np.float64), np.asarray(forecast, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != forecast.shape or not truth.size:
        raise ValueError("Metrics require equally sized, nonempty 1D arrays.")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
        raise ValueError("Metrics cannot contain missing or infinite values.")
    if mape_threshold is not None and (not np.isfinite(mape_threshold) or mape_threshold < 0):
        raise ValueError("MAPE threshold must be finite and nonnegative.")
    residual = forecast - truth
    with np.errstate(over="raise", invalid="raise"):
        result = {"count": int(len(truth)), "mae": float(np.mean(np.abs(residual))),
                  "rmse": float(np.sqrt(np.mean(residual ** 2)))}
    if mape_threshold is not None:
        mask = (truth > 0) & (truth >= mape_threshold)
        result.update({"mape_threshold": mape_threshold, "masked_mape_count": int(mask.sum()),
                       "masked_mape_percent": float(100 * np.mean(np.abs(residual[mask]) / truth[mask]))
                       if mask.any() else None})
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def ensure_new_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory must be new or empty. Use a different --out path.")


def data_fingerprint(series: pd.Series) -> str:
    """所选节点+时间轴的指纹；帮助识别模型是否用了不同数据。"""
    canonical = series.to_frame().to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%S%z",
                                       float_format="%.17g", na_rep="NaN", lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def environment_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "torch", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def split_summary(series: pd.Series) -> dict[str, Any]:
    raw = series.to_numpy(dtype=np.float64)
    result = {}
    for name, (start, stop) in split_ranges(len(raw)).items():
        targets = valid_target_indices(raw, start, stop)
        result[name] = {"start_index_0based": start, "stop_index_exclusive": stop,
                        "start_timestamp": str(series.index[start]),
                        "end_timestamp": str(series.index[stop - 1]),
                        "measurements": stop - start,
                        "missing_values": int(np.isnan(raw[start:stop]).sum()),
                        "valid_windows": int(len(targets)),
                        "skipped_windows": int(stop - start - LOOKBACK - len(targets))}
    return result

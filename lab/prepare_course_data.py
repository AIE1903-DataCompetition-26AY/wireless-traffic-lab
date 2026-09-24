#!/usr/bin/env python3
"""从已下载的竞赛训练 CSV 提取一个节点，供两种课堂模型共用。

只处理明确指定的 CSV 宽表；不下载、不上传、不猜测单位或竞赛评测规则。
默认先找唯一一个文件名带 train 的 CSV，不会把 test / submission 当训练集。
输出 timestamp,traffic；原始节点列名、文件哈希和可取得的 Git 版本另存 JSON。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from common import load_series, split_summary, write_json, data_fingerprint

COURSE_REPOSITORY = "https://github.com/AIE1903-DataCompetition-26AY/competition-data"
LEVELS = ("cell_30", "cell_217", "grid_273")
TIME_NAMES = {"timestamp", "datetime", "date_time", "time", "date", "时间", "时间戳"}
METADATA_NAMES = {"id", "index", "cell_id", "grid_id", "node_id"}
EXCLUDED_WORDS = ("test", "submission", "sample", "baseline", "template", "membership")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def level_directory(data_root: Path, level: str) -> Path:
    if level not in LEVELS:
        raise ValueError(f"Choose a level from {LEVELS}.")
    directory = (data_root / level).resolve()
    if not directory.is_dir():
        raise ValueError(f"Dataset folder not found: {directory}. "
                         "Clone the data repository first; --data-root is its root, not a level folder.")
    return directory


def csv_files(directory: Path) -> list[Path]:
    # 只列出真正位于所选数据目录中的文件，不跟随指向目录外的文件链接。
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.resolve().is_relative_to(directory.resolve())
                  and path.name.lower().endswith((".csv", ".csv.gz")))


def is_excluded(path: Path, directory: Path) -> bool:
    name = path.relative_to(directory).as_posix().lower()
    tokens = re.split(r"[^a-z]+", name)
    return any(token.startswith(word) for token in tokens for word in EXCLUDED_WORDS)


def read_preview(path: Path, rows: int = 200) -> pd.DataFrame:
    if path.name.lower().endswith(".csv"):
        with path.open("rb") as source:
            if source.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{path.name} is a Git LFS pointer, not the data file. "
                                 "Ask the instructor for the supported data download procedure.")
    try:
        return pd.read_csv(path, nrows=rows, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {path.name} as a UTF-8 comma-separated CSV: {exc}") from exc


def inspect_data(directory: Path) -> None:
    paths = sorted(path for path in directory.rglob("*") if path.is_file()
                   and path.resolve().is_relative_to(directory.resolve()))
    if not paths:
        raise ValueError(f"No files in {directory}.")
    csv_paths = set(csv_files(directory))
    print(f"Folder: {directory}")
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        print(f"\n{relative} ({path.stat().st_size:,} bytes)")
        if path in csv_paths:
            role = "excluded from automatic training selection" if is_excluded(path, directory) else "check role in course PDF"
            print(f"  Role: {role}")
            try:
                frame = read_preview(path, rows=0)
                print(f"  Columns ({len(frame.columns)}): {list(frame.columns)}")
            except ValueError as exc:
                print(f"  CHECK: {exc}")
    print("\nInspection does not verify units, train/test semantics, or competition rules.")
    print("Read the dataset PDF. This helper supports CSV wide tables only.")


def choose_source(directory: Path, relative_file: str | None = None) -> Path:
    if relative_file is not None:
        path = (directory / relative_file).resolve()
        if not path.is_relative_to(directory.resolve()) or path not in [p.resolve() for p in csv_files(directory)]:
            raise ValueError("--file must name an existing CSV inside the selected level folder.")
        if is_excluded(path, directory):
            raise ValueError("Refusing a test / submission / baseline / sample / membership file as training data.")
        return path
    candidates = [path for path in csv_files(directory)
                  if not is_excluded(path, directory) and "train" in path.name.lower()]
    if len(candidates) != 1:
        names = [str(path.relative_to(directory)) for path in candidates]
        raise ValueError(f"Expected exactly one training CSV; found {len(candidates)}: {names}. "
                         "Run --inspect, read the course PDF, then use --file with the actual training filename. "
                         "Compressed archives / Excel / long tables need an instructor-approved converter.")
    return candidates[0]


def choose_columns(preview: pd.DataFrame, time_column: str | None,
                   target: str | None) -> tuple[str, str]:
    if time_column is None:
        matches = [column for column in preview.columns if column.strip().lower() in TIME_NAMES]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely identify the time column: {matches}. "
                             "Use --time-column with the actual column name.")
        time_column = matches[0]
    if time_column not in preview.columns:
        raise ValueError(f"Time column not found: {time_column}.")
    if target is not None:
        if target not in preview.columns or target == time_column:
            raise ValueError("--target must name an existing traffic column, not the time column.")
        return time_column, target
    # 默认仅为上手选 CSV 顺序中第一列有效数值，不按模型表现挑选节点。
    for column in preview.columns:
        name = column.strip().lower()
        if column == time_column or name in METADATA_NAMES or name.startswith("unnamed:"):
            continue
        try:
            values = pd.to_numeric(preview[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError):
            continue
        observed = values[np.isfinite(values)]
        if len(observed) >= 2 and not np.isinf(values).any() and np.all(observed >= 0):
            return time_column, column
    raise ValueError("No observed numeric traffic column found in the first 200 rows. "
                     "Do not use the empty test template. For late-starting series use --target explicitly.")


def local_git_version(data_root: Path) -> dict[str, Any]:
    # 下载 ZIP 没有 Git 元数据；不要意外记录其上级项目的提交。
    if not (data_root / ".git").exists():
        return {"commit": None, "working_tree_dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(data_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "-C", str(data_root), "status", "--porcelain"],
                                capture_output=True, text=True, check=True, timeout=10).stdout
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError("Unexpected Git commit format.")
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"commit": None, "working_tree_dirty": None}


def prepare(data_root: Path, level: str, output: Path, relative_file: str | None = None,
            time_column: str | None = None, target: str | None = None) -> dict[str, Any]:
    root = data_root.resolve()
    directory = level_directory(root, level)
    source = choose_source(directory, relative_file)
    output = output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.suffix.lower() != ".csv":
        raise ValueError("--out must end in .csv.")
    if output.is_relative_to(root):
        raise ValueError("Write the classroom copy outside the original data repository.")
    if output.exists() or metadata_path.exists():
        raise ValueError("Output CSV or metadata already exists. Choose a new --out; files are not overwritten.")
    time_column, target = choose_columns(read_preview(source), time_column, target)
    # 使用训练入口相同的严格验证，不填零、不插值、不改变流量单位。
    series = load_series(source, target, time_column)
    summary = split_summary(series)  # 在写文件前验证三个本地 split 都有有效窗口。
    git_version = local_git_version(root)
    metadata = {
        "purpose": "single-series classroom backtest, NOT an official competition submission",
        "course_repository_reference": COURSE_REPOSITORY,
        "repository_origin_verified": False,
        "level": level, "source_relative_path": str(source.relative_to(root)),
        "source_file_sha256": file_sha256(source), "source_git": git_version,
        "source_time_column": time_column, "source_target_column": target,
        "output_time_column": "timestamp", "output_target_column": "traffic",
        "units": "unchanged; consult the course dataset PDF",
        "processing": "sort time; reindex to 5 minutes; missing remains NaN; no scaling or interpolation",
        "split_protocol": "60/20/20 LOCAL chronological split of the PUBLIC TRAINING file only",
        "split_summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    series.rename("traffic").rename_axis("timestamp").to_csv(output, encoding="utf-8", na_rep="")
    metadata["selected_series_sha256"] = data_fingerprint(load_series(output, "traffic"))
    metadata["output_file_sha256"] = file_sha256(output)
    write_json(metadata_path, metadata)
    print(f"Source: {source}\nOriginal time column: {time_column}\nOriginal target: {target}")
    print(f"Classroom CSV: {output}\nMetadata: {metadata_path}")
    print(f"Use BOTH models with --csv {output} --target traffic")
    print("Only the selected training series was exported; no official test/template was merged.")
    print("Verify the chosen node and units against the course PDF before class.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../competition-data"))
    parser.add_argument("--level", choices=LEVELS, default="cell_30")
    parser.add_argument("--inspect", action="store_true", help="List local files and CSV columns only; write nothing.")
    parser.add_argument("--file", help="Actual training CSV path relative to the selected level folder.")
    parser.add_argument("--time-column", help="Original time column; infer only when unambiguous.")
    parser.add_argument("--target", help="Original node column; default is first observed numeric column in file order.")
    parser.add_argument("--out", type=Path, default=Path("data/course_traffic.csv"))
    args = parser.parse_args()
    if args.inspect:
        inspect_data(level_directory(args.data_root, args.level))
    else:
        prepare(args.data_root, args.level, args.out, args.file, args.time_column, args.target)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

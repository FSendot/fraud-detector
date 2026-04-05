#!/usr/bin/env python3
"""Preprocess a raw fraud transaction CSV into deterministic interim outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import DEFAULT_PATHS_FILE, load_paths_config, project_root  # noqa: E402
from data.preprocess import default_output_paths, preprocess_raw_transactions  # noqa: E402


def _resolve_input_path(raw_candidate: Path) -> Path:
    paths = load_paths_config(DEFAULT_PATHS_FILE)
    raw_dir = paths["raw_data"].resolve()
    if raw_candidate.is_absolute():
        resolved = raw_candidate.resolve()
    else:
        project_candidate = (project_root() / raw_candidate).resolve()
        resolved = project_candidate if project_candidate.exists() else (raw_dir / raw_candidate).resolve()

    try:
        resolved.relative_to(raw_dir)
    except ValueError as exc:
        msg = f"input CSV must live under {raw_dir}"
        raise ValueError(msg) from exc

    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _resolve_output_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess one or more raw CSV files into data/interim/transactions_clean.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        action="append",
        help="Raw CSV file or dataset directory in data/raw/. Repeat for multiple files when needed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional parquet output path. Defaults to configs/paths.yaml setting.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to configs/paths.yaml setting.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_paths = [_resolve_input_path(candidate.expanduser()) for candidate in args.input]
    default_parquet_path, default_report_path = default_output_paths()
    output_path = _resolve_output_path(args.output, default_parquet_path)
    report_path = _resolve_output_path(args.report, default_report_path)

    result = preprocess_raw_transactions(
        input_path=input_paths,
        output_parquet_path=output_path,
        output_report_path=report_path,
    )
    print(
        f"rows_in={result.rows_in} rows_out={result.rows_out} "
        f"duplicates={result.duplicate_rows_dropped} invalid={result.invalid_rows_dropped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

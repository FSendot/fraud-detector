#!/usr/bin/env python3
"""Build deterministic chronological split ID tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from data.splits import DEFAULT_SPLITS_FILE, default_output_paths, make_and_write_splits  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic chronological train/valid/test split ID tables.",
    )
    parser.add_argument("--behavioral-input", type=Path, default=None, help="Optional behavioral feature parquet path.")
    parser.add_argument("--sequence-meta-input", type=Path, default=None, help="Optional sequence meta parquet path.")
    parser.add_argument("--train-ids", type=Path, default=None, help="Optional output path for train_ids.parquet.")
    parser.add_argument("--valid-ids", type=Path, default=None, help="Optional output path for valid_ids.parquet.")
    parser.add_argument("--test-ids", type=Path, default=None, help="Optional output path for test_ids.parquet.")
    parser.add_argument("--report", type=Path, default=None, help="Optional output path for split_report.json.")
    parser.add_argument("--config", type=Path, default=DEFAULT_SPLITS_FILE, help="Split configuration YAML path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    (
        default_behavioral_input,
        default_sequence_meta_input,
        default_train_ids,
        default_valid_ids,
        default_test_ids,
        default_report,
    ) = default_output_paths()

    result = make_and_write_splits(
        behavioral_input_path=_resolve_path(args.behavioral_input, default_behavioral_input),
        sequence_meta_input_path=_resolve_path(args.sequence_meta_input, default_sequence_meta_input),
        train_ids_path=_resolve_path(args.train_ids, default_train_ids),
        valid_ids_path=_resolve_path(args.valid_ids, default_valid_ids),
        test_ids_path=_resolve_path(args.test_ids, default_test_ids),
        report_path=_resolve_path(args.report, default_report),
        split_config_path=_resolve_path(args.config, DEFAULT_SPLITS_FILE),
    )
    print(
        f"mode={result.split_mode} "
        f"order_column={result.order_column} "
        f"train={result.train_rows} "
        f"valid={result.valid_rows} "
        f"test={result.test_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


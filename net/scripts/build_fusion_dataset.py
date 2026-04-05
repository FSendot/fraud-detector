#!/usr/bin/env python3
"""Build fusion-ready validation and test datasets from branch predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from fusion.build_dataset import (  # noqa: E402
    build_and_write_fusion_datasets,
    default_fusion_paths,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fusion-ready validation/test datasets from branch predictions.",
    )
    parser.add_argument("--fusion-valid-output", type=Path, default=None, help="Optional output path for fusion_valid.parquet.")
    parser.add_argument("--fusion-test-output", type=Path, default=None, help="Optional output path for fusion_test.parquet.")
    parser.add_argument("--report-output", type=Path, default=None, help="Optional output path for fusion_dataset_report.json.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    branch_paths, default_valid, default_test, default_report = default_fusion_paths()
    result = build_and_write_fusion_datasets(
        branch_prediction_paths=branch_paths,
        fusion_valid_path=_resolve_path(args.fusion_valid_output, default_valid),
        fusion_test_path=_resolve_path(args.fusion_test_output, default_test),
        report_path=_resolve_path(args.report_output, default_report),
    )
    print(
        f"branches={result.branch_count} "
        f"valid_rows={result.valid_rows} "
        f"test_rows={result.test_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

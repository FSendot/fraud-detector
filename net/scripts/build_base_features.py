#!/usr/bin/env python3
"""Build base per-transaction features from the cleaned interim parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from features.base_features import (  # noqa: E402
    build_and_write_base_features,
    default_output_paths,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic base features from data/interim/transactions_clean.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional cleaned interim parquet path. Defaults to configs/paths.yaml setting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional base feature parquet path. Defaults to configs/paths.yaml setting.",
    )
    parser.add_argument(
        "--feature-dict",
        type=Path,
        default=None,
        help="Optional feature dictionary JSON path. Defaults to configs/paths.yaml setting.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_input, default_output, default_feature_dict = default_output_paths()
    input_path = _resolve_path(args.input, default_input)
    output_path = _resolve_path(args.output, default_output)
    feature_dict_path = _resolve_path(args.feature_dict, default_feature_dict)

    result = build_and_write_base_features(
        input_path=input_path,
        output_parquet_path=output_path,
        feature_dict_path=feature_dict_path,
    )
    print(
        f"rows_out={result.rows_out} "
        f"source_features={result.source_feature_count} "
        f"derived_features={result.derived_feature_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


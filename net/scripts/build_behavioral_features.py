#!/usr/bin/env python3
"""Build historical behavioral features from the base feature parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from features.behavioral_features import (  # noqa: E402
    build_and_write_behavioral_features,
    default_output_paths,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe historical behavioral features from base_features.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional base feature parquet path. Defaults to configs/paths.yaml setting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional behavioral feature parquet path. Defaults to configs/paths.yaml setting.",
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

    result = build_and_write_behavioral_features(
        input_path=input_path,
        output_parquet_path=output_path,
        feature_dict_path=feature_dict_path,
    )
    print(
        f"rows_out={result.rows_out} "
        f"source_features={result.source_feature_count} "
        f"derived_features={result.derived_feature_count} "
        f"entity={result.entity_column}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


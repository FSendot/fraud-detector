#!/usr/bin/env python3
"""Build GRU sequence datasets from behavioral features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from sequences.builder import build_and_write_sequence_dataset, default_output_paths  # noqa: E402
from sequences.schema import DEFAULT_SEQUENCE_LENGTH  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic GRU sequence datasets from behavioral_features.parquet.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional behavioral feature parquet path.")
    parser.add_argument("--x-seq", type=Path, default=None, help="Optional output path for X_seq.npy.")
    parser.add_argument("--x-current", type=Path, default=None, help="Optional output path for X_current.parquet.")
    parser.add_argument("--y", type=Path, default=None, help="Optional output path for y.parquet.")
    parser.add_argument("--meta", type=Path, default=None, help="Optional output path for meta.parquet.")
    parser.add_argument("--schema", type=Path, default=None, help="Optional output path for sequence_schema.json.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Fixed number of prior transactions kept in each sequence window.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_input, default_x_seq, default_x_current, default_y, default_meta, default_schema = default_output_paths()

    result = build_and_write_sequence_dataset(
        input_path=_resolve_path(args.input, default_input),
        x_seq_path=_resolve_path(args.x_seq, default_x_seq),
        x_current_path=_resolve_path(args.x_current, default_x_current),
        y_path=_resolve_path(args.y, default_y),
        meta_path=_resolve_path(args.meta, default_meta),
        schema_path=_resolve_path(args.schema, default_schema),
        sequence_length=args.sequence_length,
    )
    print(
        f"samples={result.sample_count} "
        f"sequence_length={result.sequence_length} "
        f"sequence_features={result.sequence_feature_count} "
        f"current_features={result.current_feature_count} "
        f"entity={result.entity_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run batch inference from a packaged fraud model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from inference.batch_inference import run_batch_inference  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run packaged fraud-model batch inference on a parquet or CSV input dataset.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Bundle directory or manifest path. Defaults to artifacts/bundles/model_v1.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet or CSV dataset. Can be contract-aligned model input or a wider feature dataset.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Optional parquet output path for predictions.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON output path for batch summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()
    bundle_path = _resolve_path(args.bundle, paths["artifact_bundles_dir"] / "model_v1")
    predictions_output = _resolve_path(args.predictions_output, paths["outputs_batch_scoring_predictions"])
    summary_output = _resolve_path(args.summary_output, paths["outputs_batch_scoring_summary"])
    input_path = _resolve_path(args.input, project_root() / args.input if not args.input.is_absolute() else args.input)

    summary = run_batch_inference(
        bundle_path=bundle_path,
        input_path=input_path,
        predictions_output_path=predictions_output,
        summary_output_path=summary_output,
    )
    print(
        f"bundle={summary['bundle_version']} "
        f"rows_scored={summary['rows_scored']} "
        f"source_stage={summary['source_stage']} "
        f"positive_predictions={summary['score_summary']['positive_predictions']} "
        f"predictions={predictions_output} "
        f"summary={summary_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

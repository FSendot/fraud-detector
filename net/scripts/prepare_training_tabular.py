#!/usr/bin/env python3
"""Prepare tabular model inputs from behavioral features and canonical split IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from training.preprocessing import prepare_and_write_tabular_datasets  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare tabular train/valid/test inputs with train-only preprocessing.",
    )
    parser.add_argument("--behavioral-input", type=Path, default=None, help="Optional behavioral feature parquet path.")
    parser.add_argument("--train-ids", type=Path, default=None, help="Optional train split IDs parquet path.")
    parser.add_argument("--valid-ids", type=Path, default=None, help="Optional valid split IDs parquet path.")
    parser.add_argument("--test-ids", type=Path, default=None, help="Optional test split IDs parquet path.")
    parser.add_argument("--train-output", type=Path, default=None, help="Optional output path for train_tabular.parquet.")
    parser.add_argument("--valid-output", type=Path, default=None, help="Optional output path for valid_tabular.parquet.")
    parser.add_argument("--test-output", type=Path, default=None, help="Optional output path for test_tabular.parquet.")
    parser.add_argument("--scaler-output", type=Path, default=None, help="Optional output path for scaler.joblib.")
    parser.add_argument("--selector-output", type=Path, default=None, help="Optional output path for feature_selector.joblib.")
    parser.add_argument("--selected-features-output", type=Path, default=None, help="Optional output path for selected_features.json.")
    parser.add_argument("--report-output", type=Path, default=None, help="Optional output path for training_prep_report.json.")
    parser.add_argument(
        "--downsample-ratio",
        type=float,
        default=10.0,
        help="Maximum negative-to-positive ratio kept in the training set after deterministic downsampling.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Optional number of features to keep with SelectKBest. Defaults to all numeric features.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()

    result = prepare_and_write_tabular_datasets(
        behavioral_input_path=_resolve_path(args.behavioral_input, paths["processed_behavioral_features"]),
        train_ids_path=_resolve_path(args.train_ids, paths["split_train_ids"]),
        valid_ids_path=_resolve_path(args.valid_ids, paths["split_valid_ids"]),
        test_ids_path=_resolve_path(args.test_ids, paths["split_test_ids"]),
        train_output_path=_resolve_path(args.train_output, paths["model_input_train_tabular"]),
        valid_output_path=_resolve_path(args.valid_output, paths["model_input_valid_tabular"]),
        test_output_path=_resolve_path(args.test_output, paths["model_input_test_tabular"]),
        scaler_path=_resolve_path(args.scaler_output, paths["artifact_scaler"]),
        feature_selector_path=_resolve_path(args.selector_output, paths["artifact_feature_selector"]),
        selected_features_path=_resolve_path(args.selected_features_output, paths["artifact_selected_features"]),
        report_path=_resolve_path(args.report_output, paths["training_prep_report"]),
        downsample_ratio=args.downsample_ratio,
        top_k_features=args.top_k_features,
    )
    print(
        f"train={result.train_rows} "
        f"valid={result.valid_rows} "
        f"test={result.test_rows} "
        f"selected_features={result.selected_feature_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

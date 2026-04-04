#!/usr/bin/env python3
"""Evaluate the first fraud-model branch end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from data.splits import load_split_ids  # noqa: E402
from eval.branch_usefulness import (  # noqa: E402
    build_usefulness_report,
    evaluate_prediction_frame,
    write_usefulness_json,
    write_usefulness_markdown,
)
from eval.error_analysis import build_error_analysis_tables, write_error_analysis_tables  # noqa: E402
from eval.leakage_checks import run_leakage_checks  # noqa: E402
from training.train_utils import (  # noqa: E402
    classifier_prediction_frame,
    load_model_input,
    split_model_input,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the first fraud-model branch and write usefulness reports.",
    )
    parser.add_argument("--train-latent", type=Path, default=None, help="Optional latent train parquet path.")
    parser.add_argument("--valid-latent", type=Path, default=None, help="Optional latent valid parquet path.")
    parser.add_argument("--test-latent", type=Path, default=None, help="Optional latent test parquet path.")
    parser.add_argument("--train-tabular", type=Path, default=None, help="Optional train_tabular.parquet path for leakage checks.")
    parser.add_argument("--train-ids", type=Path, default=None, help="Optional train split IDs parquet path.")
    parser.add_argument("--valid-ids", type=Path, default=None, help="Optional valid split IDs parquet path.")
    parser.add_argument("--test-ids", type=Path, default=None, help="Optional test split IDs parquet path.")
    parser.add_argument("--model-input", type=Path, default=None, help="Optional Nyström model.joblib path.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional usefulness_report.json output path.")
    parser.add_argument("--report-md", type=Path, default=None, help="Optional usefulness_report.md output path.")
    parser.add_argument("--false-positives-output", type=Path, default=None, help="Optional false positives parquet output path.")
    parser.add_argument("--false-negatives-output", type=Path, default=None, help="Optional false negatives parquet output path.")
    parser.add_argument("--top-n-errors", type=int, default=50, help="Number of top false positives and negatives to save.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()

    train_latent = load_model_input(_resolve_path(args.train_latent, paths["prediction_vae_latent_train"]))
    valid_latent = load_model_input(_resolve_path(args.valid_latent, paths["prediction_vae_latent_valid"]))
    test_latent = load_model_input(_resolve_path(args.test_latent, paths["prediction_vae_latent_test"]))
    train_tabular = load_model_input(_resolve_path(args.train_tabular, paths["model_input_train_tabular"]))

    train_ids = load_split_ids(_resolve_path(args.train_ids, paths["split_train_ids"]))
    valid_ids = load_split_ids(_resolve_path(args.valid_ids, paths["split_valid_ids"]))
    test_ids = load_split_ids(_resolve_path(args.test_ids, paths["split_test_ids"]))

    model = joblib.load(_resolve_path(args.model_input, paths["artifact_nystrom_gp_model"]))

    train_txn_ids, train_labels, train_features = split_model_input(train_latent)
    valid_txn_ids, valid_labels, valid_features = split_model_input(valid_latent)
    test_txn_ids, test_labels, test_features = split_model_input(test_latent)

    train_prob = model.predict_proba(train_features)[:, 1]
    valid_prob = model.predict_proba(valid_features)[:, 1]
    test_prob = model.predict_proba(test_features)[:, 1]

    train_predictions = classifier_prediction_frame(
        transaction_ids=train_txn_ids,
        labels=train_labels,
        probabilities=train_prob,
    )
    valid_predictions = classifier_prediction_frame(
        transaction_ids=valid_txn_ids,
        labels=valid_labels,
        probabilities=valid_prob,
    )
    test_predictions = classifier_prediction_frame(
        transaction_ids=test_txn_ids,
        labels=test_labels,
        probabilities=test_prob,
    )

    metrics_by_split = {
        "train": evaluate_prediction_frame(train_predictions),
        "valid": evaluate_prediction_frame(valid_predictions),
        "test": evaluate_prediction_frame(test_predictions),
    }

    leakage_warnings = run_leakage_checks(
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
        train_features=train_tabular,
        metrics_by_split=metrics_by_split,
    )

    false_positives, false_negatives = build_error_analysis_tables(
        test_predictions,
        top_n=args.top_n_errors,
    )
    false_positive_path = _resolve_path(args.false_positives_output, paths["reports_first_branch_false_positives"])
    false_negative_path = _resolve_path(args.false_negatives_output, paths["reports_first_branch_false_negatives"])
    write_error_analysis_tables(
        false_positives=false_positives,
        false_negatives=false_negatives,
        false_positives_path=false_positive_path,
        false_negatives_path=false_negative_path,
    )

    report = build_usefulness_report(
        metrics_by_split=metrics_by_split,
        leakage_warnings=leakage_warnings,
        false_positives_count=len(false_positives),
        false_negatives_count=len(false_negatives),
        input_paths={
            "train_latent": str(_resolve_path(args.train_latent, paths["prediction_vae_latent_train"])),
            "valid_latent": str(_resolve_path(args.valid_latent, paths["prediction_vae_latent_valid"])),
            "test_latent": str(_resolve_path(args.test_latent, paths["prediction_vae_latent_test"])),
            "nystrom_model": str(_resolve_path(args.model_input, paths["artifact_nystrom_gp_model"])),
        },
    )

    report_json_path = _resolve_path(args.report_json, paths["reports_first_branch_usefulness_json"])
    report_md_path = _resolve_path(args.report_md, paths["reports_first_branch_usefulness_md"])
    write_usefulness_json(report_json_path, report)
    write_usefulness_markdown(report_md_path, report)

    print(
        f"valid_f1={report['summary']['valid_f1']:.6f} "
        f"test_f1={report['summary']['test_f1']:.6f} "
        f"useful={report['useful']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


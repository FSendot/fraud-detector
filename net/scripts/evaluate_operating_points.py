#!/usr/bin/env python3
"""Evaluate business-oriented operating points for the current fused score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from eval.business_thresholds import (  # noqa: E402
    BudgetConfig,
    business_report_table,
    evaluate_operating_points,
    load_prediction_frame,
    write_business_json,
    write_business_markdown,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _load_config(path: Path | None) -> BudgetConfig:
    base = BudgetConfig()
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    alerts_per_1k = payload.get("alerts_per_1k", list(base.alerts_per_1k))
    precision_targets = payload.get("precision_targets", list(base.precision_targets))
    if not isinstance(alerts_per_1k, list) or not isinstance(precision_targets, list):
        msg = "alerts_per_1k and precision_targets must be JSON arrays"
        raise ValueError(msg)
    return BudgetConfig(
        alerts_per_1k=tuple(float(value) for value in alerts_per_1k),
        precision_targets=tuple(float(value) for value in precision_targets),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate alert-budget and precision-target operating points from validation and test scores.",
    )
    parser.add_argument("--valid-input", type=Path, default=None, help="Optional validation prediction parquet.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional test prediction parquet.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config for business thresholds.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional output path for business_threshold_report.json.")
    parser.add_argument("--report-md", type=Path, default=None, help="Optional output path for business_threshold_report.md.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional output path for business_threshold_summary.parquet.")
    parser.add_argument(
        "--score-source-name",
        type=str,
        default="fused_raw",
        help="Short label recorded in the output report.",
    )
    parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Use fused_valid_calibrated/fused_test_calibrated instead of raw fused scores.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()

    default_valid = paths["prediction_fused_valid_calibrated"] if args.use_calibrated else paths["prediction_fused_valid"]
    default_test = paths["prediction_fused_test_calibrated"] if args.use_calibrated else paths["prediction_fused_test"]
    valid_input = _resolve_path(args.valid_input, default_valid)
    test_input = _resolve_path(args.test_input, default_test)
    report_json = _resolve_path(args.report_json, paths["business_threshold_report_json"])
    report_md = _resolve_path(args.report_md, paths["business_threshold_report_md"])
    summary_output = _resolve_path(args.summary_output, paths["business_threshold_summary"])

    config = _load_config(args.config_input)
    valid_frame = load_prediction_frame(valid_input)
    test_frame = load_prediction_frame(test_input)

    report = evaluate_operating_points(valid_frame, test_frame, config=config)
    payload = {
        "score_source": args.score_source_name,
        "input_paths": {
            "valid": str(valid_input),
            "test": str(test_input),
        },
        **report,
    }
    write_business_json(report_json, payload)
    write_business_markdown(report_md, payload, score_source_name=args.score_source_name)
    business_report_table(payload).to_parquet(summary_output, index=False)

    best_budget = payload["recommended_budget_point"]
    print(
        f"score_source={args.score_source_name} "
        f"threshold={best_budget['threshold']:.6f} "
        f"alerts_per_1k={best_budget['test']['alerts_per_1k']:.2f} "
        f"test_precision={best_budget['test']['precision']:.6f} "
        f"test_recall={best_budget['test']['recall']:.6f} "
        f"test_f1={best_budget['test']['f1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

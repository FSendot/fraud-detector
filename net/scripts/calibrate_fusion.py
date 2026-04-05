#!/usr/bin/env python3
"""Calibrate the fused fraud score on validation only and export calibrated predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration.calibrate import (  # noqa: E402
    CalibrationConfig,
    compare_calibrators,
    load_fusion_predictions,
    prediction_frame_with_scores,
)
from common.config import load_paths_config, project_root  # noqa: E402
from eval.branch_usefulness import evaluate_prediction_frame, resolve_threshold_selection  # noqa: E402
from training.train_utils import ensure_parent, write_json  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _load_config(path: Path | None) -> CalibrationConfig:
    base = CalibrationConfig()
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    methods_payload = payload.get("methods", list(base.methods))
    if not isinstance(methods_payload, list):
        msg = "methods must be a JSON array"
        raise ValueError(msg)
    return CalibrationConfig(
        methods=tuple(str(method) for method in methods_payload),
        selection_metric=str(payload.get("selection_metric", base.selection_metric)),
        threshold=float(payload.get("threshold", base.threshold)),
        threshold_strategy=str(payload.get("threshold_strategy", base.threshold_strategy)),
        threshold_candidate_count=int(payload.get("threshold_candidate_count", base.threshold_candidate_count)),
        random_state=int(payload.get("random_state", base.random_state)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit validation-only calibration for fused fraud scores.",
    )
    parser.add_argument("--valid-input", type=Path, default=None, help="Optional fused_valid.parquet path.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional fused_test.parquet path.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config override.")
    parser.add_argument("--calibrator-output", type=Path, default=None, help="Optional output path for calibrator.joblib.")
    parser.add_argument("--config-output", type=Path, default=None, help="Optional output path for config.json.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional output path for metrics.json.")
    parser.add_argument("--valid-predictions-output", type=Path, default=None, help="Optional output path for fused_valid_calibrated.parquet.")
    parser.add_argument("--test-predictions-output", type=Path, default=None, help="Optional output path for fused_test_calibrated.parquet.")
    parser.add_argument("--report-output", type=Path, default=None, help="Optional output path for calibration_report.json.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()

    valid_input = _resolve_path(args.valid_input, paths["prediction_fused_valid"])
    test_input = _resolve_path(args.test_input, paths["prediction_fused_test"])
    calibrator_output = _resolve_path(args.calibrator_output, paths["artifact_calibration_calibrator"])
    config_output = _resolve_path(args.config_output, paths["artifact_calibration_config"])
    metrics_output = _resolve_path(args.metrics_output, paths["artifact_calibration_metrics"])
    valid_predictions_output = _resolve_path(args.valid_predictions_output, paths["prediction_fused_valid_calibrated"])
    test_predictions_output = _resolve_path(args.test_predictions_output, paths["prediction_fused_test_calibrated"])
    report_output = _resolve_path(args.report_output, paths["calibration_report"])

    valid_frame = load_fusion_predictions(valid_input)
    test_frame = load_fusion_predictions(test_input)
    config = _load_config(args.config_input)

    selected_method, calibrator, comparison = compare_calibrators(valid_frame, config=config)

    raw_valid_scores = valid_frame["score"].astype(float).to_numpy()
    raw_test_scores = test_frame["score"].astype(float).to_numpy()
    calibrated_valid_scores = calibrator.predict_proba(raw_valid_scores)
    calibrated_test_scores = calibrator.predict_proba(raw_test_scores)
    threshold_selection = resolve_threshold_selection(
        valid_frame["is_fraud"],
        calibrated_valid_scores,
        strategy=config.threshold_strategy,
        fixed_threshold=config.threshold,
        candidate_count=config.threshold_candidate_count,
    )
    decision_threshold = float(threshold_selection["threshold"])

    calibrated_valid = prediction_frame_with_scores(
        valid_frame,
        calibrated_scores=calibrated_valid_scores,
        threshold=decision_threshold,
    )
    calibrated_test = prediction_frame_with_scores(
        test_frame,
        calibrated_scores=calibrated_test_scores,
        threshold=decision_threshold,
    )

    valid_metrics = evaluate_prediction_frame(calibrated_valid, threshold=decision_threshold)
    test_metrics = evaluate_prediction_frame(calibrated_test, threshold=decision_threshold)

    ensure_parent(calibrator_output)
    joblib.dump(
        {
            "selected_method": selected_method,
            "calibrator": calibrator,
            "config": config.to_payload(),
        },
        calibrator_output,
    )
    write_json(ensure_parent(config_output), config.to_payload())
    metrics_payload = {
        "branch_name": "fusion_calibration",
        "selected_method": selected_method,
        "config": config.to_payload(),
        "comparison": comparison,
        "threshold_selection": threshold_selection,
        "effective_threshold": decision_threshold,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    write_json(ensure_parent(metrics_output), metrics_payload)
    write_json(ensure_parent(report_output), metrics_payload)
    ensure_parent(valid_predictions_output)
    ensure_parent(test_predictions_output)
    calibrated_valid.to_parquet(valid_predictions_output, index=False)
    calibrated_test.to_parquet(test_predictions_output, index=False)

    print(
        f"selected_method={selected_method} "
        f"threshold={decision_threshold:.6f} "
        f"valid_brier={valid_metrics['brier_score']:.6f} "
        f"test_brier={test_metrics['brier_score']:.6f} "
        f"valid_f1={valid_metrics['f1']:.6f} "
        f"test_f1={test_metrics['f1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

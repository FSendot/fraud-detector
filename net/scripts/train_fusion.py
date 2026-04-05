#!/usr/bin/env python3
"""Train and score the ensemble fusion stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from eval.branch_usefulness import evaluate_prediction_frame, resolve_threshold_selection  # noqa: E402
from fusion.fusion_model import FusionConfig, fused_scores, labels, prediction_frame  # noqa: E402
from training.train_utils import binary_classification_metrics, ensure_parent, write_json  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _load_config(path: Path | None) -> FusionConfig:
    base = FusionConfig()
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weights = payload.get("weighted_average_weights")
    if weights is not None and not isinstance(weights, dict):
        msg = "weighted_average_weights must be a JSON object when provided"
        raise ValueError(msg)
    auto_candidates = payload.get("auto_select_candidates", list(base.auto_select_candidates))
    if not isinstance(auto_candidates, list):
        msg = "auto_select_candidates must be a JSON array when provided"
        raise ValueError(msg)
    return FusionConfig(
        mode=str(payload.get("mode", base.mode)),
        threshold=float(payload.get("threshold", base.threshold)),
        threshold_strategy=str(payload.get("threshold_strategy", base.threshold_strategy)),
        threshold_candidate_count=int(payload.get("threshold_candidate_count", base.threshold_candidate_count)),
        logistic_c=float(payload.get("logistic_c", base.logistic_c)),
        logistic_max_iter=int(payload.get("logistic_max_iter", base.logistic_max_iter)),
        random_state=int(payload.get("random_state", base.random_state)),
        weighted_average_weights={str(key): float(value) for key, value in (weights or {}).items()} or None,
        bayesian_alpha_prior=float(payload.get("bayesian_alpha_prior", base.bayesian_alpha_prior)),
        bayesian_beta_prior=float(payload.get("bayesian_beta_prior", base.bayesian_beta_prior)),
        auto_select_metric=str(payload.get("auto_select_metric", base.auto_select_metric)),
        auto_select_complexity_penalty=float(
            payload.get("auto_select_complexity_penalty", base.auto_select_complexity_penalty)
        ),
        auto_select_min_valid_average_precision=float(
            payload.get(
                "auto_select_min_valid_average_precision",
                base.auto_select_min_valid_average_precision,
            )
        ),
        auto_select_max_branches=int(payload.get("auto_select_max_branches", base.auto_select_max_branches)),
        auto_select_candidates=tuple(str(candidate) for candidate in auto_candidates),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ensemble fusion on fusion_valid.parquet and score validation/test.",
    )
    parser.add_argument("--valid-input", type=Path, default=None, help="Optional fusion_valid.parquet path.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional fusion_test.parquet path.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config override for fusion.")
    parser.add_argument("--config-output", type=Path, default=None, help="Optional output path for saved config.json.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional output path for metrics.json.")
    parser.add_argument("--valid-predictions-output", type=Path, default=None, help="Optional output path for fused_valid.parquet.")
    parser.add_argument("--test-predictions-output", type=Path, default=None, help="Optional output path for fused_test.parquet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()

    valid_input = _resolve_path(args.valid_input, paths["fusion_valid"])
    test_input = _resolve_path(args.test_input, paths["fusion_test"])
    config_output = _resolve_path(args.config_output, paths["artifact_fusion_config"])
    metrics_output = _resolve_path(args.metrics_output, paths["artifact_fusion_metrics"])
    valid_predictions_output = _resolve_path(args.valid_predictions_output, paths["prediction_fused_valid"])
    test_predictions_output = _resolve_path(args.test_predictions_output, paths["prediction_fused_test"])

    valid_frame = pd.read_parquet(valid_input)
    test_frame = pd.read_parquet(test_input)
    config = _load_config(args.config_input)

    valid_scores, fusion_details = fused_scores(valid_frame, valid_frame, config=config)
    test_scores, _ = fused_scores(valid_frame, test_frame, config=config)
    threshold_selection = resolve_threshold_selection(
        labels(valid_frame),
        valid_scores,
        strategy=config.threshold_strategy,
        fixed_threshold=config.threshold,
        candidate_count=config.threshold_candidate_count,
    )
    decision_threshold = float(threshold_selection["threshold"])

    valid_predictions = prediction_frame(valid_frame, fused_scores_array=valid_scores, threshold=decision_threshold)
    test_predictions = prediction_frame(test_frame, fused_scores_array=test_scores, threshold=decision_threshold)

    valid_metrics = binary_classification_metrics(labels(valid_frame), valid_scores)
    test_metrics = binary_classification_metrics(labels(test_frame), test_scores)
    valid_threshold_metrics = evaluate_prediction_frame(valid_predictions, threshold=decision_threshold)
    test_threshold_metrics = evaluate_prediction_frame(test_predictions, threshold=decision_threshold)

    write_json(ensure_parent(config_output), config.to_payload())
    write_json(
        ensure_parent(metrics_output),
        {
            "branch_name": "fusion",
            "config": config.to_payload(),
            "fusion_details": fusion_details,
            "threshold_selection": threshold_selection,
            "effective_threshold": decision_threshold,
            "valid": valid_metrics,
            "test": test_metrics,
            "valid_threshold_metrics": valid_threshold_metrics,
            "test_threshold_metrics": test_threshold_metrics,
            "row_counts": {
                "valid": int(len(valid_frame)),
                "test": int(len(test_frame)),
            },
        },
    )
    ensure_parent(valid_predictions_output)
    ensure_parent(test_predictions_output)
    valid_predictions.to_parquet(valid_predictions_output, index=False)
    test_predictions.to_parquet(test_predictions_output, index=False)

    print(
        f"mode={config.mode} "
        f"threshold={decision_threshold:.6f} "
        f"valid_ap={valid_metrics['average_precision']:.6f} "
        f"test_ap={test_metrics['average_precision']:.6f} "
        f"valid_f1={valid_threshold_metrics['f1']:.6f} "
        f"test_f1={test_threshold_metrics['f1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

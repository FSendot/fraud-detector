#!/usr/bin/env python3
"""Train a Nyström-kernel classifier on exported VAE embeddings."""

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

from common.config import project_root  # noqa: E402
from models.nystrom_gp import NystromGPConfig, fit_nystrom_classifier  # noqa: E402
from training.train_utils import (  # noqa: E402
    binary_classification_metrics,
    classifier_prediction_frame,
    default_nystrom_paths,
    default_nystrom_prediction_paths,
    ensure_parent,
    load_model_input,
    set_global_seed,
    split_model_input,
    write_json,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _load_config(path: Path | None) -> NystromGPConfig:
    base = NystromGPConfig()
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return NystromGPConfig(
        random_state=int(payload.get("random_state", base.random_state)),
        kernel=str(payload.get("kernel", base.kernel)),
        gamma=float(payload.get("gamma", base.gamma)),
        n_components=int(payload.get("n_components", base.n_components)),
        logistic_c=float(payload.get("logistic_c", base.logistic_c)),
        max_iter=int(payload.get("max_iter", base.max_iter)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Nyström classifier on VAE latent embeddings.",
    )
    parser.add_argument("--train-input", type=Path, default=None, help="Optional latent train parquet path.")
    parser.add_argument("--valid-input", type=Path, default=None, help="Optional latent valid parquet path.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional latent test parquet path.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config override for the classifier.")
    parser.add_argument("--model-output", type=Path, default=None, help="Optional output path for model.joblib.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional output path for metrics.json.")
    parser.add_argument("--valid-predictions-output", type=Path, default=None, help="Optional output path for valid predictions parquet.")
    parser.add_argument("--test-predictions-output", type=Path, default=None, help="Optional output path for test predictions parquet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_train, default_valid, default_test, default_model, default_metrics = default_nystrom_paths()
    default_valid_pred, default_test_pred = default_nystrom_prediction_paths()

    train_frame = load_model_input(_resolve_path(args.train_input, default_train))
    valid_frame = load_model_input(_resolve_path(args.valid_input, default_valid))
    test_frame = load_model_input(_resolve_path(args.test_input, default_test))

    train_ids, train_labels, train_features = split_model_input(train_frame)
    valid_ids, valid_labels, valid_features = split_model_input(valid_frame)
    test_ids, test_labels, test_features = split_model_input(test_frame)

    config = _load_config(args.config_input)
    set_global_seed(config.random_state)
    model = fit_nystrom_classifier(train_features, train_labels.astype(int), config=config)

    valid_probabilities = model.predict_proba(valid_features)[:, 1]
    test_probabilities = model.predict_proba(test_features)[:, 1]

    valid_metrics = binary_classification_metrics(valid_labels, valid_probabilities)
    test_metrics = binary_classification_metrics(test_labels, test_probabilities)

    valid_predictions = classifier_prediction_frame(
        transaction_ids=valid_ids,
        labels=valid_labels,
        probabilities=valid_probabilities,
    )
    test_predictions = classifier_prediction_frame(
        transaction_ids=test_ids,
        labels=test_labels,
        probabilities=test_probabilities,
    )

    model_output = _resolve_path(args.model_output, default_model)
    metrics_output = _resolve_path(args.metrics_output, default_metrics)
    valid_predictions_output = _resolve_path(args.valid_predictions_output, default_valid_pred)
    test_predictions_output = _resolve_path(args.test_predictions_output, default_test_pred)

    ensure_parent(model_output)
    joblib.dump(model, model_output)
    write_json(
        ensure_parent(metrics_output),
        {
            "config": config.to_payload(),
            "valid": valid_metrics,
            "test": test_metrics,
            "latent_feature_count": train_features.shape[1],
        },
    )
    ensure_parent(valid_predictions_output)
    ensure_parent(test_predictions_output)
    valid_predictions.to_parquet(valid_predictions_output, index=False)
    test_predictions.to_parquet(test_predictions_output, index=False)

    print(
        f"kernel={config.kernel} "
        f"components={config.n_components} "
        f"valid_ap={valid_metrics['average_precision']:.6f} "
        f"test_ap={test_metrics['average_precision']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

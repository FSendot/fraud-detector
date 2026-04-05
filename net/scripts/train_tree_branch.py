#!/usr/bin/env python3
"""Train the tree-based fraud branch on canonical tabular inputs."""

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
from contracts.feature_contract import load_feature_contract, validate_frame_against_contract  # noqa: E402
from models.tree_branch import TreeBranchConfig, feature_importance_frame, fit_tree_branch  # noqa: E402
from training.train_utils import (  # noqa: E402
    binary_classification_metrics,
    classifier_prediction_frame,
    default_tree_branch_paths,
    default_tree_branch_prediction_paths,
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


def _load_config(path: Path | None) -> TreeBranchConfig:
    base = TreeBranchConfig()
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TreeBranchConfig(
        random_state=int(payload.get("random_state", base.random_state)),
        n_estimators=int(payload.get("n_estimators", base.n_estimators)),
        criterion=str(payload.get("criterion", base.criterion)),
        max_depth=None if payload.get("max_depth", base.max_depth) is None else int(payload["max_depth"]),
        min_samples_split=int(payload.get("min_samples_split", base.min_samples_split)),
        min_samples_leaf=int(payload.get("min_samples_leaf", base.min_samples_leaf)),
        max_features=payload.get("max_features", base.max_features),
        bootstrap=bool(payload.get("bootstrap", base.bootstrap)),
        class_weight=payload.get("class_weight", base.class_weight),
        n_jobs=int(payload.get("n_jobs", base.n_jobs)),
    )


def _assert_contract(frame, contract: dict[str, object], dataset_name: str) -> None:
    result = validate_frame_against_contract(frame, contract)
    if not result.valid:
        msg = f"{dataset_name} does not satisfy feature contract: {result.errors}"
        raise ValueError(msg)


def _align_to_contract(frame, contract: dict[str, object]):
    ordered_columns = contract["dataset_requirements"]["expected_columns_in_order"]
    return frame.loc[:, ordered_columns].copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the tree-based fraud branch on canonical tabular model inputs.",
    )
    parser.add_argument("--train-input", type=Path, default=None, help="Optional prepared train parquet path.")
    parser.add_argument("--valid-input", type=Path, default=None, help="Optional prepared valid parquet path.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional prepared test parquet path.")
    parser.add_argument("--contract-input", type=Path, default=None, help="Optional feature contract JSON path.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config override.")
    parser.add_argument("--model-output", type=Path, default=None, help="Optional output path for model.joblib.")
    parser.add_argument("--config-output", type=Path, default=None, help="Optional output path for config.json.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional output path for metrics.json.")
    parser.add_argument("--valid-predictions-output", type=Path, default=None, help="Optional output path for valid predictions parquet.")
    parser.add_argument("--test-predictions-output", type=Path, default=None, help="Optional output path for test predictions parquet.")
    parser.add_argument("--feature-importances-output", type=Path, default=None, help="Optional output path for feature importances JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_train, default_valid, default_test, default_contract, default_model, default_saved_config, default_metrics = default_tree_branch_paths()
    default_valid_pred, default_test_pred, default_feature_importances = default_tree_branch_prediction_paths()

    contract = load_feature_contract(_resolve_path(args.contract_input, default_contract))
    train_frame = _align_to_contract(load_model_input(_resolve_path(args.train_input, default_train)), contract)
    valid_frame = _align_to_contract(load_model_input(_resolve_path(args.valid_input, default_valid)), contract)
    test_frame = _align_to_contract(load_model_input(_resolve_path(args.test_input, default_test)), contract)

    _assert_contract(train_frame, contract, "train")
    _assert_contract(valid_frame, contract, "valid")
    _assert_contract(test_frame, contract, "test")

    train_ids, train_labels, train_features = split_model_input(train_frame)
    valid_ids, valid_labels, valid_features = split_model_input(valid_frame)
    test_ids, test_labels, test_features = split_model_input(test_frame)

    config = _load_config(args.config_input)
    set_global_seed(config.random_state)
    model = fit_tree_branch(train_features, train_labels.astype(int), config=config)

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
    feature_importances = feature_importance_frame(
        feature_names=list(train_features.columns),
        importances=list(model.feature_importances_),
    )

    model_output = _resolve_path(args.model_output, default_model)
    config_output = _resolve_path(args.config_output, default_saved_config)
    metrics_output = _resolve_path(args.metrics_output, default_metrics)
    valid_predictions_output = _resolve_path(args.valid_predictions_output, default_valid_pred)
    test_predictions_output = _resolve_path(args.test_predictions_output, default_test_pred)
    feature_importances_output = _resolve_path(args.feature_importances_output, default_feature_importances)

    ensure_parent(model_output)
    joblib.dump(model, model_output)
    write_json(ensure_parent(config_output), config.to_payload())
    write_json(
        ensure_parent(metrics_output),
        {
            "branch_name": "tree_branch",
            "contract_name": contract["contract_name"],
            "contract_version": contract["version"],
            "config": config.to_payload(),
            "valid": valid_metrics,
            "test": test_metrics,
            "feature_count": train_features.shape[1],
            "feature_order": list(train_features.columns),
        },
    )
    write_json(
        ensure_parent(feature_importances_output),
        {
            "branch_name": "tree_branch",
            "feature_importances": feature_importances.to_dict(orient="records"),
        },
    )
    ensure_parent(valid_predictions_output)
    ensure_parent(test_predictions_output)
    valid_predictions.to_parquet(valid_predictions_output, index=False)
    test_predictions.to_parquet(test_predictions_output, index=False)

    print(
        f"model=ExtraTreesClassifier "
        f"trees={config.n_estimators} "
        f"valid_ap={valid_metrics['average_precision']:.6f} "
        f"test_ap={test_metrics['average_precision']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

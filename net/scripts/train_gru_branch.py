#!/usr/bin/env python3
"""Train the GRU fraud branch on the prepared sequence dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from models.gru_branch import GRUBranchConfig, GRUFraudClassifier  # noqa: E402
from training.sequence_loader import (  # noqa: E402
    build_sequence_split,
    downsample_training_split,
    load_sequence_resources,
    make_sequence_loader,
    maybe_cap_split,
)
from training.train_utils import (  # noqa: E402
    binary_classification_metrics,
    classifier_prediction_frame,
    default_gru_branch_paths,
    default_gru_branch_prediction_paths,
    ensure_parent,
    set_global_seed,
    write_json,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _load_config(path: Path | None, *, sequence_input_dim: int) -> GRUBranchConfig:
    base = GRUBranchConfig(sequence_input_dim=sequence_input_dim)
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GRUBranchConfig(
        sequence_input_dim=sequence_input_dim,
        hidden_dim=int(payload.get("hidden_dim", base.hidden_dim)),
        gru_layers=int(payload.get("gru_layers", base.gru_layers)),
        classifier_hidden_dim=int(payload.get("classifier_hidden_dim", base.classifier_hidden_dim)),
        dropout=float(payload.get("dropout", base.dropout)),
        learning_rate=float(payload.get("learning_rate", base.learning_rate)),
        weight_decay=float(payload.get("weight_decay", base.weight_decay)),
        batch_size=int(payload.get("batch_size", base.batch_size)),
        epochs=int(payload.get("epochs", base.epochs)),
        patience=int(payload.get("patience", base.patience)),
        downsample_ratio=float(payload.get("downsample_ratio", base.downsample_ratio)),
        max_train_samples=None if payload.get("max_train_samples", base.max_train_samples) is None else int(payload["max_train_samples"]),
        max_valid_monitor_samples=int(payload.get("max_valid_monitor_samples", base.max_valid_monitor_samples)),
        seed=int(payload.get("seed", base.seed)),
        device=str(payload.get("device", base.device)),
        pos_weight=None if payload.get("pos_weight", base.pos_weight) is None else float(payload["pos_weight"]),
    )


def _loss_function(config: GRUBranchConfig, train_labels: np.ndarray, *, device: torch.device) -> torch.nn.Module:
    if config.pos_weight is not None:
        pos_weight_value = config.pos_weight
    else:
        positives = float((train_labels == 1).sum())
        negatives = float((train_labels == 0).sum())
        pos_weight_value = 1.0 if positives == 0 else max(1.0, negatives / positives)
    return torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )


def _evaluate_loss(
    model: GRUFraudClassifier,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for sequence_batch, label_batch in loader:
            sequence_batch = sequence_batch.to(device)
            label_batch = label_batch.to(device)
            loss = loss_fn(model(sequence_batch), label_batch)
            total_loss += float(loss.item()) * len(label_batch)
            total_rows += len(label_batch)
    return total_loss / max(total_rows, 1)


def _predict_probabilities(
    model: GRUFraudClassifier,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for sequence_batch, _ in loader:
            sequence_batch = sequence_batch.to(device)
            probabilities = torch.sigmoid(model(sequence_batch)).cpu().numpy()
            outputs.append(probabilities.astype(np.float64, copy=False))
    if not outputs:
        return np.array([], dtype=np.float64)
    return np.concatenate(outputs, axis=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the GRU fraud branch on prepared sequences.",
    )
    parser.add_argument("--x-seq-input", type=Path, default=None, help="Optional X_seq.npy input path.")
    parser.add_argument("--meta-input", type=Path, default=None, help="Optional meta.parquet input path.")
    parser.add_argument("--y-input", type=Path, default=None, help="Optional y.parquet input path.")
    parser.add_argument("--schema-input", type=Path, default=None, help="Optional sequence schema JSON path.")
    parser.add_argument("--train-ids", type=Path, default=None, help="Optional train split IDs parquet path.")
    parser.add_argument("--valid-ids", type=Path, default=None, help="Optional valid split IDs parquet path.")
    parser.add_argument("--test-ids", type=Path, default=None, help="Optional test split IDs parquet path.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config override.")
    parser.add_argument("--weights-output", type=Path, default=None, help="Optional output path for weights.pt.")
    parser.add_argument("--config-output", type=Path, default=None, help="Optional output path for config.json.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional output path for metrics.json.")
    parser.add_argument("--valid-predictions-output", type=Path, default=None, help="Optional output path for valid predictions parquet.")
    parser.add_argument("--test-predictions-output", type=Path, default=None, help="Optional output path for test predictions parquet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    (
        default_x_seq,
        default_meta,
        default_y,
        default_schema,
        default_train_ids,
        default_valid_ids,
        default_test_ids,
        default_weights,
        default_config,
        default_metrics,
    ) = default_gru_branch_paths()
    default_valid_pred, default_test_pred = default_gru_branch_prediction_paths()

    resources = load_sequence_resources(
        x_seq_path=_resolve_path(args.x_seq_input, default_x_seq),
        meta_path=_resolve_path(args.meta_input, default_meta),
        y_path=_resolve_path(args.y_input, default_y),
        schema_path=_resolve_path(args.schema_input, default_schema),
    )
    config = _load_config(args.config_input, sequence_input_dim=len(resources.usable_feature_columns))
    set_global_seed(config.seed)
    requested_device = config.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    config = replace(config, device=requested_device)
    device = torch.device(config.device)

    train_split = build_sequence_split(resources, _resolve_path(args.train_ids, default_train_ids))
    valid_split = build_sequence_split(resources, _resolve_path(args.valid_ids, default_valid_ids))
    test_split = build_sequence_split(resources, _resolve_path(args.test_ids, default_test_ids))

    train_split = downsample_training_split(train_split, downsample_ratio=config.downsample_ratio)
    train_split = maybe_cap_split(train_split, max_samples=config.max_train_samples)
    valid_monitor_split = maybe_cap_split(valid_split, max_samples=config.max_valid_monitor_samples)

    model = GRUFraudClassifier(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = _loss_function(config, train_split.labels.astype(int).to_numpy(), device=device)

    train_loader = make_sequence_loader(resources, train_split, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    valid_monitor_loader = make_sequence_loader(resources, valid_monitor_split, batch_size=config.batch_size, shuffle=False, seed=config.seed)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_valid_ap = -math.inf
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for sequence_batch, label_batch in train_loader:
            sequence_batch = sequence_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(sequence_batch), label_batch)
            if not torch.isfinite(loss):
                msg = f"encountered non-finite GRU training loss at epoch {epoch + 1}"
                raise ValueError(msg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(label_batch)
            total_rows += len(label_batch)

        monitor_probabilities = _predict_probabilities(model, valid_monitor_loader, device=device)
        monitor_metrics = binary_classification_metrics(valid_monitor_split.labels, monitor_probabilities)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_rows, 1),
                "valid_monitor_loss": _evaluate_loss(model, valid_monitor_loader, device=device, loss_fn=loss_fn),
                "valid_monitor_average_precision": monitor_metrics["average_precision"],
            }
        )

        if monitor_metrics["average_precision"] > best_valid_ap:
            best_valid_ap = monitor_metrics["average_precision"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)

    valid_loader = make_sequence_loader(resources, valid_split, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    test_loader = make_sequence_loader(resources, test_split, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    valid_probabilities = _predict_probabilities(model, valid_loader, device=device)
    test_probabilities = _predict_probabilities(model, test_loader, device=device)

    valid_metrics = binary_classification_metrics(valid_split.labels, valid_probabilities)
    valid_metrics["loss"] = _evaluate_loss(model, valid_loader, device=device, loss_fn=loss_fn)
    test_metrics = binary_classification_metrics(test_split.labels, test_probabilities)
    test_metrics["loss"] = _evaluate_loss(model, test_loader, device=device, loss_fn=loss_fn)

    valid_predictions = classifier_prediction_frame(
        transaction_ids=valid_split.transaction_ids,
        labels=valid_split.labels,
        probabilities=valid_probabilities,
    )
    test_predictions = classifier_prediction_frame(
        transaction_ids=test_split.transaction_ids,
        labels=test_split.labels,
        probabilities=test_probabilities,
    )

    weights_output = _resolve_path(args.weights_output, default_weights)
    config_output = _resolve_path(args.config_output, default_config)
    metrics_output = _resolve_path(args.metrics_output, default_metrics)
    valid_predictions_output = _resolve_path(args.valid_predictions_output, default_valid_pred)
    test_predictions_output = _resolve_path(args.test_predictions_output, default_test_pred)

    ensure_parent(weights_output)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_payload(),
            "sequence_feature_columns": list(resources.usable_feature_columns),
            "sequence_length": resources.schema.sequence_length,
            "transaction_id_column": resources.schema.transaction_id_column,
            "target_column": resources.schema.target_columns[0],
        },
        weights_output,
    )
    write_json(ensure_parent(config_output), config.to_payload())
    write_json(
        ensure_parent(metrics_output),
        {
            "branch_name": "gru_branch",
            "config": config.to_payload(),
            "sequence_schema": {
                "dataset_name": resources.schema.dataset_name,
                "entity_key": resources.schema.entity_key,
                "sequence_length": resources.schema.sequence_length,
                "padding_strategy": resources.schema.padding_strategy,
                "usable_sequence_feature_columns": list(resources.usable_feature_columns),
                "leakage_prevention": resources.schema.leakage_prevention,
            },
            "split_row_counts": {
                "train_after_downsampling": train_split.row_count,
                "valid": valid_split.row_count,
                "test": test_split.row_count,
            },
            "training": {
                "best_epoch": best_epoch,
                "history": history,
            },
            "valid": valid_metrics,
            "test": test_metrics,
        },
    )
    ensure_parent(valid_predictions_output)
    ensure_parent(test_predictions_output)
    valid_predictions.to_parquet(valid_predictions_output, index=False)
    test_predictions.to_parquet(test_predictions_output, index=False)

    print(
        f"model=GRU "
        f"best_epoch={best_epoch} "
        f"valid_ap={valid_metrics['average_precision']:.6f} "
        f"test_ap={test_metrics['average_precision']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

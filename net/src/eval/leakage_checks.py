"""Leakage and sanity checks for fraud-model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.splits import validate_zero_overlap


def check_split_overlap(
    train_ids: pd.DataFrame,
    valid_ids: pd.DataFrame,
    test_ids: pd.DataFrame,
) -> None:
    """Fail loudly if any canonical split overlaps."""

    validate_zero_overlap(train_ids, valid_ids, test_ids)


def suspicious_feature_correlations(
    train_frame: pd.DataFrame,
    *,
    label_column: str = "is_fraud",
    id_column: str = "transaction_id",
    threshold: float = 0.98,
) -> list[dict[str, Any]]:
    """Flag suspiciously high absolute feature-label correlations."""

    warnings: list[dict[str, Any]] = []
    label = pd.to_numeric(train_frame[label_column], errors="coerce")

    for column in train_frame.columns:
        if column in {label_column, id_column}:
            continue
        if not pd.api.types.is_numeric_dtype(train_frame[column]):
            continue
        feature = pd.to_numeric(train_frame[column], errors="coerce")
        if feature.nunique(dropna=True) <= 1:
            continue
        correlation = feature.corr(label)
        if correlation is not None and not np.isnan(correlation) and abs(float(correlation)) >= threshold:
            warnings.append(
                {
                    "type": "suspicious_feature_target_correlation",
                    "feature": column,
                    "absolute_correlation": abs(float(correlation)),
                    "message": f"{column} has unusually high absolute correlation with the target.",
                }
            )
    return sorted(warnings, key=lambda item: item["absolute_correlation"], reverse=True)


def implausibly_high_performance_warnings(
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    f1_threshold: float = 0.995,
    ap_threshold: float = 0.999,
) -> list[dict[str, Any]]:
    """Flag performance patterns that often indicate leakage or evaluation bugs."""

    warnings: list[dict[str, Any]] = []
    for split_name in ("valid", "test"):
        metrics = metrics_by_split.get(split_name, {})
        f1_score = float(metrics.get("f1", 0.0))
        average_precision = float(metrics.get("average_precision", 0.0))
        confusion = metrics.get("confusion_matrix", {})
        fp = int(confusion.get("false_positives", 0))
        fn = int(confusion.get("false_negatives", 0))
        if f1_score >= f1_threshold or average_precision >= ap_threshold:
            warnings.append(
                {
                    "type": "implausibly_high_performance",
                    "split": split_name,
                    "message": f"{split_name} performance is unusually high and should be audited for leakage.",
                    "f1": f1_score,
                    "average_precision": average_precision,
                }
            )
        if fp == 0 and fn == 0 and confusion:
            warnings.append(
                {
                    "type": "perfect_confusion_matrix",
                    "split": split_name,
                    "message": f"{split_name} produced a perfect confusion matrix, which is unusual for fraud detection.",
                }
            )
    return warnings


def run_leakage_checks(
    *,
    train_ids: pd.DataFrame,
    valid_ids: pd.DataFrame,
    test_ids: pd.DataFrame,
    train_features: pd.DataFrame,
    metrics_by_split: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run all configured leakage and sanity checks."""

    check_split_overlap(train_ids, valid_ids, test_ids)
    warnings = suspicious_feature_correlations(train_features)
    warnings.extend(implausibly_high_performance_warnings(metrics_by_split))
    return warnings


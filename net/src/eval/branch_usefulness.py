"""Branch usefulness evaluation and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.train_utils import binary_classification_metrics, write_json


def evaluate_prediction_frame(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute branch metrics for one split."""

    y_true = predictions["is_fraud"].astype(int).to_numpy()
    y_pred = predictions["predicted_label"].astype(int).to_numpy()
    y_score = predictions["score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    probability_true, probability_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=10,
        strategy="quantile",
    )
    calibration_bins = [
        {
            "bin": index + 1,
            "mean_predicted_probability": float(probability_pred[index]),
            "observed_positive_rate": float(probability_true[index]),
        }
        for index in range(len(probability_true))
    ]
    metrics = binary_classification_metrics(pd.Series(y_true), y_score)
    metrics.update(
        {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, y_score)),
            "positive_rate": float(np.mean(y_true)),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "calibration_bins": calibration_bins,
        }
    )
    return metrics


def build_usefulness_report(
    *,
    metrics_by_split: dict[str, dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
    false_positives_count: int,
    false_negatives_count: int,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the machine-readable usefulness report payload."""

    valid_f1 = float(metrics_by_split.get("valid", {}).get("f1", 0.0))
    test_f1 = float(metrics_by_split.get("test", {}).get("f1", 0.0))
    valid_ap = float(metrics_by_split.get("valid", {}).get("average_precision", 0.0))
    test_ap = float(metrics_by_split.get("test", {}).get("average_precision", 0.0))
    useful = bool(test_ap >= 0.1 and test_f1 >= 0.05)
    return {
        "branch_name": "vae_plus_nystrom_gp",
        "useful": useful,
        "summary": {
            "valid_f1": valid_f1,
            "test_f1": test_f1,
            "valid_average_precision": valid_ap,
            "test_average_precision": test_ap,
        },
        "metrics_by_split": metrics_by_split,
        "leakage_warnings": leakage_warnings,
        "error_analysis": {
            "top_false_positives_saved": false_positives_count,
            "top_false_negatives_saved": false_negatives_count,
        },
        "input_paths": input_paths,
    }


def write_usefulness_markdown(path: Path, report: dict[str, Any]) -> None:
    """Render a concise markdown usefulness report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    warnings = report.get("leakage_warnings", [])
    lines = [
        "# First Branch Usefulness Report",
        "",
        f"- Branch: `vae_plus_nystrom_gp`",
        f"- Useful: `{report['useful']}`",
        f"- Valid F1: `{summary['valid_f1']:.4f}`",
        f"- Test F1: `{summary['test_f1']:.4f}`",
        f"- Valid Average Precision: `{summary['valid_average_precision']:.4f}`",
        f"- Test Average Precision: `{summary['test_average_precision']:.4f}`",
        "",
        "## Split Metrics",
    ]
    for split_name in ("train", "valid", "test"):
        metrics = report["metrics_by_split"][split_name]
        confusion = metrics["confusion_matrix"]
        lines.extend(
            [
                f"### {split_name.title()}",
                f"- Precision: `{metrics['precision']:.4f}`",
                f"- Recall: `{metrics['recall']:.4f}`",
                f"- F1: `{metrics['f1']:.4f}`",
                f"- Average Precision: `{metrics['average_precision']:.4f}`",
                f"- Confusion Matrix: `TN={confusion['true_negatives']}, FP={confusion['false_positives']}, FN={confusion['false_negatives']}, TP={confusion['true_positives']}`",
            ]
        )
    lines.extend(["", "## Leakage Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning['message']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Error Analysis",
            f"- Top false positives saved: `{report['error_analysis']['top_false_positives_saved']}`",
            f"- Top false negatives saved: `{report['error_analysis']['top_false_negatives_saved']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_usefulness_json(path: Path, report: dict[str, Any]) -> None:
    """Persist the machine-readable usefulness report."""

    write_json(path, report)


"""Business-oriented operating point evaluation for fraud scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from training.train_utils import write_json


@dataclass(frozen=True)
class BudgetConfig:
    """Alert-budget configuration evaluated on validation and test."""

    alerts_per_1k: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)
    precision_targets: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)


def load_prediction_frame(path: Path) -> pd.DataFrame:
    """Load a prediction frame with required score columns."""

    frame = pd.read_parquet(path)
    required = {"transaction_id", "is_fraud", "score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        msg = f"missing required columns in {path}: {missing}"
        raise ValueError(msg)
    return frame.copy()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _threshold_for_alert_rate(scores: np.ndarray, *, alerts_per_1k: float) -> float:
    """Choose a validation-derived threshold for a target review budget."""

    if scores.size == 0:
        return 1.0
    keep_count = int(np.ceil((alerts_per_1k / 1000.0) * float(scores.size)))
    keep_count = max(1, min(keep_count, int(scores.size)))
    ordered = np.sort(scores.astype(float))
    return float(ordered[-keep_count])


def _metrics_at_threshold(frame: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
    """Compute operating metrics at a fixed score threshold."""

    scores = frame["score"].to_numpy(dtype=float)
    labels = frame["is_fraud"].astype(int).to_numpy()
    predicted = scores >= threshold

    true_positives = int(np.sum((predicted == 1) & (labels == 1)))
    false_positives = int(np.sum((predicted == 1) & (labels == 0)))
    false_negatives = int(np.sum((predicted == 0) & (labels == 1)))
    alerts = int(np.sum(predicted))
    total = int(labels.size)
    positives = int(np.sum(labels))

    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, positives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall) if precision + recall > 0 else 0.0

    return {
        "threshold": float(threshold),
        "rows": total,
        "positives": positives,
        "alerts": alerts,
        "alerts_per_1k": 1000.0 * _safe_ratio(alerts, total),
        "alert_rate": _safe_ratio(alerts, total),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fraud_capture_rate": recall,
    }


def _precision_target_rows(valid_frame: pd.DataFrame, test_frame: pd.DataFrame, *, targets: tuple[float, ...]) -> list[dict[str, Any]]:
    """Find the highest-recall validation threshold meeting each precision target."""

    scores = valid_frame["score"].to_numpy(dtype=float)
    labels = valid_frame["is_fraud"].astype(int).to_numpy()
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision_values = precision[:-1]
    recall_values = recall[:-1]

    rows: list[dict[str, Any]] = []
    for target in targets:
        feasible_indices = np.flatnonzero(precision_values + 1e-12 >= target)
        best_payload: dict[str, Any] | None = None
        if feasible_indices.size > 0:
            best_index = max(
                feasible_indices.tolist(),
                key=lambda index: (float(recall_values[index]), float(thresholds[index])),
            )
            threshold = float(thresholds[best_index])
            best_payload = {
                "threshold": threshold,
                "valid": _metrics_at_threshold(valid_frame, threshold=threshold),
                "test": _metrics_at_threshold(test_frame, threshold=threshold),
            }
        rows.append(
            {
                "target_precision": float(target),
                "feasible": best_payload is not None,
                "threshold": float(best_payload["threshold"]) if best_payload is not None else None,
                "valid": best_payload["valid"] if best_payload is not None else None,
                "test": best_payload["test"] if best_payload is not None else None,
            }
        )
    return rows


def evaluate_operating_points(
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    config: BudgetConfig,
) -> dict[str, Any]:
    """Evaluate validation-derived operating points on validation and test."""

    valid_scores = valid_frame["score"].to_numpy(dtype=float)
    budget_rows: list[dict[str, Any]] = []
    for alerts_per_1k in config.alerts_per_1k:
        threshold = _threshold_for_alert_rate(valid_scores, alerts_per_1k=alerts_per_1k)
        budget_rows.append(
            {
                "target_alerts_per_1k": float(alerts_per_1k),
                "threshold": threshold,
                "valid": _metrics_at_threshold(valid_frame, threshold=threshold),
                "test": _metrics_at_threshold(test_frame, threshold=threshold),
            }
        )

    precision_rows = _precision_target_rows(valid_frame, test_frame, targets=config.precision_targets)

    best_budget = max(
        budget_rows,
        key=lambda row: (
            float(row["valid"]["f1"]),
            float(row["valid"]["precision"]),
            -float(row["target_alerts_per_1k"]),
        ),
    )
    feasible_precision_rows = [row for row in precision_rows if row["feasible"]]
    best_precision = (
        max(
            feasible_precision_rows,
            key=lambda row: (
                float(row["valid"]["recall"]),
                float(row["target_precision"]),
            ),
        )
        if feasible_precision_rows
        else None
    )

    return {
        "config": {
            "alerts_per_1k": [float(value) for value in config.alerts_per_1k],
            "precision_targets": [float(value) for value in config.precision_targets],
        },
        "row_counts": {
            "valid": int(len(valid_frame)),
            "test": int(len(test_frame)),
        },
        "budget_operating_points": budget_rows,
        "precision_target_operating_points": precision_rows,
        "recommended_budget_point": best_budget,
        "recommended_precision_point": best_precision,
    }


def business_report_table(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten budget operating points into a tabular summary."""

    rows: list[dict[str, Any]] = []
    for row in report["budget_operating_points"]:
        rows.append(
            {
                "selection_type": "alert_budget",
                "selection_value": float(row["target_alerts_per_1k"]),
                "threshold": float(row["threshold"]),
                "valid_precision": float(row["valid"]["precision"]),
                "valid_recall": float(row["valid"]["recall"]),
                "valid_f1": float(row["valid"]["f1"]),
                "test_precision": float(row["test"]["precision"]),
                "test_recall": float(row["test"]["recall"]),
                "test_f1": float(row["test"]["f1"]),
                "test_alerts_per_1k": float(row["test"]["alerts_per_1k"]),
            }
        )
    for row in report["precision_target_operating_points"]:
        rows.append(
            {
                "selection_type": "precision_target",
                "selection_value": float(row["target_precision"]),
                "threshold": float(row["threshold"]) if row["threshold"] is not None else np.nan,
                "valid_precision": float(row["valid"]["precision"]) if row["valid"] is not None else np.nan,
                "valid_recall": float(row["valid"]["recall"]) if row["valid"] is not None else np.nan,
                "valid_f1": float(row["valid"]["f1"]) if row["valid"] is not None else np.nan,
                "test_precision": float(row["test"]["precision"]) if row["test"] is not None else np.nan,
                "test_recall": float(row["test"]["recall"]) if row["test"] is not None else np.nan,
                "test_f1": float(row["test"]["f1"]) if row["test"] is not None else np.nan,
                "test_alerts_per_1k": float(row["test"]["alerts_per_1k"]) if row["test"] is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_business_markdown(path: Path, report: dict[str, Any], *, score_source_name: str) -> None:
    """Write a readable markdown summary of business operating points."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Business Threshold Report",
        "",
        f"- Score source: `{score_source_name}`",
        f"- Validation rows: `{report['row_counts']['valid']}`",
        f"- Test rows: `{report['row_counts']['test']}`",
        "",
        "## Recommended Alert-Budget Point",
    ]
    best_budget = report["recommended_budget_point"]
    lines.extend(
        [
            f"- Target alerts per 1k: `{best_budget['target_alerts_per_1k']:.2f}`",
            f"- Threshold: `{best_budget['threshold']:.6f}`",
            f"- Validation: precision=`{best_budget['valid']['precision']:.4f}`, recall=`{best_budget['valid']['recall']:.4f}`, f1=`{best_budget['valid']['f1']:.4f}`",
            f"- Test: precision=`{best_budget['test']['precision']:.4f}`, recall=`{best_budget['test']['recall']:.4f}`, f1=`{best_budget['test']['f1']:.4f}`, alerts_per_1k=`{best_budget['test']['alerts_per_1k']:.2f}`",
            "",
            "## Alert-Budget Grid",
        ]
    )
    for row in report["budget_operating_points"]:
        lines.append(
            f"- `{row['target_alerts_per_1k']:.2f}` alerts/1k: "
            f"test precision=`{row['test']['precision']:.4f}`, "
            f"recall=`{row['test']['recall']:.4f}`, "
            f"f1=`{row['test']['f1']:.4f}`, "
            f"threshold=`{row['threshold']:.6f}`"
        )

    lines.extend(["", "## Precision Targets"])
    for row in report["precision_target_operating_points"]:
        if not row["feasible"]:
            lines.append(f"- Target precision `{row['target_precision']:.2f}`: not achievable on validation.")
            continue
        lines.append(
            f"- Target precision `{row['target_precision']:.2f}`: "
            f"test precision=`{row['test']['precision']:.4f}`, "
            f"recall=`{row['test']['recall']:.4f}`, "
            f"f1=`{row['test']['f1']:.4f}`, "
            f"threshold=`{row['threshold']:.6f}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_business_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist the business operating-point report."""

    write_json(path, payload)

"""Operational drift and data-quality monitoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import DEFAULT_CONFIG_DIR, load_yaml_file


DEFAULT_MONITORING_PATH = DEFAULT_CONFIG_DIR / "monitoring.yaml"
EPSILON = 1e-9


@dataclass(frozen=True)
class ThresholdStatus:
    """Threshold evaluation outcome."""

    value: float
    warning_threshold: float
    critical_threshold: float
    severity: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "severity": self.severity,
        }


def load_monitoring_config(path: Path | None = None) -> dict[str, Any]:
    """Load monitoring configuration as a plain mapping."""

    config_path = DEFAULT_MONITORING_PATH if path is None else path
    payload = load_yaml_file(config_path)
    monitoring = payload.get("monitoring", {})
    if not isinstance(monitoring, dict):
        raise ValueError("expected 'monitoring' mapping in monitoring config")
    return monitoring


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int,
) -> float:
    """Compute PSI using reference-driven histogram bins."""

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_hist / max(ref_hist.sum(), 1), EPSILON)
    cur_pct = np.maximum(cur_hist / max(cur_hist.sum(), 1), EPSILON)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def evaluate_threshold_status(value: float, *, warning_threshold: float, critical_threshold: float) -> ThresholdStatus:
    """Map a metric value to ok/warning/critical."""

    if value >= critical_threshold:
        severity = "critical"
    elif value >= warning_threshold:
        severity = "warning"
    else:
        severity = "ok"
    return ThresholdStatus(
        value=float(value),
        warning_threshold=float(warning_threshold),
        critical_threshold=float(critical_threshold),
        severity=severity,
    )


def score_drift_report(
    reference_scores: dict[str, np.ndarray],
    current_scores: dict[str, np.ndarray],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate drift for configured score fields."""

    score_config = config.get("score_drift", {})
    bins = int(score_config.get("bins", 10))
    fields = [str(name) for name in score_config.get("score_fields", [])]
    payload: dict[str, Any] = {"enabled": bool(score_config.get("enabled", True)), "fields": {}}
    for field in fields:
        psi = population_stability_index(reference_scores[field], current_scores[field], bins=bins)
        payload["fields"][field] = evaluate_threshold_status(
            psi,
            warning_threshold=float(score_config.get("warning_threshold", 0.1)),
            critical_threshold=float(score_config.get("critical_threshold", 0.25)),
        ).to_payload()
    return payload


def feature_drift_report(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate PSI drift for selected numeric features."""

    feature_config = config.get("feature_drift", {})
    bins = int(feature_config.get("bins", 10))
    tracked = [str(name) for name in feature_config.get("tracked_features", [])]
    payload: dict[str, Any] = {"enabled": bool(feature_config.get("enabled", True)), "features": {}}
    for feature in tracked:
        if feature not in reference_frame.columns or feature not in current_frame.columns:
            payload["features"][feature] = {
                "severity": "missing",
                "warning_threshold": float(feature_config.get("warning_threshold", 0.15)),
                "critical_threshold": float(feature_config.get("critical_threshold", 0.3)),
                "value": None,
            }
            continue
        psi = population_stability_index(
            pd.to_numeric(reference_frame[feature], errors="coerce").to_numpy(),
            pd.to_numeric(current_frame[feature], errors="coerce").to_numpy(),
            bins=bins,
        )
        payload["features"][feature] = evaluate_threshold_status(
            psi,
            warning_threshold=float(feature_config.get("warning_threshold", 0.15)),
            critical_threshold=float(feature_config.get("critical_threshold", 0.3)),
        ).to_payload()
    return payload


def latency_report(latencies_ms: np.ndarray, *, config: dict[str, Any]) -> dict[str, Any]:
    """Summarize scoring latencies against configured targets."""

    latency_config = config.get("latency", {})
    values = np.asarray(latencies_ms, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"enabled": bool(latency_config.get("enabled", True)), "samples": 0}
    p50 = float(np.quantile(values, 0.5))
    p95 = float(np.quantile(values, 0.95))
    p99 = float(np.quantile(values, 0.99))
    severity = "critical" if p99 >= float(latency_config.get("critical_p99_ms", 300.0)) else "ok"
    if severity == "ok" and (
        p50 >= float(latency_config.get("target_p50_ms", 25.0))
        or p95 >= float(latency_config.get("target_p95_ms", 75.0))
        or p99 >= float(latency_config.get("target_p99_ms", 150.0))
    ):
        severity = "warning"
    return {
        "enabled": bool(latency_config.get("enabled", True)),
        "samples": int(values.size),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "severity": severity,
    }


def data_quality_report(
    *,
    rows_scored: int,
    contract_mismatch_count: int,
    rebuild_count: int,
    missing_feature_count: int,
    duplicate_transaction_id_count: int,
    null_label_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate operational data-quality ratios for scoring traffic."""

    quality = config.get("data_quality", {})
    denominator = max(rows_scored, 1)
    payload = {
        "enabled": bool(quality.get("enabled", True)),
        "rows_scored": int(rows_scored),
        "contract_mismatch_rate": contract_mismatch_count / denominator,
        "rebuild_rate": rebuild_count / denominator,
        "missing_feature_rate": missing_feature_count / denominator,
        "duplicate_transaction_id_count": int(duplicate_transaction_id_count),
        "null_label_rate": null_label_count / denominator,
    }
    breaches = []
    if payload["contract_mismatch_rate"] > float(quality.get("max_contract_mismatch_rate", 0.005)):
        breaches.append("contract_mismatch_rate")
    if payload["rebuild_rate"] > float(quality.get("max_rebuild_rate", 0.05)):
        breaches.append("rebuild_rate")
    if payload["missing_feature_rate"] > float(quality.get("max_missing_feature_rate", 0.001)):
        breaches.append("missing_feature_rate")
    if payload["duplicate_transaction_id_count"] > int(quality.get("duplicate_transaction_id_threshold", 0)):
        breaches.append("duplicate_transaction_id_count")
    payload["severity"] = "critical" if breaches else "ok"
    payload["breaches"] = breaches
    return payload

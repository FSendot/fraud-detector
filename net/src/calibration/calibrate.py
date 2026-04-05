"""Calibration utilities for fused fraud-model scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from eval.branch_usefulness import evaluate_prediction_frame
from training.train_utils import ID_COLUMN, TARGET_COLUMN


EPSILON = 1e-7


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for post-fusion calibration."""

    methods: tuple[str, ...] = ("platt", "isotonic")
    selection_metric: str = "brier_score"
    threshold: float = 0.5
    threshold_strategy: str = "validation_f1"
    threshold_candidate_count: int = 200
    random_state: int = 7

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlattCalibrator:
    """Thin wrapper around logistic regression for 1D score calibration."""

    model: LogisticRegression

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        features = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(features)[:, 1]


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Thin wrapper around isotonic regression for score calibration."""

    model: IsotonicRegression

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        features = np.asarray(scores, dtype=float)
        return np.clip(self.model.predict(features), 0.0, 1.0)


def load_fusion_predictions(path: str | bytes | Any) -> pd.DataFrame:
    """Load fused prediction parquet and enforce core columns."""

    frame = pd.read_parquet(path)
    required = {ID_COLUMN, TARGET_COLUMN, "score"}
    if not required.issubset(frame.columns):
        msg = f"expected fused prediction columns {sorted(required)} in {path}"
        raise ValueError(msg)
    frame = frame.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0).astype("Int64")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").astype(float)
    return frame


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> float:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        observed = float(y_true[mask].mean())
        predicted = float(y_prob[mask].mean())
        ece += float(mask.mean()) * abs(observed - predicted)
    return float(ece)


def fit_calibrator(
    method: str,
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> PlattCalibrator | IsotonicCalibrator:
    """Fit a supported calibration method on validation scores only."""

    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if method == "platt":
        model = LogisticRegression(random_state=random_state, class_weight="balanced")
        model.fit(x.reshape(-1, 1), y)
        return PlattCalibrator(model=model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(x, y)
        return IsotonicCalibrator(model=model)
    msg = f"unsupported calibration method: {method}"
    raise ValueError(msg)


def prediction_frame_with_scores(
    frame: pd.DataFrame,
    *,
    calibrated_scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build a calibrated prediction frame with traceability columns."""

    probabilities = np.clip(np.asarray(calibrated_scores, dtype=float), 0.0, 1.0)
    return pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].reset_index(drop=True),
            TARGET_COLUMN: frame[TARGET_COLUMN].reset_index(drop=True),
            "score": probabilities,
            "predicted_label": (probabilities >= threshold).astype(int),
        }
    )


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    prediction_frame = pd.DataFrame(
        {
            TARGET_COLUMN: y_true,
            "score": y_prob,
            "predicted_label": (y_prob >= 0.5).astype(int),
        }
    )
    metrics = evaluate_prediction_frame(prediction_frame, threshold=0.5)
    metrics["expected_calibration_error"] = _expected_calibration_error(y_true, y_prob)
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def compare_calibrators(
    valid_frame: pd.DataFrame,
    *,
    config: CalibrationConfig,
) -> tuple[str, PlattCalibrator | IsotonicCalibrator, dict[str, Any]]:
    """Fit candidate calibrators on validation only and choose the best one."""

    raw_scores = valid_frame["score"].astype(float).to_numpy()
    labels = valid_frame[TARGET_COLUMN].astype(int).to_numpy()
    methods = tuple(config.methods)
    if not methods:
        msg = "at least one calibration method must be configured"
        raise ValueError(msg)

    comparison: dict[str, Any] = {}
    fitted_models: dict[str, PlattCalibrator | IsotonicCalibrator] = {}
    for method in methods:
        calibrator = fit_calibrator(
            method,
            scores=raw_scores,
            labels=labels,
            random_state=config.random_state,
        )
        calibrated_scores = calibrator.predict_proba(raw_scores)
        fitted_models[method] = calibrator
        comparison[method] = {
            "validation_metrics": _calibration_metrics(labels, calibrated_scores),
        }

    selection_metric = config.selection_metric
    if selection_metric not in {"brier_score", "log_loss", "expected_calibration_error"}:
        msg = f"unsupported calibration selection metric: {selection_metric}"
        raise ValueError(msg)
    selected_method = min(
        methods,
        key=lambda method: float(comparison[method]["validation_metrics"][selection_metric]),
    )
    return selected_method, fitted_models[selected_method], comparison

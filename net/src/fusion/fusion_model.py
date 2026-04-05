"""Fusion model training and scoring helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fusion.bayesian_weighting import BranchReliability, estimate_branch_reliability, reliability_weighted_scores


ID_COLUMN = "transaction_id"
LABEL_COLUMN = "is_fraud"
PREDICTED_LABEL_SUFFIX = "_predicted_label"
SCORE_SUFFIX = "_score"


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for the fusion stage."""

    mode: str = "logistic_meta"
    threshold: float = 0.5
    logistic_c: float = 1.0
    logistic_max_iter: int = 500
    random_state: int = 7
    weighted_average_weights: dict[str, float] | None = None
    bayesian_alpha_prior: float = 1.0
    bayesian_beta_prior: float = 1.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def score_columns(frame: pd.DataFrame) -> list[str]:
    """Return the branch score columns in stable column order."""

    columns = [column for column in frame.columns if column.endswith(SCORE_SUFFIX)]
    if not columns:
        msg = "fusion dataset does not contain any branch score columns"
        raise ValueError(msg)
    return columns


def branch_names(frame: pd.DataFrame) -> list[str]:
    """Infer branch names from score columns."""

    return [column[: -len(SCORE_SUFFIX)] for column in score_columns(frame)]


def fusion_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the score-only feature frame used by fusion models."""

    return frame.loc[:, score_columns(frame)].astype(float)


def labels(frame: pd.DataFrame) -> pd.Series:
    """Return binary labels from a fusion dataset."""

    return pd.to_numeric(frame[LABEL_COLUMN], errors="coerce").fillna(0).astype("Int64")


def transaction_ids(frame: pd.DataFrame) -> pd.Series:
    """Return transaction IDs from a fusion dataset."""

    return frame[ID_COLUMN].astype("string")


def weighted_average_scores(frame: pd.DataFrame, *, weights: dict[str, float] | None) -> np.ndarray:
    """Fuse branch scores with a deterministic weighted average."""

    feature_frame = fusion_feature_frame(frame)
    names = branch_names(frame)
    if weights is None:
        raw_weights = np.ones(len(names), dtype=float)
    else:
        raw_weights = np.asarray([float(weights.get(name, 0.0)) for name in names], dtype=float)
    if np.any(raw_weights < 0):
        msg = "weighted average fusion weights must be non-negative"
        raise ValueError(msg)
    if np.allclose(raw_weights.sum(), 0.0):
        msg = "weighted average fusion weights must not sum to zero"
        raise ValueError(msg)
    normalized = raw_weights / raw_weights.sum()
    return feature_frame.to_numpy(dtype=float) @ normalized


def train_logistic_meta_model(frame: pd.DataFrame, *, config: FusionConfig) -> LogisticRegression:
    """Fit a logistic stacking model on validation fusion features."""

    model = LogisticRegression(
        C=config.logistic_c,
        max_iter=config.logistic_max_iter,
        random_state=config.random_state,
        class_weight="balanced",
    )
    model.fit(fusion_feature_frame(frame), labels(frame).astype(int))
    return model


def reliability_fusion_artifacts(frame: pd.DataFrame, *, config: FusionConfig) -> dict[str, Any]:
    """Compute validation-derived reliability artifacts for paper-inspired fusion."""

    names = branch_names(frame)
    score_map = {
        name: frame[f"{name}{SCORE_SUFFIX}"].to_numpy(dtype=float)
        for name in names
    }
    reliabilities = estimate_branch_reliability(
        labels(frame),
        score_map,
        alpha_prior=config.bayesian_alpha_prior,
        beta_prior=config.bayesian_beta_prior,
    )
    return {
        "branch_names": names,
        "reliabilities": reliabilities,
    }


def fused_scores(
    valid_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Produce fused scores for a target split using the configured fusion mode."""

    mode = config.mode
    if mode == "weighted_average":
        return weighted_average_scores(target_frame, weights=config.weighted_average_weights), {
            "mode": mode,
            "weights": config.weighted_average_weights,
        }
    if mode == "logistic_meta":
        model = train_logistic_meta_model(valid_frame, config=config)
        scores = model.predict_proba(fusion_feature_frame(target_frame))[:, 1]
        return scores, {
            "mode": mode,
            "coefficients": {
                name: float(value)
                for name, value in zip(score_columns(valid_frame), model.coef_[0], strict=True)
            },
            "intercept": float(model.intercept_[0]),
        }
    if mode == "bayesian_reliability":
        artifacts = reliability_fusion_artifacts(valid_frame, config=config)
        score_map = {
            name: target_frame[f"{name}{SCORE_SUFFIX}"].to_numpy(dtype=float)
            for name in artifacts["branch_names"]
        }
        scores = reliability_weighted_scores(score_map, artifacts["reliabilities"])
        return scores, {
            "mode": mode,
            "paper_inspired": True,
            "note": (
                "Inference from the cited paper: this scaffold uses validation correctness "
                "to form Beta-style reliability estimates and combines them with per-row "
                "entropy confidence for score weighting."
            ),
            "reliabilities": {
                name: branch.to_payload()
                for name, branch in artifacts["reliabilities"].items()
            },
        }
    msg = f"unsupported fusion mode: {mode}"
    raise ValueError(msg)


def prediction_frame(frame: pd.DataFrame, *, fused_scores_array: np.ndarray, threshold: float) -> pd.DataFrame:
    """Build a fused prediction export frame with traceability columns."""

    scores = np.asarray(fused_scores_array, dtype=float)
    return pd.DataFrame(
        {
            ID_COLUMN: transaction_ids(frame).reset_index(drop=True),
            LABEL_COLUMN: labels(frame).reset_index(drop=True),
            "score": scores,
            "predicted_label": (scores >= threshold).astype(int),
        }
    )

"""Fusion model training and scoring helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from fusion.bayesian_weighting import estimate_branch_reliability, reliability_weighted_scores


ID_COLUMN = "transaction_id"
LABEL_COLUMN = "is_fraud"
PREDICTED_LABEL_SUFFIX = "_predicted_label"
SCORE_SUFFIX = "_score"


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for the fusion stage."""

    mode: str = "auto_select"
    threshold: float = 0.5
    threshold_strategy: str = "validation_f1"
    threshold_candidate_count: int = 200
    logistic_c: float = 1.0
    logistic_max_iter: int = 500
    random_state: int = 7
    weighted_average_weights: dict[str, float] | None = None
    bayesian_alpha_prior: float = 1.0
    bayesian_beta_prior: float = 1.0
    auto_select_metric: str = "average_precision"
    auto_select_complexity_penalty: float = 2.5e-4
    auto_select_min_valid_average_precision: float = 1.0e-3
    auto_select_max_branches: int = 3
    auto_select_candidates: tuple[str, ...] = (
        "best_branch",
        "mean_top_k",
        "geometric_mean_top_k",
        "ap_weighted_average",
        "logistic_meta",
        "bayesian_reliability",
    )

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


def _subset_score_columns(frame: pd.DataFrame, selected_score_columns: list[str]) -> pd.DataFrame:
    base_columns = [ID_COLUMN, LABEL_COLUMN]
    predicted_columns = [column.replace(SCORE_SUFFIX, PREDICTED_LABEL_SUFFIX) for column in selected_score_columns]
    keep_columns = [column for column in [*base_columns, *selected_score_columns, *predicted_columns] if column in frame.columns]
    return frame.loc[:, keep_columns].copy()


def branch_average_precision(frame: pd.DataFrame) -> dict[str, float]:
    """Compute validation AP per branch score column."""

    y_true = labels(frame).astype(int).to_numpy()
    return {
        branch_name: float(average_precision_score(y_true, frame[f"{branch_name}{SCORE_SUFFIX}"].astype(float)))
        for branch_name in branch_names(frame)
    }


def ranked_branch_names(frame: pd.DataFrame) -> list[str]:
    """Return branch names ranked by validation average precision."""

    scores = branch_average_precision(frame)
    return sorted(scores, key=lambda name: (scores[name], name), reverse=True)


def _select_branch_pool(valid_frame: pd.DataFrame, *, config: FusionConfig) -> list[str]:
    ranked = ranked_branch_names(valid_frame)
    branch_ap = branch_average_precision(valid_frame)
    filtered = [
        branch_name
        for branch_name in ranked
        if branch_ap[branch_name] >= config.auto_select_min_valid_average_precision
    ]
    if not filtered:
        filtered = ranked[:1]
    max_branches = max(1, int(config.auto_select_max_branches))
    return filtered[:max_branches]


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


def geometric_mean_scores(frame: pd.DataFrame) -> np.ndarray:
    """Fuse branch scores with a geometric mean for conservative agreement emphasis."""

    feature_matrix = np.clip(fusion_feature_frame(frame).to_numpy(dtype=float), 1e-9, 1.0)
    return np.exp(np.mean(np.log(feature_matrix), axis=1))


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


def _candidate_metric(metric_name: str, labels_series: pd.Series, scores: np.ndarray) -> float:
    y_true = labels_series.astype(int).to_numpy()
    y_score = np.asarray(scores, dtype=float)
    if metric_name == "average_precision":
        return float(average_precision_score(y_true, y_score))
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    msg = f"unsupported auto-select metric: {metric_name}"
    raise ValueError(msg)


def _candidate_details(
    *,
    candidate_name: str,
    branch_names_used: list[str],
    validation_metric: float,
    selection_score: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_name": candidate_name,
        "branch_names_used": branch_names_used,
        "branch_count": len(branch_names_used),
        "validation_metric": validation_metric,
        "selection_score": selection_score,
    }
    if extra:
        payload.update(extra)
    return payload


def _auto_select_candidate(
    valid_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    pool = _select_branch_pool(valid_frame, config=config)
    valid_labels = labels(valid_frame)
    score_columns_by_branch = [f"{branch_name}{SCORE_SUFFIX}" for branch_name in pool]
    valid_subset = _subset_score_columns(valid_frame, score_columns_by_branch)
    target_subset = _subset_score_columns(target_frame, score_columns_by_branch)
    validation_branch_ap = branch_average_precision(valid_frame)

    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []

    best_branch = pool[0]
    best_branch_scores = target_subset[f"{best_branch}{SCORE_SUFFIX}"].to_numpy(dtype=float)
    best_branch_valid_scores = valid_subset[f"{best_branch}{SCORE_SUFFIX}"].to_numpy(dtype=float)
    best_branch_metric = _candidate_metric(config.auto_select_metric, valid_labels, best_branch_valid_scores)
    candidates.append(
        (
            best_branch_scores,
            _candidate_details(
                candidate_name="best_branch",
                branch_names_used=[best_branch],
                validation_metric=best_branch_metric,
                selection_score=best_branch_metric,
                extra={"selected_branch": best_branch},
            ),
        )
    )

    if len(pool) > 1:
        mean_valid_scores = valid_subset[[f"{name}{SCORE_SUFFIX}" for name in pool]].mean(axis=1).to_numpy(dtype=float)
        mean_target_scores = target_subset[[f"{name}{SCORE_SUFFIX}" for name in pool]].mean(axis=1).to_numpy(dtype=float)
        mean_metric = _candidate_metric(config.auto_select_metric, valid_labels, mean_valid_scores)
        candidates.append(
            (
                mean_target_scores,
                _candidate_details(
                    candidate_name="mean_top_k",
                    branch_names_used=pool,
                    validation_metric=mean_metric,
                    selection_score=mean_metric - config.auto_select_complexity_penalty * (len(pool) - 1),
                ),
            )
        )

        geo_valid_scores = geometric_mean_scores(valid_subset)
        geo_target_scores = geometric_mean_scores(target_subset)
        geo_metric = _candidate_metric(config.auto_select_metric, valid_labels, geo_valid_scores)
        candidates.append(
            (
                geo_target_scores,
                _candidate_details(
                    candidate_name="geometric_mean_top_k",
                    branch_names_used=pool,
                    validation_metric=geo_metric,
                    selection_score=geo_metric - config.auto_select_complexity_penalty * (len(pool) - 1),
                ),
            )
        )

        ap_weights = {
            name: validation_branch_ap[name]
            for name in pool
        }
        ap_valid_scores = weighted_average_scores(valid_subset, weights=ap_weights)
        ap_target_scores = weighted_average_scores(target_subset, weights=ap_weights)
        ap_metric = _candidate_metric(config.auto_select_metric, valid_labels, ap_valid_scores)
        candidates.append(
            (
                ap_target_scores,
                _candidate_details(
                    candidate_name="ap_weighted_average",
                    branch_names_used=pool,
                    validation_metric=ap_metric,
                    selection_score=ap_metric - config.auto_select_complexity_penalty * (len(pool) - 1),
                    extra={"weights": ap_weights},
                ),
            )
        )

        logistic_model = train_logistic_meta_model(valid_subset, config=config)
        logistic_valid_scores = logistic_model.predict_proba(fusion_feature_frame(valid_subset))[:, 1]
        logistic_target_scores = logistic_model.predict_proba(fusion_feature_frame(target_subset))[:, 1]
        logistic_metric = _candidate_metric(config.auto_select_metric, valid_labels, logistic_valid_scores)
        candidates.append(
            (
                logistic_target_scores,
                _candidate_details(
                    candidate_name="logistic_meta",
                    branch_names_used=pool,
                    validation_metric=logistic_metric,
                    selection_score=logistic_metric - config.auto_select_complexity_penalty * (len(pool) - 1),
                    extra={
                        "coefficients": {
                            name: float(value)
                            for name, value in zip(score_columns(valid_subset), logistic_model.coef_[0], strict=True)
                        },
                        "intercept": float(logistic_model.intercept_[0]),
                    },
                ),
            )
        )

        reliability_artifacts = reliability_fusion_artifacts(valid_subset, config=config)
        reliability_valid_score_map = {
            name: valid_subset[f"{name}{SCORE_SUFFIX}"].to_numpy(dtype=float)
            for name in pool
        }
        reliability_target_score_map = {
            name: target_subset[f"{name}{SCORE_SUFFIX}"].to_numpy(dtype=float)
            for name in pool
        }
        bayesian_valid_scores = reliability_weighted_scores(
            reliability_valid_score_map,
            reliability_artifacts["reliabilities"],
        )
        bayesian_target_scores = reliability_weighted_scores(
            reliability_target_score_map,
            reliability_artifacts["reliabilities"],
        )
        bayesian_metric = _candidate_metric(config.auto_select_metric, valid_labels, bayesian_valid_scores)
        candidates.append(
            (
                bayesian_target_scores,
                _candidate_details(
                    candidate_name="bayesian_reliability",
                    branch_names_used=pool,
                    validation_metric=bayesian_metric,
                    selection_score=bayesian_metric - config.auto_select_complexity_penalty * (len(pool) - 1),
                    extra={
                        "paper_inspired": True,
                        "reliabilities": {
                            name: branch.to_payload()
                            for name, branch in reliability_artifacts["reliabilities"].items()
                        },
                    },
                ),
            )
        )

    requested_candidates = set(config.auto_select_candidates)
    filtered_candidates = [
        candidate
        for candidate in candidates
        if candidate[1]["candidate_name"] in requested_candidates
    ]
    if not filtered_candidates:
        filtered_candidates = candidates
    selected_scores, selected_details = max(
        filtered_candidates,
        key=lambda item: (item[1]["selection_score"], item[1]["validation_metric"], -item[1]["branch_count"]),
    )
    leaderboard = sorted(
        (details for _, details in filtered_candidates),
        key=lambda details: (details["selection_score"], details["validation_metric"], -details["branch_count"]),
        reverse=True,
    )
    return selected_scores, {
        "mode": "auto_select",
        "selection_metric": config.auto_select_metric,
        "complexity_penalty_per_extra_branch": config.auto_select_complexity_penalty,
        "min_valid_average_precision": config.auto_select_min_valid_average_precision,
        "branch_pool": pool,
        "branch_validation_average_precision": validation_branch_ap,
        "selected_candidate": selected_details,
        "candidate_leaderboard": leaderboard,
    }


def fused_scores(
    valid_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Produce fused scores for a target split using the configured fusion mode."""

    mode = config.mode
    if mode == "auto_select":
        return _auto_select_candidate(valid_frame, target_frame, config=config)
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

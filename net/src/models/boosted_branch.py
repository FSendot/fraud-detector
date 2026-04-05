"""Boosted-tree fraud model branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class BoostedBranchConfig:
    """Configuration for the histogram gradient-boosted fraud branch."""

    random_state: int = 7
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 31
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    early_stopping: bool = True
    validation_fraction: float = 0.15
    n_iter_no_change: int = 20
    feature_importance_max_rows: int = 100_000
    feature_importance_repeats: int = 3
    feature_importance_scoring: str = "average_precision"
    feature_importance_n_jobs: int = 1

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def fit_boosted_branch(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    config: BoostedBranchConfig,
) -> HistGradientBoostingClassifier:
    """Fit a deterministic histogram gradient-boosted classifier."""

    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction,
        n_iter_no_change=config.n_iter_no_change,
        random_state=config.random_state,
    )
    model.fit(features, labels)
    return model


def _importance_sample(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Take a deterministic validation sample that keeps all positives when possible."""

    if len(features) <= max_rows:
        return features, labels

    positive_mask = labels.astype(int) == 1
    positive_indices = labels.index[positive_mask]
    negative_indices = labels.index[~positive_mask]
    remaining_budget = max(0, max_rows - len(positive_indices))
    if remaining_budget == 0:
        selected_indices = positive_indices[:max_rows]
    else:
        sampled_negatives = labels.loc[negative_indices].sample(
            n=min(remaining_budget, len(negative_indices)),
            random_state=random_state,
        ).index
        selected_indices = positive_indices.append(sampled_negatives)
    selected_indices = pd.Index(selected_indices).sort_values()
    return features.loc[selected_indices], labels.loc[selected_indices]


def feature_importance_frame(
    *,
    model: HistGradientBoostingClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
    config: BoostedBranchConfig,
) -> pd.DataFrame:
    """Build a stable permutation-importance table on a deterministic validation sample."""

    sampled_features, sampled_labels = _importance_sample(
        features,
        labels,
        max_rows=config.feature_importance_max_rows,
        random_state=config.random_state,
    )
    importances = permutation_importance(
        model,
        sampled_features,
        sampled_labels.astype(int),
        n_repeats=config.feature_importance_repeats,
        random_state=config.random_state,
        scoring=config.feature_importance_scoring,
        n_jobs=config.feature_importance_n_jobs,
    )
    frame = pd.DataFrame(
        {
            "feature_name": sampled_features.columns,
            "importance_mean": importances.importances_mean,
            "importance_std": importances.importances_std,
        }
    )
    return frame.sort_values(
        by=["importance_mean", "feature_name"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

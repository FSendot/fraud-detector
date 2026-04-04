"""Tree-based fraud model branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


@dataclass(frozen=True)
class TreeBranchConfig:
    """Configuration for the ExtraTrees fraud branch."""

    random_state: int = 7
    n_estimators: int = 400
    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float = "sqrt"
    bootstrap: bool = False
    class_weight: str | dict[str, float] | None = "balanced"
    n_jobs: int = -1

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def fit_tree_branch(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    config: TreeBranchConfig,
) -> ExtraTreesClassifier:
    """Fit a deterministic ExtraTrees classifier on canonical tabular features."""

    model = ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        criterion=config.criterion,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(features, labels)
    return model


def feature_importance_frame(
    *,
    feature_names: list[str],
    importances: list[float],
) -> pd.DataFrame:
    """Build a stable feature-importance table for downstream reporting."""

    frame = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    )
    return frame.sort_values(
        by=["importance", "feature_name"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

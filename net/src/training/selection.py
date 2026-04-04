"""Feature selection helpers for tabular fraud models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Fitted selector and the retained feature names."""

    selector: Any
    selected_feature_names: list[str]
    requested_top_k: int | None
    selector_name: str


def fit_feature_selector(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    top_k: int | None,
) -> FeatureSelectionResult:
    """Fit a deterministic supervised feature selector on training data only."""

    if features.empty:
        msg = "feature selection requires at least one candidate feature"
        raise ValueError(msg)

    if labels.nunique(dropna=True) < 2:
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(features)
        selected_feature_names = list(features.columns[selector.get_support()])
        return FeatureSelectionResult(
            selector=selector,
            selected_feature_names=selected_feature_names,
            requested_top_k=top_k,
            selector_name="VarianceThreshold",
        )

    effective_k: int | str
    if top_k is None:
        effective_k = "all"
    else:
        if top_k <= 0:
            msg = "top_k must be positive when provided"
            raise ValueError(msg)
        effective_k = min(top_k, features.shape[1])

    selector = SelectKBest(score_func=f_classif, k=effective_k)
    selector.fit(features, labels)
    mask = selector.get_support()
    selected_feature_names = list(features.columns[mask])
    return FeatureSelectionResult(
        selector=selector,
        selected_feature_names=selected_feature_names,
        requested_top_k=top_k,
        selector_name="SelectKBest(f_classif)",
    )

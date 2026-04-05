"""Nyström-kernel approximation classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class NystromGPConfig:
    """Configuration for the Nyström-based classifier."""

    random_state: int = 7
    kernel: str = "rbf"
    gamma: float = 0.05
    n_components: int = 256
    logistic_c: float = 2.0
    max_iter: int = 1000

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def fit_nystrom_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    config: NystromGPConfig,
) -> Pipeline:
    """Fit a Nyström-kernel approximation classifier on tabular embeddings."""

    classifier = Pipeline(
        steps=[
            (
                "nystroem",
                Nystroem(
                    kernel=config.kernel,
                    gamma=config.gamma,
                    n_components=config.n_components,
                    random_state=config.random_state,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config.logistic_c,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    classifier.fit(features, labels)
    return classifier

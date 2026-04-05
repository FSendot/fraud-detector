"""Bayesian-style reliability weighting helpers for fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-9


@dataclass(frozen=True)
class BranchReliability:
    """Validation-derived branch reliability summary."""

    branch_name: str
    alpha: float
    beta: float
    posterior_mean: float
    average_entropy: float
    reliability_weight: float

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute stable binary entropy for fraud probabilities."""

    probs = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def estimate_branch_reliability(
    labels: pd.Series,
    branch_scores: dict[str, np.ndarray],
    *,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict[str, BranchReliability]:
    """Estimate branch reliability with a Beta-style posterior over validation correctness.

    This is an inference-inspired scaffold, not a full replication of the paper.
    It uses validation-set correctness to build a reliability prior and records
    prediction entropy so later per-transaction weighting can be confidence-aware.
    """

    y_true = labels.astype(int).to_numpy()
    reliabilities: dict[str, BranchReliability] = {}
    for branch_name, scores in branch_scores.items():
        probabilities = np.clip(np.asarray(scores, dtype=float), EPSILON, 1.0 - EPSILON)
        predicted = (probabilities >= 0.5).astype(int)
        correct = (predicted == y_true).astype(float)
        alpha = float(alpha_prior + correct.sum())
        beta = float(beta_prior + len(correct) - correct.sum())
        posterior_mean = alpha / max(alpha + beta, EPSILON)
        entropy = binary_entropy(probabilities)
        average_entropy = float(entropy.mean())
        # Higher reliability and lower entropy should both increase influence.
        reliability_weight = float(posterior_mean / max(average_entropy, EPSILON))
        reliabilities[branch_name] = BranchReliability(
            branch_name=branch_name,
            alpha=alpha,
            beta=beta,
            posterior_mean=posterior_mean,
            average_entropy=average_entropy,
            reliability_weight=reliability_weight,
        )
    return reliabilities


def reliability_weighted_scores(
    branch_scores: dict[str, np.ndarray],
    reliabilities: dict[str, BranchReliability],
) -> np.ndarray:
    """Fuse branch scores using global reliability and local entropy confidence."""

    fused_numerator: np.ndarray | None = None
    fused_denominator: np.ndarray | None = None
    for branch_name, scores in branch_scores.items():
        probabilities = np.clip(np.asarray(scores, dtype=float), EPSILON, 1.0 - EPSILON)
        entropy = binary_entropy(probabilities)
        local_confidence = 1.0 / np.maximum(entropy, EPSILON)
        global_weight = reliabilities[branch_name].reliability_weight
        total_weight = global_weight * local_confidence
        weighted_scores = total_weight * probabilities
        fused_numerator = weighted_scores if fused_numerator is None else fused_numerator + weighted_scores
        fused_denominator = total_weight if fused_denominator is None else fused_denominator + total_weight
    if fused_numerator is None or fused_denominator is None:
        return np.array([], dtype=float)
    return fused_numerator / np.maximum(fused_denominator, EPSILON)

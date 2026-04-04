"""Shared utilities for model training stages."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from common.io import ensure_directory


TARGET_COLUMN = "is_fraud"
ID_COLUMN = "transaction_id"


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_input(path: Path) -> pd.DataFrame:
    """Load a prepared tabular model-input parquet file."""

    frame = pd.read_parquet(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    if not required.issubset(frame.columns):
        msg = f"expected columns {sorted(required)} in {path}"
        raise ValueError(msg)
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string")
    return frame


def split_model_input(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Split model input into ids, labels, and feature matrix."""

    ids = frame[ID_COLUMN].astype("string")
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0).astype("Int64")
    features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("Float32")
    return ids, labels, features.fillna(0.0)


def write_json(path: Path, payload: Any) -> None:
    """Persist JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def make_json_safe(payload: Any) -> Any:
    """Convert common Python objects into JSON-safe structures."""

    if is_dataclass(payload):
        return make_json_safe(asdict(payload))
    if isinstance(payload, dict):
        return {str(key): make_json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [make_json_safe(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def binary_classification_metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Compute machine-readable binary classification metrics."""

    y_true = labels.astype(int).to_numpy()
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def latent_prediction_frame(
    *,
    transaction_ids: pd.Series,
    labels: pd.Series,
    latent: np.ndarray,
) -> pd.DataFrame:
    """Build a latent-export dataframe with traceability columns."""

    data = {
        ID_COLUMN: transaction_ids.reset_index(drop=True),
        TARGET_COLUMN: labels.reset_index(drop=True),
    }
    for index in range(latent.shape[1]):
        data[f"latent_{index}"] = latent[:, index]
    return pd.DataFrame(data)


def classifier_prediction_frame(
    *,
    transaction_ids: pd.Series,
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Build a prediction dataframe with traceability columns."""

    probs = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            ID_COLUMN: transaction_ids.reset_index(drop=True),
            TARGET_COLUMN: labels.reset_index(drop=True),
            "score": probs,
            "predicted_label": (probs >= threshold).astype(int),
        }
    )


def default_vae_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return configured default paths for VAE training artifacts."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["model_input_train_tabular"],
        paths["model_input_valid_tabular"],
        paths["model_input_test_tabular"],
        paths["artifact_vae_config"],
        paths["artifact_vae_weights"],
        paths["artifact_vae_metrics"],
    )


def default_vae_prediction_paths() -> tuple[Path, Path, Path]:
    """Return configured default latent-export paths."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["prediction_vae_latent_train"],
        paths["prediction_vae_latent_valid"],
        paths["prediction_vae_latent_test"],
    )


def default_nystrom_paths() -> tuple[Path, Path, Path, Path, Path]:
    """Return configured default paths for Nyström training artifacts."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["prediction_vae_latent_train"],
        paths["prediction_vae_latent_valid"],
        paths["prediction_vae_latent_test"],
        paths["artifact_nystrom_gp_model"],
        paths["artifact_nystrom_gp_metrics"],
    )


def default_nystrom_prediction_paths() -> tuple[Path, Path]:
    """Return configured default classifier prediction export paths."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["prediction_nystrom_valid"],
        paths["prediction_nystrom_test"],
    )


def default_tabular_nystrom_paths() -> tuple[Path, Path, Path, Path, Path]:
    """Return configured default paths for direct-tabular Nyström artifacts."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["model_input_train_tabular"],
        paths["model_input_valid_tabular"],
        paths["model_input_test_tabular"],
        paths["artifact_nystrom_tabular_model"],
        paths["artifact_nystrom_tabular_metrics"],
    )


def default_tabular_nystrom_prediction_paths() -> tuple[Path, Path]:
    """Return configured default paths for direct-tabular Nyström predictions."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["prediction_nystrom_tabular_valid"],
        paths["prediction_nystrom_tabular_test"],
    )


def ensure_parent(path: Path) -> Path:
    """Create a file's parent directory if needed and return the path."""

    ensure_directory(path.parent)
    return path

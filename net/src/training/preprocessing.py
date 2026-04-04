"""Training-data preparation for tabular fraud models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from common.io import ensure_directory
from data.splits import DEFAULT_TRANSACTION_ID_COLUMNS, ensure_transaction_id_column
from training.balancing import DownsampleResult, downsample_training_frame
from training.selection import FeatureSelectionResult, fit_feature_selector


TARGET_COLUMN = "is_fraud"
TRACE_COLUMNS = {"transaction_id", "current_transaction_id", TARGET_COLUMN}
EXCLUDED_FEATURE_COLUMNS = {
    "split",
    "source_row_number",
    "transaction_order",
    "step",
    "transaction_timestamp",
    "previous_transaction_timestamp",
    "name_orig",
    "name_dest",
    "type",
}


@dataclass(frozen=True)
class TabularPreparationResult:
    """Output locations and summary stats for training prep."""

    train_output_path: str
    valid_output_path: str
    test_output_path: str
    scaler_path: str
    feature_selector_path: str
    selected_features_path: str
    report_path: str
    train_rows: int
    valid_rows: int
    test_rows: int
    selected_feature_count: int


def load_behavioral_features(input_path: Path) -> pd.DataFrame:
    """Load the behavioral feature parquet."""

    return pd.read_parquet(input_path)


def load_split_ids(path: Path) -> pd.DataFrame:
    """Load a split-ID parquet file."""

    frame = pd.read_parquet(path)
    if "transaction_id" not in frame.columns:
        msg = f"split file is missing transaction_id: {path}"
        raise ValueError(msg)
    frame["transaction_id"] = frame["transaction_id"].astype("string")
    return frame


def ensure_canonical_transaction_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the feature table carries the canonical transaction_id column."""

    working, transaction_id_column = ensure_transaction_id_column(frame, DEFAULT_TRANSACTION_ID_COLUMNS)
    if transaction_id_column != "transaction_id":
        working["transaction_id"] = working[transaction_id_column].astype("string")
    else:
        working["transaction_id"] = working["transaction_id"].astype("string")
    return working


def subset_by_split_ids(frame: pd.DataFrame, split_ids: pd.DataFrame) -> pd.DataFrame:
    """Subset the feature table using canonical transaction IDs while preserving split order."""

    indexed = frame.set_index("transaction_id", drop=False)
    subset = indexed.loc[split_ids["transaction_id"]].reset_index(drop=True)
    return subset


def choose_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Choose numeric tabular feature columns excluding traceability and label fields."""

    columns: list[str] = []
    excluded = TRACE_COLUMNS | EXCLUDED_FEATURE_COLUMNS
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if not columns:
        msg = "no numeric tabular feature columns were available"
        raise ValueError(msg)
    return columns


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    features = frame.loc[:, columns].copy()
    for column in columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("Float32")
    return features.fillna(0.0)


def _fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def _transform_features(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    scaler: StandardScaler,
    selector_result: FeatureSelectionResult,
) -> pd.DataFrame:
    numeric = _numeric_feature_frame(frame, feature_columns)
    scaled_array = scaler.transform(numeric)
    scaled = pd.DataFrame(scaled_array, columns=feature_columns, index=frame.index)
    selected_array = selector_result.selector.transform(scaled)
    selected = pd.DataFrame(
        selected_array,
        columns=selector_result.selected_feature_names,
        index=frame.index,
    )
    output = pd.concat(
        [
            frame.loc[:, ["transaction_id", TARGET_COLUMN]].reset_index(drop=True),
            selected.reset_index(drop=True),
        ],
        axis=1,
    )
    return output


def write_selected_features(path: Path, payload: dict[str, Any]) -> None:
    """Persist selected feature metadata as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Persist the training-prep report as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_output_paths() -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    """Return configured default input and output locations."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["processed_behavioral_features"],
        paths["split_train_ids"],
        paths["split_valid_ids"],
        paths["split_test_ids"],
        paths["model_input_train_tabular"],
        paths["model_input_valid_tabular"],
        paths["model_input_test_tabular"],
        paths["artifact_scaler"],
        paths["artifact_feature_selector"],
        paths["artifact_selected_features"],
    )


def prepare_and_write_tabular_datasets(
    *,
    behavioral_input_path: Path,
    train_ids_path: Path,
    valid_ids_path: Path,
    test_ids_path: Path,
    train_output_path: Path,
    valid_output_path: Path,
    test_output_path: Path,
    scaler_path: Path,
    feature_selector_path: Path,
    selected_features_path: Path,
    report_path: Path,
    downsample_ratio: float,
    top_k_features: int | None,
) -> TabularPreparationResult:
    """Build train/valid/test tabular model inputs with train-only learned preprocessing."""

    behavioral = ensure_canonical_transaction_id(load_behavioral_features(behavioral_input_path))
    train_ids = load_split_ids(train_ids_path)
    valid_ids = load_split_ids(valid_ids_path)
    test_ids = load_split_ids(test_ids_path)

    train_frame = subset_by_split_ids(behavioral, train_ids)
    valid_frame = subset_by_split_ids(behavioral, valid_ids)
    test_frame = subset_by_split_ids(behavioral, test_ids)

    feature_columns = choose_feature_columns(train_frame)
    balanced_train = downsample_training_frame(
        train_frame,
        label_column=TARGET_COLUMN,
        downsample_ratio=downsample_ratio,
    )

    train_features = _numeric_feature_frame(balanced_train.frame, feature_columns)
    train_labels = balanced_train.frame[TARGET_COLUMN].astype("Int64")

    scaler = _fit_scaler(train_features)
    scaled_train = pd.DataFrame(
        scaler.transform(train_features),
        columns=feature_columns,
        index=balanced_train.frame.index,
    )
    selector_result = fit_feature_selector(
        scaled_train,
        train_labels,
        top_k=top_k_features,
    )

    train_output = _transform_features(
        balanced_train.frame,
        feature_columns=feature_columns,
        scaler=scaler,
        selector_result=selector_result,
    )
    valid_output = _transform_features(
        valid_frame,
        feature_columns=feature_columns,
        scaler=scaler,
        selector_result=selector_result,
    )
    test_output = _transform_features(
        test_frame,
        feature_columns=feature_columns,
        scaler=scaler,
        selector_result=selector_result,
    )

    ensure_directory(train_output_path.parent)
    train_output.to_parquet(train_output_path, index=False)
    valid_output.to_parquet(valid_output_path, index=False)
    test_output.to_parquet(test_output_path, index=False)

    ensure_directory(scaler_path.parent)
    joblib.dump(scaler, scaler_path)
    joblib.dump(selector_result.selector, feature_selector_path)

    selected_features_payload = {
        "feature_columns_before_selection": feature_columns,
        "selected_feature_columns": selector_result.selected_feature_names,
        "selected_feature_count": len(selector_result.selected_feature_names),
        "top_k_requested": top_k_features,
        "feature_selector": selector_result.selector_name,
        "target_column": TARGET_COLUMN,
        "transaction_id_column": "transaction_id",
    }
    write_selected_features(selected_features_path, selected_features_payload)

    report = {
        "behavioral_input_path": str(behavioral_input_path),
        "split_inputs": {
            "train_ids": str(train_ids_path),
            "valid_ids": str(valid_ids_path),
            "test_ids": str(test_ids_path),
        },
        "row_counts": {
            "train_before_balancing": int(len(train_frame)),
            "train_after_balancing": int(len(train_output)),
            "valid": int(len(valid_output)),
            "test": int(len(test_output)),
        },
        "balancing": {
            "downsample_ratio": downsample_ratio,
            "positive_rows": balanced_train.positive_rows,
            "negative_rows_before": balanced_train.negative_rows_before,
            "negative_rows_after": balanced_train.negative_rows_after,
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "feature_selector": selector_result.selector_name,
            "top_k_requested": top_k_features,
            "selected_feature_count": len(selector_result.selected_feature_names),
        },
        "selected_feature_columns": selector_result.selected_feature_names,
    }
    write_report(report_path, report)

    return TabularPreparationResult(
        train_output_path=str(train_output_path),
        valid_output_path=str(valid_output_path),
        test_output_path=str(test_output_path),
        scaler_path=str(scaler_path),
        feature_selector_path=str(feature_selector_path),
        selected_features_path=str(selected_features_path),
        report_path=str(report_path),
        train_rows=int(len(train_output)),
        valid_rows=int(len(valid_output)),
        test_rows=int(len(test_output)),
        selected_feature_count=len(selector_result.selected_feature_names),
    )

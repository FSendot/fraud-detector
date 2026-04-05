"""Canonical feature contract export and validation helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from training.preprocessing import TARGET_COLUMN
from training.train_utils import ID_COLUMN


CONTRACT_NAME = "fraud_model_feature_contract"
CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation result for a dataset checked against a contract."""

    valid: bool
    checked_rows: int
    checked_columns: int
    errors: list[str]
    warnings: list[str]

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def _load_feature_dictionary(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_file(path)
    metadata: dict[str, dict[str, Any]] = {}
    for group_name in ("source_features", "derived_features"):
        entries = payload.get(group_name, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str):
                metadata[name] = entry
    return metadata


def _merge_feature_metadata(*maps: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for mapping in maps:
        merged.update(mapping)
    return merged


def _normalize_feature_role(name: str, metadata: dict[str, dict[str, Any]]) -> str:
    entry = metadata.get(name, {})
    group = entry.get("group")
    if group in {"source", "derived"}:
        return str(group)
    return "derived"


def _describe_feature(name: str, metadata: dict[str, dict[str, Any]]) -> str:
    entry = metadata.get(name, {})
    description = entry.get("description")
    if isinstance(description, str) and description.strip():
        return description
    return f"Model input feature {name!r}."


def _dtype_string(series: pd.Series) -> str:
    return str(series.dtype)


def _feature_null_rule() -> dict[str, Any]:
    return {
        "allowed_in_contract_dataset": False,
        "upstream_training_handling": {
            "numeric_coercion": "pd.to_numeric(errors='coerce')",
            "fill_strategy_before_scaling": "fillna",
            "fill_value": 0.0,
        },
    }


def _feature_preprocessing_expectations(selected_features_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_stage": "model_input_tabular_after_train_only_preprocessing",
        "numeric_coercion": "pd.to_numeric(errors='coerce')",
        "null_fill_before_scaling": 0.0,
        "scaling": {
            "name": "StandardScaler",
            "fit_scope": "train_only",
        },
        "feature_selection": {
            "name": "VarianceThreshold+SelectKBest(f_classif)",
            "fit_scope": "train_only",
            "selected_feature_count": int(selected_features_payload.get("selected_feature_count", 0)),
            "top_k_requested": selected_features_payload.get("top_k_requested"),
        },
    }


def _transaction_id_contract() -> dict[str, Any]:
    return {
        "name": ID_COLUMN,
        "dtype": "string",
        "required": True,
        "nullable": False,
        "description": "Canonical transaction identifier used to join splits, predictions, and serving traces.",
        "conventions": {
            "uniqueness_expectation": "expected_unique_within_dataset",
            "format": "string",
        },
    }


def _label_contract() -> dict[str, Any]:
    return {
        "name": TARGET_COLUMN,
        "dtype": "boolean",
        "required": True,
        "nullable": False,
        "description": "Ground-truth supervised fraud label carried separately from model features.",
        "conventions": {
            "positive_class": 1,
            "negative_class": 0,
            "accepted_runtime_forms": ["boolean", "integer_0_1"],
        },
    }


def _build_feature_entries(
    frame: pd.DataFrame,
    feature_names: list[str],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        series = frame[name]
        entries.append(
            {
                "name": name,
                "index": index,
                "dtype": _dtype_string(series),
                "nullable": bool(series.isna().any()),
                "role": _normalize_feature_role(name, metadata),
                "description": _describe_feature(name, metadata),
                "point_in_time_safe": True,
                "null_handling": _feature_null_rule(),
                "preprocessing_expectations": {
                    "numeric": True,
                    "scaled": True,
                    "selected_for_current_contract": True,
                },
            }
        )
    return entries


def _infer_expected_columns(contract: dict[str, Any]) -> list[str]:
    feature_names = [entry["name"] for entry in contract["features"]]
    return [
        contract["transaction_id"]["name"],
        contract["label"]["name"],
        *feature_names,
    ]


def build_feature_contract(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_features_payload: dict[str, Any],
    feature_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a versioned feature contract from current tabular model inputs."""

    expected_columns = list(train_frame.columns)
    for split_name, frame in (("valid", valid_frame), ("test", test_frame)):
        if list(frame.columns) != expected_columns:
            msg = f"{split_name} columns do not match train columns exactly"
            raise ValueError(msg)

    feature_names = [
        column
        for column in expected_columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    selected_feature_names = list(selected_features_payload.get("selected_feature_columns", feature_names))
    if feature_names != selected_feature_names:
        msg = "selected feature metadata does not match current model-input columns"
        raise ValueError(msg)

    raw_feature_count = sum(_normalize_feature_role(name, feature_metadata) == "source" for name in feature_names)
    derived_feature_count = len(feature_names) - raw_feature_count

    return {
        "contract_name": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "dataset_stage": "tabular_model_input",
        "transaction_id": _transaction_id_contract(),
        "label": _label_contract(),
        "features": _build_feature_entries(train_frame, feature_names, feature_metadata),
        "feature_order": feature_names,
        "summary": {
            "feature_count": len(feature_names),
            "raw_feature_count": raw_feature_count,
            "derived_feature_count": derived_feature_count,
        },
        "dataset_requirements": {
            "expected_columns_in_order": expected_columns,
            "strict_column_order": True,
            "allow_extra_columns": False,
            "row_granularity": "one_row_per_transaction",
        },
        "preprocessing_expectations": _feature_preprocessing_expectations(selected_features_payload),
        "artifacts": {
            "selected_features": selected_features_payload,
        },
        "serving_notes": {
            "go_integration_ready": True,
            "numeric_features_must_arrive_in_declared_order": True,
            "transaction_id_and_label_are_not_model_features": True,
        },
    }


def render_feature_contract_markdown(contract: dict[str, Any]) -> str:
    """Render a concise markdown summary for the feature contract."""

    lines = [
        f"# Feature Contract {contract['version']}",
        "",
        f"- Contract: `{contract['contract_name']}`",
        f"- Dataset stage: `{contract['dataset_stage']}`",
        f"- Feature count: `{contract['summary']['feature_count']}`",
        f"- Raw features: `{contract['summary']['raw_feature_count']}`",
        f"- Derived features: `{contract['summary']['derived_feature_count']}`",
        "",
        "## Conventions",
        f"- Transaction ID: `{contract['transaction_id']['name']}` as `{contract['transaction_id']['dtype']}`",
        f"- Label: `{contract['label']['name']}` as `{contract['label']['dtype']}`",
        "",
        "## Preprocessing Expectations",
        f"- Numeric coercion: `{contract['preprocessing_expectations']['numeric_coercion']}`",
        f"- Null fill before scaling: `{contract['preprocessing_expectations']['null_fill_before_scaling']}`",
        f"- Scaling: `{contract['preprocessing_expectations']['scaling']['name']}` fit on `{contract['preprocessing_expectations']['scaling']['fit_scope']}`",
        f"- Feature selection: `{contract['preprocessing_expectations']['feature_selection']['name']}`",
        "",
        "## Feature Order",
    ]
    for entry in contract["features"]:
        lines.append(f"- `{entry['index']:02d}` `{entry['name']}` [{entry['role']}, {entry['dtype']}]")
    return "\n".join(lines) + "\n"


def write_feature_contract(
    *,
    contract: dict[str, Any],
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """Persist the machine-readable and human-readable contract."""

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with json_output_path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(render_feature_contract_markdown(contract), encoding="utf-8")


def load_feature_contract(path: Path) -> dict[str, Any]:
    """Load a persisted feature contract."""

    payload = _load_json_file(path)
    if payload.get("contract_name") != CONTRACT_NAME:
        msg = f"unexpected contract_name in {path}"
        raise ValueError(msg)
    return payload


def load_default_feature_metadata() -> dict[str, dict[str, Any]]:
    """Load merged feature metadata from the current artifact dictionaries."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return _merge_feature_metadata(
        _load_feature_dictionary(paths["artifact_feature_dict"]),
        _load_feature_dictionary(paths["artifact_behavioral_feature_dict"]),
    )


def load_contract_export_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Load default inputs used for exporting the canonical feature contract."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    train_frame = pd.read_parquet(paths["model_input_train_tabular"])
    valid_frame = pd.read_parquet(paths["model_input_valid_tabular"])
    test_frame = pd.read_parquet(paths["model_input_test_tabular"])
    selected_features_payload = _load_json_file(paths["artifact_selected_features"])
    feature_metadata = load_default_feature_metadata()
    return train_frame, valid_frame, test_frame, selected_features_payload, feature_metadata


def _validate_column_layout(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    allow_extra_columns: bool,
) -> tuple[list[str], list[str]]:
    expected = _infer_expected_columns(contract)
    actual = list(frame.columns)
    errors: list[str] = []
    warnings: list[str] = []

    missing = [name for name in expected if name not in actual]
    extra = [name for name in actual if name not in expected]
    if missing:
        errors.append(f"missing required columns: {missing}")
    if extra and not allow_extra_columns:
        errors.append(f"unexpected extra columns: {extra}")
    if not missing:
        comparable = [name for name in actual if name in expected]
        if comparable[: len(expected)] != expected:
            errors.append(f"column order does not match contract expected order {expected}")
    elif extra:
        warnings.append("column order check skipped because required columns are missing")
    return errors, warnings


def _is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any():
        return False
    values = set(coerced.astype(int).tolist())
    return values.issubset({0, 1})


def _validate_transaction_id(frame: pd.DataFrame, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    column = contract["transaction_id"]["name"]
    if column not in frame.columns:
        return errors
    series = frame[column]
    string_series = series.astype("string")
    if string_series.isna().any():
        errors.append(f"{column} contains null values")
    if string_series.str.len().eq(0).any():
        errors.append(f"{column} contains empty string values")
    if string_series.duplicated().any():
        errors.append(f"{column} contains duplicate values")
    return errors


def _validate_label(frame: pd.DataFrame, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    column = contract["label"]["name"]
    if column not in frame.columns:
        return errors
    series = frame[column]
    if series.isna().any():
        errors.append(f"{column} contains null values")
    if not _is_boolean_like(series):
        errors.append(f"{column} is not boolean or 0/1 integer-like")
    return errors


def _validate_feature_column(frame: pd.DataFrame, entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    name = entry["name"]
    if name not in frame.columns:
        return errors
    numeric = pd.to_numeric(frame[name], errors="coerce")
    if numeric.isna().any():
        errors.append(f"{name} contains null or non-numeric values")
        return errors
    if (~np.isfinite(numeric.to_numpy(dtype=float))).any():
        errors.append(f"{name} contains non-finite values")
    return errors


def validate_frame_against_contract(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    allow_extra_columns: bool = False,
) -> ValidationResult:
    """Validate a dataframe against the canonical feature contract."""

    errors, warnings = _validate_column_layout(frame, contract, allow_extra_columns=allow_extra_columns)
    errors.extend(_validate_transaction_id(frame, contract))
    errors.extend(_validate_label(frame, contract))
    for entry in contract["features"]:
        errors.extend(_validate_feature_column(frame, entry))
    return ValidationResult(
        valid=not errors,
        checked_rows=int(len(frame)),
        checked_columns=int(len(frame.columns)),
        errors=errors,
        warnings=warnings,
    )


def load_dataframe_for_contract_validation(path: Path) -> pd.DataFrame:
    """Load a parquet or CSV dataset for contract validation."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    msg = f"unsupported dataset format for contract validation: {path}"
    raise ValueError(msg)

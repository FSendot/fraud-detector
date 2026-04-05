"""Deterministic preprocessing pipeline for raw fraud transaction CSV files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import DEFAULT_PATHS_FILE, load_paths_config, project_root
from common.io import ensure_directory
from data.schema import build_transaction_schema, normalize_columns
from data.validators import (
    combine_reason_masks,
    count_reason_masks,
    missing_required_fields_mask,
    negative_any_mask,
    negative_value_mask,
    parse_failure_mask,
)


BOOLEAN_TRUE_VALUES = {"1", "true", "t", "yes", "y", "found"}
BOOLEAN_FALSE_VALUES = {"0", "false", "f", "no", "n", "notfound", "not_found"}
IEEE_REFERENCE_TIMESTAMP = pd.Timestamp("2017-12-01T00:00:00Z")


@dataclass(frozen=True)
class PreprocessResult:
    """Output locations and summary for a preprocessing run."""

    input_path: str
    parquet_path: str
    report_path: str
    rows_in: int
    rows_out: int
    duplicate_rows_dropped: int
    invalid_rows_dropped: int
    drop_reasons: dict[str, int]


def load_raw_csv(input_path: Path) -> pd.DataFrame:
    """Load a raw CSV as strings so coercion stays explicit and deterministic."""

    return pd.read_csv(input_path, dtype="string", low_memory=False)


def _ieee_train_files(input_path: Path) -> tuple[Path, Path | None] | None:
    if input_path.is_dir():
        transaction_path = input_path / "train_transaction.csv"
        identity_path = input_path / "train_identity.csv"
        if transaction_path.is_file():
            return transaction_path, identity_path if identity_path.is_file() else None
        return None
    if input_path.name == "train_transaction.csv":
        identity_candidate = input_path.with_name("train_identity.csv")
        return input_path, identity_candidate if identity_candidate.is_file() else None
    return None


def _load_ieee_training_frame(transaction_path: Path, identity_path: Path | None) -> pd.DataFrame:
    transaction_frame = load_raw_csv(transaction_path)
    if identity_path is None:
        return transaction_frame
    identity_frame = load_raw_csv(identity_path)
    return transaction_frame.merge(identity_frame, on="TransactionID", how="left", validate="one_to_one")


def _compose_entity_id(frame: pd.DataFrame, columns: list[str], *, fallback_column: str | None = None) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    result = pd.Series([""] * len(frame), index=frame.index, dtype="string")
    has_value = pd.Series(False, index=frame.index, dtype="boolean")

    for column in available:
        values = frame[column].astype("string").fillna("").str.strip()
        mask = values.ne("")
        segment = pd.Series(f"{column}=", index=frame.index, dtype="string") + values
        updated = np.where(mask & has_value.to_numpy(dtype=bool), result + "|" + segment, result)
        result = pd.Series(updated, index=frame.index, dtype="string")
        updated = np.where(mask & ~has_value.to_numpy(dtype=bool), segment, result)
        result = pd.Series(updated, index=frame.index, dtype="string")
        has_value = has_value | mask

    if fallback_column is not None and fallback_column in frame.columns:
        fallback_values = frame[fallback_column].astype("string").fillna("").str.strip()
        fallback_mask = ~has_value.to_numpy(dtype=bool) & fallback_values.ne("").to_numpy(dtype=bool)
        result = pd.Series(np.where(fallback_mask, fallback_values, result), index=frame.index, dtype="string")
        has_value = has_value | pd.Series(fallback_mask, index=frame.index, dtype="boolean")

    return result.mask(~has_value, pd.NA)


def canonicalize_ieee_cis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map IEEE-CIS training columns into the pipeline's canonical transaction schema."""

    working = frame.copy()
    rename_map = {
        "transaction_amt": "amount",
        "product_cd": "type",
    }
    working = working.rename(columns={key: value for key, value in rename_map.items() if key in working.columns})

    if "transaction_dt" in working.columns and "step" not in working.columns:
        working["step"] = _coerce_integer(working["transaction_dt"])
    if "step" in working.columns and "transaction_timestamp" not in working.columns:
        step_seconds = pd.to_numeric(working["step"], errors="coerce")
        working["transaction_timestamp"] = pd.to_datetime(
            IEEE_REFERENCE_TIMESTAMP + pd.to_timedelta(step_seconds, unit="s"),
            utc=True,
        )
    if "transaction_id" not in working.columns and "source_row_number" in working.columns:
        working["transaction_id"] = "ieee_" + working["source_row_number"].astype("Int64").astype("string")

    if "name_orig" not in working.columns:
        working["name_orig"] = _compose_entity_id(
            working,
            ["card1", "card2", "card3", "card5", "card6"],
            fallback_column="transaction_id",
        )
    if "name_dest" not in working.columns:
        working["name_dest"] = _compose_entity_id(
            working,
            ["type", "addr1", "addr2", "p_emaildomain", "r_emaildomain"],
            fallback_column="transaction_id",
        )
    if "is_flagged_fraud" not in working.columns:
        working["is_flagged_fraud"] = pd.Series(False, index=working.index, dtype="boolean")
    return working


def _load_input_frames(input_paths: list[Path]) -> list[tuple[pd.DataFrame, str]]:
    if len(input_paths) == 1:
        ieee_files = _ieee_train_files(input_paths[0])
        if ieee_files is not None:
            transaction_path, identity_path = ieee_files
            merged = _load_ieee_training_frame(transaction_path, identity_path)
            return [(merged, f"{input_paths[0].name}:train")]

    ieee_transaction = next((path for path in input_paths if path.name == "train_transaction.csv"), None)
    if ieee_transaction is not None:
        identity_path = next((path for path in input_paths if path.name == "train_identity.csv"), None)
        merged = _load_ieee_training_frame(ieee_transaction, identity_path)
        source_name = ieee_transaction.parent.name if identity_path is not None else ieee_transaction.name
        return [(merged, f"{source_name}:train")]

    return [(load_raw_csv(path), path.name) for path in input_paths]


def prepare_raw_frame(raw_frame: pd.DataFrame, *, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize headers and trim string values without losing the original frame."""

    normalized = raw_frame.copy()
    normalized.columns = normalize_columns(normalized.columns)
    normalized.insert(0, "source_file", pd.Series([source_name] * len(normalized), dtype="string"))
    normalized.insert(1, "source_file_row_number", pd.Series(range(1, len(normalized) + 1), dtype="Int64"))
    normalized.insert(0, "source_row_number", pd.Series(range(1, len(normalized) + 1), dtype="Int64"))
    for column in normalized.columns:
        if column in {"source_row_number", "source_file_row_number"}:
            continue
        normalized[column] = normalized[column].astype("string").str.strip()
    if {"transaction_id", "is_fraud", "transaction_dt"}.issubset(set(normalized.columns)):
        normalized = canonicalize_ieee_cis_frame(normalized)
    return normalized, normalized.copy()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _coerce_integer(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    integer_like = numeric.notna() & numeric.mod(1).eq(0)
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result.loc[integer_like] = numeric.loc[integer_like].astype("int64")
    return result


def _coerce_boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(BOOLEAN_TRUE_VALUES)] = True
    result.loc[normalized.isin(BOOLEAN_FALSE_VALUES)] = False
    return result


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def coerce_frame_types(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series], Any]:
    """Coerce known columns and return row-level parse-failure masks."""

    coerced = frame.copy()
    schema = build_transaction_schema(coerced.columns)
    reason_masks: dict[str, pd.Series] = {}

    for column in schema.string_columns:
        coerced[column] = coerced[column].astype("string")

    for column in schema.numeric_columns:
        raw_series = frame[column]
        coerced[column] = _coerce_numeric(raw_series)
        reason_masks[f"invalid_{column}"] = parse_failure_mask(raw_series, coerced[column])

    for column in schema.integer_columns:
        raw_series = frame[column]
        coerced[column] = _coerce_integer(raw_series)
        reason_masks[f"invalid_{column}"] = parse_failure_mask(raw_series, coerced[column])

    for column in schema.boolean_columns:
        raw_series = frame[column]
        coerced[column] = _coerce_boolean(raw_series)
        reason_masks[f"invalid_{column}"] = parse_failure_mask(raw_series, coerced[column])

    for column in schema.timestamp_columns:
        raw_series = frame[column]
        coerced[column] = _coerce_timestamp(raw_series)
        reason_masks[f"invalid_{column}"] = parse_failure_mask(raw_series, coerced[column])

    return coerced, reason_masks, schema


def deduplicate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop exact duplicates using a stable first-row-wins rule."""

    metadata_columns = {"source_row_number", "source_file", "source_file_row_number"}
    subset = [column for column in frame.columns if column not in metadata_columns]
    duplicate_mask = frame.duplicated(subset=subset, keep="first")
    return frame.loc[~duplicate_mask].copy(), duplicate_mask


def build_invalid_reason_masks(
    raw_frame: pd.DataFrame,
    coerced_frame: pd.DataFrame,
    parse_masks: dict[str, pd.Series],
    schema: Any,
) -> dict[str, pd.Series]:
    """Build semantic invalid-row masks from the parsed frame."""

    reason_masks: dict[str, pd.Series] = {
        reason: mask.copy()
        for reason, mask in parse_masks.items()
        if int(mask.fillna(False).astype(bool).sum()) > 0
    }

    missing_ids = missing_required_fields_mask(raw_frame, schema.required_id_columns)
    if int(missing_ids.fillna(False).astype(bool).sum()) > 0:
        reason_masks["missing_required_id"] = missing_ids

    if "amount" in coerced_frame.columns:
        negative_amount = negative_value_mask(coerced_frame["amount"])
        if int(negative_amount.fillna(False).astype(bool).sum()) > 0:
            reason_masks["negative_amount"] = negative_amount

    negative_balances = negative_any_mask(coerced_frame, schema.balance_columns)
    if int(negative_balances.fillna(False).astype(bool).sum()) > 0:
        reason_masks["negative_balance"] = negative_balances

    if schema.balance_columns:
        malformed_balance = combine_reason_masks(
            {
                f"invalid_{column}": parse_masks[f"invalid_{column}"]
                for column in schema.balance_columns
                if f"invalid_{column}" in parse_masks
            },
            index=coerced_frame.index,
        )
        if int(malformed_balance.fillna(False).astype(bool).sum()) > 0:
            reason_masks["malformed_balance"] = malformed_balance

    return reason_masks


def assign_transaction_order(frame: pd.DataFrame, schema: Any) -> pd.DataFrame:
    """Create a deterministic transaction order using timestamps or step columns when present."""

    ordered = frame.copy()
    sort_columns: list[str] = []

    timestamp_columns = [column for column in schema.order_columns if "timestamp" in column]
    if timestamp_columns:
        primary_timestamp = timestamp_columns[0]
        ordered = ordered.rename(columns={primary_timestamp: "transaction_timestamp"})
        sort_columns.append("transaction_timestamp")
    elif "step" in ordered.columns:
        sort_columns.append("step")

    if "source_file" in ordered.columns:
        sort_columns.append("source_file")
    if "source_file_row_number" in ordered.columns:
        sort_columns.append("source_file_row_number")
    sort_columns.append("source_row_number")
    ordered = ordered.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)
    ordered["transaction_order"] = pd.Series(range(1, len(ordered) + 1), dtype="Int64")
    return ordered


def _make_json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            safe[key] = _make_json_safe(value)
        elif isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    """Persist the preprocessing report with stable formatting."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(report), handle, indent=2, sort_keys=True)
        handle.write("\n")


def preprocess_raw_transactions(
    *,
    input_path: Path | list[Path] | tuple[Path, ...],
    output_parquet_path: Path,
    output_report_path: Path,
) -> PreprocessResult:
    """Run the full deterministic preprocessing pipeline."""

    input_paths = [input_path] if isinstance(input_path, Path) else list(input_path)
    if not input_paths:
        msg = "at least one raw input path is required"
        raise ValueError(msg)

    prepared_frames: list[pd.DataFrame] = []
    coercion_frames: list[pd.DataFrame] = []
    input_row_counts: dict[str, int] = {}
    for raw_input, source_name in _load_input_frames(sorted(input_paths, key=lambda item: item.as_posix())):
        prepared_raw_frame, prepared_coercion_frame = prepare_raw_frame(raw_input, source_name=source_name)
        prepared_frames.append(prepared_raw_frame)
        coercion_frames.append(prepared_coercion_frame)
        input_row_counts[source_name] = int(len(raw_input))

    prepared_raw = pd.concat(prepared_frames, axis=0, ignore_index=True)
    prepared_for_coercion = pd.concat(coercion_frames, axis=0, ignore_index=True)
    deduplicated_raw, duplicate_mask = deduplicate_frame(prepared_raw)
    deduplicated_coercion = prepared_for_coercion.loc[~duplicate_mask].copy()

    coerced_frame, parse_masks, schema = coerce_frame_types(deduplicated_coercion)
    reason_masks = build_invalid_reason_masks(deduplicated_raw, coerced_frame, parse_masks, schema)
    invalid_mask = combine_reason_masks(reason_masks, index=coerced_frame.index)
    cleaned = coerced_frame.loc[~invalid_mask].copy()
    cleaned = assign_transaction_order(cleaned, schema)

    ensure_directory(output_parquet_path.parent)
    cleaned.to_parquet(output_parquet_path, index=False)

    report = {
        "input_path": ",".join(str(path) for path in input_paths),
        "input_paths": [str(path) for path in input_paths],
        "input_row_counts": input_row_counts,
        "output_parquet_path": str(output_parquet_path),
        "rows_in": int(len(prepared_raw)),
        "duplicate_rows_dropped": int(duplicate_mask.astype(bool).sum()),
        "invalid_rows_dropped": int(invalid_mask.fillna(False).astype(bool).sum()),
        "rows_out": int(len(cleaned)),
        "drop_reasons": count_reason_masks(reason_masks),
        "columns_out": list(cleaned.columns),
    }
    write_report(output_report_path, report)

    return PreprocessResult(
        input_path=",".join(str(path) for path in input_paths),
        parquet_path=str(output_parquet_path),
        report_path=str(output_report_path),
        rows_in=report["rows_in"],
        rows_out=report["rows_out"],
        duplicate_rows_dropped=report["duplicate_rows_dropped"],
        invalid_rows_dropped=report["invalid_rows_dropped"],
        drop_reasons=report["drop_reasons"],
    )


def default_output_paths() -> tuple[Path, Path]:
    """Return the configured default interim output locations."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return paths["interim_transactions_clean"], paths["interim_preprocessing_report"]


def result_as_report(result: PreprocessResult) -> dict[str, Any]:
    """Convert a result dataclass into a plain mapping."""

    return asdict(result)

"""Deterministic preprocessing pipeline for raw fraud transaction CSV files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


BOOLEAN_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
BOOLEAN_FALSE_VALUES = {"0", "false", "f", "no", "n"}


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
    for path in sorted(input_paths, key=lambda item: item.as_posix()):
        raw_input = load_raw_csv(path)
        prepared_raw_frame, prepared_coercion_frame = prepare_raw_frame(raw_input, source_name=path.name)
        prepared_frames.append(prepared_raw_frame)
        coercion_frames.append(prepared_coercion_frame)
        input_row_counts[str(path)] = int(len(raw_input))

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

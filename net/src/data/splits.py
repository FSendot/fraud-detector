"""Deterministic chronological dataset splitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import DEFAULT_PATHS_FILE, dump_yaml_file, load_paths_config, load_yaml_file
from common.io import ensure_directory


DEFAULT_SPLITS_FILE = Path(__file__).resolve().parents[2] / "configs" / "splits.yaml"
DEFAULT_ORDER_COLUMNS = ("transaction_timestamp", "step", "transaction_order", "source_row_number")
DEFAULT_TRANSACTION_ID_COLUMNS = ("current_transaction_id", "transaction_id", "event_id", "payment_id")


@dataclass(frozen=True)
class SplitBuildResult:
    """Output locations and row counts for a split build."""

    behavioral_input_path: str
    sequence_meta_input_path: str
    train_ids_path: str
    valid_ids_path: str
    test_ids_path: str
    report_path: str
    split_mode: str
    transaction_id_column: str
    order_column: str
    train_rows: int
    valid_rows: int
    test_rows: int


def load_behavioral_features(input_path: Path) -> pd.DataFrame:
    """Load the behavioral feature parquet."""

    return pd.read_parquet(input_path)


def load_sequence_meta(input_path: Path) -> pd.DataFrame:
    """Load the sequence metadata parquet."""

    return pd.read_parquet(input_path)


def load_split_ids(path: Path) -> pd.DataFrame:
    """Load a canonical split-ID parquet file."""

    frame = pd.read_parquet(path)
    if "transaction_id" not in frame.columns:
        msg = f"split file is missing transaction_id: {path}"
        raise ValueError(msg)
    frame["transaction_id"] = frame["transaction_id"].astype("string")
    return frame


def load_split_config(path: Path = DEFAULT_SPLITS_FILE) -> dict[str, Any]:
    """Load the canonical split configuration."""

    payload = load_yaml_file(path)
    strategy = payload.get("split_strategy", {})
    if not isinstance(strategy, dict):
        msg = f"expected 'split_strategy' to be a mapping in {path}"
        raise ValueError(msg)
    payload["split_strategy"] = strategy
    return payload


def choose_transaction_id_column(frame: pd.DataFrame, preferences: list[str] | tuple[str, ...]) -> str:
    """Choose the transaction identifier column used for canonical split IDs."""

    for column in preferences:
        if column in frame.columns:
            return column
    msg = "could not determine a transaction id column for splits"
    raise ValueError(msg)


def choose_order_column(frame: pd.DataFrame, preferences: list[str] | tuple[str, ...]) -> str:
    """Choose the primary chronological split column."""

    for column in preferences:
        if column in frame.columns:
            return column
    msg = "could not determine a chronological order column for splits"
    raise ValueError(msg)


def _derive_transaction_id(frame: pd.DataFrame) -> pd.Series:
    if "transaction_order" in frame.columns:
        return "txn_" + frame["transaction_order"].astype("Int64").astype("string")
    if "source_row_number" in frame.columns:
        return "txn_" + frame["source_row_number"].astype("Int64").astype("string")
    msg = "could not derive transaction ids from behavioral features"
    raise ValueError(msg)


def ensure_transaction_id_column(frame: pd.DataFrame, preferences: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    """Ensure a canonical transaction id column exists in the frame."""

    working = frame.copy()
    for column in preferences:
        if column in working.columns:
            working[column] = working[column].astype("string")
            return working, column
    working["current_transaction_id"] = _derive_transaction_id(working)
    return working, "current_transaction_id"


def build_canonical_split_frame(
    behavioral_frame: pd.DataFrame,
    sequence_meta_frame: pd.DataFrame,
    *,
    transaction_id_preferences: list[str] | tuple[str, ...],
    order_column_preferences: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, str, str]:
    """Build the canonical row-level frame used to define splits."""

    behavioral, transaction_id_column = ensure_transaction_id_column(
        behavioral_frame,
        transaction_id_preferences,
    )
    sequence_meta, sequence_transaction_id_column = ensure_transaction_id_column(
        sequence_meta_frame,
        transaction_id_preferences,
    )
    order_column = choose_order_column(behavioral, order_column_preferences)

    sequence_ids = set(sequence_meta[sequence_transaction_id_column].dropna().astype("string"))
    canonical = behavioral.loc[
        behavioral[transaction_id_column].astype("string").isin(sequence_ids)
    ].copy()
    if canonical.empty:
        msg = "no overlapping transaction ids were found between behavioral features and sequence metadata"
        raise ValueError(msg)

    duplicate_ids = canonical[transaction_id_column].astype("string").duplicated(keep=False)
    if bool(duplicate_ids.any()):
        msg = "transaction ids must be unique before splitting"
        raise ValueError(msg)

    sort_columns = [order_column]
    for fallback in DEFAULT_ORDER_COLUMNS:
        if fallback in canonical.columns and fallback not in sort_columns:
            sort_columns.append(fallback)
    canonical = canonical.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)
    return canonical, transaction_id_column, order_column


def _normalize_cutoff(value: Any, *, order_series: pd.Series) -> Any:
    if value is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(order_series):
        return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value)
    return value


def _split_by_proportions(frame: pd.DataFrame, proportions: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    train_size = int(total * proportions["train"])
    valid_size = int(total * proportions["valid"])
    test_size = total - train_size - valid_size

    train_end = train_size
    valid_end = train_size + valid_size
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:valid_end].copy(),
        frame.iloc[valid_end : valid_end + test_size].copy(),
    )


def _split_by_cutoffs(
    frame: pd.DataFrame,
    *,
    order_column: str,
    cutoffs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_series = frame[order_column]
    train_end = _normalize_cutoff(cutoffs.get("train_end"), order_series=order_series)
    valid_end = _normalize_cutoff(cutoffs.get("valid_end"), order_series=order_series)

    if train_end is None or valid_end is None:
        msg = "date/order cutoff mode requires both train_end and valid_end"
        raise ValueError(msg)

    train_mask = order_series <= train_end
    valid_mask = (order_series > train_end) & (order_series <= valid_end)
    test_mask = order_series > valid_end

    return (
        frame.loc[train_mask].copy(),
        frame.loc[valid_mask].copy(),
        frame.loc[test_mask].copy(),
    )


def split_canonical_frame(
    frame: pd.DataFrame,
    *,
    order_column: str,
    strategy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the canonical ordered frame by proportions or explicit cutoffs."""

    mode = strategy.get("mode", "proportions")
    if mode == "proportions":
        proportions = strategy.get("proportions", {})
        required = {"train", "valid", "test"}
        if set(proportions) != required:
            msg = "proportion mode requires train, valid, and test proportions"
            raise ValueError(msg)
        if abs(sum(float(proportions[key]) for key in required) - 1.0) > 1e-9:
            msg = "train, valid, and test proportions must sum to 1.0"
            raise ValueError(msg)
        return _split_by_proportions(frame, {key: float(proportions[key]) for key in required})
    if mode == "cutoffs":
        return _split_by_cutoffs(frame, order_column=order_column, cutoffs=strategy.get("cutoffs", {}))
    msg = f"unsupported split mode: {mode}"
    raise ValueError(msg)


def _split_id_frame(frame: pd.DataFrame, *, transaction_id_column: str, split_name: str) -> pd.DataFrame:
    id_frame = pd.DataFrame(
        {
            "transaction_id": frame[transaction_id_column].astype("string").reset_index(drop=True),
            "split": pd.Series([split_name] * len(frame), dtype="string"),
        }
    )
    return id_frame


def validate_zero_overlap(train_ids: pd.DataFrame, valid_ids: pd.DataFrame, test_ids: pd.DataFrame) -> None:
    """Ensure the canonical splits are mutually exclusive."""

    train_set = set(train_ids["transaction_id"])
    valid_set = set(valid_ids["transaction_id"])
    test_set = set(test_ids["transaction_id"])
    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        msg = "train/valid/test splits overlap"
        raise ValueError(msg)


def _time_range_payload(frame: pd.DataFrame, *, order_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"min": None, "max": None}
    minimum = frame[order_column].min()
    maximum = frame[order_column].max()
    if hasattr(minimum, "isoformat"):
        return {"min": minimum.isoformat(), "max": maximum.isoformat()}
    return {"min": minimum.item() if hasattr(minimum, "item") else minimum, "max": maximum.item() if hasattr(maximum, "item") else maximum}


def write_split_report(path: Path, payload: dict[str, Any]) -> None:
    """Persist the split report as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_output_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return configured default input and output locations."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["processed_behavioral_features"],
        paths["processed_sequences_meta"],
        paths["split_train_ids"],
        paths["split_valid_ids"],
        paths["split_test_ids"],
        paths["split_report"],
    )


def make_and_write_splits(
    *,
    behavioral_input_path: Path,
    sequence_meta_input_path: Path,
    train_ids_path: Path,
    valid_ids_path: Path,
    test_ids_path: Path,
    report_path: Path,
    split_config_path: Path = DEFAULT_SPLITS_FILE,
) -> SplitBuildResult:
    """Build and persist deterministic chronological splits."""

    config = load_split_config(split_config_path)
    strategy = config["split_strategy"]
    transaction_id_preferences = tuple(strategy.get("transaction_id_preference", DEFAULT_TRANSACTION_ID_COLUMNS))
    order_column_preferences = tuple(strategy.get("order_column_preference", DEFAULT_ORDER_COLUMNS))

    behavioral = load_behavioral_features(behavioral_input_path)
    sequence_meta = load_sequence_meta(sequence_meta_input_path)
    canonical, transaction_id_column, order_column = build_canonical_split_frame(
        behavioral,
        sequence_meta,
        transaction_id_preferences=transaction_id_preferences,
        order_column_preferences=order_column_preferences,
    )

    train_frame, valid_frame, test_frame = split_canonical_frame(
        canonical,
        order_column=order_column,
        strategy=strategy,
    )
    train_ids = _split_id_frame(train_frame, transaction_id_column=transaction_id_column, split_name="train")
    valid_ids = _split_id_frame(valid_frame, transaction_id_column=transaction_id_column, split_name="valid")
    test_ids = _split_id_frame(test_frame, transaction_id_column=transaction_id_column, split_name="test")

    validate_zero_overlap(train_ids, valid_ids, test_ids)

    ensure_directory(train_ids_path.parent)
    train_ids.to_parquet(train_ids_path, index=False)
    valid_ids.to_parquet(valid_ids_path, index=False)
    test_ids.to_parquet(test_ids_path, index=False)

    report = {
        "split_mode": strategy.get("mode", "proportions"),
        "transaction_id_column": transaction_id_column,
        "order_column": order_column,
        "row_counts": {
            "train": int(len(train_ids)),
            "valid": int(len(valid_ids)),
            "test": int(len(test_ids)),
            "total": int(len(canonical)),
        },
        "time_ranges": {
            "train": _time_range_payload(train_frame, order_column=order_column),
            "valid": _time_range_payload(valid_frame, order_column=order_column),
            "test": _time_range_payload(test_frame, order_column=order_column),
        },
        "zero_overlap_verified": True,
        "config_path": str(split_config_path),
    }
    write_split_report(report_path, report)
    dump_yaml_file(split_config_path, config)

    return SplitBuildResult(
        behavioral_input_path=str(behavioral_input_path),
        sequence_meta_input_path=str(sequence_meta_input_path),
        train_ids_path=str(train_ids_path),
        valid_ids_path=str(valid_ids_path),
        test_ids_path=str(test_ids_path),
        report_path=str(report_path),
        split_mode=strategy.get("mode", "proportions"),
        transaction_id_column=transaction_id_column,
        order_column=order_column,
        train_rows=int(len(train_ids)),
        valid_rows=int(len(valid_ids)),
        test_rows=int(len(test_ids)),
    )

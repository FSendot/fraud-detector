"""Deterministic GRU sequence dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from common.io import ensure_directory
from sequences.schema import (
    DEFAULT_PADDING_STRATEGY,
    DEFAULT_SEQUENCE_LENGTH,
    ENTITY_KEY_CANDIDATES,
    EXCLUDED_CURRENT_COLUMNS,
    EXCLUDED_SEQUENCE_COLUMNS,
    ORDER_COLUMN_CANDIDATES,
    SequenceSchema,
    TARGET_COLUMN_CANDIDATES,
    TRANSACTION_ID_CANDIDATES,
    write_sequence_schema,
)


@dataclass(frozen=True)
class SequenceBuildResult:
    """Output locations and core dimensions for a sequence build."""

    input_path: str
    x_seq_path: str
    x_current_path: str
    y_path: str
    meta_path: str
    schema_path: str
    sample_count: int
    sequence_length: int
    sequence_feature_count: int
    current_feature_count: int
    entity_key: str


def load_behavioral_features(input_path: Path) -> pd.DataFrame:
    """Load the behavioral feature parquet."""

    return pd.read_parquet(input_path)


def choose_entity_key(frame: pd.DataFrame) -> str:
    """Choose the default grouping key for historical sequences."""

    for column in ENTITY_KEY_CANDIDATES:
        if column in frame.columns:
            return column
    msg = "could not determine a default sequence entity key"
    raise ValueError(msg)


def choose_order_columns(frame: pd.DataFrame) -> list[str]:
    """Choose the stable chronological ordering columns."""

    columns = [column for column in ORDER_COLUMN_CANDIDATES if column in frame.columns]
    if not columns:
        msg = "sequence building requires at least one stable ordering column"
        raise ValueError(msg)
    return columns


def choose_target_columns(frame: pd.DataFrame) -> list[str]:
    """Choose supervised target columns preserved separately from model inputs."""

    columns = [column for column in TARGET_COLUMN_CANDIDATES if column in frame.columns]
    if not columns:
        msg = "sequence building requires a target column such as is_fraud"
        raise ValueError(msg)
    return columns


def choose_transaction_id(frame: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """Choose or derive a deterministic transaction identifier."""

    for column in TRANSACTION_ID_CANDIDATES:
        if column in frame.columns:
            return frame[column].astype("string"), column, "source"
    if "transaction_order" in frame.columns:
        series = "txn_" + frame["transaction_order"].astype("Int64").astype("string")
        return series, "current_transaction_id", "derived_from_transaction_order"
    if "source_row_number" in frame.columns:
        series = "txn_" + frame["source_row_number"].astype("Int64").astype("string")
        return series, "current_transaction_id", "derived_from_source_row_number"
    msg = "could not determine a stable transaction identifier"
    raise ValueError(msg)


def choose_sequence_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Choose numeric history features to include in the GRU tensor."""

    columns: list[str] = []
    for column in frame.columns:
        if column in EXCLUDED_SEQUENCE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if not columns:
        msg = "no numeric sequence feature columns were available"
        raise ValueError(msg)
    return columns


def choose_current_feature_columns(frame: pd.DataFrame, *, target_columns: list[str]) -> list[str]:
    """Choose current-row features for the non-sequential model branch."""

    excluded = set(EXCLUDED_CURRENT_COLUMNS) | set(target_columns)
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if not columns:
        msg = "no numeric current feature columns were available"
        raise ValueError(msg)
    return columns


def _float_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame.loc[:, columns].copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce").astype("Float32")
    return converted


def _build_sequence_tensor(
    ordered_frame: pd.DataFrame,
    *,
    entity_key: str,
    sequence_feature_columns: list[str],
    sequence_length: int,
) -> np.ndarray:
    """Build left-padded prior-history tensors for each transaction row."""

    feature_values = _float_frame(ordered_frame, sequence_feature_columns).fillna(0.0).to_numpy(dtype=np.float32)
    sequence_tensor = np.zeros(
        (len(ordered_frame), sequence_length, len(sequence_feature_columns)),
        dtype=np.float32,
    )

    start = 0
    entity_values = ordered_frame[entity_key].astype("string")
    while start < len(ordered_frame):
        end = start + 1
        while end < len(ordered_frame) and entity_values.iloc[end] == entity_values.iloc[start]:
            end += 1

        entity_matrix = feature_values[start:end]
        for offset in range(end - start):
            history_start = max(0, offset - sequence_length)
            history = entity_matrix[history_start:offset]
            history_length = len(history)
            if history_length > 0:
                # Leakage prevention: the current row at `offset` is excluded.
                # Only strictly earlier rows from the same entity populate the sequence.
                sequence_tensor[start + offset, -history_length:, :] = history
        start = end

    return sequence_tensor


def build_sequence_schema(
    *,
    entity_key: str,
    transaction_id_column: str,
    transaction_id_source: str,
    order_columns: list[str],
    sequence_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    meta_columns: list[str],
    sequence_length: int,
) -> SequenceSchema:
    """Create the machine-readable schema for the sequence dataset."""

    return SequenceSchema(
        dataset_name="gru_sequences",
        entity_key=entity_key,
        transaction_id_column=transaction_id_column,
        transaction_id_source=transaction_id_source,
        sequence_length=sequence_length,
        padding_strategy=DEFAULT_PADDING_STRATEGY,
        order_columns=tuple(order_columns),
        sequence_feature_columns=tuple(sequence_feature_columns),
        current_feature_columns=tuple(current_feature_columns),
        target_columns=tuple(target_columns),
        meta_columns=tuple(meta_columns),
        leakage_prevention=(
            "Sequences are grouped by the entity key and sorted by stable chronological columns. "
            "Each sample's X_seq contains up to the previous N transactions for that entity, "
            "never the current row or any future row. Short histories are left-padded with zeros."
        ),
    )


def build_sequence_dataset(
    frame: pd.DataFrame,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, SequenceSchema]:
    """Build deterministic GRU-ready sequence and current-row datasets."""

    entity_key = choose_entity_key(frame)
    order_columns = choose_order_columns(frame)
    target_columns = choose_target_columns(frame)
    transaction_id, transaction_id_column, transaction_id_source = choose_transaction_id(frame)

    working = frame.copy()
    working["current_transaction_id"] = transaction_id
    working["_original_position"] = pd.Series(range(len(working)), dtype="Int64")

    sort_columns = [entity_key, *order_columns]
    ordered = working.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)

    sequence_feature_columns = choose_sequence_feature_columns(ordered)
    current_feature_columns = choose_current_feature_columns(ordered, target_columns=target_columns)

    # Grouping key: sequences are built per entity, defaulting to the origin-side
    # account when present, so each sample only sees that account's own history.
    # Padding strategy: short histories are left-padded with zeros so the most
    # recent prior transaction is aligned at the end of the fixed-length window.
    # Sequence length: defaults to 10 prior transactions per sample.
    # Leakage prevention assumption: history windows exclude the current row and
    # all future rows, and ordering is deterministic via stable chronological keys.
    x_seq = _build_sequence_tensor(
        ordered,
        entity_key=entity_key,
        sequence_feature_columns=sequence_feature_columns,
        sequence_length=sequence_length,
    )

    x_current = _float_frame(ordered, current_feature_columns)
    y = ordered.loc[:, target_columns].copy()
    meta = ordered.loc[:, ["current_transaction_id", entity_key, *order_columns, "_original_position"]].copy()
    meta["sequence_length"] = sequence_length
    meta["history_length"] = (
        ordered.groupby(entity_key, sort=False, dropna=False).cumcount().clip(upper=sequence_length).astype("Int64")
    )

    restore_order = ordered["_original_position"].sort_values(kind="mergesort").index

    x_seq = x_seq[restore_order.to_numpy()]
    x_current = x_current.iloc[restore_order].reset_index(drop=True)
    y = y.iloc[restore_order].reset_index(drop=True)
    meta = meta.iloc[restore_order].drop(columns="_original_position").reset_index(drop=True)

    schema = build_sequence_schema(
        entity_key=entity_key,
        transaction_id_column="current_transaction_id",
        transaction_id_source=transaction_id_source,
        order_columns=order_columns,
        sequence_feature_columns=sequence_feature_columns,
        current_feature_columns=current_feature_columns,
        target_columns=target_columns,
        meta_columns=list(meta.columns),
        sequence_length=sequence_length,
    )
    return x_seq, x_current, y, meta, schema


def default_output_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return configured default input and output locations."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["processed_behavioral_features"],
        paths["processed_sequences_x_seq"],
        paths["processed_sequences_x_current"],
        paths["processed_sequences_y"],
        paths["processed_sequences_meta"],
        paths["artifact_sequence_schema"],
    )


def build_and_write_sequence_dataset(
    *,
    input_path: Path,
    x_seq_path: Path,
    x_current_path: Path,
    y_path: Path,
    meta_path: Path,
    schema_path: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> SequenceBuildResult:
    """Build the sequence dataset and persist all artifacts."""

    frame = load_behavioral_features(input_path)
    x_seq, x_current, y, meta, schema = build_sequence_dataset(frame, sequence_length=sequence_length)

    ensure_directory(x_seq_path.parent)
    np.save(x_seq_path, x_seq)
    x_current.to_parquet(x_current_path, index=False)
    y.to_parquet(y_path, index=False)
    meta.to_parquet(meta_path, index=False)
    write_sequence_schema(schema_path, schema)

    return SequenceBuildResult(
        input_path=str(input_path),
        x_seq_path=str(x_seq_path),
        x_current_path=str(x_current_path),
        y_path=str(y_path),
        meta_path=str(meta_path),
        schema_path=str(schema_path),
        sample_count=int(len(meta)),
        sequence_length=sequence_length,
        sequence_feature_count=len(schema.sequence_feature_columns),
        current_feature_count=len(schema.current_feature_columns),
        entity_key=schema.entity_key,
    )

"""Historical behavioral feature engineering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from common.io import ensure_directory
from features.feature_registry import FeatureRegistry, FeatureSpec, write_feature_registry


PREFERRED_ENTITY_COLUMNS = (
    "name_orig",
    "sender_id",
    "sender",
    "account_id",
    "customer_id",
    "user_id",
)
PREFERRED_DIVERSITY_COLUMNS = (
    "name_dest",
    "merchant_id",
    "merchant",
    "merchant_name",
)
SOURCE_COLUMN_SPECS: dict[str, str] = {
    "transaction_id": "Stable transaction identifier retained for downstream split reuse.",
    "source_row_number": "Original row number from the raw input after deterministic preprocessing.",
    "transaction_order": "Stable global transaction order assigned during preprocessing.",
    "transaction_timestamp": "Parsed event timestamp when the source dataset provides one.",
    "step": "Source-provided transaction step or sequence counter.",
    "type": "Observed transaction type label.",
    "name_orig": "Default origin-side behavioral entity for historical grouping.",
    "name_dest": "Destination-side account or entity identifier.",
    "amount": "Observed transaction amount used for historical rolling statistics.",
    "is_fraud": "Ground-truth fraud label carried through for supervised training.",
    "is_flagged_fraud": "Observed source-system fraud flag carried through as a passthrough field.",
}


@dataclass(frozen=True)
class BehavioralFeatureBuildResult:
    """Output locations and feature counts for a build run."""

    input_path: str
    output_parquet_path: str
    feature_dict_path: str
    rows_out: int
    source_feature_count: int
    derived_feature_count: int
    entity_column: str


def load_base_features(input_path: Path) -> pd.DataFrame:
    """Load the base feature parquet."""

    return pd.read_parquet(input_path)


def choose_entity_column(frame: pd.DataFrame) -> str:
    """Choose the default entity used to define per-account history."""

    for column in PREFERRED_ENTITY_COLUMNS:
        if column in frame.columns:
            return column
    msg = "could not determine a default behavioral entity column"
    raise ValueError(msg)


def choose_diversity_column(frame: pd.DataFrame, *, entity_column: str) -> str | None:
    """Choose a counterparty-like column for prior diversity features."""

    for column in PREFERRED_DIVERSITY_COLUMNS:
        if column in frame.columns and column != entity_column:
            return column
    return None


def _ordered_history_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for candidate in ("transaction_timestamp", "step", "transaction_order", "source_row_number"):
        if candidate in frame.columns and candidate not in columns:
            columns.append(candidate)
    return columns


def _prior_unique_count(series: pd.Series, window: int) -> pd.Series:
    """Count unique non-null values over the prior fixed-length history window."""

    history: deque[str] = deque(maxlen=window)
    counts: list[int] = []

    for value in series.astype("string"):
        counts.append(len(set(history)))
        if pd.notna(value) and value != "":
            history.append(str(value))
    return pd.Series(counts, index=series.index, dtype="Int64")


def _groupwise_prior_unique_count(frame: pd.DataFrame, *, entity_column: str, value_column: str, window: int) -> pd.Series:
    """Apply prior-window unique counts per entity and preserve the row index."""

    pieces: list[pd.Series] = []
    for _, group in frame.groupby(entity_column, sort=False, dropna=False):
        pieces.append(_prior_unique_count(group[value_column], window))
    return pd.concat(pieces).sort_index()


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _float_result(series: pd.Series) -> pd.Series:
    return pd.Series(series, index=series.index, dtype="Float64")


def _source_columns(frame: pd.DataFrame, *, entity_column: str) -> list[str]:
    columns = list(frame.columns)
    if entity_column not in columns:
        columns.append(entity_column)
    return columns


def _source_specs(columns: list[str], frame: pd.DataFrame, *, entity_column: str) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for column in columns:
        description = SOURCE_COLUMN_SPECS.get(column, "Passthrough source column retained for downstream joins.")
        if column == entity_column and column not in SOURCE_COLUMN_SPECS:
            description = "Default behavioral grouping entity retained for downstream joins."
        specs.append(
            FeatureSpec(
                name=column,
                group="source",
                dtype=str(frame[column].dtype),
                source="source",
                point_in_time_safe=True,
                description=description,
            )
        )
    return specs


def _derived_descriptions(*, entity_column: str, diversity_column: str | None) -> dict[str, str]:
    descriptions = {
        "previous_transaction_amount": f"Amount from the immediately preceding transaction for entity {entity_column}.",
        "previous_transaction_timestamp": f"Timestamp from the immediately preceding transaction for entity {entity_column}.",
        "seconds_since_previous_transaction": f"Elapsed seconds since the prior transaction for entity {entity_column}.",
        "prior_5_transaction_count": f"Count of available prior transactions for entity {entity_column}, capped at 5.",
        "prior_10_transaction_count": f"Count of available prior transactions for entity {entity_column}, capped at 10.",
        "prior_5_amount_sum": f"Sum of transaction amounts over the prior 5 transactions for entity {entity_column}.",
        "prior_5_amount_mean": f"Mean transaction amount over the prior 5 transactions for entity {entity_column}.",
        "prior_5_amount_std": f"Population standard deviation of transaction amounts over the prior 5 transactions for entity {entity_column}.",
        "prior_10_amount_sum": f"Sum of transaction amounts over the prior 10 transactions for entity {entity_column}.",
        "prior_10_amount_mean": f"Mean transaction amount over the prior 10 transactions for entity {entity_column}.",
        "prior_10_amount_std": f"Population standard deviation of transaction amounts over the prior 10 transactions for entity {entity_column}.",
    }
    if diversity_column is not None:
        descriptions[f"prior_5_unique_{diversity_column}_count"] = (
            f"Unique {diversity_column} values seen over the prior 5 transactions for entity {entity_column}."
        )
        descriptions[f"prior_10_unique_{diversity_column}_count"] = (
            f"Unique {diversity_column} values seen over the prior 10 transactions for entity {entity_column}."
        )
    return descriptions


def _derived_specs(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    diversity_column: str | None,
    source_columns: list[str],
) -> list[FeatureSpec]:
    descriptions = _derived_descriptions(entity_column=entity_column, diversity_column=diversity_column)
    return [
        FeatureSpec(
            name=column,
            group="derived",
            dtype=str(frame[column].dtype),
            source="engineered",
            point_in_time_safe=True,
            description=descriptions[column],
        )
        for column in frame.columns
        if column not in source_columns and column in descriptions
    ]


def build_behavioral_feature_frame(base_frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureRegistry]:
    """Build strictly past-dependent behavioral features."""

    entity_column = choose_entity_column(base_frame)
    diversity_column = choose_diversity_column(base_frame, entity_column=entity_column)
    source_columns = _source_columns(base_frame, entity_column=entity_column)

    working = base_frame.loc[:, source_columns].copy()
    working["_original_position"] = pd.Series(range(len(working)), dtype="Int64")

    sort_columns = [entity_column, *_ordered_history_columns(working)]
    working = working.sort_values(sort_columns, kind="mergesort", na_position="last").copy()

    grouped = working.groupby(entity_column, sort=False, dropna=False)
    prior_count = grouped.cumcount().astype("Int64")

    if "amount" not in working.columns:
        msg = "behavioral feature generation requires an amount column"
        raise ValueError(msg)
    working["amount"] = _safe_float(working["amount"])
    grouped = working.groupby(entity_column, sort=False, dropna=False)

    # Leakage prevention: shift the amount series before any rolling aggregation
    # so each window can only see transactions that happened strictly earlier.
    lagged_amount = grouped["amount"].shift(1)
    lagged_amount_grouped = lagged_amount.groupby(working[entity_column], sort=False, dropna=False)

    working["previous_transaction_amount"] = _float_result(lagged_amount)
    working["prior_5_transaction_count"] = prior_count.clip(upper=5).astype("Int64")
    working["prior_10_transaction_count"] = prior_count.clip(upper=10).astype("Int64")

    for window in (5, 10):
        rolling = lagged_amount_grouped.rolling(window=window, min_periods=1)
        working[f"prior_{window}_amount_sum"] = _float_result(rolling.sum().reset_index(level=0, drop=True))
        working[f"prior_{window}_amount_mean"] = _float_result(rolling.mean().reset_index(level=0, drop=True))
        working[f"prior_{window}_amount_std"] = _float_result(
            rolling.std(ddof=0).reset_index(level=0, drop=True)
        )

    if "transaction_timestamp" in working.columns:
        # Leakage prevention: the "previous" timestamp is a strict one-row lag
        # within each entity history, never the current row's timestamp.
        working["previous_transaction_timestamp"] = grouped["transaction_timestamp"].shift(1)
        seconds_delta = (
            working["transaction_timestamp"] - working["previous_transaction_timestamp"]
        ).dt.total_seconds()
        working["seconds_since_previous_transaction"] = _float_result(seconds_delta)
    else:
        working["previous_transaction_timestamp"] = pd.Series(
            pd.NaT,
            index=working.index,
            dtype="datetime64[ns, UTC]",
        )
        working["seconds_since_previous_transaction"] = pd.Series(pd.NA, index=working.index, dtype="Float64")

    if diversity_column is not None:
        # Leakage prevention: each unique-count window is populated from a
        # history buffer before the current counterparty is appended.
        for window in (5, 10):
            working[f"prior_{window}_unique_{diversity_column}_count"] = _groupwise_prior_unique_count(
                working,
                entity_column=entity_column,
                value_column=diversity_column,
                window=window,
            )

    output_columns = [column for column in working.columns if column != "_original_position"]
    output_frame = (
        working.loc[:, output_columns + ["_original_position"]]
        .sort_values("_original_position", kind="mergesort")
        .drop(columns="_original_position")
        .reset_index(drop=True)
    )

    registry = FeatureRegistry(
        dataset_name="behavioral_features",
        source_features=tuple(_source_specs(source_columns, output_frame, entity_column=entity_column)),
        derived_features=tuple(
            _derived_specs(
                output_frame,
                entity_column=entity_column,
                diversity_column=diversity_column,
                source_columns=source_columns,
            )
        ),
    )
    return output_frame, registry


def default_output_paths() -> tuple[Path, Path, Path]:
    """Return configured default input and output locations."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["processed_base_features"],
        paths["processed_behavioral_features"],
        paths["artifact_behavioral_feature_dict"],
    )


def build_and_write_behavioral_features(
    *,
    input_path: Path,
    output_parquet_path: Path,
    feature_dict_path: Path,
) -> BehavioralFeatureBuildResult:
    """Build behavioral features and persist both the frame and its registry."""

    base_frame = load_base_features(input_path)
    feature_frame, registry = build_behavioral_feature_frame(base_frame)

    ensure_directory(output_parquet_path.parent)
    feature_frame.to_parquet(output_parquet_path, index=False)
    write_feature_registry(feature_dict_path, registry)

    entity_column = choose_entity_column(base_frame)
    return BehavioralFeatureBuildResult(
        input_path=str(input_path),
        output_parquet_path=str(output_parquet_path),
        feature_dict_path=str(feature_dict_path),
        rows_out=int(len(feature_frame)),
        source_feature_count=len(registry.source_features),
        derived_feature_count=len(registry.derived_features),
        entity_column=entity_column,
    )

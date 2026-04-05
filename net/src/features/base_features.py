"""Base per-transaction feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from common.io import ensure_directory
from features.feature_registry import FeatureRegistry, FeatureSpec, write_feature_registry


SOURCE_COLUMN_SPECS: dict[str, str] = {
    "transaction_id": "Stable transaction identifier carried through from the source dataset.",
    "source_row_number": "Original row number from the raw input after deterministic preprocessing.",
    "transaction_order": "Stable transaction order assigned during preprocessing.",
    "transaction_timestamp": "Parsed transaction timestamp when the raw dataset provides one.",
    "step": "Transaction step or sequence counter from the source dataset.",
    "type": "Transaction type label from the source dataset.",
    "name_orig": "Source-side account or entity identifier.",
    "name_dest": "Destination-side account or entity identifier.",
    "amount": "Observed transaction amount.",
    "oldbalance_org": "Source-side balance before the transaction.",
    "newbalance_orig": "Source-side balance after the transaction.",
    "oldbalance_dest": "Destination-side balance before the transaction.",
    "newbalance_dest": "Destination-side balance after the transaction.",
    "is_fraud": "Ground-truth fraud label carried through for supervised training.",
    "is_flagged_fraud": "Source-system fraud flag carried through as an observed field.",
}


@dataclass(frozen=True)
class BaseFeatureBuildResult:
    """Output locations and feature counts for a build run."""

    input_path: str
    output_parquet_path: str
    feature_dict_path: str
    rows_out: int
    source_feature_count: int
    derived_feature_count: int


def load_interim_transactions(input_path: Path) -> pd.DataFrame:
    """Load the cleaned transaction parquet."""

    return pd.read_parquet(input_path)


def _float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("Float64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    ratio = pd.Series(pd.NA, index=numerator.index, dtype="Float64")
    ratio.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return ratio


def _signed_log1p(series: pd.Series) -> pd.Series:
    valid = series.notna()
    transformed = pd.Series(pd.NA, index=series.index, dtype="Float64")
    transformed.loc[valid] = np.sign(series.loc[valid]) * np.log1p(np.abs(series.loc[valid]))
    return transformed


def _bounded_ratio(series: pd.Series, *, lower: float = -10.0, upper: float = 10.0) -> pd.Series:
    bounded = series.clip(lower=lower, upper=upper)
    return bounded.astype("Float64")


def _feature_specs_for_columns(columns: list[str], frame: pd.DataFrame) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for column in columns:
        specs.append(
            FeatureSpec(
                name=column,
                group="source",
                dtype=str(frame[column].dtype),
                source="source",
                point_in_time_safe=True,
                description=SOURCE_COLUMN_SPECS.get(column, "Passthrough source column retained for downstream modeling."),
            )
        )
    return specs


def _derived_feature_specs(frame: pd.DataFrame) -> list[FeatureSpec]:
    descriptions = {
        "balance_delta_org": "Source-side post-minus-pre balance change.",
        "balance_delta_dest": "Destination-side post-minus-pre balance change.",
        "amount_to_oldbalance_ratio": "Transaction amount divided by source-side pre-transaction balance when non-zero.",
        "amount_to_newbalance_ratio": "Transaction amount divided by source-side post-transaction balance when non-zero.",
        "amount_to_dest_oldbalance_ratio": "Transaction amount divided by destination-side pre-transaction balance when non-zero.",
        "amount_to_dest_newbalance_ratio": "Transaction amount divided by destination-side post-transaction balance when non-zero.",
        "amount_log1p": "Signed log1p transform of amount for large-value stability.",
        "oldbalance_org_log1p": "Signed log1p transform of source-side pre-transaction balance.",
        "newbalance_orig_log1p": "Signed log1p transform of source-side post-transaction balance.",
        "oldbalance_dest_log1p": "Signed log1p transform of destination-side pre-transaction balance.",
        "newbalance_dest_log1p": "Signed log1p transform of destination-side post-transaction balance.",
        "balance_delta_org_log1p": "Signed log1p transform of source-side balance delta.",
        "balance_delta_dest_log1p": "Signed log1p transform of destination-side balance delta.",
        "amount_to_oldbalance_ratio_bounded": "Clipped version of amount_to_oldbalance_ratio for stability.",
        "amount_to_newbalance_ratio_bounded": "Clipped version of amount_to_newbalance_ratio for stability.",
        "amount_to_dest_oldbalance_ratio_bounded": "Clipped version of amount_to_dest_oldbalance_ratio for stability.",
        "amount_to_dest_newbalance_ratio_bounded": "Clipped version of amount_to_dest_newbalance_ratio for stability.",
    }
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
        if column in descriptions
    ]


def build_base_feature_frame(clean_frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureRegistry]:
    """Build point-in-time-safe scalar base features from cleaned transactions."""

    source_columns = list(clean_frame.columns)
    feature_frame = clean_frame.loc[:, source_columns].copy()

    if {"oldbalance_org", "newbalance_orig"} <= set(clean_frame.columns):
        feature_frame["balance_delta_org"] = _float_series(clean_frame, "newbalance_orig") - _float_series(clean_frame, "oldbalance_org")

    if {"oldbalance_dest", "newbalance_dest"} <= set(clean_frame.columns):
        feature_frame["balance_delta_dest"] = _float_series(clean_frame, "newbalance_dest") - _float_series(clean_frame, "oldbalance_dest")

    if {"amount", "oldbalance_org"} <= set(clean_frame.columns):
        feature_frame["amount_to_oldbalance_ratio"] = _safe_ratio(
            _float_series(clean_frame, "amount"),
            _float_series(clean_frame, "oldbalance_org"),
        )

    if {"amount", "newbalance_orig"} <= set(clean_frame.columns):
        feature_frame["amount_to_newbalance_ratio"] = _safe_ratio(
            _float_series(clean_frame, "amount"),
            _float_series(clean_frame, "newbalance_orig"),
        )

    if {"amount", "oldbalance_dest"} <= set(clean_frame.columns):
        feature_frame["amount_to_dest_oldbalance_ratio"] = _safe_ratio(
            _float_series(clean_frame, "amount"),
            _float_series(clean_frame, "oldbalance_dest"),
        )

    if {"amount", "newbalance_dest"} <= set(clean_frame.columns):
        feature_frame["amount_to_dest_newbalance_ratio"] = _safe_ratio(
            _float_series(clean_frame, "amount"),
            _float_series(clean_frame, "newbalance_dest"),
        )

    for column in ("amount", "oldbalance_org", "newbalance_orig", "oldbalance_dest", "newbalance_dest"):
        if column in clean_frame.columns:
            feature_frame[f"{column}_log1p"] = _signed_log1p(_float_series(clean_frame, column))

    for column in ("balance_delta_org", "balance_delta_dest"):
        if column in feature_frame.columns:
            feature_frame[f"{column}_log1p"] = _signed_log1p(_float_series(feature_frame, column))

    for column in (
        "amount_to_oldbalance_ratio",
        "amount_to_newbalance_ratio",
        "amount_to_dest_oldbalance_ratio",
        "amount_to_dest_newbalance_ratio",
    ):
        if column in feature_frame.columns:
            feature_frame[f"{column}_bounded"] = _bounded_ratio(_float_series(feature_frame, column))

    registry = FeatureRegistry(
        dataset_name="base_features",
        source_features=tuple(_feature_specs_for_columns(source_columns, feature_frame)),
        derived_features=tuple(_derived_feature_specs(feature_frame)),
    )
    return feature_frame, registry


def default_output_paths() -> tuple[Path, Path, Path]:
    """Return configured default input and output locations."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return (
        paths["interim_transactions_clean"],
        paths["processed_base_features"],
        paths["artifact_feature_dict"],
    )


def build_and_write_base_features(
    *,
    input_path: Path,
    output_parquet_path: Path,
    feature_dict_path: Path,
) -> BaseFeatureBuildResult:
    """Build base features and persist both the frame and its registry."""

    clean_frame = load_interim_transactions(input_path)
    feature_frame, registry = build_base_feature_frame(clean_frame)

    ensure_directory(output_parquet_path.parent)
    feature_frame.to_parquet(output_parquet_path, index=False)
    write_feature_registry(feature_dict_path, registry)

    return BaseFeatureBuildResult(
        input_path=str(input_path),
        output_parquet_path=str(output_parquet_path),
        feature_dict_path=str(feature_dict_path),
        rows_out=int(len(feature_frame)),
        source_feature_count=len(registry.source_features),
        derived_feature_count=len(registry.derived_features),
    )

"""Validation helpers for deterministic transaction preprocessing."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def non_empty_mask(series: pd.Series) -> pd.Series:
    """Return a mask for cells that contain a non-empty value."""

    normalized = series.astype("string").str.strip()
    return normalized.notna() & normalized.ne("")


def parse_failure_mask(raw_series: pd.Series, parsed_series: pd.Series) -> pd.Series:
    """Flag rows that had a value but could not be parsed into the target type."""

    return non_empty_mask(raw_series) & parsed_series.isna()


def missing_required_fields_mask(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.Series:
    """Flag rows with null or blank values in required identifier columns."""

    if not columns:
        return pd.Series(False, index=frame.index, dtype="boolean")

    masks = [~non_empty_mask(frame[column]) for column in columns]
    return pd.concat(masks, axis=1).any(axis=1)


def negative_value_mask(series: pd.Series) -> pd.Series:
    """Flag numeric rows that are negative."""

    return series.notna() & series.lt(0)


def negative_any_mask(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.Series:
    """Flag rows with any negative value across the given numeric columns."""

    if not columns:
        return pd.Series(False, index=frame.index, dtype="boolean")
    masks = [negative_value_mask(frame[column]) for column in columns]
    return pd.concat(masks, axis=1).any(axis=1)


def count_reason_masks(reason_masks: Mapping[str, pd.Series]) -> dict[str, int]:
    """Count flagged rows for each validation reason."""

    return {
        reason: int(mask.fillna(False).astype(bool).sum())
        for reason, mask in reason_masks.items()
        if int(mask.fillna(False).astype(bool).sum()) > 0
    }


def combine_reason_masks(reason_masks: Mapping[str, pd.Series], *, index: pd.Index) -> pd.Series:
    """Combine several boolean masks into a single invalid-row mask."""

    if not reason_masks:
        return pd.Series(False, index=index, dtype="boolean")
    return pd.concat(list(reason_masks.values()), axis=1).fillna(False).any(axis=1)


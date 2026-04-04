"""Deterministic class balancing helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DownsampleResult:
    """Result of deterministic majority-class downsampling."""

    frame: pd.DataFrame
    rows_before: int
    rows_after: int
    positive_rows: int
    negative_rows_before: int
    negative_rows_after: int
    downsample_ratio: float


def _deterministic_take_positions(size: int, target_size: int) -> np.ndarray:
    """Choose stable, evenly spaced positions from an ordered frame."""

    if target_size >= size:
        return np.arange(size, dtype=int)
    if target_size <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, size - 1, num=target_size, dtype=int)


def downsample_training_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    downsample_ratio: float,
) -> DownsampleResult:
    """Deterministically downsample the majority class in the training set only."""

    if downsample_ratio <= 0:
        msg = "downsample_ratio must be greater than 0"
        raise ValueError(msg)

    positives = frame.loc[frame[label_column] == 1].copy()
    negatives = frame.loc[frame[label_column] == 0].copy()
    if positives.empty or negatives.empty:
        return DownsampleResult(
            frame=frame.copy().reset_index(drop=True),
            rows_before=int(len(frame)),
            rows_after=int(len(frame)),
            positive_rows=int(len(positives)),
            negative_rows_before=int(len(negatives)),
            negative_rows_after=int(len(negatives)),
            downsample_ratio=downsample_ratio,
        )

    target_negative_rows = min(len(negatives), int(len(positives) * downsample_ratio))
    take_positions = _deterministic_take_positions(len(negatives), target_negative_rows)
    sampled_negatives = negatives.iloc[take_positions].copy()

    balanced = (
        pd.concat([positives, sampled_negatives], axis=0)
        .sort_index(kind="mergesort")
        .reset_index(drop=True)
    )
    return DownsampleResult(
        frame=balanced,
        rows_before=int(len(frame)),
        rows_after=int(len(balanced)),
        positive_rows=int(len(positives)),
        negative_rows_before=int(len(negatives)),
        negative_rows_after=int(len(sampled_negatives)),
        downsample_ratio=downsample_ratio,
    )


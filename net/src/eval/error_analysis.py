"""Error analysis helpers for the first fraud-model branch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_error_analysis_tables(
    predictions: pd.DataFrame,
    *,
    top_n: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ranked false-positive and false-negative tables."""

    false_positives = predictions.loc[
        (predictions["is_fraud"] == 0) & (predictions["predicted_label"] == 1)
    ].sort_values(["score", "transaction_id"], ascending=[False, True], kind="mergesort")
    false_negatives = predictions.loc[
        (predictions["is_fraud"] == 1) & (predictions["predicted_label"] == 0)
    ].sort_values(["score", "transaction_id"], ascending=[True, True], kind="mergesort")
    return false_positives.head(top_n).reset_index(drop=True), false_negatives.head(top_n).reset_index(drop=True)


def write_error_analysis_tables(
    *,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    false_positives_path: Path,
    false_negatives_path: Path,
) -> None:
    """Persist error analysis tables as parquet files."""

    false_positives_path.parent.mkdir(parents=True, exist_ok=True)
    false_negatives_path.parent.mkdir(parents=True, exist_ok=True)
    false_positives.to_parquet(false_positives_path, index=False)
    false_negatives.to_parquet(false_negatives_path, index=False)


"""Sequence dataset loading and split helpers for GRU-style branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.splits import load_split_ids


INTERNAL_SEQUENCE_COLUMNS = {
    "_original_position",
}


@dataclass(frozen=True)
class LoadedSequenceSchema:
    """Loaded sequence schema payload with typed convenience fields."""

    dataset_name: str
    entity_key: str
    transaction_id_column: str
    sequence_length: int
    padding_strategy: str
    order_columns: tuple[str, ...]
    sequence_feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    meta_columns: tuple[str, ...]
    leakage_prevention: str


@dataclass(frozen=True)
class SequenceResources:
    """Core sequence artifacts loaded from disk."""

    x_seq: np.memmap
    meta: pd.DataFrame
    targets: pd.Series
    schema: LoadedSequenceSchema
    usable_feature_columns: tuple[str, ...]
    usable_feature_indices: tuple[int, ...]


@dataclass(frozen=True)
class SequenceSplit:
    """Deterministic split slice for one branch stage."""

    positions: np.ndarray
    transaction_ids: pd.Series
    labels: pd.Series

    @property
    def row_count(self) -> int:
        return int(len(self.positions))


class IndexedSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapper that slices a memmapped sequence tensor by stable positions."""

    def __init__(
        self,
        x_seq: np.memmap,
        *,
        positions: np.ndarray,
        feature_indices: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self._x_seq = x_seq
        self._positions = positions.astype(np.int64, copy=False)
        self._feature_indices = feature_indices.astype(np.int64, copy=False)
        self._labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self._positions))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = int(self._positions[index])
        sequence = np.asarray(self._x_seq[position][:, self._feature_indices], dtype=np.float32)
        label = np.float32(self._labels[index])
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.float32)


def _load_schema(path: Path) -> LoadedSequenceSchema:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return LoadedSequenceSchema(
        dataset_name=str(payload["dataset_name"]),
        entity_key=str(payload["entity_key"]),
        transaction_id_column=str(payload["transaction_id_column"]),
        sequence_length=int(payload["sequence_length"]),
        padding_strategy=str(payload["padding_strategy"]),
        order_columns=tuple(payload["order_columns"]),
        sequence_feature_columns=tuple(payload["sequence_feature_columns"]),
        target_columns=tuple(payload["target_columns"]),
        meta_columns=tuple(payload["meta_columns"]),
        leakage_prevention=str(payload["leakage_prevention"]),
    )


def _usable_sequence_feature_indices(schema: LoadedSequenceSchema) -> tuple[tuple[str, ...], tuple[int, ...]]:
    columns: list[str] = []
    indices: list[int] = []
    for index, name in enumerate(schema.sequence_feature_columns):
        if name.startswith("_") or name in INTERNAL_SEQUENCE_COLUMNS:
            continue
        columns.append(name)
        indices.append(index)
    if not columns:
        msg = "no usable sequence feature columns remain after filtering internal columns"
        raise ValueError(msg)
    return tuple(columns), tuple(indices)


def load_sequence_resources(
    *,
    x_seq_path: Path,
    meta_path: Path,
    y_path: Path,
    schema_path: Path,
) -> SequenceResources:
    """Load sequence artifacts and infer the usable model feature layout."""

    x_seq = np.load(x_seq_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path)
    targets_frame = pd.read_parquet(y_path)
    schema = _load_schema(schema_path)

    if schema.transaction_id_column not in meta.columns:
        msg = f"meta file is missing transaction id column {schema.transaction_id_column!r}"
        raise ValueError(msg)
    target_column = schema.target_columns[0]
    if target_column not in targets_frame.columns:
        msg = f"target file is missing target column {target_column!r}"
        raise ValueError(msg)
    if len(meta) != len(targets_frame) or len(meta) != x_seq.shape[0]:
        msg = "sequence artifacts do not share the same row count"
        raise ValueError(msg)

    meta = meta.copy()
    meta[schema.transaction_id_column] = meta[schema.transaction_id_column].astype("string")
    usable_feature_columns, usable_feature_indices = _usable_sequence_feature_indices(schema)

    return SequenceResources(
        x_seq=x_seq,
        meta=meta,
        targets=targets_frame[target_column].astype("Int64"),
        schema=schema,
        usable_feature_columns=usable_feature_columns,
        usable_feature_indices=usable_feature_indices,
    )


def _deterministic_take_positions(size: int, target_size: int) -> np.ndarray:
    if target_size >= size:
        return np.arange(size, dtype=np.int64)
    if target_size <= 0:
        return np.array([], dtype=np.int64)
    return np.linspace(0, size - 1, num=target_size, dtype=np.int64)


def build_sequence_split(resources: SequenceResources, split_ids_path: Path) -> SequenceSplit:
    """Create a deterministic split slice from canonical transaction IDs."""

    split_ids = load_split_ids(split_ids_path)
    position_lookup = pd.Series(resources.meta.index.to_numpy(), index=resources.meta[resources.schema.transaction_id_column])
    if position_lookup.index.duplicated().any():
        msg = "sequence transaction ids must be unique"
        raise ValueError(msg)
    missing = split_ids.loc[~split_ids["transaction_id"].isin(position_lookup.index), "transaction_id"]
    if not missing.empty:
        msg = f"split ids not found in sequence metadata: {missing.iloc[:5].tolist()}"
        raise ValueError(msg)
    positions = position_lookup.loc[split_ids["transaction_id"]].to_numpy(dtype=np.int64)
    labels = resources.targets.iloc[positions].reset_index(drop=True)
    return SequenceSplit(
        positions=positions,
        transaction_ids=split_ids["transaction_id"].reset_index(drop=True).astype("string"),
        labels=labels,
    )


def downsample_training_split(split: SequenceSplit, *, downsample_ratio: float) -> SequenceSplit:
    """Deterministically downsample the majority class in the training split."""

    if downsample_ratio <= 0:
        msg = "downsample_ratio must be greater than 0"
        raise ValueError(msg)
    labels = split.labels.astype(int).to_numpy()
    positive_positions = np.flatnonzero(labels == 1)
    negative_positions = np.flatnonzero(labels == 0)
    if len(positive_positions) == 0 or len(negative_positions) == 0:
        return split
    target_negative_rows = min(len(negative_positions), int(len(positive_positions) * downsample_ratio))
    take_positions = _deterministic_take_positions(len(negative_positions), target_negative_rows)
    selected = np.concatenate([positive_positions, negative_positions[take_positions]])
    selected.sort(kind="mergesort")
    return SequenceSplit(
        positions=split.positions[selected],
        transaction_ids=split.transaction_ids.iloc[selected].reset_index(drop=True),
        labels=split.labels.iloc[selected].reset_index(drop=True),
    )


def maybe_cap_split(split: SequenceSplit, *, max_samples: int | None) -> SequenceSplit:
    """Optionally take a deterministic evenly spaced cap from a split."""

    if max_samples is None or max_samples >= len(split.positions):
        return split
    selected = _deterministic_take_positions(len(split.positions), max_samples)
    return SequenceSplit(
        positions=split.positions[selected],
        transaction_ids=split.transaction_ids.iloc[selected].reset_index(drop=True),
        labels=split.labels.iloc[selected].reset_index(drop=True),
    )


def make_sequence_loader(
    resources: SequenceResources,
    split: SequenceSplit,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic data loader for one sequence split."""

    dataset = IndexedSequenceDataset(
        resources.x_seq,
        positions=split.positions,
        feature_indices=np.asarray(resources.usable_feature_indices, dtype=np.int64),
        labels=split.labels.astype(float).to_numpy(),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

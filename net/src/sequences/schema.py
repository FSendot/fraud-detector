"""Sequence dataset schema definitions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_PADDING_STRATEGY = "left_pad_with_zeros"
TRANSACTION_ID_CANDIDATES = (
    "transaction_id",
    "event_id",
    "payment_id",
)
ENTITY_KEY_CANDIDATES = (
    "name_orig",
    "sender_id",
    "sender",
    "account_id",
    "customer_id",
    "user_id",
)
ORDER_COLUMN_CANDIDATES = (
    "transaction_timestamp",
    "step",
    "transaction_order",
    "source_row_number",
)
TARGET_COLUMN_CANDIDATES = ("is_fraud",)
EXCLUDED_SEQUENCE_COLUMNS = {
    "is_fraud",
    "is_flagged_fraud",
    "transaction_id",
    "event_id",
    "payment_id",
    "name_orig",
    "name_dest",
    "type",
    "transaction_timestamp",
    "previous_transaction_timestamp",
}
EXCLUDED_CURRENT_COLUMNS = {
    "is_fraud",
}


@dataclass(frozen=True)
class SequenceSchema:
    """Machine-readable definition of the sequence dataset contract."""

    dataset_name: str
    entity_key: str
    transaction_id_column: str
    transaction_id_source: str
    sequence_length: int
    padding_strategy: str
    order_columns: tuple[str, ...]
    sequence_feature_columns: tuple[str, ...]
    current_feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    meta_columns: tuple[str, ...]
    leakage_prevention: str

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON payload."""

        return asdict(self)


def write_sequence_schema(path: Path, schema: SequenceSchema) -> None:
    """Persist the sequence schema to JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(schema.to_payload(), handle, indent=2, sort_keys=True)
        handle.write("\n")


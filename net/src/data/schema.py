"""Schema definitions for fraud transaction preprocessing."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


REQUIRED_ID_CANDIDATES = ("name_orig", "name_dest")
NUMERIC_CANDIDATES = (
    "amount",
    "oldbalance_org",
    "newbalance_orig",
    "oldbalance_dest",
    "newbalance_dest",
)
INTEGER_CANDIDATES = ("step",)
BOOLEAN_CANDIDATES = ("is_fraud", "is_flagged_fraud")
TIMESTAMP_CANDIDATES = (
    "timestamp",
    "transaction_timestamp",
    "event_timestamp",
    "created_at",
    "updated_at",
    "date",
    "datetime",
)
STRING_CANDIDATES = ("type", "name_orig", "name_dest")
ORDER_CANDIDATES = ("transaction_timestamp", "timestamp", "event_timestamp", "step")


@dataclass(frozen=True)
class TransactionSchema:
    """Resolved schema groups for a normalized dataset."""

    columns: tuple[str, ...]
    required_id_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    balance_columns: tuple[str, ...]
    integer_columns: tuple[str, ...]
    boolean_columns: tuple[str, ...]
    timestamp_columns: tuple[str, ...]
    string_columns: tuple[str, ...]
    order_columns: tuple[str, ...]


def normalize_column_name(name: str) -> str:
    """Convert arbitrary column names into stable snake_case labels."""

    candidate = name.strip()
    candidate = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", candidate)
    candidate = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", candidate)
    candidate = re.sub(r"[^0-9A-Za-z]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_").lower()
    return candidate or "column"


def normalize_columns(columns: Iterable[str]) -> list[str]:
    """Normalize column names and make collisions deterministic."""

    seen: defaultdict[str, int] = defaultdict(int)
    normalized: list[str] = []
    for column in columns:
        base = normalize_column_name(column)
        index = seen[base]
        seen[base] += 1
        normalized.append(base if index == 0 else f"{base}_{index + 1}")
    return normalized


def build_transaction_schema(columns: Iterable[str]) -> TransactionSchema:
    """Build a schema description from normalized column names."""

    normalized = tuple(columns)
    balance_columns = tuple(column for column in normalized if "balance" in column)
    return TransactionSchema(
        columns=normalized,
        required_id_columns=tuple(column for column in REQUIRED_ID_CANDIDATES if column in normalized),
        numeric_columns=tuple(column for column in NUMERIC_CANDIDATES if column in normalized),
        balance_columns=balance_columns,
        integer_columns=tuple(column for column in INTEGER_CANDIDATES if column in normalized),
        boolean_columns=tuple(column for column in BOOLEAN_CANDIDATES if column in normalized),
        timestamp_columns=tuple(column for column in TIMESTAMP_CANDIDATES if column in normalized),
        string_columns=tuple(column for column in STRING_CANDIDATES if column in normalized),
        order_columns=tuple(column for column in ORDER_CANDIDATES if column in normalized),
    )


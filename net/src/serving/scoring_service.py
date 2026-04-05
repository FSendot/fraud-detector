"""Shadow-mode scoring service for the packaged fraud bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.feature_contract import load_feature_contract
from inference.pipeline import (
    _bundle_file_map,
    apply_fusion_and_calibration,
    load_bundle,
    predict_required_branches,
    prepare_contract_features,
)
from training.train_utils import ID_COLUMN, TARGET_COLUMN


SHADOW_MODE_NAME = "shadow_only"


@dataclass(frozen=True)
class ScoringRequestRecord:
    """Normalized request record for shadow scoring."""

    transaction_id: str
    features: dict[str, Any]
    metadata: dict[str, Any]
    label: int | None


def _coerce_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)
    raise ValueError(f"invalid label value: {value!r}")


def _parse_request_records(payload: dict[str, Any]) -> list[ScoringRequestRecord]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("request payload must contain a non-empty 'records' array")

    normalized: list[ScoringRequestRecord] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        transaction_id = record.get("transaction_id")
        features = record.get("features")
        metadata = record.get("metadata", {})
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError(f"record {index} is missing a non-empty transaction_id")
        if not isinstance(features, dict) or not features:
            raise ValueError(f"record {index} is missing a non-empty features object")
        if not isinstance(metadata, dict):
            raise ValueError(f"record {index} metadata must be an object when provided")
        label = _coerce_label(record.get("is_fraud"))
        normalized.append(
            ScoringRequestRecord(
                transaction_id=transaction_id,
                features=features,
                metadata=metadata,
                label=label,
            )
        )
    return normalized


def _records_to_frame(records: list[ScoringRequestRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            ID_COLUMN: record.transaction_id,
            TARGET_COLUMN: record.label,
            **record.features,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string")
    return frame


class ShadowScoringService:
    """Shadow-only scorer that wraps the packaged bundle runtime."""

    def __init__(self, *, bundle_path: Path) -> None:
        self.bundle = load_bundle(bundle_path)
        self.bundle_files = _bundle_file_map(self.bundle.manifest, self.bundle.bundle_root)
        contract_path = self.bundle_files.get("contracts/feature_contract_v1.json")
        if contract_path is None:
            raise FileNotFoundError("bundle is missing contracts/feature_contract_v1.json")
        self.contract = load_feature_contract(contract_path)

    @property
    def bundle_version(self) -> str:
        return str(self.bundle.manifest.get("bundle_version", "unknown"))

    def score_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score a JSON request payload and return a JSON-safe response."""

        request_id = payload.get("request_id")
        normalized_records = _parse_request_records(payload)
        request_frame = _records_to_frame(normalized_records)

        prepared = prepare_contract_features(
            request_frame,
            contract=self.contract,
            bundle_files=self.bundle_files,
        )
        branch_frame = predict_required_branches(
            prepared.contract_frame,
            contract=self.contract,
            bundle_files=self.bundle_files,
            manifest=self.bundle.manifest,
        )
        predictions = apply_fusion_and_calibration(
            branch_frame,
            bundle_files=self.bundle_files,
            manifest=self.bundle.manifest,
        )

        response_records: list[dict[str, Any]] = []
        metadata_by_id = {record.transaction_id: record.metadata for record in normalized_records}
        branch_score_columns = [column for column in branch_frame.columns if column.endswith("_score")]
        for row_index in range(len(predictions)):
            transaction_id = str(predictions.iloc[row_index][ID_COLUMN])
            branch_outputs = {}
            for column in branch_score_columns:
                branch_name = column[: -len("_score")]
                branch_outputs[branch_name] = {
                    "score": float(branch_frame.iloc[row_index][column]),
                    "predicted_label_at_0_5": int(branch_frame.iloc[row_index][f"{branch_name}_predicted_label"]),
                }
            label_value = predictions.iloc[row_index][TARGET_COLUMN]
            response_records.append(
                {
                    "transaction_id": transaction_id,
                    "is_fraud": None if pd.isna(label_value) else int(label_value),
                    "model_version": self.bundle_version,
                    "shadow_mode": True,
                    "branch_outputs": branch_outputs,
                    "raw_fused_score": float(predictions.iloc[row_index]["raw_fused_score"]),
                    "calibrated_score": float(predictions.iloc[row_index]["calibrated_score"]),
                    "decision_threshold": float(predictions.iloc[row_index]["decision_threshold"]),
                    "predicted_label": int(predictions.iloc[row_index]["predicted_label"]),
                    "metadata": metadata_by_id.get(transaction_id, {}),
                }
            )

        return {
            "request_id": request_id,
            "shadow_mode": True,
            "mode": SHADOW_MODE_NAME,
            "model_version": self.bundle_version,
            "bundle_manifest": str(self.bundle.manifest_path),
            "runtime_defaults": self.bundle.manifest["runtime_metadata"]["operational_defaults"],
            "contract_check": {
                "valid": prepared.contract_valid,
                "errors": prepared.validation_errors,
                "warnings": prepared.validation_warnings,
                "rebuild_applied": prepared.rebuild_applied,
                "source_stage": prepared.source_stage,
            },
            "records": response_records,
        }


def load_request_payload(path: Path) -> dict[str, Any]:
    """Load a JSON scoring request payload from disk."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload

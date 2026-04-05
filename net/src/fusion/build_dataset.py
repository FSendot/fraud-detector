"""Fusion-ready dataset builder for ensemble branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import DEFAULT_PATHS_FILE, load_paths_config
from common.io import ensure_directory


ID_COLUMN = "transaction_id"
LABEL_COLUMN = "is_fraud"
REQUIRED_PREDICTION_COLUMNS = (ID_COLUMN, LABEL_COLUMN, "score", "predicted_label")
DEFAULT_BRANCH_FILES = {
    "vae_nystrom": {
        "valid": "prediction_nystrom_valid",
        "test": "prediction_nystrom_test",
    },
    "tabular_nystrom": {
        "valid": "prediction_nystrom_tabular_valid",
        "test": "prediction_nystrom_tabular_test",
    },
    "tree_branch": {
        "valid": "prediction_tree_branch_valid",
        "test": "prediction_tree_branch_test",
    },
    "boosted_branch": {
        "valid": "prediction_boosted_branch_valid",
        "test": "prediction_boosted_branch_test",
    },
    "gru_branch": {
        "valid": "prediction_gru_branch_valid",
        "test": "prediction_gru_branch_test",
    },
}


@dataclass(frozen=True)
class FusionBuildResult:
    """Output summary for fusion dataset materialization."""

    fusion_valid_path: str
    fusion_test_path: str
    report_path: str
    valid_rows: int
    test_rows: int
    branch_count: int


def _load_prediction_frame(path: Path, *, branch_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"{branch_name} prediction file missing required columns {missing}: {path}"
        raise ValueError(msg)
    working = frame.loc[:, list(REQUIRED_PREDICTION_COLUMNS)].copy()
    working[ID_COLUMN] = working[ID_COLUMN].astype("string")
    if working[ID_COLUMN].duplicated().any():
        msg = f"{branch_name} prediction file has duplicate transaction_id values: {path}"
        raise ValueError(msg)
    return working


def _rename_branch_columns(frame: pd.DataFrame, *, branch_name: str) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "score": f"{branch_name}_score",
            "predicted_label": f"{branch_name}_predicted_label",
        }
    )
    return renamed


def _validate_label_consistency(base: pd.DataFrame, candidate: pd.DataFrame, *, branch_name: str) -> None:
    comparison = base[[ID_COLUMN, LABEL_COLUMN]].merge(
        candidate[[ID_COLUMN, LABEL_COLUMN]],
        on=ID_COLUMN,
        how="left",
        suffixes=("", f"_{branch_name}"),
    )
    candidate_label = f"{LABEL_COLUMN}_{branch_name}"
    if comparison[candidate_label].isna().any():
        msg = f"{branch_name} is missing labels for some base transaction ids"
        raise ValueError(msg)
    if not comparison[LABEL_COLUMN].astype(int).equals(comparison[candidate_label].astype(int)):
        msg = f"{branch_name} label values do not match the base branch"
        raise ValueError(msg)


def _join_branch_frames(frames: dict[str, pd.DataFrame], *, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    branch_names = list(frames)
    base_branch = branch_names[0]
    base = _rename_branch_columns(frames[base_branch], branch_name=base_branch)

    diagnostics: dict[str, Any] = {
        "split": split_name,
        "base_branch": base_branch,
        "base_rows": int(len(base)),
        "branch_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "missing_transaction_ids_by_branch": {},
    }

    fused = base.copy()
    base_id_set = set(base[ID_COLUMN])
    for branch_name in branch_names[1:]:
        candidate = _rename_branch_columns(frames[branch_name], branch_name=branch_name)
        _validate_label_consistency(base, candidate, branch_name=branch_name)
        candidate_id_set = set(candidate[ID_COLUMN])
        missing_from_branch = sorted(base_id_set - candidate_id_set)
        extra_in_branch = sorted(candidate_id_set - base_id_set)
        diagnostics["missing_transaction_ids_by_branch"][branch_name] = {
            "missing_from_branch_count": int(len(missing_from_branch)),
            "extra_in_branch_count": int(len(extra_in_branch)),
            "sample_missing_transaction_ids": missing_from_branch[:5],
            "sample_extra_transaction_ids": extra_in_branch[:5],
        }
        if missing_from_branch:
            msg = f"{branch_name} is missing {len(missing_from_branch)} transaction ids during {split_name} fusion join"
            raise ValueError(msg)
        fused = fused.merge(
            candidate.drop(columns=[LABEL_COLUMN]),
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

    expected_columns = [
        ID_COLUMN,
        LABEL_COLUMN,
        *[
            column
            for branch_name in branch_names
            for column in (f"{branch_name}_score", f"{branch_name}_predicted_label")
        ],
    ]
    fused = fused.loc[:, expected_columns]

    missing_feature_counts = {
        column: int(fused[column].isna().sum())
        for column in fused.columns
        if column not in {ID_COLUMN, LABEL_COLUMN}
    }
    diagnostics["missing_values_by_column"] = missing_feature_counts
    diagnostics["rows_after_join"] = int(len(fused))
    diagnostics["all_branch_outputs_present"] = not any(count > 0 for count in missing_feature_counts.values())
    if not diagnostics["all_branch_outputs_present"]:
        msg = f"missing joined branch outputs detected in {split_name} fusion dataset"
        raise ValueError(msg)
    return fused, diagnostics


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_fusion_paths() -> tuple[dict[str, dict[str, Path]], Path, Path, Path]:
    """Return configured default branch inputs and fusion outputs."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    branch_paths = {
        branch_name: {
            split_name: paths[path_key]
            for split_name, path_key in split_mapping.items()
        }
        for branch_name, split_mapping in DEFAULT_BRANCH_FILES.items()
    }
    return (
        branch_paths,
        paths["fusion_valid"],
        paths["fusion_test"],
        paths["fusion_dataset_report"],
    )


def build_and_write_fusion_datasets(
    *,
    branch_prediction_paths: dict[str, dict[str, Path]],
    fusion_valid_path: Path,
    fusion_test_path: Path,
    report_path: Path,
) -> FusionBuildResult:
    """Build fusion-ready validation/test datasets from branch predictions."""

    split_frames: dict[str, dict[str, pd.DataFrame]] = {"valid": {}, "test": {}}
    for branch_name, split_paths in branch_prediction_paths.items():
        for split_name in ("valid", "test"):
            split_frames[split_name][branch_name] = _load_prediction_frame(
                split_paths[split_name],
                branch_name=branch_name,
            )

    fusion_valid, valid_diagnostics = _join_branch_frames(split_frames["valid"], split_name="valid")
    fusion_test, test_diagnostics = _join_branch_frames(split_frames["test"], split_name="test")

    ensure_directory(fusion_valid_path.parent)
    fusion_valid.to_parquet(fusion_valid_path, index=False)
    fusion_test.to_parquet(fusion_test_path, index=False)

    report_payload = {
        "branch_names": list(branch_prediction_paths),
        "branch_count": len(branch_prediction_paths),
        "valid": valid_diagnostics,
        "test": test_diagnostics,
    }
    _write_report(report_path, report_payload)

    return FusionBuildResult(
        fusion_valid_path=str(fusion_valid_path),
        fusion_test_path=str(fusion_test_path),
        report_path=str(report_path),
        valid_rows=int(len(fusion_valid)),
        test_rows=int(len(fusion_test)),
        branch_count=len(branch_prediction_paths),
    )

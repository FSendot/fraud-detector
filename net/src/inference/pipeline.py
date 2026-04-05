"""Batch inference pipeline for packaged fraud model bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from calibration.calibrate import prediction_frame_with_scores
from contracts.feature_contract import validate_frame_against_contract
from fusion.bayesian_weighting import BranchReliability, reliability_weighted_scores
from fusion.fusion_model import PREDICTED_LABEL_SUFFIX, SCORE_SUFFIX
from models.vae import TabularVAE, VAEConfig
from training.train_utils import ID_COLUMN, TARGET_COLUMN, binary_classification_metrics, write_json


@dataclass(frozen=True)
class BundleContext:
    """Loaded bundle metadata and root paths."""

    bundle_root: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PreparedFeatures:
    """Prepared contract-aligned feature frame plus validation details."""

    contract_frame: pd.DataFrame
    source_stage: str
    contract_valid: bool
    validation_errors: list[str]
    validation_warnings: list[str]
    rebuild_applied: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def _bundle_file_map(manifest: dict[str, Any], bundle_root: Path) -> dict[str, Path]:
    return {
        entry["path"]: bundle_root / entry["path"]
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }


def load_bundle(bundle_path: Path) -> BundleContext:
    """Load a bundle directory or manifest path."""

    candidate = bundle_path.expanduser().resolve()
    manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    return BundleContext(
        bundle_root=manifest_path.parent,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_inference_input(path: Path) -> pd.DataFrame:
    """Load a parquet or CSV dataset for scoring."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported input format: {path}")
    return frame


def _required_contract_columns(contract: dict[str, Any]) -> list[str]:
    return [
        contract["transaction_id"]["name"],
        contract["label"]["name"],
        *[entry["name"] for entry in contract["features"]],
    ]


def _required_contract_columns_without_label(contract: dict[str, Any]) -> list[str]:
    return [
        contract["transaction_id"]["name"],
        *[entry["name"] for entry in contract["features"]],
    ]


def _contract_subset(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    required = _required_contract_columns(contract)
    available = [column for column in required if column in frame.columns]
    subset = frame.loc[:, available].copy()
    if ID_COLUMN in subset.columns:
        subset[ID_COLUMN] = subset[ID_COLUMN].astype("string")
    if TARGET_COLUMN in subset.columns:
        subset[TARGET_COLUMN] = pd.to_numeric(subset[TARGET_COLUMN], errors="coerce").astype("Int64")
    return subset


def _rebuild_contract_frame(frame: pd.DataFrame, *, contract: dict[str, Any], bundle_files: dict[str, Path]) -> PreparedFeatures:
    """Rebuild the model-input feature frame from a wider feature dataset."""

    missing = []
    if ID_COLUMN not in frame.columns:
        missing.append(ID_COLUMN)
    selected_features_payload = _load_json(bundle_files["preprocessing/selected_features.json"])
    preselection_columns = list(selected_features_payload.get("feature_columns_before_selection", []))
    missing.extend(column for column in preselection_columns if column not in frame.columns)
    if missing:
        raise ValueError(f"cannot rebuild contract frame; missing columns: {sorted(set(missing))}")

    scaler = joblib.load(bundle_files["preprocessing/scaler.joblib"])
    selector = joblib.load(bundle_files["preprocessing/feature_selector.joblib"])

    numeric = frame.loc[:, preselection_columns].copy()
    for column in preselection_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce").astype("Float32")
    numeric = numeric.fillna(0.0)

    scaled = pd.DataFrame(
        scaler.transform(numeric),
        columns=preselection_columns,
        index=frame.index,
    )
    selected_feature_names = list(selected_features_payload.get("selected_feature_columns", []))
    selected_array = selector.transform(scaled)
    selected = pd.DataFrame(
        selected_array,
        columns=selected_feature_names,
        index=frame.index,
    )

    labels = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype("Int64")
        if TARGET_COLUMN in frame.columns
        else pd.Series(pd.array([pd.NA] * len(frame), dtype="Int64"), index=frame.index)
    )
    contract_frame = pd.concat(
        [
            frame[ID_COLUMN].astype("string").rename(ID_COLUMN),
            labels.rename(TARGET_COLUMN),
            selected,
        ],
        axis=1,
    ).reset_index(drop=True)

    validation = validate_frame_against_contract(contract_frame.dropna(subset=[TARGET_COLUMN]), contract) if TARGET_COLUMN in frame.columns else None
    return PreparedFeatures(
        contract_frame=contract_frame,
        source_stage="rebuild_from_feature_dataset",
        contract_valid=bool(validation.valid) if validation is not None else True,
        validation_errors=list(validation.errors) if validation is not None else [],
        validation_warnings=(["label missing; contract feature columns were rebuilt without supervised validation"]),
        rebuild_applied=True,
    )


def prepare_contract_features(frame: pd.DataFrame, *, contract: dict[str, Any], bundle_files: dict[str, Path]) -> PreparedFeatures:
    """Use the input directly if it matches the contract, otherwise rebuild it."""

    subset = _contract_subset(frame, contract)
    expected_with_label = _required_contract_columns(contract)
    expected_without_label = _required_contract_columns_without_label(contract)
    if list(subset.columns) == expected_without_label:
        contract_frame = subset.copy()
        contract_frame[TARGET_COLUMN] = pd.Series(pd.array([pd.NA] * len(contract_frame), dtype="Int64"))
        contract_frame = contract_frame.loc[:, expected_with_label]
        return PreparedFeatures(
            contract_frame=contract_frame.reset_index(drop=True),
            source_stage="contract_aligned_input_without_labels",
            contract_valid=False,
            validation_errors=["label missing; supervised contract validation skipped for inference input"],
            validation_warnings=[],
            rebuild_applied=False,
        )
    if list(subset.columns) == expected_with_label:
        validation = validate_frame_against_contract(subset, contract)
        if validation.valid:
            return PreparedFeatures(
                contract_frame=subset.reset_index(drop=True),
                source_stage="contract_aligned_input",
                contract_valid=True,
                validation_errors=[],
                validation_warnings=list(validation.warnings),
                rebuild_applied=False,
            )
        if TARGET_COLUMN not in subset.columns or subset[TARGET_COLUMN].isna().all():
            return PreparedFeatures(
                contract_frame=subset.reset_index(drop=True),
                source_stage="contract_aligned_input_without_labels",
                contract_valid=False,
                validation_errors=["label missing; supervised contract validation skipped for inference input"],
                validation_warnings=list(validation.warnings),
                rebuild_applied=False,
            )

    return _rebuild_contract_frame(frame, contract=contract, bundle_files=bundle_files)


def _contract_feature_matrix(contract_frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    feature_order = list(contract["feature_order"])
    features = contract_frame.loc[:, feature_order].copy()
    for column in feature_order:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0).astype("Float32")
    return features


def _labels_for_output(contract_frame: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in contract_frame.columns:
        return pd.Series(pd.array([pd.NA] * len(contract_frame), dtype="Int64"))
    return pd.to_numeric(contract_frame[TARGET_COLUMN], errors="coerce").astype("Int64")


def _predict_with_joblib_model(model_path: Path, features: pd.DataFrame) -> np.ndarray:
    model = joblib.load(model_path)
    probabilities = model.predict_proba(features)[:, 1]
    return np.asarray(probabilities, dtype=float)


def _predict_vae_nystrom(bundle_files: dict[str, Path], features: pd.DataFrame) -> np.ndarray:
    vae_config_payload = _load_json(bundle_files["models/vae/config.json"])
    config = VAEConfig(
        input_dim=int(vae_config_payload["input_dim"]),
        latent_dim=int(vae_config_payload.get("latent_dim", 16)),
        hidden_dims=tuple(int(value) for value in vae_config_payload.get("hidden_dims", [128, 64])),
        learning_rate=float(vae_config_payload.get("learning_rate", 5e-4)),
        batch_size=int(vae_config_payload.get("batch_size", 512)),
        epochs=int(vae_config_payload.get("epochs", 30)),
        beta=float(vae_config_payload.get("beta", 0.05)),
        seed=int(vae_config_payload.get("seed", 7)),
        device=str(vae_config_payload.get("device", "cpu")),
        logvar_min=float(vae_config_payload.get("logvar_min", -10.0)),
        logvar_max=float(vae_config_payload.get("logvar_max", 10.0)),
    )
    model = TabularVAE(config)
    state = torch.load(bundle_files["models/vae/weights.pt"], map_location="cpu")
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()

    inputs = torch.as_tensor(features.to_numpy(dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        latent = model.latent_mean(inputs).cpu().numpy()
    nystrom_model = joblib.load(bundle_files["models/nystrom_gp/model.joblib"])
    probabilities = nystrom_model.predict_proba(latent)[:, 1]
    return np.asarray(probabilities, dtype=float)


def predict_required_branches(
    contract_frame: pd.DataFrame,
    *,
    contract: dict[str, Any],
    bundle_files: dict[str, Path],
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Run the branch models required by the packaged fusion recipe."""

    runtime = manifest["runtime_metadata"]["fusion_runtime"]["selected_candidate"]
    branch_names_used = list(runtime.get("branch_names_used", []))
    if not branch_names_used and isinstance(runtime.get("selected_branch"), str):
        branch_names_used = [runtime["selected_branch"]]
    if not branch_names_used:
        raise ValueError("bundle fusion runtime does not declare any branch_names_used")

    features = _contract_feature_matrix(contract_frame, contract)
    labels = _labels_for_output(contract_frame)
    outputs = {
        ID_COLUMN: contract_frame[ID_COLUMN].astype("string").reset_index(drop=True),
        TARGET_COLUMN: labels.reset_index(drop=True),
    }

    for branch_name in branch_names_used:
        if branch_name == "boosted_branch":
            scores = _predict_with_joblib_model(bundle_files["models/boosted_branch/model.joblib"], features)
        elif branch_name == "tree_branch":
            scores = _predict_with_joblib_model(bundle_files["models/tree_branch/model.joblib"], features)
        elif branch_name == "tabular_nystrom":
            scores = _predict_with_joblib_model(bundle_files["models/nystrom_tabular/model.joblib"], features)
        elif branch_name == "vae_nystrom":
            scores = _predict_vae_nystrom(bundle_files, features)
        else:
            raise ValueError(f"unsupported required branch for batch inference: {branch_name}")
        outputs[f"{branch_name}{SCORE_SUFFIX}"] = scores
        outputs[f"{branch_name}{PREDICTED_LABEL_SUFFIX}"] = (scores >= 0.5).astype(int)

    return pd.DataFrame(outputs)


def _apply_selected_fusion(branch_frame: pd.DataFrame, *, manifest: dict[str, Any]) -> np.ndarray:
    selected = manifest["runtime_metadata"]["fusion_runtime"]["selected_candidate"]
    candidate_name = str(selected["candidate_name"])
    branch_names_used = list(selected.get("branch_names_used", []))

    if candidate_name == "best_branch":
        branch_name = str(selected.get("selected_branch", branch_names_used[0]))
        return branch_frame[f"{branch_name}{SCORE_SUFFIX}"].to_numpy(dtype=float)

    if candidate_name == "mean_top_k":
        matrix = branch_frame[[f"{name}{SCORE_SUFFIX}" for name in branch_names_used]].to_numpy(dtype=float)
        return matrix.mean(axis=1)

    if candidate_name == "geometric_mean_top_k":
        matrix = np.clip(
            branch_frame[[f"{name}{SCORE_SUFFIX}" for name in branch_names_used]].to_numpy(dtype=float),
            1e-9,
            1.0,
        )
        return np.exp(np.mean(np.log(matrix), axis=1))

    if candidate_name == "ap_weighted_average":
        weights = selected.get("weights", {})
        raw_weights = np.asarray([float(weights.get(name, 0.0)) for name in branch_names_used], dtype=float)
        raw_weights = raw_weights / raw_weights.sum()
        matrix = branch_frame[[f"{name}{SCORE_SUFFIX}" for name in branch_names_used]].to_numpy(dtype=float)
        return matrix @ raw_weights

    if candidate_name == "logistic_meta":
        coefficients = selected.get("coefficients", {})
        intercept = float(selected.get("intercept", 0.0))
        matrix = branch_frame[[f"{name}{SCORE_SUFFIX}" for name in branch_names_used]].to_numpy(dtype=float)
        coef = np.asarray([float(coefficients[f"{name}{SCORE_SUFFIX}"]) for name in branch_names_used], dtype=float)
        logits = matrix @ coef + intercept
        return 1.0 / (1.0 + np.exp(-logits))

    if candidate_name == "bayesian_reliability":
        reliabilities_payload = selected.get("reliabilities", {})
        reliability_map = {
            name: BranchReliability(**payload)
            for name, payload in reliabilities_payload.items()
        }
        score_map = {
            name: branch_frame[f"{name}{SCORE_SUFFIX}"].to_numpy(dtype=float)
            for name in branch_names_used
        }
        return reliability_weighted_scores(score_map, reliability_map)

    raise ValueError(f"unsupported packaged fusion candidate: {candidate_name}")


def apply_fusion_and_calibration(branch_frame: pd.DataFrame, *, bundle_files: dict[str, Path], manifest: dict[str, Any]) -> pd.DataFrame:
    """Apply the packaged fusion recipe, calibrator, and serving threshold."""

    fused_scores = _apply_selected_fusion(branch_frame, manifest=manifest)

    calibrator_payload = joblib.load(bundle_files["models/calibration/calibrator.joblib"])
    calibrator = calibrator_payload["calibrator"]
    calibrated_scores = calibrator.predict_proba(fused_scores)

    threshold = float(manifest["runtime_metadata"]["operational_defaults"]["decision_threshold"])
    final_predictions = prediction_frame_with_scores(
        pd.DataFrame(
            {
                ID_COLUMN: branch_frame[ID_COLUMN],
                TARGET_COLUMN: branch_frame[TARGET_COLUMN],
            }
        ),
        calibrated_scores=calibrated_scores,
        threshold=threshold,
    )
    final_predictions["raw_fused_score"] = np.asarray(fused_scores, dtype=float)
    final_predictions["calibrated_score"] = np.asarray(calibrated_scores, dtype=float)
    final_predictions["decision_threshold"] = threshold
    return final_predictions


def build_summary(
    *,
    bundle: BundleContext,
    input_path: Path,
    prepared: PreparedFeatures,
    branch_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Build a machine-readable batch-inference summary."""

    labels = pd.to_numeric(predictions[TARGET_COLUMN], errors="coerce").astype("Int64")
    has_labels = not labels.isna().all()
    metrics: dict[str, Any] | None = None
    if has_labels:
        clean_labels = labels.fillna(0)
        metrics = binary_classification_metrics(clean_labels, predictions["score"].to_numpy(dtype=float))

    score_series = predictions["score"].astype(float)
    summary = {
        "bundle_version": bundle.manifest.get("bundle_version"),
        "bundle_manifest": str(bundle.manifest_path),
        "input_path": str(input_path),
        "output_predictions_path": str(output_path),
        "rows_in": int(len(prepared.contract_frame)),
        "rows_scored": int(len(predictions)),
        "source_stage": prepared.source_stage,
        "rebuild_applied": prepared.rebuild_applied,
        "contract_validation": {
            "valid": prepared.contract_valid,
            "errors": prepared.validation_errors,
            "warnings": prepared.validation_warnings,
        },
        "branches_used": [
            column[: -len(SCORE_SUFFIX)]
            for column in branch_frame.columns
            if column.endswith(SCORE_SUFFIX)
        ],
        "runtime_defaults": bundle.manifest["runtime_metadata"]["operational_defaults"],
        "score_summary": {
            "min": float(score_series.min()),
            "max": float(score_series.max()),
            "mean": float(score_series.mean()),
            "positive_predictions": int(predictions["predicted_label"].sum()),
        },
        "labels_available": bool(has_labels),
        "metrics": metrics,
    }
    return summary


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Persist the batch inference summary."""

    write_json(path, payload)

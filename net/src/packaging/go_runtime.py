"""Export the active packaged model path into a Go-runnable runtime spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from common.config import DEFAULT_CONFIG_DIR
from contracts.feature_contract import load_feature_contract
from inference.pipeline import _bundle_file_map, load_bundle
from ops.policy_engine import load_policy_config
from training.train_utils import write_json


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _export_hist_gradient_boosting_model(model_path: Path) -> dict[str, Any]:
    model = joblib.load(model_path)
    trees: list[dict[str, Any]] = []
    for predictor_group in model._predictors:
        tree = predictor_group[0]
        nodes = []
        for node in tree.nodes:
            nodes.append(
                {
                    "value": float(node["value"]),
                    "feature_idx": int(node["feature_idx"]),
                    "threshold": float(node["num_threshold"]),
                    "missing_go_to_left": bool(node["missing_go_to_left"]),
                    "left": int(node["left"]),
                    "right": int(node["right"]),
                    "is_leaf": bool(node["is_leaf"]),
                }
            )
        trees.append({"nodes": nodes})
    return {
        "type": "hist_gradient_boosting_binary",
        "baseline_prediction": float(model._baseline_prediction[0][0]),
        "n_features": int(model.n_features_in_),
        "trees": trees,
    }


def _export_isotonic_calibrator(calibrator_path: Path) -> dict[str, Any]:
    payload = joblib.load(calibrator_path)
    calibrator = payload["calibrator"]
    model = calibrator.model
    return {
        "type": "isotonic_regression",
        "x_thresholds": [float(value) for value in model.X_thresholds_.tolist()],
        "y_thresholds": [float(value) for value in model.y_thresholds_.tolist()],
        "out_of_bounds": str(model.out_of_bounds),
    }


def _export_policy() -> dict[str, Any]:
    policy = load_policy_config(DEFAULT_CONFIG_DIR / "production_thresholds.yaml")
    return {
        "score_field": policy.score_field,
        "shadow_enabled": policy.shadow_enabled,
        "emit_decision_recommendation_only": policy.emit_decision_recommendation_only,
        "actions": [
            {
                "name": action.name,
                "min_score_inclusive": action.min_score_inclusive,
                "max_score_exclusive": action.max_score_exclusive,
                "rationale": action.rationale,
            }
            for action in policy.actions
        ],
        "fallback": {
            "on_missing_score": policy.on_missing_score,
            "on_contract_mismatch": policy.on_contract_mismatch,
        },
    }


def export_go_runtime_spec(*, bundle_path: Path, output_path: Path) -> Path:
    """Export the active serving path into a JSON spec consumable by Go."""

    bundle = load_bundle(bundle_path)
    bundle_files = _bundle_file_map(bundle.manifest, bundle.bundle_root)
    contract = load_feature_contract(bundle_files["contracts/feature_contract_v1.json"])

    selected = bundle.manifest["runtime_metadata"]["fusion_runtime"]["selected_candidate"]
    selected_branch = str(selected.get("selected_branch", ""))
    if selected.get("candidate_name") != "best_branch" or selected_branch != "boosted_branch":
        raise ValueError(
            "Go runtime export currently supports only the active best-branch boosted_branch serving path"
        )

    runtime_spec = {
        "format_version": 1,
        "model_version": str(bundle.manifest["bundle_version"]),
        "bundle_manifest": str(bundle.manifest_path),
        "feature_contract": {
            "version": str(contract["version"]),
            "transaction_id_field": str(contract["transaction_id"]["name"]),
            "label_field": str(contract["label"]["name"]),
            "feature_order": list(contract["feature_order"]),
        },
        "branch_runtime": {
            "branch_name": "boosted_branch",
            "model": _export_hist_gradient_boosting_model(bundle_files["models/boosted_branch/model.joblib"]),
        },
        "fusion_runtime": {
            "mode": "best_branch",
            "selected_branch": "boosted_branch",
            "raw_fused_score_source": "boosted_branch",
        },
        "calibration_runtime": _export_isotonic_calibrator(bundle_files["models/calibration/calibrator.joblib"]),
        "decision_runtime": {
            "decision_threshold": float(bundle.manifest["runtime_metadata"]["operational_defaults"]["decision_threshold"]),
            "business_threshold": bundle.manifest["runtime_metadata"]["operational_defaults"].get("business_threshold"),
            "policy": _export_policy(),
        },
    }
    write_json(output_path, runtime_spec)
    return output_path

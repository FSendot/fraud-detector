"""Model bundle packaging for deterministic fraud inference."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import DEFAULT_PATHS_FILE, load_paths_config, project_root
from common.io import copy_file, ensure_directory, iter_regular_files, sha256_file
from training.train_utils import write_json


@dataclass(frozen=True)
class BundleFileSpec:
    """Description of one file copied into a bundle."""

    source: Path
    destination: Path
    category: str
    required: bool = True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def _bundle_specs(paths: dict[str, Path]) -> list[BundleFileSpec]:
    """Return the required bundle payload."""

    return [
        BundleFileSpec(paths["artifact_feature_contract_json"], Path("contracts/feature_contract_v1.json"), "contract"),
        BundleFileSpec(paths["artifact_feature_contract_md"], Path("contracts/feature_contract_v1.md"), "contract"),
        BundleFileSpec(paths["artifact_scaler"], Path("preprocessing/scaler.joblib"), "preprocessing"),
        BundleFileSpec(paths["artifact_feature_selector"], Path("preprocessing/feature_selector.joblib"), "preprocessing"),
        BundleFileSpec(paths["artifact_selected_features"], Path("preprocessing/selected_features.json"), "preprocessing"),
        BundleFileSpec(paths["artifact_sequence_schema"], Path("schemas/sequence_schema.json"), "schema"),
        BundleFileSpec(paths["artifact_vae_config"], Path("models/vae/config.json"), "branch_model"),
        BundleFileSpec(paths["artifact_vae_weights"], Path("models/vae/weights.pt"), "branch_model"),
        BundleFileSpec(paths["artifact_vae_metrics"], Path("models/vae/metrics.json"), "branch_model"),
        BundleFileSpec(paths["artifact_nystrom_gp_model"], Path("models/nystrom_gp/model.joblib"), "branch_model"),
        BundleFileSpec(paths["artifact_nystrom_gp_metrics"], Path("models/nystrom_gp/metrics.json"), "branch_model"),
        BundleFileSpec(paths["artifact_nystrom_tabular_model"], Path("models/nystrom_tabular/model.joblib"), "branch_model"),
        BundleFileSpec(paths["artifact_nystrom_tabular_metrics"], Path("models/nystrom_tabular/metrics.json"), "branch_model"),
        BundleFileSpec(paths["artifact_tree_branch_model"], Path("models/tree_branch/model.joblib"), "branch_model"),
        BundleFileSpec(paths["artifact_tree_branch_config"], Path("models/tree_branch/config.json"), "branch_model"),
        BundleFileSpec(paths["artifact_tree_branch_metrics"], Path("models/tree_branch/metrics.json"), "branch_model"),
        BundleFileSpec(paths["artifact_tree_branch_feature_importances"], Path("models/tree_branch/feature_importances.json"), "branch_model"),
        BundleFileSpec(paths["artifact_boosted_branch_model"], Path("models/boosted_branch/model.joblib"), "branch_model"),
        BundleFileSpec(paths["artifact_boosted_branch_config"], Path("models/boosted_branch/config.json"), "branch_model"),
        BundleFileSpec(paths["artifact_boosted_branch_metrics"], Path("models/boosted_branch/metrics.json"), "branch_model"),
        BundleFileSpec(paths["artifact_boosted_branch_feature_importances"], Path("models/boosted_branch/feature_importances.json"), "branch_model"),
        BundleFileSpec(paths["artifact_gru_branch_weights"], Path("models/gru_branch/weights.pt"), "branch_model"),
        BundleFileSpec(paths["artifact_gru_branch_config"], Path("models/gru_branch/config.json"), "branch_model"),
        BundleFileSpec(paths["artifact_gru_branch_metrics"], Path("models/gru_branch/metrics.json"), "branch_model"),
        BundleFileSpec(paths["artifact_fusion_config"], Path("models/fusion/config.json"), "fusion"),
        BundleFileSpec(paths["artifact_fusion_metrics"], Path("models/fusion/metrics.json"), "fusion"),
        BundleFileSpec(paths["artifact_calibration_calibrator"], Path("models/calibration/calibrator.joblib"), "calibration"),
        BundleFileSpec(paths["artifact_calibration_config"], Path("models/calibration/config.json"), "calibration"),
        BundleFileSpec(paths["artifact_calibration_metrics"], Path("models/calibration/metrics.json"), "calibration"),
        BundleFileSpec(paths["calibration_report"], Path("reports/calibration_report.json"), "report"),
        BundleFileSpec(paths["reports_ablations_report_json"], Path("reports/ablation_report.json"), "report"),
        BundleFileSpec(paths["business_threshold_report_json"], Path("reports/business_threshold_report.json"), "report", required=False),
        BundleFileSpec(
            paths["reports_dir"] / "business_threshold_report_calibrated.json",
            Path("reports/business_threshold_report_calibrated.json"),
            "report",
            required=False,
        ),
    ]


def _validate_required_files(specs: list[BundleFileSpec]) -> None:
    missing = [str(spec.source) for spec in specs if spec.required and not spec.source.exists()]
    if missing:
        msg = "missing required bundle files:\n" + "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(msg)


def _load_runtime_metadata(paths: dict[str, Path]) -> dict[str, Any]:
    """Build machine-readable runtime metadata needed for later inference."""

    fusion_metrics = _load_json(paths["artifact_fusion_metrics"])
    calibration_metrics = _load_json(paths["artifact_calibration_metrics"])
    ablation_report = _load_json(paths["reports_ablations_report_json"])

    business_report_path = paths["reports_dir"] / "business_threshold_report_calibrated.json"
    business_report = _load_json(business_report_path) if business_report_path.exists() else None

    fusion_selected = fusion_metrics.get("fusion_details", {}).get("selected_candidate")
    if not isinstance(fusion_selected, dict):
        msg = "fusion metrics are missing fusion_details.selected_candidate"
        raise ValueError(msg)

    calibration_method = calibration_metrics.get("selected_method")
    if not isinstance(calibration_method, str) or not calibration_method:
        msg = "calibration metrics are missing selected_method"
        raise ValueError(msg)

    calibration_threshold = calibration_metrics.get("effective_threshold")
    if not isinstance(calibration_threshold, (int, float)):
        msg = "calibration metrics are missing effective_threshold"
        raise ValueError(msg)

    ablation_variant = ablation_report.get("recommended_variant")
    if not isinstance(ablation_variant, str) or not ablation_variant:
        msg = "ablation report is missing recommended_variant"
        raise ValueError(msg)

    operational_defaults: dict[str, Any] = {
        "decision_variant": "fused_calibrated",
        "score_source": "fused_calibrated",
        "calibration_method": calibration_method,
        "decision_threshold": float(calibration_threshold),
    }
    if isinstance(business_report, dict):
        recommended_budget = business_report.get("recommended_budget_point")
        if isinstance(recommended_budget, dict):
            operational_defaults["business_threshold"] = {
                "selection_type": "alert_budget",
                "target_alerts_per_1k": float(recommended_budget.get("target_alerts_per_1k", 0.0)),
                "threshold": float(recommended_budget.get("threshold", calibration_threshold)),
                "test_precision": float(recommended_budget.get("test", {}).get("precision", 0.0)),
                "test_recall": float(recommended_budget.get("test", {}).get("recall", 0.0)),
                "test_f1": float(recommended_budget.get("test", {}).get("f1", 0.0)),
            }

    return {
        "project_root": str(project_root()),
        "feature_contract_version": "v1",
        "fusion_runtime": {
            "mode": fusion_metrics.get("config", {}).get("mode"),
            "selected_candidate": fusion_selected,
            "effective_threshold": float(fusion_metrics.get("effective_threshold", 0.5)),
        },
        "calibration_runtime": {
            "selected_method": calibration_method,
            "effective_threshold": float(calibration_threshold),
            "selection_metric": calibration_metrics.get("config", {}).get("selection_metric"),
        },
        "operational_defaults": operational_defaults,
        "evaluation_summary": {
            "ablation_recommended_variant": ablation_variant,
        },
    }


def _copy_specs_to_bundle(specs: list[BundleFileSpec], bundle_root: Path) -> list[dict[str, Any]]:
    """Copy required files into the bundle and return manifest entries."""

    entries: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.source.exists():
            if spec.required:
                raise FileNotFoundError(spec.source)
            continue
        destination = bundle_root / spec.destination
        copy_file(spec.source, destination)
        entries.append(
            {
                "path": spec.destination.as_posix(),
                "category": spec.category,
                "size_bytes": int(destination.stat().st_size),
                "sha256": sha256_file(destination),
                "source_path": str(spec.source),
            }
        )
    entries.sort(key=lambda item: item["path"])
    return entries


def _validate_bundle(bundle_root: Path, manifest_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate that copied files match the manifest entries."""

    expected_paths = {entry["path"] for entry in manifest_entries}
    actual_paths = {
        candidate.relative_to(bundle_root).as_posix()
        for candidate in iter_regular_files(bundle_root)
        if candidate.name != "manifest.json"
    }
    if expected_paths != actual_paths:
        msg = f"bundle file mismatch: expected {sorted(expected_paths)}, got {sorted(actual_paths)}"
        raise ValueError(msg)

    hash_mismatches: list[str] = []
    for entry in manifest_entries:
        path = bundle_root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            hash_mismatches.append(entry["path"])
    if hash_mismatches:
        msg = "bundle hash mismatch:\n" + "\n".join(f"- {path}" for path in hash_mismatches)
        raise ValueError(msg)

    return {
        "file_count": len(manifest_entries),
        "hashes_verified": True,
        "unexpected_files": [],
    }


def package_model_bundle(
    *,
    bundle_version: str,
    output_root: Path | None = None,
    paths_file: Path | None = None,
) -> Path:
    """Create a versioned deterministic model bundle and return its manifest path."""

    paths = load_paths_config(paths_file or DEFAULT_PATHS_FILE)
    bundle_parent = output_root or paths["artifact_bundles_dir"]
    bundle_root = bundle_parent / bundle_version

    specs = _bundle_specs(paths)
    _validate_required_files(specs)
    runtime_metadata = _load_runtime_metadata(paths)

    ensure_directory(bundle_parent)
    with tempfile.TemporaryDirectory(prefix=f"{bundle_version}_", dir=bundle_parent) as temp_dir:
        staging_root = Path(temp_dir) / bundle_version
        ensure_directory(staging_root)
        manifest_entries = _copy_specs_to_bundle(specs, staging_root)
        validation = _validate_bundle(staging_root, manifest_entries)
        manifest = {
            "bundle_version": bundle_version,
            "format_version": 1,
            "project": "fraud-detector-net",
            "paths_file": str(paths_file or DEFAULT_PATHS_FILE),
            "runtime_metadata": runtime_metadata,
            "validation": validation,
            "files": manifest_entries,
        }
        write_json(staging_root / "manifest.json", manifest)

        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        shutil.move(str(staging_root), str(bundle_root))

    return bundle_root / "manifest.json"

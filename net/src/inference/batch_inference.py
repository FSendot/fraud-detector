"""High-level orchestration for packaged batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io import ensure_directory
from contracts.feature_contract import load_feature_contract
from inference.pipeline import (
    _bundle_file_map,
    apply_fusion_and_calibration,
    build_summary,
    load_bundle,
    load_inference_input,
    predict_required_branches,
    prepare_contract_features,
    write_summary,
)


def run_batch_inference(
    *,
    bundle_path: Path,
    input_path: Path,
    predictions_output_path: Path,
    summary_output_path: Path,
) -> dict[str, Any]:
    """Run bundled fraud inference end to end and persist outputs."""

    bundle = load_bundle(bundle_path)
    bundle_files = _bundle_file_map(bundle.manifest, bundle.bundle_root)
    required_bundle_paths = [
        "contracts/feature_contract_v1.json",
        "preprocessing/scaler.joblib",
        "preprocessing/feature_selector.joblib",
        "preprocessing/selected_features.json",
        "models/calibration/calibrator.joblib",
    ]
    missing = [path for path in required_bundle_paths if path not in bundle_files]
    if missing:
        raise FileNotFoundError(f"bundle is missing required files: {missing}")

    contract = load_feature_contract(bundle_files["contracts/feature_contract_v1.json"])
    input_frame = load_inference_input(input_path)
    prepared = prepare_contract_features(input_frame, contract=contract, bundle_files=bundle_files)
    branch_frame = predict_required_branches(
        prepared.contract_frame,
        contract=contract,
        bundle_files=bundle_files,
        manifest=bundle.manifest,
    )
    predictions = apply_fusion_and_calibration(branch_frame, bundle_files=bundle_files, manifest=bundle.manifest)

    ensure_directory(predictions_output_path.parent)
    predictions.to_parquet(predictions_output_path, index=False)

    summary = build_summary(
        bundle=bundle,
        input_path=input_path,
        prepared=prepared,
        branch_frame=branch_frame,
        predictions=predictions,
        output_path=predictions_output_path,
    )
    write_summary(summary_output_path, summary)
    return summary

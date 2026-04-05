"""Ablation runner for comparable fraud-pipeline variant evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from calibration.calibrate import CalibrationConfig, compare_calibrators, prediction_frame_with_scores
from common.config import DEFAULT_PATHS_FILE, load_paths_config
from eval.branch_usefulness import evaluate_prediction_frame, resolve_threshold_selection
from fusion.fusion_model import FusionConfig, fused_scores
from training.train_utils import write_json


BRANCH_PATH_KEYS = {
    "vae_nystrom": ("prediction_nystrom_valid", "prediction_nystrom_test"),
    "tabular_nystrom": ("prediction_nystrom_tabular_valid", "prediction_nystrom_tabular_test"),
    "tree_branch": ("prediction_tree_branch_valid", "prediction_tree_branch_test"),
    "boosted_branch": ("prediction_boosted_branch_valid", "prediction_boosted_branch_test"),
    "gru_branch": ("prediction_gru_branch_valid", "prediction_gru_branch_test"),
}
VARIANT_ORDER = (
    "full",
    "no_vae",
    "no_tree_branch",
    "no_boosted_branch",
    "no_gru",
    "no_fusion",
    "no_calibration",
)


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for ablation execution."""

    fusion_config: FusionConfig
    calibration_config: CalibrationConfig
    standalone_branch_mode: str = "best_by_valid_average_precision"
    standalone_branch_name: str = "vae_nystrom"

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fusion_config"] = self.fusion_config.to_payload()
        payload["calibration_config"] = self.calibration_config.to_payload()
        return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"expected a JSON object in {path}"
        raise ValueError(msg)
    return payload


def load_default_ablation_inputs() -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.DataFrame]], AblationConfig]:
    """Load default artifacts used by the ablation runner."""

    paths = load_paths_config(DEFAULT_PATHS_FILE)
    fusion_valid = pd.read_parquet(paths["fusion_valid"])
    fusion_test = pd.read_parquet(paths["fusion_test"])

    branch_predictions: dict[str, dict[str, pd.DataFrame]] = {}
    for branch_name, (valid_key, test_key) in BRANCH_PATH_KEYS.items():
        branch_predictions[branch_name] = {
            "valid": pd.read_parquet(paths[valid_key]),
            "test": pd.read_parquet(paths[test_key]),
        }

    fusion_payload = _load_json(paths["artifact_fusion_config"])
    calibration_payload = _load_json(paths["artifact_calibration_config"])
    config = AblationConfig(
        fusion_config=FusionConfig(
            mode=str(fusion_payload.get("mode", "logistic_meta")),
            threshold=float(fusion_payload.get("threshold", 0.5)),
            threshold_strategy=str(fusion_payload.get("threshold_strategy", "validation_f1")),
            threshold_candidate_count=int(fusion_payload.get("threshold_candidate_count", 200)),
            logistic_c=float(fusion_payload.get("logistic_c", 1.0)),
            logistic_max_iter=int(fusion_payload.get("logistic_max_iter", 500)),
            random_state=int(fusion_payload.get("random_state", 7)),
            weighted_average_weights=(
                {str(key): float(value) for key, value in fusion_payload.get("weighted_average_weights", {}).items()}
                if isinstance(fusion_payload.get("weighted_average_weights"), dict)
                else None
            ),
            bayesian_alpha_prior=float(fusion_payload.get("bayesian_alpha_prior", 1.0)),
            bayesian_beta_prior=float(fusion_payload.get("bayesian_beta_prior", 1.0)),
            auto_select_metric=str(fusion_payload.get("auto_select_metric", "average_precision")),
            auto_select_complexity_penalty=float(
                fusion_payload.get("auto_select_complexity_penalty", 3.5e-4)
            ),
            auto_select_min_valid_average_precision=float(
                fusion_payload.get("auto_select_min_valid_average_precision", 1.0e-3)
            ),
            auto_select_max_branches=int(fusion_payload.get("auto_select_max_branches", 2)),
            auto_select_candidates=tuple(
                str(candidate)
                for candidate in fusion_payload.get(
                    "auto_select_candidates",
                    [
                        "best_branch",
                        "mean_top_k",
                        "geometric_mean_top_k",
                        "ap_weighted_average",
                        "logistic_meta",
                        "bayesian_reliability",
                    ],
                )
            ),
        ),
        calibration_config=CalibrationConfig(
            methods=tuple(str(method) for method in calibration_payload.get("methods", ["platt", "isotonic"])),
            selection_metric=str(calibration_payload.get("selection_metric", "brier_score")),
            threshold=float(calibration_payload.get("threshold", 0.5)),
            threshold_strategy=str(calibration_payload.get("threshold_strategy", "validation_f1")),
            threshold_candidate_count=int(calibration_payload.get("threshold_candidate_count", 200)),
            random_state=int(calibration_payload.get("random_state", 7)),
        ),
    )
    return paths, fusion_valid, fusion_test, branch_predictions, config


def _score_columns_for_branch(branch_name: str) -> list[str]:
    return [f"{branch_name}_score", f"{branch_name}_predicted_label"]


def _drop_branch(frame: pd.DataFrame, branch_name: str) -> pd.DataFrame:
    return frame.drop(columns=_score_columns_for_branch(branch_name)).copy()


def _select_standalone_branch(
    branch_predictions: dict[str, dict[str, pd.DataFrame]],
    *,
    mode: str,
    fallback_name: str,
) -> str:
    if mode == "fixed":
        return fallback_name
    if mode == "best_by_valid_average_precision":
        scored = []
        for branch_name, payload in branch_predictions.items():
            metrics = evaluate_prediction_frame(payload["valid"])
            scored.append((float(metrics["average_precision"]), branch_name))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][1]
    msg = f"unsupported standalone branch mode: {mode}"
    raise ValueError(msg)


def _calibrate_predictions(
    valid_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    calibration_config: CalibrationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_method, calibrator, comparison = compare_calibrators(valid_predictions, config=calibration_config)
    calibrated_valid = prediction_frame_with_scores(
        valid_predictions,
        calibrated_scores=calibrator.predict_proba(valid_predictions["score"].to_numpy(dtype=float)),
        threshold=calibration_config.threshold,
    )
    calibrated_test = prediction_frame_with_scores(
        test_predictions,
        calibrated_scores=calibrator.predict_proba(test_predictions["score"].to_numpy(dtype=float)),
        threshold=calibration_config.threshold,
    )
    return calibrated_valid, calibrated_test, {
        "selected_method": selected_method,
        "comparison": comparison,
    }


def _fusion_predictions(
    fusion_valid: pd.DataFrame,
    fusion_test: pd.DataFrame,
    *,
    fusion_config: FusionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid_scores, fusion_details = fused_scores(fusion_valid, fusion_valid, config=fusion_config)
    test_scores, _ = fused_scores(fusion_valid, fusion_test, config=fusion_config)
    valid_predictions = prediction_frame_with_scores(
        fusion_valid,
        calibrated_scores=valid_scores,
        threshold=fusion_config.threshold,
    )
    test_predictions = prediction_frame_with_scores(
        fusion_test,
        calibrated_scores=test_scores,
        threshold=fusion_config.threshold,
    )
    return valid_predictions, test_predictions, fusion_details


def _variant_result(
    name: str,
    valid_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    description: str,
    uses_fusion: bool,
    uses_calibration: bool,
    branch_removed: str | None,
    threshold_strategy: str,
    fixed_threshold: float,
    threshold_candidate_count: int,
    extra_details: dict[str, Any],
) -> dict[str, Any]:
    threshold_selection = resolve_threshold_selection(
        valid_predictions["is_fraud"],
        valid_predictions["score"].to_numpy(dtype=float),
        strategy=threshold_strategy,
        fixed_threshold=fixed_threshold,
        candidate_count=threshold_candidate_count,
    )
    decision_threshold = float(threshold_selection["threshold"])
    thresholded_valid = prediction_frame_with_scores(
        valid_predictions,
        calibrated_scores=valid_predictions["score"].to_numpy(dtype=float),
        threshold=decision_threshold,
    )
    thresholded_test = prediction_frame_with_scores(
        test_predictions,
        calibrated_scores=test_predictions["score"].to_numpy(dtype=float),
        threshold=decision_threshold,
    )
    valid_metrics = evaluate_prediction_frame(thresholded_valid, threshold=decision_threshold)
    test_metrics = evaluate_prediction_frame(thresholded_test, threshold=decision_threshold)
    return {
        "variant": name,
        "description": description,
        "uses_fusion": uses_fusion,
        "uses_calibration": uses_calibration,
        "branch_removed": branch_removed,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "details": {
            **extra_details,
            "threshold_selection": threshold_selection,
            "effective_threshold": decision_threshold,
        },
    }


def run_ablations(
    *,
    fusion_valid: pd.DataFrame,
    fusion_test: pd.DataFrame,
    branch_predictions: dict[str, dict[str, pd.DataFrame]],
    config: AblationConfig,
) -> list[dict[str, Any]]:
    """Run the configured ablation suite and return machine-readable results."""

    results: list[dict[str, Any]] = []

    full_valid_raw, full_test_raw, full_fusion_details = _fusion_predictions(
        fusion_valid,
        fusion_test,
        fusion_config=config.fusion_config,
    )
    full_valid_calibrated, full_test_calibrated, full_calibration_details = _calibrate_predictions(
        full_valid_raw,
        full_test_raw,
        calibration_config=config.calibration_config,
    )
    results.append(
        _variant_result(
            "full",
            full_valid_calibrated,
            full_test_calibrated,
            description="Full fusion pipeline with calibration.",
            uses_fusion=True,
            uses_calibration=True,
            branch_removed=None,
            threshold_strategy=config.calibration_config.threshold_strategy,
            fixed_threshold=config.calibration_config.threshold,
            threshold_candidate_count=config.calibration_config.threshold_candidate_count,
            extra_details={
                "fusion": full_fusion_details,
                "calibration": full_calibration_details,
            },
        )
    )

    for variant_name, branch_name in (
        ("no_vae", "vae_nystrom"),
        ("no_tree_branch", "tree_branch"),
        ("no_boosted_branch", "boosted_branch"),
        ("no_gru", "gru_branch"),
    ):
        ablated_valid = _drop_branch(fusion_valid, branch_name)
        ablated_test = _drop_branch(fusion_test, branch_name)
        ablated_valid_raw, ablated_test_raw, ablated_fusion_details = _fusion_predictions(
            ablated_valid,
            ablated_test,
            fusion_config=config.fusion_config,
        )
        ablated_valid_calibrated, ablated_test_calibrated, ablated_calibration_details = _calibrate_predictions(
            ablated_valid_raw,
            ablated_test_raw,
            calibration_config=config.calibration_config,
        )
        results.append(
            _variant_result(
                variant_name,
                ablated_valid_calibrated,
                ablated_test_calibrated,
                description=f"Fusion pipeline without the {branch_name} branch.",
                uses_fusion=True,
                uses_calibration=True,
                branch_removed=branch_name,
                threshold_strategy=config.calibration_config.threshold_strategy,
                fixed_threshold=config.calibration_config.threshold,
                threshold_candidate_count=config.calibration_config.threshold_candidate_count,
                extra_details={
                    "fusion": ablated_fusion_details,
                    "calibration": ablated_calibration_details,
                },
            )
        )

    standalone_branch = _select_standalone_branch(
        branch_predictions,
        mode=config.standalone_branch_mode,
        fallback_name=config.standalone_branch_name,
    )
    standalone_valid = branch_predictions[standalone_branch]["valid"]
    standalone_test = branch_predictions[standalone_branch]["test"]
    standalone_valid_calibrated, standalone_test_calibrated, standalone_calibration_details = _calibrate_predictions(
        standalone_valid,
        standalone_test,
        calibration_config=config.calibration_config,
    )
    results.append(
        _variant_result(
            "no_fusion",
            standalone_valid_calibrated,
            standalone_test_calibrated,
            description="Best standalone branch without ensemble fusion, still calibrated on validation only.",
            uses_fusion=False,
            uses_calibration=True,
            branch_removed=None,
            threshold_strategy=config.calibration_config.threshold_strategy,
            fixed_threshold=config.calibration_config.threshold,
            threshold_candidate_count=config.calibration_config.threshold_candidate_count,
            extra_details={
                "standalone_branch": standalone_branch,
                "calibration": standalone_calibration_details,
            },
        )
    )

    results.append(
        _variant_result(
            "no_calibration",
            full_valid_raw,
            full_test_raw,
            description="Full fusion pipeline without post-fusion calibration.",
            uses_fusion=True,
            uses_calibration=False,
            branch_removed=None,
            threshold_strategy=config.fusion_config.threshold_strategy,
            fixed_threshold=config.fusion_config.threshold,
            threshold_candidate_count=config.fusion_config.threshold_candidate_count,
            extra_details={
                "fusion": full_fusion_details,
            },
        )
    )

    result_map = {result["variant"]: result for result in results}
    return [result_map[name] for name in VARIANT_ORDER if name in result_map]


def ablation_summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert ablation results to a flat summary table."""

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "variant": result["variant"],
                "description": result["description"],
                "uses_fusion": result["uses_fusion"],
                "uses_calibration": result["uses_calibration"],
                "branch_removed": result["branch_removed"],
                "valid_average_precision": result["valid_metrics"]["average_precision"],
                "valid_roc_auc": result["valid_metrics"].get("roc_auc"),
                "valid_brier_score": result["valid_metrics"]["brier_score"],
                "valid_f1": result["valid_metrics"]["f1"],
                "valid_threshold": result["valid_metrics"]["threshold"],
                "test_average_precision": result["test_metrics"]["average_precision"],
                "test_roc_auc": result["test_metrics"].get("roc_auc"),
                "test_brier_score": result["test_metrics"]["brier_score"],
                "test_f1": result["test_metrics"]["f1"],
                "test_threshold": result["test_metrics"]["threshold"],
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["test_average_precision", "variant"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _recommended_variant(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a validation-driven default operating variant."""

    return max(
        results,
        key=lambda result: (
            float(result["valid_metrics"]["f1"]),
            float(result["valid_metrics"]["average_precision"]),
            -float(result["valid_metrics"]["brier_score"]),
            int(bool(result["uses_fusion"])),
            int(bool(result["uses_calibration"])),
            int(result["branch_removed"] is None),
            int(result["variant"] == "full"),
        ),
    )


def write_ablation_outputs(
    *,
    summary_frame: pd.DataFrame,
    results: list[dict[str, Any]],
    config: AblationConfig,
    summary_path: Path,
    report_json_path: Path,
    report_md_path: Path,
) -> None:
    """Persist ablation summary artifacts."""

    recommended = _recommended_variant(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_parquet(summary_path, index=False)

    payload = {
        "config": config.to_payload(),
        "recommended_variant": recommended["variant"],
        "summary": summary_frame.to_dict(orient="records"),
        "results": results,
    }
    write_json(report_json_path, payload)

    lines = [
        "# Ablation Report",
        "",
        f"- Variants evaluated: `{len(results)}`",
        f"- Standalone branch mode: `{config.standalone_branch_mode}`",
        f"- Validation-selected default: `{recommended['variant']}`",
        "",
        "## Summary",
    ]
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            f"- `{row['variant']}`: test AP=`{row['test_average_precision']:.4f}`, "
            f"test ROC AUC=`{row['test_roc_auc']:.4f}`, test Brier=`{row['test_brier_score']:.4f}`, "
            f"test F1=`{row['test_f1']:.4f}`, threshold=`{row['test_threshold']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            (
                f"- Default to `{recommended['variant']}` for validation-selected operation. "
                "It wins on the validation-led tie-break over F1, average precision, and Brier score "
                "while preserving the full pipeline structure."
            ),
            (
                "- Keep the raw fused score artifact as well when downstream consumers care about "
                "fine-grained score ranking, because calibration can introduce ties even when it "
                "preserves the same thresholded decisions."
            ),
        ]
    )
    lines.extend(["", "## Variant Notes"])
    for result in results:
        lines.append(f"- `{result['variant']}`: {result['description']}")
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

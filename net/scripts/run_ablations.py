#!/usr/bin/env python3
"""Run comparable ablations for the fraud-detection pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from eval.ablation_runner import (  # noqa: E402
    AblationConfig,
    ablation_summary_frame,
    load_default_ablation_inputs,
    run_ablations,
    write_ablation_outputs,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run comparable ablations over the current fraud pipeline artifacts.",
    )
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON override for ablation config.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional output path for ablation_summary.parquet.")
    parser.add_argument("--report-json-output", type=Path, default=None, help="Optional output path for ablation_report.json.")
    parser.add_argument("--report-md-output", type=Path, default=None, help="Optional output path for ablation_report.md.")
    return parser


def _load_override(path: Path | None, base: AblationConfig) -> AblationConfig:
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    standalone_branch_mode = str(payload.get("standalone_branch_mode", base.standalone_branch_mode))
    standalone_branch_name = str(payload.get("standalone_branch_name", base.standalone_branch_name))
    return AblationConfig(
        fusion_config=base.fusion_config,
        calibration_config=base.calibration_config,
        standalone_branch_mode=standalone_branch_mode,
        standalone_branch_name=standalone_branch_name,
    )


def main() -> int:
    args = build_parser().parse_args()
    paths, fusion_valid, fusion_test, branch_predictions, base_config = load_default_ablation_inputs()
    config = _load_override(args.config_input, base_config)
    results = run_ablations(
        fusion_valid=fusion_valid,
        fusion_test=fusion_test,
        branch_predictions=branch_predictions,
        config=config,
    )
    summary = ablation_summary_frame(results)
    write_ablation_outputs(
        summary_frame=summary,
        results=results,
        config=config,
        summary_path=_resolve_path(args.summary_output, paths["reports_ablations_summary"]),
        report_json_path=_resolve_path(args.report_json_output, paths["reports_ablations_report_json"]),
        report_md_path=_resolve_path(args.report_md_output, paths["reports_ablations_report_md"]),
    )
    best = summary.iloc[0]
    print(
        f"variants={len(summary)} "
        f"best_variant={best['variant']} "
        f"best_test_ap={best['test_average_precision']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

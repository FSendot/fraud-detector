#!/usr/bin/env python3
"""Export the canonical feature contract for fraud model inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from contracts.feature_contract import (  # noqa: E402
    build_feature_contract,
    load_contract_export_inputs,
    write_feature_contract,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the canonical feature contract from current tabular model inputs.",
    )
    parser.add_argument("--json-output", type=Path, default=None, help="Optional output path for feature_contract_v1.json.")
    parser.add_argument("--markdown-output", type=Path, default=None, help="Optional output path for feature_contract_v1.md.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()
    train_frame, valid_frame, test_frame, selected_features_payload, feature_metadata = load_contract_export_inputs()
    contract = build_feature_contract(
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        selected_features_payload=selected_features_payload,
        feature_metadata=feature_metadata,
    )
    json_output = _resolve_path(args.json_output, paths["artifact_feature_contract_json"])
    markdown_output = _resolve_path(args.markdown_output, paths["artifact_feature_contract_md"])
    write_feature_contract(
        contract=contract,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )
    print(
        f"version={contract['version']} "
        f"features={contract['summary']['feature_count']} "
        f"json={json_output} "
        f"markdown={markdown_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

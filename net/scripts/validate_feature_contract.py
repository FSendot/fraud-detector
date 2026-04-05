#!/usr/bin/env python3
"""Validate a dataset against the canonical fraud feature contract."""

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
from contracts.feature_contract import (  # noqa: E402
    load_dataframe_for_contract_validation,
    load_feature_contract,
    validate_frame_against_contract,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a dataset against the canonical feature contract.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Parquet or CSV dataset to validate.")
    parser.add_argument("--contract", type=Path, default=None, help="Optional feature contract JSON path.")
    parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help="Allow extra columns outside the contract, while still validating required columns and order.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()
    contract_path = _resolve_path(args.contract, paths["artifact_feature_contract_json"])
    dataset_path = _resolve_path(args.dataset, args.dataset)
    contract = load_feature_contract(contract_path)
    frame = load_dataframe_for_contract_validation(dataset_path)
    result = validate_frame_against_contract(
        frame,
        contract,
        allow_extra_columns=args.allow_extra_columns,
    )
    print(
        json.dumps(
            {
                "contract": str(contract_path),
                "dataset": str(dataset_path),
                **result.to_payload(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

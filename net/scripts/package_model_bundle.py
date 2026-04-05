#!/usr/bin/env python3
"""Package the deterministic fraud inference model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from packaging.model_bundle import package_model_bundle  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package the trained fraud pipeline into a versioned inference bundle.",
    )
    parser.add_argument(
        "--bundle-version",
        type=str,
        default="model_v1",
        help="Versioned output directory name under artifacts/bundles/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional bundle root directory. Defaults to artifacts/bundles.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()
    output_root = _resolve_path(args.output_root, paths["artifact_bundles_dir"])
    manifest_path = package_model_bundle(
        bundle_version=args.bundle_version,
        output_root=output_root,
    )
    print(f"bundle={args.bundle_version} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export the packaged active model path into a Go-runnable runtime spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from packaging.go_runtime import export_go_runtime_spec  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the active packaged fraud model into a Go runtime spec.",
    )
    parser.add_argument("--bundle", type=Path, default=None, help="Bundle directory or manifest path.")
    parser.add_argument("--output", type=Path, default=None, help="Runtime spec output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()
    bundle_path = _resolve_path(args.bundle, paths["artifact_bundles_dir"] / "model_v1")
    output_path = _resolve_path(args.output, paths["outputs_go_runtime_spec"])
    exported = export_go_runtime_spec(bundle_path=bundle_path, output_path=output_path)
    print(f"runtime_spec={exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

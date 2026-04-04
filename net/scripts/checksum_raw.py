#!/usr/bin/env python3
"""Recompute SHA-256 checksums for every raw dataset and sync the manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import (  # noqa: E402
    DEFAULT_MANIFEST_FILE,
    DEFAULT_PATHS_FILE,
    dump_yaml_file,
    load_dataset_manifest,
    load_paths_config,
    project_root,
)
from common.io import ensure_directory, iter_regular_files, sha256_file  # noqa: E402


def _resolve_raw_dir() -> Path:
    paths = load_paths_config(DEFAULT_PATHS_FILE)
    return ensure_directory(paths["raw_data"])


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_dataset_manifest(manifest_path)
    manifest.setdefault("datasets", {})
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute SHA-256 checksums for all files in data/raw and update the manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="Path to the dataset manifest YAML.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_dir = _resolve_raw_dir()
    manifest_path = args.manifest if args.manifest.is_absolute() else project_root() / args.manifest
    manifest = _load_manifest(manifest_path)
    datasets = manifest["datasets"]

    updated: dict[str, Any] = {}
    for raw_file in iter_regular_files(raw_dir):
        if raw_file.name.startswith("."):
            continue
        relative_path = raw_file.relative_to(project_root()).as_posix()
        entry = dict(datasets.get(relative_path, {}))
        entry.update(
            {
                "filename": raw_file.name,
                "relative_path": relative_path,
                "size_bytes": raw_file.stat().st_size,
                "sha256": sha256_file(raw_file),
            }
        )
        updated[relative_path] = entry
        print(f"{relative_path} sha256={entry['sha256']}")

    manifest["datasets"] = dict(sorted(updated.items()))
    dump_yaml_file(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Copy a dataset into data/raw or register an already-present raw file."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
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
from common.io import copy_file, ensure_directory, sha256_file  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_raw_dir() -> Path:
    paths = load_paths_config(DEFAULT_PATHS_FILE)
    raw_dir = paths["raw_data"]
    return ensure_directory(raw_dir)


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_dataset_manifest(manifest_path)
    manifest.setdefault("datasets", {})
    return manifest


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    dump_yaml_file(manifest_path, manifest)


def _register_entry(
    *,
    manifest_path: Path,
    relative_path: Path,
    source_path: Path,
    acquisition: str,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    datasets = manifest["datasets"]
    key = relative_path.as_posix()
    file_size = source_path.stat().st_size
    checksum = sha256_file(source_path)
    existing = dict(datasets.get(key, {}))
    existing.update(
        {
            "filename": relative_path.name,
            "relative_path": key,
            "source_path": str(source_path),
            "acquisition": acquisition,
            "size_bytes": file_size,
            "sha256": checksum,
            "registered_utc": _utc_now(),
        }
    )
    datasets[key] = existing
    _write_manifest(manifest_path, manifest)
    return existing


def _copy_into_raw(source: Path, raw_dir: Path) -> Path:
    destination = raw_dir / source.name
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == sha256_file(source):
            return destination
        msg = f"refusing to overwrite existing raw file: {destination}"
        raise FileExistsError(msg)
    copy_file(source, destination)
    return destination


def _register_existing(raw_candidate: Path, raw_dir: Path) -> Path:
    if raw_candidate.is_absolute():
        resolved = raw_candidate
    else:
        project_candidate = (project_root() / raw_candidate).resolve()
        resolved = project_candidate if project_candidate.exists() else (raw_dir / raw_candidate)
    resolved = resolved.resolve()
    raw_dir = raw_dir.resolve()
    try:
        resolved.relative_to(raw_dir)
    except ValueError as exc:
        msg = f"registered files must live under {raw_dir}"
        raise ValueError(msg) from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a local dataset into data/raw or register a file that is already there.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source",
        type=Path,
        help="Local file to copy into data/raw/ before registering it.",
    )
    group.add_argument(
        "--register",
        type=Path,
        help="File that already exists inside data/raw/ and only needs to be registered.",
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

    if args.source is not None:
        source = args.source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        raw_file = _copy_into_raw(source, raw_dir)
        entry = _register_entry(
            manifest_path=manifest_path,
            relative_path=raw_file.relative_to(project_root()),
            source_path=source,
            acquisition="copied",
        )
    else:
        raw_file = _register_existing(args.register.expanduser(), raw_dir)
        entry = _register_entry(
            manifest_path=manifest_path,
            relative_path=raw_file.relative_to(project_root()),
            source_path=raw_file,
            acquisition="registered",
        )

    print(f"{entry['relative_path']} sha256={entry['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

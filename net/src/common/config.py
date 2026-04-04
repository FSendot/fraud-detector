"""Configuration helpers for the net module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_PATHS_FILE = DEFAULT_CONFIG_DIR / "paths.yaml"
DEFAULT_MANIFEST_FILE = DEFAULT_CONFIG_DIR / "dataset_manifest.yaml"


def project_root() -> Path:
    """Return the root directory of the net module."""

    return PROJECT_ROOT


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        msg = f"expected a YAML mapping in {path}"
        raise ValueError(msg)
    return payload


def dump_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML mapping to disk with stable key ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            sort_keys=True,
            allow_unicode=False,
            default_flow_style=False,
        )


def load_paths_config(path: Path | None = None) -> dict[str, Path]:
    """Load the path registry and resolve relative paths from the project root."""

    config_path = path or DEFAULT_PATHS_FILE
    payload = load_yaml_file(config_path)
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        msg = f"expected 'paths' to be a mapping in {config_path}"
        raise ValueError(msg)

    resolved: dict[str, Path] = {}
    for key, value in paths.items():
        location = Path(value)
        resolved[key] = location if location.is_absolute() else (PROJECT_ROOT / location).resolve()
    return resolved


def load_dataset_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the dataset manifest and validate its basic shape."""

    manifest_path = path or DEFAULT_MANIFEST_FILE
    payload = load_yaml_file(manifest_path)
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        msg = f"expected 'datasets' to be a mapping in {manifest_path}"
        raise ValueError(msg)
    payload["datasets"] = datasets
    return payload


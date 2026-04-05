#!/usr/bin/env python3
"""Run the pipeline from raw dataset registration to first-branch validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402


def _resolve_path(candidate: Path) -> Path:
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _default_raw_csvs() -> list[Path]:
    raw_dir = load_paths_config()["raw_data"]
    candidates = sorted(path.resolve() for path in raw_dir.glob("*.csv"))
    if candidates:
        return candidates
    if not candidates:
        msg = "no raw CSV found in data/raw; pass --source or --register."
        raise FileNotFoundError(msg)
    return candidates


def _run_step(script_name: str, extra_args: list[str]) -> None:
    script_path = project_root() / "scripts" / script_name
    command = [sys.executable, str(script_path), *extra_args]
    subprocess.run(command, cwd=project_root(), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fraud pipeline from raw dataset registration through first-branch usefulness evaluation.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="Local CSV to copy into data/raw/ before running the pipeline. Repeat for multiple files.",
    )
    parser.add_argument(
        "--register",
        type=Path,
        action="append",
        help="CSV that already exists inside data/raw/ and should be used. Repeat for multiple files.",
    )
    parser.add_argument(
        "--downsample-ratio",
        type=float,
        default=3.0,
        help="Negative-to-positive cap used during tabular training preparation.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Optional number of tabular features to retain during selection.",
    )
    parser.add_argument("--vae-config", type=Path, default=None, help="Optional JSON config override for VAE training.")
    parser.add_argument("--nystrom-config", type=Path, default=None, help="Optional JSON config override for Nyström training.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip dataset registration and use an existing raw CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.source and args.register:
        msg = "use either --source or --register, not both"
        raise ValueError(msg)
    raw_csvs: list[Path]

    if args.skip_fetch:
        if args.register is not None:
            raw_csvs = [_resolve_path(path) for path in args.register]
        else:
            raw_csvs = _default_raw_csvs()
    else:
        if args.source is not None:
            raw_csvs = []
            for source_path in [_resolve_path(path) for path in args.source]:
                _run_step("fetch_dataset.py", ["--source", str(source_path)])
                raw_csvs.append((load_paths_config()["raw_data"] / source_path.name).resolve())
        elif args.register is not None:
            raw_csvs = [_resolve_path(path) for path in args.register]
            for raw_csv in raw_csvs:
                _run_step("fetch_dataset.py", ["--register", str(raw_csv)])
        else:
            raw_csvs = _default_raw_csvs()

    _run_step("checksum_raw.py", [])
    preprocess_args: list[str] = []
    for raw_csv in raw_csvs:
        preprocess_args.extend(["--input", str(raw_csv)])
    _run_step("preprocess_raw.py", preprocess_args)
    _run_step("build_base_features.py", [])
    _run_step("build_behavioral_features.py", [])
    _run_step("build_sequences.py", [])
    _run_step("make_splits.py", [])

    prep_args = ["--downsample-ratio", str(args.downsample_ratio)]
    if args.top_k_features is not None:
        prep_args.extend(["--top-k-features", str(args.top_k_features)])
    _run_step("prepare_training_tabular.py", prep_args)

    vae_args: list[str] = []
    if args.vae_config is not None:
        vae_args.extend(["--config-input", str(_resolve_path(args.vae_config))])
    _run_step("train_vae.py", vae_args)

    nystrom_args: list[str] = []
    if args.nystrom_config is not None:
        nystrom_args.extend(["--config-input", str(_resolve_path(args.nystrom_config))])
    _run_step("train_nystrom_gp.py", nystrom_args)
    _run_step("evaluate_first_branch.py", [])

    print("pipeline_complete=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

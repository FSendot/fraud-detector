# Fraud Detection Research Reproduction

This repository is the `net` module of a larger system. It contains the Python
research and training workflow, the data lifecycle layout, and the contracts
that future serving code will consume.

## Project stages

1. **Acquire** raw datasets into `data/raw/`.
   Raw files are treated as immutable inputs and are tracked through
   `configs/dataset_manifest.yaml` plus SHA-256 checksums.
2. **Validate** and fingerprint raw inputs.
   `scripts/checksum_raw.py` walks `data/raw/` and keeps the manifest aligned
   with the files on disk.
3. **Prepare** data in `data/interim/` and `data/processed/`.
   Interim data is for temporary cleaning or joining steps. Processed data is
   the final, model-ready representation used by training.
4. **Train** models from `training/`.
   This is where the paper reproduction pipeline, evaluation, and experiment
   orchestration will live.
5. **Publish** immutable artifacts to `artifacts/`.
   Model binaries, metrics, plots, and reports should be written here.
6. **Serve** through the future Go integration.
   Serving contracts live in `serving/contracts/` so the Python training side
   and the Go runtime can agree on schemas, feature order, and model metadata.

## Data layout

- `data/raw/`: immutable source datasets and checksum records.
- `data/interim/`: temporary cleaned or joined data.
- `data/processed/`: final training-ready tables.

## Utility scripts

- `scripts/fetch_dataset.py` copies a local file into `data/raw/` or registers
  a file that is already there.
- `scripts/checksum_raw.py` recomputes SHA-256 hashes for every raw file and
  updates the manifest.

## Environment

Use Pipenv from this directory:

```bash
pipenv install
```


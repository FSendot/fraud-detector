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
   `scripts/preprocess_raw.py` creates a deterministic cleaned parquet file and
   a JSON report with duplicate and invalid-row counts.
4. **Train** models from `training/`.
   This is where the paper reproduction pipeline, evaluation, and experiment
   orchestration will live.
   `scripts/build_base_features.py` produces the first point-in-time-safe
   per-transaction feature layer and its machine-readable feature dictionary.
   `scripts/build_behavioral_features.py` adds leakage-safe historical
   behavioral aggregates over prior transactions for each entity.
   `scripts/build_sequences.py` converts the behavioral feature table into a
   fixed-length GRU sequence dataset plus current-row, target, and metadata files.
   `scripts/make_splits.py` creates the canonical chronological train/valid/test
   split IDs that both the tabular and sequence branches should reuse.
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
- `scripts/preprocess_raw.py` normalizes a raw CSV, drops impossible rows, and
  writes `data/interim/transactions_clean.parquet` plus a preprocessing report.
- `scripts/build_base_features.py` reads the cleaned interim parquet and writes
  `data/processed/base_features.parquet` plus `artifacts/feature_dict/base_features.json`.
- `scripts/build_behavioral_features.py` reads the base feature parquet and
  writes `data/processed/behavioral_features.parquet` plus
  `artifacts/feature_dict/behavioral_features.json`.
- `scripts/build_sequences.py` reads the behavioral feature parquet and writes
  `data/processed/sequences/` artifacts plus `artifacts/sequence_schema.json`.
- `scripts/make_splits.py` reads the behavioral feature table and sequence
  metadata, then writes canonical split ID files in `data/splits/` plus
  `reports/split_report.json`.

## Environment

Use Pipenv from this directory:

```bash
pipenv install
```

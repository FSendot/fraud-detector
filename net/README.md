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
   `scripts/prepare_training_tabular.py` applies train-only balancing,
   scaling, and feature selection for the tabular model branches.
   `scripts/train_vae.py` learns a reusable latent representation from the
   tabular branch and exports embeddings for downstream models.
   `scripts/train_nystrom_gp.py` trains a Nyström-kernel classifier on the
   exported latent representation and writes prediction files.
   `scripts/train_nystrom_tabular.py` trains a direct Nyström-kernel baseline
   on the selected tabular features so the latent branch can be compared
   against a simpler non-VAE alternative.
   `scripts/train_tree_branch.py` trains a contract-validated ExtraTrees-style
   tabular branch and exports predictions plus feature importances for later
   fusion experiments.
   `scripts/train_gru_branch.py` trains a configurable GRU branch on the
   prepared sequence dataset and exports branch predictions for later fusion.
   `scripts/build_fusion_dataset.py` joins branch prediction files into clean
   validation/test fusion datasets with strict missing-join diagnostics.
   `scripts/export_feature_contract.py` exports the canonical versioned model
   feature contract used by training and future serving integrations.
   `scripts/validate_feature_contract.py` validates a parquet or CSV dataset
   against that contract before training or serving.
   `scripts/evaluate_first_branch.py` runs branch-level usefulness checks,
   leakage warnings, calibration summaries, and error analysis outputs.
   `scripts/run_until_branch_validation.py` orchestrates the end-to-end path
   from raw CSV to the first branch usefulness report.
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
- `scripts/prepare_training_tabular.py` reads the behavioral feature table plus
  canonical split IDs and writes train/valid/test tabular model inputs, learned
  preprocessing artifacts, and `reports/training_prep_report.json`.
- `scripts/train_vae.py` reads the prepared tabular train/valid/test splits and
  writes VAE config, weights, metrics, and latent embedding parquets.
- `scripts/train_nystrom_gp.py` reads the latent embedding parquets and writes
  the Nyström classifier artifact, metrics, and valid/test prediction parquets.
- `scripts/train_nystrom_tabular.py` reads the prepared tabular train/valid/test
  splits and writes a direct-tabular Nyström baseline artifact, metrics, and
  valid/test prediction parquets for branch comparison.
- `scripts/train_tree_branch.py` reads the canonical feature contract plus the
  prepared tabular train/valid/test splits, then writes the tree model,
  config, metrics, feature importances, and valid/test prediction parquets.
- `scripts/train_gru_branch.py` reads `data/processed/sequences/` plus the
  canonical split IDs, then writes GRU weights, config, metrics, and
  valid/test prediction parquets.
- `scripts/build_fusion_dataset.py` reads branch prediction parquets, joins
  them by `transaction_id`, and writes fusion-ready validation/test tables plus
  `reports/fusion_dataset_report.json`.
- `scripts/export_feature_contract.py` reads the current train/valid/test
  model-input tables plus feature metadata and writes a canonical feature
  contract to `artifacts/contracts/`.
- `scripts/validate_feature_contract.py` checks that a dataset matches the
  canonical feature names, order, dtypes, and null-handling expectations.
- `scripts/evaluate_first_branch.py` evaluates the first branch, checks for
  overlap or suspicious signals, saves top false positives/negatives, and
  writes `reports/first_branch/usefulness_report.json` plus markdown.
- `scripts/run_until_branch_validation.py` runs the full branch pipeline from
  raw dataset registration to usefulness evaluation.

## Environment

Use Pipenv from this directory:

```bash
pipenv install
```

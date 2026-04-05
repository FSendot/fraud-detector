# Fraud Detection Pipeline Guide

This repository is the `net` module of a larger fraud-detection system. It
contains the Python training pipeline, the packaged inference flow, the shadow
scoring boundary, and the operational readiness layer that a future Go service
can integrate with.

## Main goal of this module

The goal of the project is to score financial transactions for fraud risk.

At a high level, the pipeline does this:

1. Ingest raw transaction data.
2. Clean and normalize it deterministically.
3. Build point-in-time-safe features.
4. Train several model branches.
5. Combine branch outputs into one fused score.
6. Calibrate that score so probabilities are more trustworthy.
7. Package the result into a reusable model bundle.
8. Expose the bundle through a shadow-mode scoring boundary for later service integration.

## Core Fraud-Detection Concepts

- Fraud detection is usually an imbalanced problem.
  Most transactions are legitimate, and only a small fraction are fraud.
  Because of that, plain accuracy is not enough.

- Point-in-time-safe features are features that only use information that would
  have been available at the moment of the transaction.
  This avoids leakage.

- Leakage means the model accidentally uses future information, labels, or
  downstream effects that would not exist in live scoring.
  Leakage makes offline results look unrealistically good.

- Calibration means adjusting model scores so that a score like `0.80` behaves
  more like a real 80% risk estimate.
  A model can rank transactions well but still be poorly calibrated.

- Shadow mode means the model runs and logs outputs, but does not trigger real
  business actions.
  This is the safest way to introduce a new fraud model.

- Drift means production data no longer looks like the data the model was
  trained on.
  Fraud systems must monitor drift continuously.

The project has multiple model branches:

- VAE:
  Variational Autoencoder. It learns a compressed latent representation of the
  tabular input. In this project it is mainly used as a feature-learning step.

- Nyström classifier:
  A kernel-approximation approach that can capture nonlinear structure without
  using a full expensive kernel method.

- ExtraTrees:
  An ensemble of randomized decision trees. Good for tabular data and useful as
  a robust baseline.

- Histogram Gradient Boosting:
  A boosted-tree model optimized for tabular problems. In the current IEEE run,
  this is the strongest branch.

- GRU:
  Gated Recurrent Unit. A recurrent neural network for sequential behavior.
  Here it is used to model prior transaction history.

- Fusion:
  The stage that combines branch outputs into one risk score.

- Calibration:
  The stage that converts the fused score into a more reliable probability-like
  score for decisioning.

## Current Recommended Production Path

With the current IEEE-CIS setup, the practical default is:

- primary score: calibrated fused score
- strongest underlying branch: boosted tree
- serving mode: shadow only until business rollout approves live actions

The policy layer currently defines these recommendation bands:

- `allow`: low risk
- `review`: medium risk
- `block`: high risk

These are recommendations only until the outer system enables real side effects.

## Repository Structure

Important directories:

- `configs/`
  Runtime, split, monitoring, threshold, and retraining configuration.

- `data/raw/`
  Immutable input datasets.

- `data/interim/`
  Cleaned but not yet fully model-ready data.

- `data/processed/`
  Feature layers and sequence datasets.

- `data/model_input/`
  Final tabular datasets used by model training.

- `artifacts/`
  Generated model files, metrics, contracts, preprocessing artifacts, and bundles.
  These are now ignored by git by default.

- `predictions/`
  Validation and test predictions from model branches and fusion stages.
  These are ignored by git.

- `reports/`
  Evaluation, calibration, ablation, and shadow log schemas.
  These are ignored by git.

- `outputs/`
  Batch inference outputs.
  These are ignored by git.

- `serving/contracts/`
  JSON request and response contracts for service integration.

- `serving/go/`
  Integration notes for a future Go client.

- `src/`
  Python source code by pipeline domain.

## Pipeline Stages

The main pipeline is organized like this:

1. Raw data registration and checksum tracking
2. Deterministic preprocessing
3. Base feature generation
4. Behavioral historical feature generation
5. Sequence dataset building
6. Chronological splitting
7. Tabular training preparation
8. Model branch training
9. Fusion and calibration
10. Ablation analysis
11. Bundle packaging
12. Batch inference
13. Shadow scoring
14. Production readiness configs and ops logic

## Dataset: IEEE-CIS

The project currently expects the IEEE-CIS fraud training data in:

- `data/raw/ieee-fraud-detection/train_transaction.csv`
- `data/raw/ieee-fraud-detection/train_identity.csv`

Current dataset download link provided for this project:

- [IEEE dataset download](https://drive.google.com/file/d/1OPr1oBPYKxCsQM1TlUA0wh9tCm-IGbCs/view?usp=sharing)

Expected setup after download:

1. Download the archive from the link above.
2. Extract it manually.
3. Place the extracted training files under:

```text
net/data/raw/ieee-fraud-detection/
  train_transaction.csv
  train_identity.csv
```

Note:

- This pipeline uses the IEEE training files only.
- Generated datasets, artifacts, reports, predictions, outputs, and bundles are
  intentionally ignored by git.

## Environment Setup From Scratch

Run everything from:

```bash
cd /path/to/fraud-detector/net
```

Install dependencies with Pipenv (install Pipenv if not present in your computer):

```bash
pipenv install
```

## Step-By-Step Tutorial: Full Training Pipeline

### 1. Preprocess the raw IEEE data

This merges the IEEE transaction and identity files, normalizes the schema, and
writes the cleaned interim parquet.

```bash
pipenv run python scripts/preprocess_raw.py --input data/raw/ieee-fraud-detection
```

Expected outputs:

- `data/interim/transactions_clean.parquet`
- `data/interim/transactions_clean.report.json`

### 2. Build base transaction features

```bash
pipenv run python scripts/build_base_features.py
```

### 3. Build historical behavioral features

This creates past-only transaction history features per entity.

```bash
pipenv run python scripts/build_behavioral_features.py
```

### 4. Build sequence datasets for the GRU branch

```bash
pipenv run python scripts/build_sequences.py
```

### 5. Create chronological train/validation/test splits

```bash
pipenv run python scripts/make_splits.py
```

### 6. Prepare tabular model inputs

This step applies train-only balancing, scaling, and feature selection.

```bash
pipenv run python scripts/prepare_training_tabular.py
```

### 7. Export the canonical feature contract

This contract is what training, batch inference, and serving use to agree on
feature names, order, and expectations.

```bash
pipenv run python scripts/export_feature_contract.py
```

### 8. Train the model branches

```bash
pipenv run python scripts/train_vae.py
pipenv run python scripts/train_nystrom_gp.py
pipenv run python scripts/train_nystrom_tabular.py
pipenv run python scripts/train_tree_branch.py
pipenv run python scripts/train_boosted_branch.py
pipenv run python scripts/train_gru_branch.py
```

### 9. Build fusion data and train fusion

```bash
pipenv run python scripts/build_fusion_dataset.py
pipenv run python scripts/train_fusion.py
```

### 10. Calibrate the fused score

```bash
pipenv run python scripts/calibrate_fusion.py
```

### 11. Evaluate business operating points

```bash
pipenv run python scripts/evaluate_operating_points.py
pipenv run python scripts/evaluate_operating_points.py \
  --use-calibrated \
  --score-source-name fused_calibrated \
  --report-json reports/business_threshold_report_calibrated.json \
  --report-md reports/business_threshold_report_calibrated.md \
  --summary-output reports/business_threshold_summary_calibrated.parquet
```

### 12. Run ablations

```bash
pipenv run python scripts/run_ablations.py
```

### 13. Package the model bundle

```bash
pipenv run python scripts/package_model_bundle.py
```

Expected bundle:

- `artifacts/bundles/model_v1/`

### 14. Export the Go runtime spec

This converts the active packaged serving path into a pure Go runtime spec so
the critical scoring path can run without Python.

```bash
pipenv run python scripts/export_go_runtime.py
```

Output:

- `outputs/go_runtime/model_v1/runtime_spec.json`

## Step-By-Step Tutorial: Batch Inference From The Bundle

If you already have a packaged bundle, you can score a dataset without retraining.

### Contract-aligned input case

If your dataset already matches the packaged feature contract:

```bash
pipenv run python scripts/run_batch_inference.py \
  --input data/model_input/test_tabular.parquet
```

Outputs:

- `outputs/batch_scoring/predictions.parquet`
- `outputs/batch_scoring/summary.json`

### Wider feature dataset case

If your dataset has a wider feature table that still includes the needed
preselection columns, the pipeline can rebuild the contract-aligned model input
using the packaged scaler and selector.

The summary JSON will tell you whether:

- the contract matched directly
- a rebuild was applied
- any contract mismatches were found

## Step-By-Step Tutorial: Shadow Scoring Service

The shadow scorer is the future service integration boundary.

### One-shot request from file

```bash
pipenv run python scripts/serve_shadow_scoring.py \
  --request-file /tmp/shadow_request.json
```

### Local HTTP service mode

```bash
pipenv run python scripts/serve_shadow_scoring.py --host 127.0.0.1 --port 8080
```

Available endpoints:

- `GET /health`
- `POST /score-shadow`

### Request contract

See:

- `serving/contracts/scoring_request.json`

Each request record contains:

- `transaction_id`
- optional `is_fraud`
- `features`
- optional `metadata`

### Response contract

See:

- `serving/contracts/scoring_response.json`

Each response record contains:

- branch outputs
- raw fused score
- calibrated score
- decision threshold
- predicted label
- model version
- echoed metadata

Important:

- The service is shadow only.
- It does not perform real allow/review/block side effects.
- It only returns the recommendation payload.

## Pure Go Runtime Option

If you want the serving path to run in Go instead of Python:

1. Train and package the model in Python.
2. Export the runtime spec with `scripts/export_go_runtime.py`.
3. Build the Go scorer in `serving/go/`.
4. Run the Go shadow service with the exported runtime spec.

The reusable import path is:

```go
import "frauddetector/serving/go/pkg/fraudruntime"
```

Minimal local integration example:

1. Export the runtime spec:

```bash
cd /Users/martin.zahnd/Documents/cloud/fraud-detector/net
pipenv run python scripts/export_go_runtime.py
```

2. In your Go service `go.mod`, add a local replace during development:

```go
module your-service

go 1.24.2

require frauddetector/serving/go v0.0.0

replace frauddetector/serving/go => /path/to/fraud-detector/net/serving/go
```

3. Load the exported runtime spec and score requests directly:

```go
package main

import (
	"fmt"
	"log"

	"frauddetector/serving/go/pkg/fraudruntime"
)

func main() {
	scorer, err := fraudruntime.NewScorerFromSpecPath(
		"/path/to/fraud-detector/net/outputs/go_runtime/model_v1/runtime_spec.json",
	)
	if err != nil {
		log.Fatal(err)
	}

	inputs := []fraudruntime.ScoreInput{
		{
			TransactionID: "123456789",
			Label:         nil,
			Features: map[string]float64{
				"transaction_amt": 59.95,
				"c1":              1,
				"c2":              3,
			},
			Metadata: map[string]any{
				"source": "shadow-test",
			},
		},
	}

	responses, err := scorer.ScoreMany(inputs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("model=%s calibrated_score=%.6f decision=%v\n",
		responses[0].ModelVersion,
		responses[0].CalibratedScore,
		responses[0].PredictedLabel,
	)
}
```

4. If your service prefers HTTP routing instead of direct calls, mount the shadow handler:

```go
package main

import (
	"log"
	"net/http"

	"frauddetector/serving/go/pkg/fraudruntime"
)

func main() {
	scorer, err := fraudruntime.NewScorerFromSpecPath(
		"/path/to/fraud-detector/net/outputs/go_runtime/model_v1/runtime_spec.json",
	)
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/score-shadow", fraudruntime.NewShadowHTTPHandler(scorer))

	log.Fatal(http.ListenAndServe(":8081", mux))
}
```

The exported runtime spec already contains:

- feature order expected by the active model
- boosted-branch scoring logic
- calibration tables
- packaged threshold metadata
- model version information

See:

- `serving/go/README.md`
- `serving/go/scoring_contract.md`

## Step-By-Step Tutorial: Integrating With A Go Service From Scratch

This section assumes:

- Python stays responsible for the packaged model runtime
- Go acts as the outer application or API layer
- integration starts in shadow mode

### Option A: Call the Python shadow service over HTTP

This is the simplest first integration.

1. Start the Python shadow scorer:

```bash
cd /Users/martin.zahnd/Documents/cloud/fraud-detector/net
pipenv run python scripts/serve_shadow_scoring.py --host 127.0.0.1 --port 8080
```

2. From Go, send a `POST` request to `/score-shadow`.

3. Log the response, but do not trigger business actions yet.

Minimal Go example:

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type Record struct {
	TransactionID string                 `json:"transaction_id"`
	IsFraud       *int                   `json:"is_fraud"`
	Features      map[string]float64     `json:"features"`
	Metadata      map[string]any         `json:"metadata,omitempty"`
}

type Request struct {
	RequestID string   `json:"request_id"`
	Records   []Record `json:"records"`
}

func main() {
	reqBody := Request{
		RequestID: "go-shadow-001",
		Records: []Record{
			{
				TransactionID: "txn-123",
				IsFraud:       nil,
				Features: map[string]float64{
					"amount": 100.0,
				},
				Metadata: map[string]any{
					"source": "go-api",
				},
			},
		},
	}

	payload, _ := json.Marshal(reqBody)
	resp, err := http.Post("http://127.0.0.1:8080/score-shadow", "application/json", bytes.NewReader(payload))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	var decoded map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		panic(err)
	}

	fmt.Printf("shadow response: %+v\n", decoded)
}
```

### Option B: Batch score offline data first

Before wiring live requests, many teams validate with backfills:

```bash
pipenv run python scripts/run_batch_inference.py \
  --bundle artifacts/bundles/model_v1 \
  --input data/model_input/test_tabular.parquet
```

This is useful for:

- validating contract compatibility
- checking offline output quality
- testing bundle loading and rollback flows

## Production Readiness Layer

This repository now includes operational readiness files, but not full cloud deployment.

### Threshold policy

- `configs/production_thresholds.yaml`
- `src/ops/policy_engine.py`

This maps scores into:

- `allow`
- `review`
- `block`

The current policy is still intended for shadow-first rollout.

### Monitoring

- `configs/monitoring.yaml`
- `src/ops/drift_monitoring.py`

This defines expected checks for:

- score drift
- feature drift
- latency
- data quality

### Retraining triggers

- `configs/retraining.yaml`

This defines when the system should consider retraining or rollback.

### Versioning and rollback

- `src/ops/versioning.py`

This logic:

- discovers bundle versions
- selects the newest verified one by default
- supports explicit version pinning
- supports rollback based on configured metric drops

### Production rollout guide

See:

- `docs/production_readiness.md`

## Important Files

If someone only wants the most important integration and ops files, start here:

- `artifacts/bundles/model_v1/manifest.json`
- `artifacts/contracts/feature_contract_v1.json`
- `serving/contracts/scoring_request.json`
- `serving/contracts/scoring_response.json`
- `src/serving/scoring_service.py`
- `src/inference/pipeline.py`
- `configs/production_thresholds.yaml`
- `configs/monitoring.yaml`
- `configs/retraining.yaml`
- `docs/production_readiness.md`

Generated artifacts are intentionally ignored by git now.

That includes:

- model artifacts
- prediction files
- reports
- batch outputs
- packaged bundles

Only source code, configs, contracts, and docs should be committed by default.

## Quick Start Checklist

If you want the shortest possible path from zero to a working shadow scorer:

1. Download and extract the IEEE dataset from the Google Drive link above.
2. Put the files in `data/raw/ieee-fraud-detection/`.
3. Run the full training pipeline.
4. Package the model with `scripts/package_model_bundle.py`.
5. Export the Go runtime with `scripts/export_go_runtime.py` if you want a pure Go serving path.
6. Start either the Python shadow scorer or the Go shadow scorer.
7. Call `/score-shadow` from the Go client.
8. Log outputs in shadow mode only.

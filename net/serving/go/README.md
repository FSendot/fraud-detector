# Go Runtime Serving

This directory contains a pure Go runtime for the active fraud-serving path.

Important boundary:

- Python is still used for training, evaluation, and export.
- Go is used for runtime scoring after export.
- The critical scoring path does not need Python once the runtime spec has been exported.

## What Is Exported

The current Go export supports the active packaged serving path:

- selected branch: `boosted_branch`
- fusion mode: `best_branch`
- calibration: isotonic
- decision threshold: packaged calibrated threshold

The exporter writes:

- `outputs/go_runtime/model_v1/runtime_spec.json`

That JSON contains:

- feature order
- boosted-tree structure
- isotonic calibration thresholds
- packaged threshold metadata
- policy metadata

## Export The Runtime Spec

From the `net` directory:

```bash
pipenv run python scripts/export_go_runtime.py
```

## Build The Go Binary

From the `net` directory:

```bash
mkdir -p .cache/go-build
cd serving/go
GOCACHE=/Users/martin.zahnd/Documents/cloud/fraud-detector/net/.cache/go-build go build ./cmd/shadow-score
```

## Import As A Library In Another Go Service

The reusable package is:

```go
import "frauddetector/serving/go/pkg/fraudruntime"
```

For local development in another Go service, add a `replace` rule to that
service's `go.mod`:

```go
require frauddetector/serving/go v0.0.0

replace frauddetector/serving/go => /Users/martin.zahnd/Documents/cloud/fraud-detector/net/serving/go
```

Then load the exported runtime spec in your server:

```go
package main

import (
	"log"
	"net/http"

	"frauddetector/serving/go/pkg/fraudruntime"
)

func main() {
	scorer, err := fraudruntime.NewScorerFromSpecPath(
		"/Users/martin.zahnd/Documents/cloud/fraud-detector/net/outputs/go_runtime/model_v1/runtime_spec.json",
	)
	if err != nil {
		log.Fatal(err)
	}

	http.Handle("/fraud/shadow/", http.StripPrefix("/fraud/shadow", fraudruntime.NewShadowHTTPHandler(scorer)))
	log.Fatal(http.ListenAndServe(":8080", nil))
}
```

If you do not want the helper HTTP handler, you can call the scorer directly:

```go
response, err := scorer.ScoreMany("request-123", []fraudruntime.ScoreInput{
	{
		TransactionID: "txn-1",
		Features: map[string]float64{
			"amount": 10.5,
		},
		Metadata: map[string]any{
			"source": "go-service",
		},
	},
})
```

Your calling service is responsible for:

- loading the right `runtime_spec.json`
- constructing the feature map in the packaged feature order/names
- deciding whether to expose the helper handler or call the scorer internally
- keeping the integration in shadow mode until rollout is approved

## Run One-Shot Scoring

```bash
cd /Users/martin.zahnd/Documents/cloud/fraud-detector/net/serving/go
GOCACHE=/Users/martin.zahnd/Documents/cloud/fraud-detector/net/.cache/go-build ./shadow-score \
  --spec /Users/martin.zahnd/Documents/cloud/fraud-detector/net/outputs/go_runtime/model_v1/runtime_spec.json \
  --request-file /tmp/shadow_request.json
```

## Run As A Local Go Shadow Service

```bash
cd /Users/martin.zahnd/Documents/cloud/fraud-detector/net/serving/go
GOCACHE=/Users/martin.zahnd/Documents/cloud/fraud-detector/net/.cache/go-build ./shadow-score \
  --spec /Users/martin.zahnd/Documents/cloud/fraud-detector/net/outputs/go_runtime/model_v1/runtime_spec.json \
  --host 127.0.0.1 \
  --port 8081
```

Endpoints:

- `GET /health`
- `POST /score-shadow`

## Runtime Guarantees

The Go scorer reproduces:

- boosted branch probability
- raw fused score
- isotonic calibrated score
- packaged decision threshold
- final predicted label

This is enough to run the current shadow serving path without Python in the
request path.

## Current Limitation

The exporter currently supports the active serving route only.

That means:

- it exports the currently selected `boosted_branch` path
- it does not yet export arbitrary future fusion combinations
- if the active packaged bundle changes to a different unsupported route, the
  exporter fails loudly instead of generating a misleading runtime

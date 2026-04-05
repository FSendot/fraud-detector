# Shadow Scoring Contract

This contract applies to both:

- the Python shadow scorer
- the exported Go shadow scorer

The current Go runtime reads the exported runtime spec and serves the same JSON
shape without Python in the request path.

## Request

The client sends one JSON object per call.

- `request_id`: caller-generated correlation id
- `records`: array of transactions to score

Each record contains:

- `transaction_id`: canonical id used for traceability
- `is_fraud`: optional offline label, `0`, `1`, or `null`
- `features`: map of feature name to numeric value
- `metadata`: optional opaque object echoed back unchanged

The expected feature names come from the packaged feature contract in the model
bundle. The safest client behavior is to send the exact contract features and no
extras.

## Response

The service always responds in shadow mode.

Top-level fields:

- `shadow_mode`: always `true`
- `mode`: always `shadow_only`
- `model_version`: bundle version such as `model_v1`
- `bundle_manifest`: absolute manifest path for traceability
- `runtime_defaults`: packaged threshold and score-source metadata
- `contract_check`: validation and rebuild diagnostics
- `records`: per-transaction scoring results

Each response record includes:

- `transaction_id`
- `model_version`
- `shadow_mode`
- `branch_outputs`
- `raw_fused_score`
- `calibrated_score`
- `decision_threshold`
- `predicted_label`
- `metadata`

## Go Mapping Guidance

- Use `string` for ids and version fields.
- Use `float64` for all score and threshold values.
- Use `map[string]float64` or typed structs for feature payloads.
- Treat `branch_outputs` as a map keyed by branch name.
- Preserve `metadata` as `map[string]any` if the caller needs it back.

## Operational Notes

- This contract is for shadow logging and offline comparison only.
- A `predicted_label` is returned for convenience, but it must not trigger live
  fraud decisions yet.
- If `contract_check.valid` is false, the caller should treat the response as a
  diagnostic failure and not compare it against production outcomes.

## Go Runtime Flow

1. Train and package the model in Python.
2. Export the serving path into `runtime_spec.json`.
3. Either import `frauddetector/serving/go/pkg/fraudruntime` into another Go
   service or start the bundled Go scorer with that runtime spec.
4. Send the same request JSON that the Python shadow scorer expects.

The Go runtime currently supports the active exported serving path:

- boosted-tree branch
- best-branch fusion
- isotonic calibration

## Import Pattern For Another Go Service

Local development can use a `replace` directive:

```go
require frauddetector/serving/go v0.0.0

replace frauddetector/serving/go => /path/to/fraud-detector/net/serving/go
```

Import path:

```go
import "frauddetector/serving/go/pkg/fraudruntime"
```

Typical flow:

1. Load `runtime_spec.json`.
2. Create a scorer with `fraudruntime.NewScorerFromSpecPath(...)`.
3. Call `ScoreMany(...)` directly or mount `fraudruntime.NewShadowHTTPHandler(...)`.

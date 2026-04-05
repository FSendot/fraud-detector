// Package fraudruntime exposes the exported fraud-scoring runtime for use from
// other Go services.
//
// The intended flow is:
//  1. Train and package the model in Python.
//  2. Export runtime_spec.json from the packaged bundle.
//  3. Load the runtime spec in Go with NewScorerFromSpecPath.
//  4. Score contract-aligned requests directly inside the calling Go server.
package fraudruntime

# Processor

## Local DynamoDB

Fastest end-to-end local smoke test:

```bash
cd processor
./run_local.sh
```

That script will:

- refuse to start if another process is already listening on the configured `GRPC_PORT`
- start DynamoDB Local in Docker
- create the `user_profiles` table if needed
- start the processor with the local DynamoDB endpoint and ML runtime spec
- send two test gRPC requests so the second call exercises the persisted profile path

High-volume dynamic test run:

```bash
cd processor
./run_loadtest.sh
```

That run will:

- refuse to start if another process is already listening on the configured `GRPC_PORT`
- start local DynamoDB and bootstrap `user_profiles`
- start the processor locally
- generate a large dynamic request dataset with richer ML feature coverage across trusted, burst, cross-border, device-shift, merchant-fanout, sparse new-user, and extreme-risk scenarios
- send the requests concurrently against the current pipeline
- write `requests.jsonl`, `results.jsonl`, and `summary.json` under `processor/output/loadtest/<run_id>/`

The load test now reads the runtime spec feature contract and synthesizes a broad slice of the model inputs, including:

- `c*`, `d*`, `m*`, `v*`, and `id_*` feature families
- engineered behavior features such as previous amount, prior 5/10 counts, prior 5/10 sums and standard deviations, seconds since previous transaction, and unique destination counts
- per-user transaction history so stable users evolve across the run while local DynamoDB state evolves across repeated runs

Useful overrides:

```bash
LOADTEST_REQUESTS=5000 LOADTEST_CONCURRENCY=80 ./run_loadtest.sh
```

Use a custom runtime spec for feature generation:

```bash
LOADTEST_FEATURE_SPEC_PATH=/path/to/runtime_spec.json ./run_loadtest.sh
```

Override the gRPC target without editing scripts:

```bash
GRPC_HOST=127.0.0.1 GRPC_PORT=50051 ./run_loadtest.sh
```

Stable users keep the same IDs across runs by default, so re-running the load test lets local DynamoDB state evolve over time.

The generated `summary.json` now includes:

- `decision_by_scenario` to show whether each scenario family is producing different outcomes
- `scenario_audit` to compare each scenario against its expected decision and measure match rate
- `calibrated_score` percentiles to inspect score spread
- `request_feature_count` percentiles to verify how much of the feature contract each synthetic request is populating

Start DynamoDB Local from the `processor` directory:

```bash
docker compose up -d
```

Create the `user_profiles` table used by the processor:

```bash
cd processor
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export DYNAMODB_ENDPOINT=http://localhost:8000
go run ./cmd/bootstrap-dynamodb
```

Run the processor against local DynamoDB and the exported ML runtime spec:

```bash
cd processor
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export DYNAMODB_ENDPOINT=http://localhost:8000
export FRAUD_RUNTIME_SPEC_PATH="$(cd .. && pwd)/net/outputs/go_runtime/model_v1/runtime_spec.json"
go run ./cmd/server
```

In another terminal, send a test transaction:

```bash
cd processor
go run ./cmd/client \
  -tx-id tx_local_001 \
  -user-id u_local_001 \
  -person-id p_local_001 \
  -account-id acc_local_001 \
  -amount 15000 \
  -currency ARS \
  -timestamp 2026-04-03T10:22:00Z \
  -channel web \
  -dest acc_dest_001 \
  -country BR \
  -request-id req_local_001 \
  -source-system local-payments \
  -source-component manual-test \
  -source-region local
```

The first call should succeed with a default profile. Re-running the same request exercises the persisted local DynamoDB profile update path.

Stop the local database:

```bash
docker compose down
```

Reset the local database completely:

```bash
docker compose down -v
```

## Bundle Server With ML Artifacts

Create a deployable bundle from the `processor` directory:

```bash
cd processor
./bundle_server.sh
```

What the bundler does:

- builds the current gRPC server binary for the selected `GOOS` and `GOARCH`
- finds a `runtime_spec.json` automatically or asks you to choose one if multiple are present
- copies the selected ML runtime directory into the bundle
- generates `.env.example`, `run_server.sh`, `README.md`, and `manifest.txt`
- creates a `.tar.gz` archive next to the bundle directory

Useful overrides:

```bash
BUNDLE_RUNTIME_SPEC_PATH=../net/outputs/go_runtime/model_v1/runtime_spec.json ./bundle_server.sh
```

```bash
BUNDLE_GOOS=linux BUNDLE_GOARCH=amd64 ./bundle_server.sh
```

Inside the generated bundle:

- copy `.env.example` to `.env`
- fill in the production values
- run `./run_server.sh`

## Production Configuration

The processor uses the normal AWS SDK credential chain in production, so prefer IAM roles or workload identity. You only need to set a small set of processor-specific environment variables.

Required:

- `AWS_REGION`: AWS region for DynamoDB, for example `us-east-1`
- `FRAUD_RUNTIME_SPEC_PATH`: path to the bundled `runtime_spec.json`

Recommended:

- `GRPC_PORT`: server listen port, default `50051`
- `DYNAMODB_TABLE_NAME`: DynamoDB table name, default `user_profiles`

Optional:

- `DYNAMODB_ENDPOINT`: only set this for local or non-production endpoint overrides
- `AWS_PROFILE`: useful when running manually from a shell
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`: only if you are not using IAM role-based auth

Example production shell:

```bash
cd processor
export AWS_REGION=us-east-1
export DYNAMODB_TABLE_NAME=user_profiles
export FRAUD_RUNTIME_SPEC_PATH=/opt/fraud-processor/ml/model_v1/runtime_spec.json
export GRPC_PORT=50051
go run ./cmd/server
```

Example using a generated bundle:

```bash
cd /opt/fraud-processor
cp .env.example .env
./run_server.sh
```

## Production Behavior

If `DYNAMODB_ENDPOINT` is not set, the processor uses the normal AWS SDK configuration and talks to the configured AWS DynamoDB service. If `DYNAMODB_TABLE_NAME` is not set, it defaults to `user_profiles`.

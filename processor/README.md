# Processor Local Development

## Local DynamoDB

Fastest end-to-end local smoke test:

```bash
cd /path/to/fraud-detector/processor
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
cd /path/to/fraud-detector/processor
./run_loadtest.sh
```

That run will:

- refuse to start if another process is already listening on the configured `GRPC_PORT`
- start local DynamoDB and bootstrap `user_profiles`
- start the processor locally
- generate a large dynamic request dataset with new users, returning users, bursts, country changes, destination changes, and higher-risk feature templates
- send the requests concurrently against the current pipeline
- write `requests.jsonl`, `results.jsonl`, and `summary.json` under `processor/output/loadtest/<run_id>/`

Useful overrides:

```bash
LOADTEST_REQUESTS=5000 LOADTEST_CONCURRENCY=80 ./run_loadtest.sh
```

Stable users keep the same IDs across runs by default, so re-running the load test lets local DynamoDB state evolve over time.

Start DynamoDB Local from the `processor` directory:

```bash
docker compose up -d
```

Create the `user_profiles` table used by the processor:

```bash
cd /path/to/fraud-detector/processor
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export DYNAMODB_ENDPOINT=http://localhost:8000
go run ./cmd/bootstrap-dynamodb
```

Run the processor against local DynamoDB and the exported ML runtime spec:

```bash
cd /path/to/fraud-detector/processor
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=local
export AWS_SECRET_ACCESS_KEY=local
export DYNAMODB_ENDPOINT=http://localhost:8000
export FRAUD_RUNTIME_SPEC_PATH=/path/to/fraud-detector/net/outputs/go_runtime/model_v1/runtime_spec.json
go run ./cmd/server
```

In another terminal, send a test transaction:

```bash
cd /path/to/fraud-detector/processor
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

## Production Behavior

If `DYNAMODB_ENDPOINT` is not set, the processor uses the normal AWS SDK configuration and talks to the configured AWS DynamoDB service.

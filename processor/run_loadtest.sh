#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-local}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-local}"
export DYNAMODB_ENDPOINT="${DYNAMODB_ENDPOINT:-http://localhost:8000}"
export FRAUD_RUNTIME_SPEC_PATH="${FRAUD_RUNTIME_SPEC_PATH:-${PROJECT_ROOT}/net/outputs/go_runtime/model_v1/runtime_spec.json}"
export LOADTEST_FEATURE_SPEC_PATH="${LOADTEST_FEATURE_SPEC_PATH:-${FRAUD_RUNTIME_SPEC_PATH}}"
export GRPC_HOST="${GRPC_HOST:-127.0.0.1}"
export GRPC_PORT="${GRPC_PORT:-50051}"
export GRPC_ADDRESS="${GRPC_ADDRESS:-${GRPC_HOST}:${GRPC_PORT}}"
export LOADTEST_REQUESTS="${LOADTEST_REQUESTS:-2500}"
export LOADTEST_CONCURRENCY="${LOADTEST_CONCURRENCY:-40}"
export LOADTEST_STABLE_USERS="${LOADTEST_STABLE_USERS:-200}"
export LOADTEST_USER_PREFIX="${LOADTEST_USER_PREFIX:-loadtest}"
export LOADTEST_OUTPUT_DIR="${LOADTEST_OUTPUT_DIR:-${SCRIPT_DIR}/output/loadtest}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

ensure_server_port_free() {
  local existing_listener
  existing_listener="$(lsof -nP -iTCP:"${GRPC_PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${existing_listener}" ]]; then
    echo "port ${GRPC_PORT} is already in use by another process:" >&2
    echo "${existing_listener}" >&2
    echo "stop that process first or choose another GRPC_PORT before running this script." >&2
    return 1
  fi
}

wait_for_dynamodb() {
  local host port
  local retries=30
  local delay=1
  host="${DYNAMODB_ENDPOINT#http://}"
  host="${host#https://}"
  host="${host%%:*}"
  port="${DYNAMODB_ENDPOINT##*:}"

  for ((i=1; i<=retries; i++)); do
    if nc -z "${host}" "${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay}"
  done

  echo "dynamodb local did not become ready at ${DYNAMODB_ENDPOINT}" >&2
  return 1
}

bootstrap_table() {
  local retries=10
  local delay=2

  for ((i=1; i<=retries; i++)); do
    echo "bootstrap attempt ${i}/${retries}..."
    if go run ./cmd/bootstrap-dynamodb; then
      return 0
    fi
    echo "bootstrap attempt ${i}/${retries} failed, retrying in ${delay}s..."
    sleep "${delay}"
  done

  echo "failed to bootstrap ${DYNAMODB_ENDPOINT} after ${retries} attempts" >&2
  return 1
}

wait_for_server() {
  local address="$1"
  local retries=30
  local delay=1

  for ((i=1; i<=retries; i++)); do
    if nc -z "${address%:*}" "${address##*:}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay}"
  done

  echo "processor server did not become ready at ${address}" >&2
  return 1
}

trap cleanup EXIT

cd "${SCRIPT_DIR}"

echo "Checking that gRPC port ${GRPC_PORT} is free..."
ensure_server_port_free

echo "Starting local DynamoDB..."
docker compose up -d

echo "Waiting for DynamoDB Local at ${DYNAMODB_ENDPOINT}..."
wait_for_dynamodb

echo "Bootstrapping ${DYNAMODB_ENDPOINT}..."
bootstrap_table

echo "Starting processor server on :${GRPC_PORT}..."
go run ./cmd/server &
SERVER_PID=$!

wait_for_server "${GRPC_ADDRESS}"

echo "Running high-volume load test..."
go run ./cmd/loadtest \
  -addr "${GRPC_ADDRESS}" \
  -requests "${LOADTEST_REQUESTS}" \
  -concurrency "${LOADTEST_CONCURRENCY}" \
  -stable-users "${LOADTEST_STABLE_USERS}" \
  -user-prefix "${LOADTEST_USER_PREFIX}" \
  -output-dir "${LOADTEST_OUTPUT_DIR}" \
  -feature-spec "${LOADTEST_FEATURE_SPEC_PATH}"

echo
echo "Load test completed."
pkill -x server
echo "Server killed successfully"
echo "DynamoDB Local is still running in Docker."
echo "Stop it with: cd ${SCRIPT_DIR} && docker compose down"

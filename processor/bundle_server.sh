#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_DIST_DIR="${SCRIPT_DIR}/dist"
DEFAULT_BUNDLE_NAME="fraud-processor-bundle"
DEFAULT_BINARY_NAME="fraud-processor"
DEFAULT_GOOS="$(go env GOOS)"
DEFAULT_GOARCH="$(go env GOARCH)"
DEFAULT_GOCACHE="${SCRIPT_DIR}/.cache/go-build"

prompt_if_tty() {
  local prompt="$1"
  local default_value="${2:-}"
  if [[ -t 0 ]]; then
    read -r -p "${prompt}" response
    if [[ -n "${response}" ]]; then
      printf '%s\n' "${response}"
      return
    fi
  fi
  printf '%s\n' "${default_value}"
}

confirm_if_tty() {
  local prompt="$1"
  local default_answer="${2:-Y}"
  local fallback="${3:-yes}"
  if [[ ! -t 0 ]]; then
    [[ "${fallback}" == "yes" ]]
    return
  fi

  local suffix="[Y/n]"
  if [[ "${default_answer}" == "N" ]]; then
    suffix="[y/N]"
  fi

  local response
  read -r -p "${prompt} ${suffix} " response
  response="${response:-${default_answer}}"
  [[ "${response}" =~ ^[Yy]$ ]]
}

find_runtime_specs() {
  find "${PROJECT_ROOT}" -path '*/outputs/go_runtime/*/runtime_spec.json' -type f | sort
}

choose_runtime_spec() {
  local selected="${BUNDLE_RUNTIME_SPEC_PATH:-}"
  if [[ -n "${selected}" ]]; then
    if [[ ! -f "${selected}" ]]; then
      echo "BUNDLE_RUNTIME_SPEC_PATH does not point to a file: ${selected}" >&2
      exit 1
    fi
    realpath "${selected}"
    return
  fi

  local detected=()
  while IFS= read -r path; do
    detected+=("${path}")
  done < <(find_runtime_specs)

  if [[ ${#detected[@]} -eq 1 ]]; then
    realpath "${detected[0]}"
    return
  fi

  if [[ ${#detected[@]} -gt 1 ]]; then
    if [[ -t 0 ]]; then
      echo "Multiple runtime specs found:"
      local index=1
      for candidate in "${detected[@]}"; do
        echo "  ${index}. ${candidate}"
        index=$((index + 1))
      done
      local choice
      choice="$(prompt_if_tty "Choose the runtime spec number: " "1")"
      if [[ ! "${choice}" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#detected[@]} )); then
        echo "Invalid selection: ${choice}" >&2
        exit 1
      fi
      realpath "${detected[$((choice - 1))]}"
      return
    fi

    echo "Multiple runtime specs found; set BUNDLE_RUNTIME_SPEC_PATH to choose one." >&2
    printf '  %s\n' "${detected[@]}" >&2
    exit 1
  fi

  if [[ -t 0 ]]; then
    local manual
    manual="$(prompt_if_tty "No runtime spec found automatically. Enter the full path to runtime_spec.json: " "")"
    if [[ -z "${manual}" ]] || [[ ! -f "${manual}" ]]; then
      echo "Runtime spec not found: ${manual}" >&2
      exit 1
    fi
    realpath "${manual}"
    return
  fi

  echo "No runtime spec found automatically; set BUNDLE_RUNTIME_SPEC_PATH." >&2
  exit 1
}

write_bundle_env() {
  local path="$1"
  local ml_dir_name="$2"
  cat > "${path}" <<EOF
# Listener config
GRPC_PORT=50051

# ML runtime config
FRAUD_RUNTIME_SPEC_PATH=./ml/${ml_dir_name}/runtime_spec.json

# DynamoDB config
AWS_REGION=us-east-1
DYNAMODB_TABLE_NAME=user_profiles

# Leave DYNAMODB_ENDPOINT unset in production.
# Set it only for local or test environments.
# DYNAMODB_ENDPOINT=http://localhost:8000

# AWS credentials:
# Prefer an IAM role or container/task role in production.
# If needed for non-production shells, export the standard AWS SDK vars:
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_SESSION_TOKEN=
# AWS_PROFILE=

# RDS (PostgreSQL) config — optional, transactions are persisted here
# Leave RDS_HOST unset to skip RDS entirely.
RDS_HOST=
RDS_PORT=5432
RDS_USER=fraud
RDS_PASSWORD=
RDS_DBNAME=fraud
EOF
}

write_bundle_run_script() {
  local path="$1"
  local ml_dir_name="$2"
  cat > "${path}" <<EOF
#!/bin/bash

set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "\${BUNDLE_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "\${BUNDLE_DIR}/.env"
fi

export FRAUD_RUNTIME_SPEC_PATH="\${FRAUD_RUNTIME_SPEC_PATH:-\${BUNDLE_DIR}/ml/${ml_dir_name}/runtime_spec.json}"

exec "\${BUNDLE_DIR}/bin/fraud-processor"
EOF
}

write_bundle_readme() {
  local path="$1"
  local bundle_name="$2"
  local binary_name="$3"
  local ml_dir_name="$4"
  cat > "${path}" <<EOF
# ${bundle_name}

Contents:
- \`bin/${binary_name}\`: processor server binary
- \`ml/${ml_dir_name}/runtime_spec.json\`: bundled ML runtime spec
- \`.env.example\`: runtime environment template
- \`run_server.sh\`: convenience launcher that sources \`.env\` if present

Recommended start flow:

\`\`\`bash
cp .env.example .env
./run_server.sh
\`\`\`

Required runtime env:
- \`AWS_REGION\`
- \`DYNAMODB_TABLE_NAME\` if different from \`user_profiles\`
- \`FRAUD_RUNTIME_SPEC_PATH\` if you move the bundled ML directory

Optional RDS env (transactions are persisted when set):
- \`RDS_HOST\`: PostgreSQL host (e.g. your RDS endpoint)
- \`RDS_PORT\`: default 5432
- \`RDS_USER\`, \`RDS_PASSWORD\`, \`RDS_DBNAME\`: connection credentials

AWS auth:
- prefer IAM role / instance profile / task role
- otherwise use standard AWS SDK variables such as \`AWS_ACCESS_KEY_ID\`, \`AWS_SECRET_ACCESS_KEY\`, \`AWS_SESSION_TOKEN\`, or \`AWS_PROFILE\`

Local/test-only override:
- \`DYNAMODB_ENDPOINT\`
EOF
}

main() {
  local runtime_spec
  runtime_spec="$(choose_runtime_spec)"
  local runtime_dir
  runtime_dir="$(dirname "${runtime_spec}")"
  local runtime_dir_name
  runtime_dir_name="$(basename "${runtime_dir}")"

  local bundle_name
  bundle_name="$(prompt_if_tty "Bundle name [${DEFAULT_BUNDLE_NAME}]: " "${DEFAULT_BUNDLE_NAME}")"

  local output_root
  output_root="$(prompt_if_tty "Bundle output directory [${DEFAULT_DIST_DIR}]: " "${DEFAULT_DIST_DIR}")"
  mkdir -p "${output_root}"
  output_root="$(cd "${output_root}" && pwd)"

  local target_goos="${BUNDLE_GOOS:-${DEFAULT_GOOS}}"
  local target_goarch="${BUNDLE_GOARCH:-${DEFAULT_GOARCH}}"
  if [[ -t 0 ]]; then
    target_goos="$(prompt_if_tty "Target GOOS [${target_goos}]: " "${target_goos}")"
    target_goarch="$(prompt_if_tty "Target GOARCH [${target_goarch}]: " "${target_goarch}")"
  fi

  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local bundle_dir="${output_root}/${bundle_name}-${target_goos}-${target_goarch}-${timestamp}"

  if [[ -e "${bundle_dir}" ]]; then
    echo "Bundle output already exists: ${bundle_dir}" >&2
    exit 1
  fi

  echo "Bundling processor server"
  echo "  runtime spec: ${runtime_spec}"
  echo "  target: ${target_goos}/${target_goarch}"
  echo "  output: ${bundle_dir}"

  mkdir -p "${bundle_dir}/bin" "${bundle_dir}/ml"
  mkdir -p "${DEFAULT_GOCACHE}"

  (
    cd "${SCRIPT_DIR}"
    env GOCACHE="${BUNDLE_GOCACHE:-${DEFAULT_GOCACHE}}" CGO_ENABLED=0 GOOS="${target_goos}" GOARCH="${target_goarch}" \
      go build -o "${bundle_dir}/bin/${DEFAULT_BINARY_NAME}" ./cmd/server
  )

  cp -R "${runtime_dir}" "${bundle_dir}/ml/"
  write_bundle_env "${bundle_dir}/.env.example" "${runtime_dir_name}"
  write_bundle_run_script "${bundle_dir}/run_server.sh" "${runtime_dir_name}"
  write_bundle_readme "${bundle_dir}/README.md" "${bundle_name}" "${DEFAULT_BINARY_NAME}" "${runtime_dir_name}"

  chmod +x "${bundle_dir}/bin/${DEFAULT_BINARY_NAME}" "${bundle_dir}/run_server.sh"

  local manifest="${bundle_dir}/manifest.txt"
  {
    echo "bundle_name=${bundle_name}"
    echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "goos=${target_goos}"
    echo "goarch=${target_goarch}"
    echo "server_binary=bin/${DEFAULT_BINARY_NAME}"
    echo "bundled_runtime_spec=ml/${runtime_dir_name}/runtime_spec.json"
    echo "runtime_source_dir=${runtime_dir}"
  } > "${manifest}"

  local archive_path="${bundle_dir}.tar.gz"
  tar -C "${output_root}" -czf "${archive_path}" "$(basename "${bundle_dir}")"

  echo
  echo "Bundle created successfully."
  echo "Directory: ${bundle_dir}"
  echo "Archive:   ${archive_path}"
  echo
  echo "Next step:"
  echo "  cp '${bundle_dir}/.env.example' '${bundle_dir}/.env'"
  echo "  edit '${bundle_dir}/.env' for the target environment"
  echo "  run '${bundle_dir}/run_server.sh'"
}

main "$@"

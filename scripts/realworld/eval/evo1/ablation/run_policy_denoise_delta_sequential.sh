#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ "${REPO_ROOT}" != "/" ]]; do
  if [[ -d "${REPO_ROOT}/.venv" && -d "${REPO_ROOT}/openpi" && -d "${REPO_ROOT}/steering" ]]; then
    break
  fi
  REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [[ ! -d "${REPO_ROOT}/openpi" ]]; then
  echo "Could not find repo root from ${SCRIPT_DIR}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
INPUT_DIR="${INPUT_DIR:-output/evo1/raw_fastwam}"
INDEX="${INDEX:-0}"
NUM_INDICES="${NUM_INDICES:-10}"
DIMS="${DIMS:-all}"
METRIC="${METRIC:-chunk_l2}"
SEED="${SEED:-123}"
NUM_SIGMAS="${NUM_SIGMAS:-21}"
SIGMAS="${SIGMAS:-}"
ALL_POLICY_STEPS="${ALL_POLICY_STEPS:-false}"
SERVER_COOLDOWN="${SERVER_COOLDOWN:-5}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

BASE_OUTPUT_PREFIX="${OUTPUT_PREFIX:-output/evo1/denoise_delta/denoise_delta}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
FINAL_OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${BASE_OUTPUT_PREFIX}")/${RUN_TIMESTAMP}}"
PLOTS_OUTPUT_DIR="${PLOTS_OUTPUT_DIR:-${FINAL_OUTPUT_DIR}/plots}"
FINAL_BASENAME="$(basename "${BASE_OUTPUT_PREFIX}")"
OUTPUT_PREFIX=""
PHASE1_ACTIONS=""
PLOT_SCRIPT="${SCRIPT_DIR}/plot_policy_denoise_delta.py"
SERVER_DIR="${REPO_ROOT}/scripts/realworld/eval/evo1/server"
CONDITION_FRS="false"
CONDITION_RANDOM_NOISE_MODE="none"

cd "${REPO_ROOT}"
mkdir -p "${FINAL_OUTPUT_DIR}" "${PLOTS_OUTPUT_DIR}"

SERVER_PID=""
SERVER_LOG=""
CONDITION_OUTPUT_PREFIX=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    sleep "${SERVER_COOLDOWN}"
  fi
}

cleanup() {
  cleanup_server
}
trap cleanup EXIT

mapfile -t INPUT_INDICES < <(
  "${PYTHON_BIN}" - "${INPUT_DIR}" "${INDEX}" "${NUM_INDICES}" <<'PY'
from pathlib import Path
import re
import sys

input_dir = Path(sys.argv[1])
start = int(sys.argv[2])
limit = int(sys.argv[3])
pattern = re.compile(r"input_(\d+)\.json$")
indices = []
for path in sorted(input_dir.glob("input_*.json")):
    match = pattern.match(path.name)
    if match:
        value = int(match.group(1))
        if value >= start:
            indices.append(value)
if limit > 0:
    indices = indices[:limit]
for value in indices:
    print(value)
PY
)
if [[ "${#INPUT_INDICES[@]}" -eq 0 ]]; then
  echo "No input_*.json files found in ${INPUT_DIR} at or after INDEX=${INDEX}." >&2
  exit 1
fi
printf 'selected input indices:'
for input_index in "${INPUT_INDICES[@]}"; do
  printf ' %s' "${input_index}"
done
printf '\n'

start_server() {
  local label="$1"
  local script="$2"
  SERVER_LOG="${CONDITION_OUTPUT_PREFIX}_${label}_server.log"
  ensure_port_free "${label}"
  echo "starting ${label} server on ${POLICY_HOST}:${POLICY_PORT}; frs=${CONDITION_FRS}; log=${SERVER_LOG}"
  if [[ "${label}" == "dp" || "${label}" == "dp_ddim" ]]; then
    local ddim="false"
    if [[ "${label}" == "dp_ddim" ]]; then
      ddim="true"
    fi
    setsid env FRS="${CONDITION_FRS}" DDIM="${ddim}" bash "${script}" --port="${POLICY_PORT}" >"${SERVER_LOG}" 2>&1 &
  else
    setsid env FRS="${CONDITION_FRS}" bash "${script}" --port="${POLICY_PORT}" >"${SERVER_LOG}" 2>&1 &
  fi
  SERVER_PID="$!"
  wait_for_server "${label}"
}

expected_policy_type() {
  local label="$1"
  case "${label}" in
    dp|dp_ddim|diffusion_policy) printf '%s\n' "diffusion_policy" ;;
    openpi) printf '%s\n' "openpi" ;;
    fastwam) printf '%s\n' "fastwam" ;;
    *) printf '%s\n' "${label}" ;;
  esac
}

read_server_policy_type() {
  timeout 8s "${PYTHON_BIN}" - "${REPO_ROOT}" "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
for path in (
    repo_root,
    repo_root / "openpi" / "src",
    repo_root / "openpi" / "packages" / "openpi-client" / "src",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(host=sys.argv[2], port=int(sys.argv[3]))
metadata = policy.get_server_metadata()
policy_type = metadata.get("phase2_policy_type") or metadata.get("policy_type") or ""
print(str(policy_type).strip().lower())
PY
}

read_server_scheduler() {
  timeout 8s "${PYTHON_BIN}" - "${REPO_ROOT}" "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
for path in (
    repo_root,
    repo_root / "openpi" / "src",
    repo_root / "openpi" / "packages" / "openpi-client" / "src",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(host=sys.argv[2], port=int(sys.argv[3]))
metadata = policy.get_server_metadata()
scheduler = metadata.get("scheduler")
nested = metadata.get("diffusion_policy")
if not scheduler and isinstance(nested, dict):
    scheduler = nested.get("scheduler")
print(str(scheduler or "").strip())
PY
}

port_in_use() {
  "${PYTHON_BIN}" - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1.0):
        pass
except OSError:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_port_free() {
  local label="$1"
  if port_in_use; then
    echo "Cannot start ${label}: ${POLICY_HOST}:${POLICY_PORT} is already accepting connections." >&2
    echo "Stop the existing policy server or set POLICY_PORT to a free port before rerunning." >&2
    if command -v ss >/dev/null 2>&1; then
      ss -ltnp "( sport = :${POLICY_PORT} )" >&2 || true
    fi
    exit 1
  fi
}

tail_server_log() {
  if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
    echo "last server log lines (${SERVER_LOG}):" >&2
    tail -80 "${SERVER_LOG}" >&2 || true
  fi
}

wait_for_server() {
  local label="$1"
  local expected
  expected="$(expected_policy_type "${label}")"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  local actual=""

  echo "waiting for ${label} server metadata; expecting policy_type=${expected}"
  while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "${label} server exited before becoming ready." >&2
      tail_server_log
      exit 1
    fi

    actual="$(read_server_policy_type 2>/dev/null || true)"
    actual="${actual##*$'\n'}"
    if [[ -n "${actual}" ]]; then
      if [[ "${actual}" == "${expected}" ]]; then
        if [[ "${label}" == "dp" || "${label}" == "dp_ddim" ]]; then
          local scheduler
          scheduler="$(read_server_scheduler 2>/dev/null || true)"
          scheduler="${scheduler##*$'\n'}"
          local expected_scheduler="DDPMScheduler"
          if [[ "${label}" == "dp_ddim" ]]; then
            expected_scheduler="DDIMScheduler"
          fi
          if [[ "${scheduler}" != "${expected_scheduler}" ]]; then
            echo "${label} server scheduler is ${scheduler:-<empty>}, expected ${expected_scheduler}." >&2
            tail_server_log
            exit 1
          fi
        fi
        echo "${label} server ready: policy_type=${actual}"
        return 0
      fi
      echo "Connected to ${POLICY_HOST}:${POLICY_PORT}, but expected ${expected} for ${label} and got ${actual}." >&2
      echo "This usually means an old server is still bound to the port or the new server failed to start." >&2
      tail_server_log
      exit 1
    fi
    sleep 5
  done

  echo "Timed out after ${SERVER_START_TIMEOUT}s waiting for ${label} server on ${POLICY_HOST}:${POLICY_PORT}." >&2
  tail_server_log
  exit 1
}

plot_args() {
  local index="$1"
  local -a args=(
    --input-dir "${INPUT_DIR}"
    --index "${index}"
    --output-prefix "${OUTPUT_PREFIX}"
    --dims "${DIMS}"
    --metric "${METRIC}"
    --seed "${SEED}"
    --skip-plot
  )
  if [[ -n "${SIGMAS}" ]]; then
    args+=(--sigmas "${SIGMAS}")
  else
    args+=(--num-sigmas "${NUM_SIGMAS}")
  fi
  if [[ "${ALL_POLICY_STEPS}" == "1" || "${ALL_POLICY_STEPS}" == "true" || "${ALL_POLICY_STEPS}" == "yes" ]]; then
    args+=(--all-policy-steps)
  fi
  if [[ "${CONDITION_FRS}" == "1" || "${CONDITION_FRS}" == "true" || "${CONDITION_FRS}" == "yes" ]]; then
    args+=(--frs)
  fi
  args+=(--random-noise-ratio-mode "${CONDITION_RANDOM_NOISE_MODE}")
  printf '%s\n' "${args[@]}"
}

run_policy() {
  local index="$1"
  local label="$2"
  local phase1_mode="$3"
  shift 3
  local -a args
  mapfile -t args < <(plot_args "${index}")
  args+=(--endpoint "${label}=${POLICY_HOST}:${POLICY_PORT}")
  if [[ "${phase1_mode}" == "steer" ]]; then
    args+=(--phase1-policy "${label}")
  else
    args+=(--phase1-actions "${PHASE1_ACTIONS}" --append)
  fi
  echo "running ${label} sweep"
  "${PYTHON_BIN}" "${PLOT_SCRIPT}" "${args[@]}"
}

run_condition() {
  local condition_label="$1"
  CONDITION_FRS="$2"
  CONDITION_RANDOM_NOISE_MODE="$3"
  CONDITION_OUTPUT_PREFIX="${FINAL_OUTPUT_DIR}/${FINAL_BASENAME}_${condition_label}"
  local aggregate_prefix="${CONDITION_OUTPUT_PREFIX}_mean"

  echo "=== condition ${condition_label}: frs=${CONDITION_FRS} random_noise_ratio_mode=${CONDITION_RANDOM_NOISE_MODE} ==="

  start_server "dp" "${SERVER_DIR}/steer_server_dp.sh"
  for input_index in "${INPUT_INDICES[@]}"; do
    printf -v index_padded "%06d" "${input_index}"
    OUTPUT_PREFIX="${CONDITION_OUTPUT_PREFIX}_input${index_padded}"
    PHASE1_ACTIONS="${OUTPUT_PREFIX}_phase1.npy"
    run_policy "${input_index}" "dp" "steer"
  done
  cleanup_server

  start_server "dp_ddim" "${SERVER_DIR}/steer_server_dp.sh"
  for input_index in "${INPUT_INDICES[@]}"; do
    printf -v index_padded "%06d" "${input_index}"
    OUTPUT_PREFIX="${CONDITION_OUTPUT_PREFIX}_input${index_padded}"
    PHASE1_ACTIONS="${OUTPUT_PREFIX}_phase1.npy"
    if [[ ! -f "${PHASE1_ACTIONS}" ]]; then
      echo "Expected phase1 actions not found: ${PHASE1_ACTIONS}" >&2
      exit 1
    fi
    run_policy "${input_index}" "dp_ddim" "reuse"
  done
  cleanup_server

  start_server "openpi" "${SERVER_DIR}/steer_server_openpi.sh"
  for input_index in "${INPUT_INDICES[@]}"; do
    printf -v index_padded "%06d" "${input_index}"
    OUTPUT_PREFIX="${CONDITION_OUTPUT_PREFIX}_input${index_padded}"
    PHASE1_ACTIONS="${OUTPUT_PREFIX}_phase1.npy"
    if [[ ! -f "${PHASE1_ACTIONS}" ]]; then
      echo "Expected phase1 actions not found: ${PHASE1_ACTIONS}" >&2
      exit 1
    fi
    run_policy "${input_index}" "openpi" "reuse"
  done
  cleanup_server

  start_server "fastwam" "${SERVER_DIR}/steer_server_fastwam_right.sh"
  for input_index in "${INPUT_INDICES[@]}"; do
    printf -v index_padded "%06d" "${input_index}"
    OUTPUT_PREFIX="${CONDITION_OUTPUT_PREFIX}_input${index_padded}"
    PHASE1_ACTIONS="${OUTPUT_PREFIX}_phase1.npy"
    if [[ ! -f "${PHASE1_ACTIONS}" ]]; then
      echo "Expected phase1 actions not found: ${PHASE1_ACTIONS}" >&2
      exit 1
    fi
    run_policy "${input_index}" "fastwam" "reuse"
  done
  cleanup_server

  local -a aggregate_args=(--output-prefix "${aggregate_prefix}" --skip-plot)
  for input_index in "${INPUT_INDICES[@]}"; do
    printf -v index_padded "%06d" "${input_index}"
    aggregate_args+=(--aggregate-jsonl "${CONDITION_OUTPUT_PREFIX}_input${index_padded}.jsonl")
    aggregate_args+=(--normalization-jsonl "${FINAL_OUTPUT_DIR}/${FINAL_BASENAME}_no_frs_input${index_padded}.jsonl")
  done
  "${PYTHON_BIN}" "${PLOT_SCRIPT}" "${aggregate_args[@]}"
  echo "condition ${condition_label} done"
  echo "mean jsonl: ${aggregate_prefix}.jsonl"
}

shared_y_max() {
  "${PYTHON_BIN}" - \
    "${FINAL_OUTPUT_DIR}/${FINAL_BASENAME}_no_frs_mean.jsonl" \
    "${FINAL_OUTPUT_DIR}/${FINAL_BASENAME}_frs_mean.jsonl" \
    "${FINAL_OUTPUT_DIR}/${FINAL_BASENAME}_frs_random_noise_mean.jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path

max_y = 0.0
for path_text in sys.argv[1:]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            value = float(json.loads(line).get("plot_y_normalized", 0.0) or 0.0)
            if math.isfinite(value):
                max_y = max(max_y, value)
print(max(1.0, max_y * 1.05))
PY
}

plot_condition_mean() {
  local condition_label="$1"
  local y_max="$2"
  local mean_jsonl="${FINAL_OUTPUT_DIR}/${FINAL_BASENAME}_${condition_label}_mean.jsonl"
  local final_png="${PLOTS_OUTPUT_DIR}/${FINAL_BASENAME}_${condition_label}_mean.png"
  "${PYTHON_BIN}" "${PLOT_SCRIPT}" \
    --plot-jsonl "${mean_jsonl}" \
    --plot-path "${final_png}" \
    --y-max "${y_max}"
  echo "plot: ${final_png}"
}

run_condition "no_frs" "false" "none"
run_condition "frs" "true" "none"
run_condition "frs_random_noise" "true" "sigma"

Y_MAX="$(shared_y_max)"
echo "shared y max: ${Y_MAX}"
plot_condition_mean "no_frs" "${Y_MAX}"
plot_condition_mean "frs" "${Y_MAX}"
plot_condition_mean "frs_random_noise" "${Y_MAX}"

echo "done"
printf 'indices:'
for input_index in "${INPUT_INDICES[@]}"; do
  printf ' %s' "${input_index}"
done
printf '\n'
echo "plots:"
echo "  ${PLOTS_OUTPUT_DIR}/${FINAL_BASENAME}_no_frs_mean.png"
echo "  ${PLOTS_OUTPUT_DIR}/${FINAL_BASENAME}_frs_mean.png"
echo "  ${PLOTS_OUTPUT_DIR}/${FINAL_BASENAME}_frs_random_noise_mean.png"

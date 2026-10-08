#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -z "${RP_DATA_ROOT:-}" ]]; then
  echo "[noise_train] RP_DATA_ROOT is not set." >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  LAUNCH_PYTHON="${LAUNCH_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
else
  LAUNCH_PYTHON="${LAUNCH_PYTHON:-python3}"
fi
NOISE_ACTION_CONFIG="${NOISE_ACTION_CONFIG:-noise_action}"
NOISE_FULL_CONFIG="${NOISE_FULL_CONFIG:-noise_full}"
TIME="${TIME:-$(date +%Y%m%d)}"
export TIME
NOISE_ACTION_CKPT="${NOISE_ACTION_CKPT:-${RP_DATA_ROOT}/evo1/noise_action_${TIME}/step_final}"
NOISE_FULL_CKPT="${NOISE_FULL_CKPT:-${RP_DATA_ROOT}/evo1/noise_full_${TIME}/step_final}"

if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
cd "${SCRIPT_DIR}"

echo "[noise_train] TIME: ${TIME}"
echo "[noise_train] Launcher python: ${LAUNCH_PYTHON}"
echo "[noise_train] Phase 1: train action expert on real robot noise data for 5000 steps"
echo "[noise_train] Phase 1 config: ${NOISE_ACTION_CONFIG}"
"${LAUNCH_PYTHON}" train.py --config "${NOISE_ACTION_CONFIG}"

if [[ ! -d "${NOISE_ACTION_CKPT}" ]]; then
  echo "[noise_train] Missing phase 1 checkpoint: ${NOISE_ACTION_CKPT}" >&2
  exit 1
fi

echo "[noise_train] Phase 1 checkpoint: ${NOISE_ACTION_CKPT}"
echo "[noise_train] Phase 2: unfreeze VLM and train on real robot noise data for 50000 steps"
echo "[noise_train] Phase 2 config: ${NOISE_FULL_CONFIG}"
"${LAUNCH_PYTHON}" train.py --config "${NOISE_FULL_CONFIG}"

if [[ ! -d "${NOISE_FULL_CKPT}" ]]; then
  echo "[noise_train] Missing phase 2 checkpoint: ${NOISE_FULL_CKPT}" >&2
  exit 1
fi

echo "[noise_train] Final checkpoint: ${NOISE_FULL_CKPT}"

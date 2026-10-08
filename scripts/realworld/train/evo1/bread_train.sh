#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -z "${RP_DATA_ROOT:-}" ]]; then
  echo "[bread_train] RP_DATA_ROOT is not set." >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  LAUNCH_PYTHON="${LAUNCH_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
else
  LAUNCH_PYTHON="${LAUNCH_PYTHON:-python3}"
fi
BREAD_ACTION_CONFIG="${BREAD_ACTION_CONFIG:-bread_action}"
BREAD_FULL_CONFIG="${BREAD_FULL_CONFIG:-bread_full}"
TIME="${TIME:-$(date +%Y%m%d)}"
export TIME
BREAD_ACTION_CKPT="${BREAD_ACTION_CKPT:-${RP_DATA_ROOT}/evo1/bread_action_${TIME}/step_final}"
BREAD_FULL_CKPT="${BREAD_FULL_CKPT:-${RP_DATA_ROOT}/evo1/bread_full_${TIME}/step_final}"

if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
cd "${SCRIPT_DIR}"

echo "[bread_train] TIME: ${TIME}"
echo "[bread_train] Launcher python: ${LAUNCH_PYTHON}"
echo "[bread_train] Phase 1 config: ${BREAD_ACTION_CONFIG}"
"${LAUNCH_PYTHON}" train.py --config "${BREAD_ACTION_CONFIG}"

if [[ ! -d "${BREAD_ACTION_CKPT}" ]]; then
  echo "[bread_train] Missing phase 1 checkpoint: ${BREAD_ACTION_CKPT}" >&2
  exit 1
fi

echo "[bread_train] Phase 1 checkpoint: ${BREAD_ACTION_CKPT}"
echo "[bread_train] Phase 2 config: ${BREAD_FULL_CONFIG}"
"${LAUNCH_PYTHON}" train.py --config "${BREAD_FULL_CONFIG}"

if [[ ! -d "${BREAD_FULL_CKPT}" ]]; then
  echo "[bread_train] Missing phase 2 checkpoint: ${BREAD_FULL_CKPT}" >&2
  exit 1
fi

echo "[bread_train] Final checkpoint: ${BREAD_FULL_CKPT}"

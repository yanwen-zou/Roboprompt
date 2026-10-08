#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/runtime.sh"
cd "${REPO_ROOT}"
if ! is_true "${DRY_RUN:-false}" && ! "${PYTHON_BIN}" -c 'import flexivrdk'; then
  echo "Install Flexiv RDK in PYTHON_BIN=${PYTHON_BIN}." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/real_robot_data/play_toys}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-put toys into the box}"
FPS="${FPS:-10}"
RESOLUTION_H="${RESOLUTION_H:-224}"
RESOLUTION_W="${RESOLUTION_W:-224}"

run_python "${REPO_ROOT}/hardware/record.py" \
  --output "${OUTPUT_DIR}" \
  --task-description "${TASK_DESCRIPTION}" \
  --fps "${FPS}" \
  --resolution "${RESOLUTION_H}" "${RESOLUTION_W}" \
  "$@"

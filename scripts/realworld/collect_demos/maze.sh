#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/runtime.sh"
cd "${REPO_ROOT}"
if ! is_true "${DRY_RUN:-false}" && ! "${PYTHON_BIN}" -c 'import flexivrdk'; then
  echo "Install Flexiv RDK in PYTHON_BIN=${PYTHON_BIN}." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/real_robot_data/maze_c}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-push cube to the red flag through the maze}"
FPS="${FPS:-20}"
RESOLUTION_H="${RESOLUTION_H:-224}"
RESOLUTION_W="${RESOLUTION_W:-224}"
DISABLE_GRIPPER_CMD="${DISABLE_GRIPPER_CMD:-true}"
Z_FREEZE_CLUTCH_THRESHOLD="${Z_FREEZE_CLUTCH_THRESHOLD:--0.2}"

args=(
  --output "${OUTPUT_DIR}"
  --task-description "${TASK_DESCRIPTION}"
  --fps "${FPS}"
  --resolution "${RESOLUTION_H}" "${RESOLUTION_W}"
  --z-freeze-clutch-threshold "${Z_FREEZE_CLUTCH_THRESHOLD}"
)

if [[ "${DISABLE_GRIPPER_CMD}" == "1" || "${DISABLE_GRIPPER_CMD}" == "true" || "${DISABLE_GRIPPER_CMD}" == "yes" ]]; then
  args+=(--disable-gripper-cmd)
fi

run_python "${REPO_ROOT}/hardware/record.py" \
  "${args[@]}" \
  "$@"

#!/usr/bin/env bash
# Usage:
#   scripts/realworld/collect_demos/to_lerobot_compute_norm.sh [<path> ...] [--output DIR] [--fps FPS]
#
# This wrapper forwards all arguments to `convert_hdf5_to_lerobot.py`.
# It converts HDF5 demos into LeRobot format and then computes `norm_stats.json`
# in the generated LeRobot dataset directory.
#
# If --fps is not provided, the converter uses data.attrs["env_info"]["fps"]
# from the input HDF5 file. Multiple HDF5 inputs must have the same fps unless
# --fps is provided explicitly.
#
# It uses `.venv/bin/python` when available, otherwise falls back to `python`.
#
# By default, output is written under the input location:
# - directory input: `<input_dir>/lerobot-<input_dir_name>`
# - single HDF5 file: `<input_dir>/lerobot-<timestamp>`
# - multiple HDF5 inputs: one shared `lerobot-<common_parent_name>` directory
#
# If no path is provided, the newest `*.hdf5` / `*.h5` file under
# `${REPO_ROOT}/real_robot_data` is used automatically.
#
# Example:
#   scripts/realworld/collect_demos/to_lerobot_compute_norm.sh \
#     path/to/demo_a.hdf5 path/to/demo_b.hdf5
#
#   scripts/realworld/collect_demos/to_lerobot_compute_norm.sh \
#     path/to/demo_dir --output path/to/lerobot --fps 30
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_PYTHON_BIN="python"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

if [[ "$#" -eq 0 ]]; then
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/real_robot_data}"
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "No input paths were provided and DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    exit 1
  fi

  latest_hdf5="$(
    find "${DATA_ROOT}" -type f \( -name '*.hdf5' -o -name '*.h5' \) -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "${latest_hdf5}" ]]; then
    echo "No input paths were provided and no HDF5 files were found under ${DATA_ROOT}" >&2
    exit 1
  fi

  echo "No input path provided; using latest HDF5: ${latest_hdf5}" >&2
  set -- "${latest_hdf5}"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_hdf5_to_lerobot.py" "$@"

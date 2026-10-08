#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON_BIN="python"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
TRAJ_SCRIPT="${SCRIPT_DIR}/labeling/traj_label_rw.py"
WRIST_SCRIPT="${SCRIPT_DIR}/labeling/wrist_label_rw.py"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rw_label_batch.sh DATASET_DIR [--overwrite] [--workers N] [--max-episodes N] [-- traj_label_rw args...]

Batch-run traj_label_rw.py and wrist_label_rw.py over a standard LeRobot dataset.

Arguments:
  DATASET_DIR      Path to dataset root (contains data/, videos/, meta/, extras/).

Options:
  --overwrite      Pass --overwrite to both label scripts so existing outputs are replaced.
  --workers N      Number of episodes to process in parallel for traj. Default: 8.
  --max-episodes N Process at most N episodes after sorting by index. Default: all.

Examples:
  bash scripts/run_rw_label_batch.sh /path/to/lerobot-dataset
  bash scripts/run_rw_label_batch.sh /path/to/lerobot-dataset --overwrite
  bash scripts/run_rw_label_batch.sh /path/to/lerobot-dataset --workers 8
  bash scripts/run_rw_label_batch.sh /path/to/lerobot-dataset -- --future-len 80

Environment:
  PYTHON_BIN       Python executable to use. Default: repo .venv/bin/python when available.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

DATASET_DIR="$1"
shift

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${TRAJ_SCRIPT}" ]]; then
  echo "Missing script: ${TRAJ_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${WRIST_SCRIPT}" ]]; then
  echo "Missing script: ${WRIST_SCRIPT}" >&2
  exit 1
fi

WORKERS=8
MAX_EPISODES=""
OVERWRITE=0
EXTRA_ARGS=()
seen_double_dash=0

while [[ $# -gt 0 ]]; do
  if [[ ${seen_double_dash} -eq 1 ]]; then
    EXTRA_ARGS+=("$1")
    shift
    continue
  fi

  case "$1" in
    --)
      seen_double_dash=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --workers" >&2
        exit 1
      fi
      WORKERS="$2"
      shift 2
      ;;
    --max-episodes)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --max-episodes" >&2
        exit 1
      fi
      MAX_EPISODES="$2"
      shift 2
      ;;
    *)
      echo "Unexpected argument: $1. Use '--' before extra label script arguments." >&2
      exit 1
      ;;
  esac
done

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --workers value: ${WORKERS}. Expected a positive integer." >&2
  exit 1
fi

if [[ -n "${MAX_EPISODES}" ]] && ! [[ "${MAX_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --max-episodes value: ${MAX_EPISODES}. Expected a positive integer." >&2
  exit 1
fi

# Discover episodes from data/chunk-*/episode_*.parquet
mapfile -t EPISODE_INDICES < <(find "${DATASET_DIR}/data" -maxdepth 2 -name 'episode_*.parquet' -print | awk -F'episode_' '{sub(/\.parquet$/, ""); print $2+0}' | sort -n | uniq)

TOTAL=${#EPISODE_INDICES[@]}
if [[ ${TOTAL} -eq 0 ]]; then
  echo "No episodes found under ${DATASET_DIR}/data. Expected data/chunk-*/episode_*.parquet." >&2
  exit 1
fi

if [[ -n "${MAX_EPISODES}" && ${TOTAL} -gt ${MAX_EPISODES} ]]; then
  EPISODE_INDICES=("${EPISODE_INDICES[@]:0:${MAX_EPISODES}}")
  TOTAL="${MAX_EPISODES}"
fi

echo "Matched ${TOTAL} episode(s) under ${DATASET_DIR}."
echo "Python: ${PYTHON_BIN}"
echo "Workers: ${WORKERS}"
echo "Overwrite: $([[ ${OVERWRITE} -eq 1 ]] && echo yes || echo no)"
if [[ -n "${MAX_EPISODES}" ]]; then
  echo "Max episodes: ${MAX_EPISODES}"
else
  echo "Max episodes: all"
fi

# ------------------------------------------------------------------
# Run wrist_label_rw.py once (it handles all episodes internally)
# ------------------------------------------------------------------
WRIST_EXTRA_ARGS=()
if [[ ${OVERWRITE} -eq 1 ]]; then
  WRIST_EXTRA_ARGS+=("--overwrite")
fi

echo "[wrist] start (all episodes)"
if "${PYTHON_BIN}" "${WRIST_SCRIPT}" --dataset-dir "${DATASET_DIR}" "${WRIST_EXTRA_ARGS[@]}"; then
  echo "[wrist] done"
else
  echo "[wrist] failed" >&2
fi

# ------------------------------------------------------------------
# Run traj_label_rw.py per-episode (can be parallelized)
# ------------------------------------------------------------------
status_dir="$(mktemp -d)"
trap 'rm -rf "${status_dir}"' EXIT

export PYTHON_BIN TRAJ_SCRIPT OVERWRITE DATASET_DIR

xargs_status=0
printf '%s\n' "${EPISODE_INDICES[@]}" \
  | xargs -P "${WORKERS}" -I "{}" bash -c '
      idx="$1"
      status_dir="$2"
      ep_name=$(printf "episode_%06d" "${idx}")
      extras_dir="${DATASET_DIR}/extras/${ep_name}"

      traj_needed=0
      if [[ ${OVERWRITE} -eq 1 ]]; then
        traj_needed=1
      elif [[ ! -f "${extras_dir}/target_pixels.npy" ]]; then
        traj_needed=1
      fi

      if [[ ${traj_needed} -eq 0 ]]; then
        echo "[${ep_name}] skip (target_pixels exists)"
        touch "${status_dir}/processed.${ep_name}"
        exit 0
      fi

      echo "[${ep_name}] start traj"
      if "${PYTHON_BIN}" "${TRAJ_SCRIPT}" --dataset-dir "${DATASET_DIR}" --episode-index "${idx}" "${EXTRA_ARGS[@]:+${EXTRA_ARGS[@]}}"; then
        echo "[${ep_name}] traj done"
        touch "${status_dir}/processed.${ep_name}"
      else
        echo "[${ep_name}] traj failed" >&2
        touch "${status_dir}/failed.${ep_name}"
        exit 1
      fi
    ' _ "{}" "${status_dir}" "${EXTRA_ARGS[@]:+${EXTRA_ARGS[@]}}" \
  || xargs_status=$?

processed_count=$(find "${status_dir}" -type f -name 'processed.*' | wc -l | tr -d '[:space:]')
failed_count=$(find "${status_dir}" -type f -name 'failed.*' | wc -l | tr -d '[:space:]')

echo
echo "Done."
echo "Processed episodes: ${processed_count}"
echo "Failed episodes: ${failed_count}"

if [[ ${failed_count} -gt 0 || ${xargs_status} -ne 0 ]]; then
  exit 1
fi

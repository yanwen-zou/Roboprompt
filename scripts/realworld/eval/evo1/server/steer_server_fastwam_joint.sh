#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FASTWAM_CONFIG="${FASTWAM_CONFIG:-bread_joint_2cam224_1e-4}"
exec bash "${SCRIPT_DIR}/steer_server_fastwam.sh" "$@"

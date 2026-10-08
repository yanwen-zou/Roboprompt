#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PI05_CONFIG="${PI05_CONFIG:-pi05_flexiv_maze}"
exec bash "${SCRIPT_DIR}/steer_server_openpi.sh" "$@"

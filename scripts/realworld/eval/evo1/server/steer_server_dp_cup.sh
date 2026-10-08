#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DP_CONFIG="${DP_CONFIG:-train_diffusion_transformer_hybrid_flexiv_cup_lerobot_v21_image}"
exec bash "${SCRIPT_DIR}/steer_server_dp.sh" "$@"

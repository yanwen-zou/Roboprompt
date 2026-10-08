#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../lib/runtime.sh"
cd "${REPO_ROOT}"
require_path EVO1_CKPT
require_path PI05_CKPT
: "${PI05_CONFIG:?Set PI05_CONFIG to the config used to train the policy}"
args=(
  --port="${PORT:-8000}"
  --steerer.mode=evo
  --steerer.ckpt="${EVO1_CKPT}"
  --steerer.seed="${STEERER_SEED:-0}"
  --policy.type=openpi
  --policy.config="${PI05_CONFIG}"
  --policy.dir="${PI05_CKPT}"
  --random-noise-ratio-step="${RANDOM_NOISE_RATIO_STEP:-0.2}"
)
if is_true "${FRS:-false}"; then args+=(--frs); fi
if is_true "${NON_LINEAR_NOISE:-false}"; then
  echo "NON_LINEAR_NOISE is not supported by steering/scripts/serve_policy.py." >&2
  exit 2
fi
run_python steering/scripts/serve_policy.py "${args[@]}" "$@"

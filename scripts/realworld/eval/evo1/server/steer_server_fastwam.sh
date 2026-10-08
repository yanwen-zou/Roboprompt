#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../lib/runtime.sh"
cd "${REPO_ROOT}"
require_path EVO1_CKPT
require_path FASTWAM_CKPT
: "${FASTWAM_CONFIG:?Set FASTWAM_CONFIG to the config used to train the policy}"
require_path FASTWAM_STATS
args=(
  --port="${PORT:-8000}"
  --steerer.mode=evo
  --steerer.ckpt="${EVO1_CKPT}"
  --steerer.seed="${STEERER_SEED:-0}"
  --policy.type=fastwam
  --policy.config="${FASTWAM_CONFIG}"
  --policy.dir="${FASTWAM_CKPT}"
  --policy.stats="${FASTWAM_STATS}"
  --policy.sigma-shift="${SIGMA_SHIFT:-1.0}"
  --random-noise-ratio-step="${RANDOM_NOISE_RATIO_STEP:-0.2}"
)
if is_true "${FRS:-true}"; then args+=(--frs); fi
if is_true "${NON_LINEAR_NOISE:-false}"; then
  echo "NON_LINEAR_NOISE is not supported by steering/scripts/serve_policy.py." >&2
  exit 2
fi
run_python steering/scripts/serve_policy.py "${args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../lib/runtime.sh"

POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-output/evo1/openpi}"
TASK="${TASK:-cook bread}"
STEER="${STEER:-evo}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
FPS="${FPS:-20.0}"
NUM_EPISODES="${NUM_EPISODES:-20}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2000}"
RENDER_HEIGHT="${RENDER_HEIGHT:-224}"
RENDER_WIDTH="${RENDER_WIDTH:-224}"
ON_SCREEN="${ON_SCREEN:-true}"
PROMPT_UI="${PROMPT_UI:-local}"
WEB_STEER_URL="${WEB_STEER_URL:-http://127.0.0.1:8765}"
WEB_STEER_PUBLISH_HZ="${WEB_STEER_PUBLISH_HZ:-5.0}"
# Keep local policy/Web requests direct even when HTTP(S) proxies are enabled.
# Preserve both spellings because Python libraries may prefer lowercase.
export no_proxy="${no_proxy:+${no_proxy},}${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1,0.0.0.0"
export NO_PROXY="${no_proxy}"
ENABLE_RTC="${ENABLE_RTC:-false}"
EXECUTE_OBSERVATION_TARGET_DELTA_ACTIONS="${EXECUTE_OBSERVATION_TARGET_DELTA_ACTIONS:-false}"
RTC_DEFAULT_DELAY_STEPS="${RTC_DEFAULT_DELAY_STEPS:-4}"
RTC_LATENCY_PERCENTILE="${RTC_LATENCY_PERCENTILE:-0.95}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-exp}"

cd "${REPO_ROOT}"

args=(
  --args.host "${POLICY_HOST}"
  --args.port "${POLICY_PORT}"
  --args.action-horizon "${ACTION_HORIZON}"
  --args.fps "${FPS}"
  --args.num-episodes "${NUM_EPISODES}"
  --args.max-episode-steps "${MAX_EPISODE_STEPS}"
  --args.render-height "${RENDER_HEIGHT}"
  --args.render-width "${RENDER_WIDTH}"
  --args.output-dir "${OUTPUT_DIR}"
  --args.task "${TASK}"
  --args.steer "${STEER}"
  --args.prompt-ui "${PROMPT_UI}"
  --args.web-steer-url "${WEB_STEER_URL}"
  --args.web-steer-publish-hz "${WEB_STEER_PUBLISH_HZ}"
  --args.rtc-default-delay-steps "${RTC_DEFAULT_DELAY_STEPS}"
  --args.rtc-latency-percentile "${RTC_LATENCY_PERCENTILE}"
  --args.rtc-max-guidance-weight "${RTC_MAX_GUIDANCE_WEIGHT}"
  --args.rtc-prefix-attention-schedule "${RTC_PREFIX_ATTENTION_SCHEDULE}"
)

if [[ "${ON_SCREEN}" == "1" || "${ON_SCREEN}" == "true" || "${ON_SCREEN}" == "yes" ]]; then
  args+=(--args.on-screen)
fi

if [[ "${ENABLE_RTC}" == "1" || "${ENABLE_RTC}" == "true" || "${ENABLE_RTC}" == "yes" ]]; then
  args+=(--args.enable-rtc)
fi

if [[ "${EXECUTE_OBSERVATION_TARGET_DELTA_ACTIONS}" == "1" || "${EXECUTE_OBSERVATION_TARGET_DELTA_ACTIONS}" == "true" || "${EXECUTE_OBSERVATION_TARGET_DELTA_ACTIONS}" == "yes" ]]; then
  args+=(--args.execute-observation-target-delta-actions)
fi

run_python openpi/examples/flexiv_real/main.py "${args[@]}" "$@"

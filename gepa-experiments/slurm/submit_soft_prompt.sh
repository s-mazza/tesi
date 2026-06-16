#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/soft_prompt_topical_chat_engagingness_smoke.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
SLURM_NODE="${SLURM_NODE:-}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gepa-experiments/results/slurm}"
JOB_SLUG="${JOB_SLUG:-soft-prompt-${DATASET:-topical_chat}-${DIMENSION:-engagingness}}"
JOB_SLUG="${JOB_SLUG// /_}"
SOFT_PROMPT_IMAGE="${SOFT_PROMPT_IMAGE:-geval_gepa_softprompt:latest}"
TELEGRAM_MONITOR="${TELEGRAM_MONITOR:-1}"
TELEGRAM_MONITOR_SCRIPT="${TELEGRAM_MONITOR_SCRIPT:-gepa-experiments/slurm/telegram_monitor.py}"
TELEGRAM_MONITOR_ROOT="${TELEGRAM_MONITOR_ROOT:-gepa-experiments/results/monitor}"
TELEGRAM_MONITOR_POLL_SECONDS="${TELEGRAM_MONITOR_POLL_SECONDS:-60}"
TELEGRAM_MONITOR_CREDENTIALS="${TELEGRAM_MONITOR_CREDENTIALS:-$HOME/.telegram_credentials}"

NODE_ARGS=()
if [[ -n "$SLURM_NODE" && "$SLURM_NODE" != "auto" && "$SLURM_NODE" != "AUTO" ]]; then
  NODE_ARGS=(-w "$SLURM_NODE")
fi

EXCLUDE_ARGS=()
if [[ -n "$SLURM_EXCLUDE" ]]; then
  EXCLUDE_ARGS=(--exclude="$SLURM_EXCLUDE")
fi

DEPENDENCY_ARGS=()
if [[ -n "$SLURM_DEPENDENCY" ]]; then
  DEPENDENCY_ARGS=(--dependency="$SLURM_DEPENDENCY")
fi

mkdir -p "$OUTPUT_ROOT"

SBATCH_OUTPUT="$(sbatch \
  --parsable \
  -N 1 \
  --gpus="$GPU_SPEC" \
  "${NODE_ARGS[@]}" \
  "${EXCLUDE_ARGS[@]}" \
  "${DEPENDENCY_ARGS[@]}" \
  --time="$SLURM_TIME" \
  --mem="$SLURM_MEM" \
  --output="${OUTPUT_ROOT}/slurm-%j-${JOB_SLUG}.out" \
  --export=ALL,CONFIG_FILE="$CONFIG_FILE",IMAGE_NAME="$SOFT_PROMPT_IMAGE" \
  gepa-experiments/slurm/run_docker.sh \
  "bash gepa-experiments/slurm/run_soft_prompt_job.sh")"

JOB_ID="${SBATCH_OUTPUT%%;*}"
echo "Submitted batch job ${JOB_ID}"

start_telegram_monitor() {
  if [[ "$TELEGRAM_MONITOR" == "0" || "$TELEGRAM_MONITOR" == "false" ]]; then
    echo "Telegram monitor disabled by TELEGRAM_MONITOR=${TELEGRAM_MONITOR}."
    return 0
  fi
  if [[ ! -f "$TELEGRAM_MONITOR_CREDENTIALS" ]]; then
    echo "Telegram monitor skipped: missing credentials at ${TELEGRAM_MONITOR_CREDENTIALS}." >&2
    return 0
  fi
  if [[ ! -f "$TELEGRAM_MONITOR_SCRIPT" ]]; then
    echo "Telegram monitor skipped: missing script ${TELEGRAM_MONITOR_SCRIPT}." >&2
    return 0
  fi

  mkdir -p "$TELEGRAM_MONITOR_ROOT"
  local monitor_label="${TELEGRAM_MONITOR_LABEL:-GEPA soft prompt ${DATASET:-topical_chat}/${DIMENSION:-engagingness} ${JOB_ID}}"
  local monitor_log="${TELEGRAM_MONITOR_ROOT}/telegram_monitor_${JOB_ID}.out"
  local monitor_pid="${TELEGRAM_MONITOR_ROOT}/telegram_monitor_${JOB_ID}.pid"
  local monitor_state="${TELEGRAM_MONITOR_ROOT}/.state_${JOB_ID}"
  local slurm_log="${OUTPUT_ROOT}/slurm-${JOB_ID}-${JOB_SLUG}.out"

  nohup python3 "$TELEGRAM_MONITOR_SCRIPT" "$JOB_ID" \
    --label "$monitor_label" \
    --poll-seconds "$TELEGRAM_MONITOR_POLL_SECONDS" \
    --credentials "$TELEGRAM_MONITOR_CREDENTIALS" \
    --state-dir "$monitor_state" \
    --pid-file "$monitor_pid" \
    --replace \
    --log-glob "$slurm_log" \
    >"$monitor_log" 2>&1 < /dev/null &

  echo "Telegram monitor started for job ${JOB_ID}"
  echo "  pid file: ${monitor_pid}"
  echo "  log: ${monitor_log}"
  echo "  watched log: ${slurm_log}"
}

start_telegram_monitor

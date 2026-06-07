#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/geval_gepa_engaging_qwen25.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: $CONFIG_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
SLURM_NODE="${SLURM_NODE:-}"
IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gepa-experiments/results/slurm}"
JOB_SLUG="${JOB_SLUG:-geval-gepa-${DATASET:-topical_chat}-${DIMENSION:-${LABEL:-run}}}"
JOB_SLUG="${JOB_SLUG// /_}"
SLURM_TIME="${SLURM_TIME:-}"
SLURM_MEM="${SLURM_MEM:-}"
TELEGRAM_MONITOR="${TELEGRAM_MONITOR:-1}"
TELEGRAM_MONITOR_SCRIPT="${TELEGRAM_MONITOR_SCRIPT:-gepa-experiments/slurm/telegram_monitor.py}"
TELEGRAM_MONITOR_ROOT="${TELEGRAM_MONITOR_ROOT:-gepa-experiments/results/monitor}"
TELEGRAM_MONITOR_POLL_SECONDS="${TELEGRAM_MONITOR_POLL_SECONDS:-60}"
TELEGRAM_MONITOR_CREDENTIALS="${TELEGRAM_MONITOR_CREDENTIALS:-$HOME/.telegram_credentials}"

mkdir -p "$OUTPUT_ROOT"

NODE_ARGS=()
if [[ -n "$SLURM_NODE" ]]; then
  NODE_ARGS=(-w "$SLURM_NODE")
fi

TIME_ARGS=()
if [[ -n "$SLURM_TIME" ]]; then
  TIME_ARGS=(--time="$SLURM_TIME")
fi

MEM_ARGS=()
if [[ -n "$SLURM_MEM" ]]; then
  MEM_ARGS=(--mem="$SLURM_MEM")
fi

SBATCH_OUTPUT="$(sbatch \
  --parsable \
  -N 1 \
  --gpus="$GPU_SPEC" \
  "${NODE_ARGS[@]}" \
  "${TIME_ARGS[@]}" \
  "${MEM_ARGS[@]}" \
  --output="${OUTPUT_ROOT}/slurm-%j-${JOB_SLUG}.out" \
  --export=ALL,IMAGE_NAME="$IMAGE_NAME",CONFIG_FILE="$CONFIG_FILE" \
  gepa-experiments/slurm/run_docker.sh \
  "bash gepa-experiments/slurm/run_gepa_engaging_job.sh")"

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
  local monitor_label="${TELEGRAM_MONITOR_LABEL:-GEPA ${DATASET:-topical_chat}/${DIMENSION:-${LABEL:-run}} ${JOB_ID}}"
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

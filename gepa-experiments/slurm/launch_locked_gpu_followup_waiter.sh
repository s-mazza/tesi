#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mazzacano/tesi}"
CURRENT_QUEUE_PID="${CURRENT_QUEUE_PID:-856633}"
FOLLOWUP_RUN_ID="${FOLLOWUP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_DEVICE="${GPU_DEVICE:-0,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gepa-experiments/results/locked_gpu_followup_${FOLLOWUP_RUN_ID}}"
MONITOR_NAME="${MONITOR_NAME:-locked_gpu_followup_${FOLLOWUP_RUN_ID}}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/monitor"

bash -n gepa-experiments/slurm/run_locked_gpu_followup_queue.sh

nohup env \
  REPO_ROOT="${REPO_ROOT}" \
  CURRENT_QUEUE_PID="${CURRENT_QUEUE_PID}" \
  LOCKED_GPU_RUN_ID="${FOLLOWUP_RUN_ID}" \
  DIRECT_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  GPU_DEVICE="${GPU_DEVICE}" \
  bash -c '
    set -euo pipefail
    cd "${REPO_ROOT}"
    echo $$ > "${DIRECT_OUTPUT_ROOT}/followup_waiter.pid"
    while kill -0 "${CURRENT_QUEUE_PID}" 2>/dev/null; do
      echo "$(date -Is) waiting for current locked queue pid ${CURRENT_QUEUE_PID}"
      sleep 300
    done
    echo "$(date -Is) current queue ended, waiting cleanup grace"
    sleep 120
    bash gepa-experiments/slurm/run_locked_gpu_followup_queue.sh
  ' > "${OUTPUT_ROOT}/followup_waiter.out" 2>&1 &
waiter_pid="$!"

sleep 2
if [[ ! -s "${OUTPUT_ROOT}/followup_waiter.pid" ]]; then
  printf 'follow-up waiter did not create pid file: %s\n' "${OUTPUT_ROOT}/followup_waiter.pid" >&2
  exit 1
fi

nohup python3 gepa-experiments/slurm/telegram_pid_monitor.py \
  --replace \
  --label "${MONITOR_NAME}" \
  --pid-file "${OUTPUT_ROOT}/followup_waiter.pid" \
  --poll-seconds 300 \
  --credentials /home/mazzacano/.telegram_credentials \
  --alert-cooldown-seconds 7200 \
  --max-alerts-per-poll 1 \
  --log-glob "${OUTPUT_ROOT}/logs/*.log" \
  --state-dir "${OUTPUT_ROOT}/monitor/state" \
  --monitor-pid-file "${OUTPUT_ROOT}/monitor/telegram_pid_monitor.pid" \
  > "${OUTPUT_ROOT}/telegram_monitor.out" 2>&1 &
monitor_pid="$!"

printf 'followup_run_id=%s\n' "${FOLLOWUP_RUN_ID}"
printf 'followup_root=%s\n' "${OUTPUT_ROOT}"
printf 'followup_waiter_launcher_pid=%s\n' "${waiter_pid}"
printf 'followup_waiter_pid_file=%s\n' "$(cat "${OUTPUT_ROOT}/followup_waiter.pid")"
printf 'telegram_monitor_pid=%s\n' "${monitor_pid}"
tail -5 "${OUTPUT_ROOT}/followup_waiter.out"

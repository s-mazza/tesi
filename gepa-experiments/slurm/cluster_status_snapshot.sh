#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/tesi}"
SLURM_USER="${SLURM_USER:-mazzacano}"
JOB_IDS="${JOB_IDS:-11912917 11912918 11912947 11912948 11913130 11913131 11913161 11913262 11913284}"
NODE_IDS="${NODE_IDS:-faretra moro232}"
RESULT_ROOT="${RESULT_ROOT:-gepa-experiments/results}"

cd "$PROJECT_DIR"

section() {
  printf '\n===== %s =====\n' "$1"
}

section "snapshot"
date
hostname

section "queue"
squeue -u "$SLURM_USER" -o "%.18i %.9P %.48j %.8T %.12M %.9l %.24R" || true

section "nodes"
for node in $NODE_IDS; do
  echo "---- ${node}"
  scontrol show node="$node" 2>&1 | sed -n '1,45p' || true
done

section "jobs"
for job_id in $JOB_IDS; do
  echo "---- ${job_id}"
  scontrol show job "$job_id" 2>&1 | sed -n '1,85p' || true
done

section "recent slurm logs"
find "${RESULT_ROOT}/slurm" -maxdepth 1 -type f \
  \( -name "slurm-119129*.out" -o -name "slurm-119131*.out" -o -name "slurm-119132*.out" -o -name "slurm-*-slurm-stdout-smoke.out" \) \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort | tail -40 || true

section "current job log tails"
for job_id in $JOB_IDS; do
  while IFS= read -r log_path; do
    [[ -n "$log_path" ]] || continue
    echo "---- ${log_path}"
    ls -l "$log_path" 2>/dev/null || true
    tail -80 "$log_path" 2>/dev/null || true
  done < <(find "${RESULT_ROOT}/slurm" -maxdepth 1 -type f -name "slurm-${job_id}-*.out" 2>/dev/null | sort)
done

section "recent GEPA artifacts"
for dir in \
  "${RESULT_ROOT}/geval_gepa_summeval_consistency_ppl_real_nla_smoke" \
  "${RESULT_ROOT}/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke" \
  "${RESULT_ROOT}/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke" \
  "${RESULT_ROOT}/geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke" \
  "${RESULT_ROOT}/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b" \
  "${RESULT_ROOT}/geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke" \
  "${RESULT_ROOT}/experimental_nla_candidate_content_6_topical_chat_smoke" \
  "${RESULT_ROOT}/experimental_nla_candidate_content_10_topical_chat_smoke" \
  "${RESULT_ROOT}/experimental_nla_hybrid_context_dedup_6_topical_chat_smoke"; do
  echo "---- ${dir}"
  find "$dir" -maxdepth 2 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort | tail -35 || true
done

section "monitor logs"
for job_id in $JOB_IDS; do
  monitor_log="${RESULT_ROOT}/monitor/telegram_monitor_${job_id}.out"
  monitor_pid="${RESULT_ROOT}/monitor/telegram_monitor_${job_id}.pid"
  echo "---- ${job_id}"
  ls -l "$monitor_log" "$monitor_pid" 2>/dev/null || true
  tail -40 "$monitor_log" 2>/dev/null || true
done

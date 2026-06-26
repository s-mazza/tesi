#!/usr/bin/env bash
set -euo pipefail

# Follow-up queue for the two GPU slots reserved on faretra.
# It intentionally starts only after the main locked-GPU queue has finished.

RUN_ID="${LOCKED_GPU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${DIRECT_OUTPUT_ROOT:-gepa-experiments/results/locked_gpu_followup_${RUN_ID}}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
TMP_CONFIG_ROOT="${OUTPUT_ROOT}/configs"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_CONFIG_ROOT}"

GPU_DEVICE="${GPU_DEVICE:-0,2}"
COMMON_CMD="${COMMON_CMD:-bash gepa-experiments/slurm/run_gepa_engaging_job.sh}"
NLA_STRATEGY_CMD="${NLA_STRATEGY_CMD:-bash gepa-experiments/slurm/run_experimental_nla_strategy_job.sh}"

declare -a MANIFEST_LINES=()

log_manifest() {
  local line="$1"
  MANIFEST_LINES+=("${line}")
  printf '%s\n' "${line}" | tee -a "${OUTPUT_ROOT}/manifest.tsv"
}

preflight() {
  local missing=0
  local files=(
    "gepa-experiments/slurm/run_gepa_engaging_job.sh"
    "gepa-experiments/slurm/run_experimental_nla_strategy_job.sh"
    "gepa-experiments/slurm/run_docker.sh"
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_nla_auxjudge_llamacpp35b.env"
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control.env"
    "gepa-experiments/config/geval_gepa_summeval_consistency_ppl_real_nla_smoke.env"
    "gepa-experiments/config/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke.env"
    "gepa-experiments/config/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke.env"
    "gepa-experiments/config/experimental_nla_candidate_content_10_topical_chat_smoke.env"
    "gepa-experiments/config/experimental_nla_hybrid_context_dedup_6_topical_chat_smoke.env"
  )

  for file in "${files[@]}"; do
    if [[ ! -f "${file}" ]]; then
      printf 'missing required file: %s\n' "${file}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    return 1
  fi

  bash -n gepa-experiments/slurm/run_gepa_engaging_job.sh
  bash -n gepa-experiments/slurm/run_experimental_nla_strategy_job.sh
  bash -n gepa-experiments/slurm/run_docker.sh
}

write_config() {
  local label="$1"
  local base_config="$2"
  local server_port="$3"
  local proposer_port="$4"
  shift 4

  local job_output="${OUTPUT_ROOT}/${label}"
  local job_log="${LOG_ROOT}"
  local config="${TMP_CONFIG_ROOT}/${label}.env"

  {
    printf 'source %q\n' "${base_config}"
    printf 'OUTPUT_DIR=%q\n' "${job_output}"
    printf 'LOG_DIR=%q\n' "${job_log}"
    printf 'SERVER_PORT=%q\n' "${server_port}"
    printf 'PROPOSER_PORT=%q\n' "${proposer_port}"
    printf 'GPU_DEVICE=%q\n' "${GPU_DEVICE}"
    printf 'JOB_SLUG=%q\n' "${label}"
    printf 'KEEP_FAILED_CONTAINERS=0\n'
    printf 'LLAMACPP_BATCH_SIZE=64\n'
    printf 'LLAMACPP_FLASH_ATTN=off\n'
    local override
    for override in "$@"; do
      printf '%s\n' "${override}"
    done
  } > "${config}"

  printf '%s\n' "${config}"
}

run_job() {
  local label="$1"
  local command="$2"
  local base_config="$3"
  local server_port="$4"
  local proposer_port="$5"
  local critical="$6"
  shift 6

  local config
  config="$(write_config "${label}" "${base_config}" "${server_port}" "${proposer_port}" "$@")"
  local log_file="${LOG_ROOT}/${label}.log"
  local start_epoch
  start_epoch="$(date +%s)"

  log_manifest "START	${label}	$(date -Is)	config=${config}	log=${log_file}	critical=${critical}"
  set +e
  GEPA_CONFIG="${config}" ${command} 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  local end_epoch
  end_epoch="$(date +%s)"
  log_manifest "END	${label}	$(date -Is)	status=${status}	elapsed_sec=$((end_epoch - start_epoch))"

  if [[ "${status}" -ne 0 && "${critical}" == "critical" ]]; then
    printf 'critical job failed: %s\n' "${label}" >&2
    return "${status}"
  fi

  return 0
}

preflight
log_manifest "QUEUE_START	locked_gpu_followup	$(date -Is)	gpus=${GPU_DEVICE}	output=${OUTPUT_ROOT}"

# Priority 1: clean rerun of the most relevant long branch after llama.cpp
# sidecar hardening. This removes the previous D4 caveat caused by sidecar
# crashes near the end of optimization.
run_job \
  "F1_clean_aux_long_ppl_nla_seed42" \
  "${COMMON_CMD}" \
  "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_nla_auxjudge_llamacpp35b.env" \
  "19300" "19301" "noncritical"

# Priority 2: same branch with a different seed. This gives a robustness point
# for the thesis without changing the core setting.
run_job \
  "F2_clean_aux_long_ppl_nla_seed43" \
  "${COMMON_CMD}" \
  "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_nla_auxjudge_llamacpp35b.env" \
  "19310" "19311" "noncritical" \
  "SEED=43"

# Priority 3: dataset-coverage smoke tests for the G-Eval matrix extension.
run_job \
  "F3_summeval_consistency_real_nla_smoke" \
  "${COMMON_CMD}" \
  "gepa-experiments/config/geval_gepa_summeval_consistency_ppl_real_nla_smoke.env" \
  "19320" "19321" "noncritical"

run_job \
  "F4_qags_cnn_consistency_real_nla_smoke" \
  "${COMMON_CMD}" \
  "gepa-experiments/config/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke.env" \
  "19330" "19331" "noncritical"

run_job \
  "F5_qags_xsum_consistency_real_nla_smoke" \
  "${COMMON_CMD}" \
  "gepa-experiments/config/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke.env" \
  "19340" "19341" "noncritical"

# Priority 4: two NLA token-selection probes that can help explain whether the
# current verbalization placement is the weak point.
run_job \
  "F6_candidate_content_10_strategy_probe" \
  "${NLA_STRATEGY_CMD}" \
  "gepa-experiments/config/experimental_nla_candidate_content_10_topical_chat_smoke.env" \
  "19350" "19351" "noncritical"

run_job \
  "F7_hybrid_context_dedup_6_strategy_probe" \
  "${NLA_STRATEGY_CMD}" \
  "gepa-experiments/config/experimental_nla_hybrid_context_dedup_6_topical_chat_smoke.env" \
  "19360" "19361" "noncritical"

# Priority 5: lower-priority clean control. It is useful if time remains, but
# the thesis already has a stronger historical PPL-only baseline.
run_job \
  "F8_clean_ppl_only_control_long" \
  "${COMMON_CMD}" \
  "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control.env" \
  "19370" "19371" "noncritical"

log_manifest "QUEUE_END	locked_gpu_followup	$(date -Is)	gpus=${GPU_DEVICE}	output=${OUTPUT_ROOT}"

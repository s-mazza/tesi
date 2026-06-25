#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

LOCKED_GPU_DEVICES="${LOCKED_GPU_DEVICES:-0,2}"
JUDGE_GPU_DEVICE="${JUDGE_GPU_DEVICE:-0}"
PROPOSER_GPU_DEVICE="${PROPOSER_GPU_DEVICE:-2}"
IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-llama.cpp:localcuda}"
LLM_CACHE_DIR="${LLM_CACHE_DIR:-/llms}"
RUN_ID="${LOCKED_GPU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DIRECT_OUTPUT_ROOT="${DIRECT_OUTPUT_ROOT:-gepa-experiments/results/locked_gpu_${RUN_ID}}"
CONFIG_ROOT="${DIRECT_OUTPUT_ROOT}/configs"
LOG_ROOT="${DIRECT_OUTPUT_ROOT}/logs"
MANIFEST="${DIRECT_OUTPUT_ROOT}/manifest.tsv"
CONTINUE_AFTER_NONCRITICAL_FAILURE="${CONTINUE_AFTER_NONCRITICAL_FAILURE:-1}"

mkdir -p "$CONFIG_ROOT" "$LOG_ROOT"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 2
  fi
}

preflight() {
  log "Locked GPU queue preflight"
  log "project_dir=${PROJECT_DIR}"
  log "locked_gpus=${LOCKED_GPU_DEVICES} judge_gpu=${JUDGE_GPU_DEVICE} proposer_gpu=${PROPOSER_GPU_DEVICE}"
  log "output_root=${DIRECT_OUTPUT_ROOT}"

  command -v docker >/dev/null 2>&1 || {
    echo "docker is not available on this node." >&2
    exit 2
  }
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is not available on this node." >&2
    exit 2
  }

  docker image inspect "$IMAGE_NAME" >/dev/null
  docker image inspect "$LLAMACPP_IMAGE" >/dev/null

  require_dir "${LLM_CACHE_DIR}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
  require_file "${LLM_CACHE_DIR}/llamacpp-cache/opensota_Qwen3.6-35B-A3B-GGUF_Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

  IFS=',' read -r -a gpu_ids <<< "$LOCKED_GPU_DEVICES"
  for gpu_id in "${gpu_ids[@]}"; do
    log "GPU ${gpu_id} status"
    nvidia-smi --id="$gpu_id" --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
  done

  log "Docker GPU exposure check"
  docker run --rm --gpus "device=${JUDGE_GPU_DEVICE}" "$IMAGE_NAME" \
    python -c 'import torch; print("judge_cuda_available", torch.cuda.is_available()); print("judge_device_count", torch.cuda.device_count())'
  docker run --rm --gpus "device=${PROPOSER_GPU_DEVICE}" "$IMAGE_NAME" \
    python -c 'import torch; print("proposer_cuda_available", torch.cuda.is_available()); print("proposer_device_count", torch.cuda.device_count())'
}

write_config() {
  local label="$1"
  local base_config="$2"
  local server_port="$3"
  local proposer_port="$4"
  local output_dir="$5"
  local config_path="${CONFIG_ROOT}/${label}.env"

  require_file "$base_config"
  cat >"$config_path" <<EOF
source ${base_config}
OUTPUT_DIR=${output_dir}
LOG_DIR=${output_dir}/logs
SERVER_PORT=${server_port}
PROPOSER_PORT=${proposer_port}
JOB_SLUG=${label}
KEEP_MAIN_CONTAINER_ON_FAIL=1
EOF

  printf '%s\n' "$config_path"
}

execute_config_job() {
  local label="$1"
  local config_path="$2"
  local command="$3"
  local critical="$4"
  local output_dir="${DIRECT_OUTPUT_ROOT}/${label}"
  local job_instance_id
  job_instance_id="$(date -u +%Y%m%d%H%M%S)$((RANDOM % 9000 + 1000))"
  local log_path="${LOG_ROOT}/${label}_${job_instance_id}.log"
  local start_epoch
  start_epoch="$(date +%s)"

  log "START label=${label} config=${config_path} output=${output_dir}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "START" "$label" "$config_path" "$output_dir" "$log_path" >>"$MANIFEST"

  set +e
  CUDA_VISIBLE_DEVICES="$LOCKED_GPU_DEVICES" \
    JUDGE_GPU_DEVICE="$JUDGE_GPU_DEVICE" \
    PROPOSER_GPU_DEVICE="$PROPOSER_GPU_DEVICE" \
    IMAGE_NAME="$IMAGE_NAME" \
    LLAMACPP_IMAGE="$LLAMACPP_IMAGE" \
    PROJECT_DIR="$PROJECT_DIR" \
    LLM_CACHE_DIR="$LLM_CACHE_DIR" \
    CONFIG_FILE="$config_path" \
    SLURM_JOB_ID="$job_instance_id" \
    CONTAINER_NAME="geval_gepa_${label}_${job_instance_id}" \
    SIDECAR_NAME="llamacpp_${label}_${job_instance_id}" \
    bash gepa-experiments/slurm/run_docker.sh "$command" 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e

  local end_epoch
  end_epoch="$(date +%s)"
  local elapsed=$((end_epoch - start_epoch))
  log "END label=${label} status=${status} elapsed_seconds=${elapsed}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "END:${status}:${elapsed}" "$label" "$config_path" "$output_dir" "$log_path" >>"$MANIFEST"

  if [[ "$status" -ne 0 ]]; then
    if [[ "$critical" == "critical" ]]; then
      echo "Critical job failed: ${label}. Stopping locked-GPU queue." >&2
      exit "$status"
    fi
    if [[ "$CONTINUE_AFTER_NONCRITICAL_FAILURE" != "1" ]]; then
      echo "Non-critical job failed and continuation is disabled: ${label}." >&2
      exit "$status"
    fi
  fi
}

run_gepa_job() {
  local label="$1"
  local base_config="$2"
  local command="$3"
  local server_port="$4"
  local proposer_port="$5"
  local critical="$6"
  local output_dir="${DIRECT_OUTPUT_ROOT}/${label}"
  local config_path
  config_path="$(write_config "$label" "$base_config" "$server_port" "$proposer_port" "$output_dir")"

  execute_config_job "$label" "$config_path" "$command" "$critical"
}

run_strategy_job() {
  local strategy="$1"
  local index="$2"
  local server_port=$((19200 + index * 2))
  local proposer_port=$((server_port + 1))
  local label="B_sweep_${strategy}"
  local output_dir="${DIRECT_OUTPUT_ROOT}/${label}"
  local base_config="gepa-experiments/config/experimental_nla_position_sweep_topical_chat_smoke.env"
  local config_path="${CONFIG_ROOT}/${label}.env"

  require_file "$base_config"
  cat >"$config_path" <<EOF
source ${base_config}
EXPERIMENTAL_NLA_TOKEN_STRATEGY=${strategy}
OUTPUT_DIR=${output_dir}
LOG_DIR=${output_dir}/logs
SERVER_PORT=${server_port}
PROPOSER_PORT=${proposer_port}
JOB_SLUG=${label}
KEEP_MAIN_CONTAINER_ON_FAIL=1
EOF

  execute_config_job \
    "$label" \
    "$config_path" \
    "bash gepa-experiments/slurm/run_experimental_nla_strategy_job.sh" \
    "noncritical"
}

main() {
  preflight

  if [[ "${LOCKED_GPU_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    log "Preflight-only mode completed."
    return 0
  fi

  {
    printf 'timestamp\tevent\tlabel\tconfig\toutput_dir\tlog\n'
  } >"$MANIFEST"

  run_gepa_job \
    "D1_aux_judge_fixed_smoke_ppl_nla" \
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke.env" \
    "bash gepa-experiments/slurm/run_gepa_engaging_job.sh" \
    19010 \
    19011 \
    "critical"

  run_gepa_job \
    "D4_aux_judge_fixed_long_ppl_nla" \
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_nla_auxjudge_llamacpp35b.env" \
    "bash gepa-experiments/slurm/run_gepa_engaging_job.sh" \
    19020 \
    19021 \
    "noncritical"

  run_gepa_job \
    "D2_matched_no_aux_smoke_ppl_nla" \
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke.env" \
    "bash gepa-experiments/slurm/run_gepa_engaging_job.sh" \
    19030 \
    19031 \
    "noncritical"

  run_gepa_job \
    "D3_aux_judge_only_smoke_ppl_aux_no_nla" \
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_ppl_auxjudge_llamacpp35b_smoke.env" \
    "bash gepa-experiments/slurm/run_gepa_engaging_job.sh" \
    19040 \
    19041 \
    "noncritical"

  run_gepa_job \
    "A1_nla_strategy_wiring_probe" \
    "gepa-experiments/config/experimental_nla_candidate_content_6_topical_chat_smoke.env" \
    "bash gepa-experiments/slurm/run_experimental_nla_strategy_job.sh" \
    19050 \
    19051 \
    "noncritical"

  local strategies=(
    candidate_first_1
    candidate_middle_1
    candidate_last_1
    candidate_fml_3
    candidate_quintile_5
    candidate_even_8
    source_fml_3
    reference_fml_3
    balanced_fml_9
    prompt_tail_6
    evaluation_tail_3
    hybrid_context_dedup_8
  )

  local index=0
  for strategy in "${strategies[@]}"; do
    run_strategy_job "$strategy" "$index"
    index=$((index + 1))
  done

  run_gepa_job \
    "D5_matched_no_aux_long_ppl_nla" \
    "gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b.env" \
    "bash gepa-experiments/slurm/run_gepa_engaging_job.sh" \
    19080 \
    19081 \
    "noncritical"

  log "Locked GPU queue completed. Manifest: ${MANIFEST}"
}

main "$@"

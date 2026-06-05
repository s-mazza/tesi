#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
LLM_CACHE_DIR="${LLM_CACHE_DIR:-/llms}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-geval_gepa_${SLURM_JOB_ID:-$$}}"
SIDECAR_NAME="${SIDECAR_NAME:-llamacpp_proposer_${SLURM_JOB_ID:-$$}}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: sbatch -N 1 --gpus=<gpu>:1 gepa-experiments/slurm/run_docker.sh '<command>'" >&2
  exit 2
fi

if [[ -n "${CONFIG_FILE:-}" && -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

PROPOSER_BACKEND="${PROPOSER_BACKEND:-}"
MAIN_VISIBLE_DEVICES="$VISIBLE_DEVICES"
DOCKER_NETWORK_ARGS=()

cleanup() {
  docker stop --time 10 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if [[ -n "${SIDECAR_LOG_PID:-}" ]]; then
    kill "$SIDECAR_LOG_PID" >/dev/null 2>&1 || true
  fi
  docker stop --time 10 "$SIDECAR_NAME" >/dev/null 2>&1 || true
  docker kill "$SIDECAR_NAME" >/dev/null 2>&1 || true
}
trap 'cleanup; exit 143' INT TERM
trap cleanup EXIT

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-180}"
  local sleep_seconds="${4:-5}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS -H "Authorization: Bearer ${PROPOSER_API_KEY:-${LLAMA_API_KEY:-}}" "$url" >/dev/null 2>&1; then
      echo "${label} is ready: ${url}"
      return 0
    fi
    sleep "$sleep_seconds"
  done
  echo "Timed out waiting for ${label}: ${url}" >&2
  return 1
}

wait_for_sidecar_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-180}"
  local sleep_seconds="${4:-5}"
  local log_path="$5"
  for _ in $(seq 1 "$attempts"); do
    if ! docker inspect -f '{{.State.Running}}' "$SIDECAR_NAME" >/dev/null 2>&1; then
      echo "${label} container exited before readiness. Last log lines:" >&2
      tail -100 "$log_path" >&2 || true
      return 1
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "${label} is ready: ${url}"
      return 0
    fi
    sleep "$sleep_seconds"
  done
  echo "Timed out waiting for ${label}: ${url}" >&2
  return 1
}

start_llamacpp_sidecar() {
  IFS=',' read -r -a allocated_gpus <<< "$VISIBLE_DEVICES"
  if [[ "${#allocated_gpus[@]}" -lt 2 ]]; then
    echo "PROPOSER_BACKEND=llamacpp requires two allocated GPUs; CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}" >&2
    exit 2
  fi

  JUDGE_GPU_DEVICE="${JUDGE_GPU_DEVICE:-${allocated_gpus[0]}}"
  PROPOSER_GPU_DEVICE="${PROPOSER_GPU_DEVICE:-${allocated_gpus[1]}}"
  MAIN_VISIBLE_DEVICES="$JUDGE_GPU_DEVICE"
  DOCKER_NETWORK_ARGS=(--network=host)

  LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-llama.cpp:localcuda}"
  LLAMACPP_MODEL_DIR="${LLAMACPP_MODEL_DIR:-$LLM_CACHE_DIR}"
  LLAMACPP_MODEL_NAME="${LLAMACPP_MODEL_NAME:-}"
  LLAMACPP_HF_MODEL="${LLAMACPP_HF_MODEL:-opensota/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M}"
  LLAMACPP_HF_CACHE_DIR="${LLAMACPP_HF_CACHE_DIR:-${LLM_CACHE_DIR}/llamacpp-cache}"
  LLAMACPP_HOST="${LLAMACPP_HOST:-127.0.0.1}"
  PROPOSER_PORT="${PROPOSER_PORT:-8080}"
  LLAMA_API_KEY="${LLAMA_API_KEY:-local-llamacpp-key}"
  LLAMACPP_CTX_SIZE="${LLAMACPP_CTX_SIZE:-8192}"
  LLAMACPP_PARALLEL="${LLAMACPP_PARALLEL:-1}"
  LLAMACPP_BATCH_SIZE="${LLAMACPP_BATCH_SIZE:-512}"
  LLAMACPP_N_GPU_LAYERS="${LLAMACPP_N_GPU_LAYERS:-999}"
  LLAMACPP_FLASH_ATTN="${LLAMACPP_FLASH_ATTN:-on}"
  LLAMACPP_READY_ATTEMPTS="${LLAMACPP_READY_ATTEMPTS:-2160}"
  LLAMACPP_READY_SLEEP_SECONDS="${LLAMACPP_READY_SLEEP_SECONDS:-5}"
  PROPOSER_MODEL="${PROPOSER_MODEL:-local-llamacpp}"
  PROPOSER_API_BASE="${PROPOSER_API_BASE:-http://${LLAMACPP_HOST}:${PROPOSER_PORT}/v1}"
  PROPOSER_API_KEY="${PROPOSER_API_KEY:-$LLAMA_API_KEY}"

  local sidecar_log_dir="${LOG_DIR:-${OUTPUT_DIR:-gepa-experiments/results/geval_gepa_engaging_qwen25}/logs}"
  mkdir -p "$sidecar_log_dir" "$LLAMACPP_HF_CACHE_DIR"
  local sidecar_log="${sidecar_log_dir}/llamacpp_${SLURM_JOB_ID:-local}.log"

  local model_args=()
  if [[ -n "$LLAMACPP_MODEL_NAME" ]]; then
    model_args=(-m "/models/${LLAMACPP_MODEL_NAME}")
    echo "Starting llama.cpp proposer from local GGUF: ${LLAMACPP_MODEL_DIR}/${LLAMACPP_MODEL_NAME}"
  else
    model_args=(-hf "$LLAMACPP_HF_MODEL")
    echo "Starting llama.cpp proposer from Hugging Face GGUF: ${LLAMACPP_HF_MODEL}"
  fi
  echo "  sidecar GPU: ${PROPOSER_GPU_DEVICE}"
  echo "  judge GPU: ${JUDGE_GPU_DEVICE}"
  echo "  sidecar log: ${sidecar_log}"

  docker run \
    --rm \
    --detach \
    --name "$SIDECAR_NAME" \
    --init \
    --network=host \
    --ipc=host \
    --gpus "device=${PROPOSER_GPU_DEVICE}" \
    -v "${LLAMACPP_MODEL_DIR}:/models:ro" \
    -v "${LLAMACPP_HF_CACHE_DIR}:/root/.cache/llama.cpp" \
    "$LLAMACPP_IMAGE" \
    "${model_args[@]}" \
    --api-key "$LLAMA_API_KEY" \
    --host "$LLAMACPP_HOST" \
    --port "$PROPOSER_PORT" \
    --n-gpu-layers "$LLAMACPP_N_GPU_LAYERS" \
    --ctx-size "$LLAMACPP_CTX_SIZE" \
    --parallel "$LLAMACPP_PARALLEL" \
    --flash-attn "$LLAMACPP_FLASH_ATTN" \
    --batch-size "$LLAMACPP_BATCH_SIZE" \
    >/dev/null

  docker logs -f "$SIDECAR_NAME" >"$sidecar_log" 2>&1 &
  SIDECAR_LOG_PID="$!"

  wait_for_sidecar_http "${PROPOSER_API_BASE%/}/models" \
    "llama.cpp proposer" \
    "$LLAMACPP_READY_ATTEMPTS" \
    "$LLAMACPP_READY_SLEEP_SECONDS" \
    "$sidecar_log" || {
    echo "Last llama.cpp log lines:" >&2
    tail -100 "$sidecar_log" >&2 || true
    exit 1
  }
}

if [[ "$PROPOSER_BACKEND" == "llamacpp" ]]; then
  start_llamacpp_sidecar
fi

docker run \
  --rm \
  --name "$CONTAINER_NAME" \
  --init \
  --ipc=host \
  "${DOCKER_NETWORK_ARGS[@]}" \
  --gpus "device=${MAIN_VISIBLE_DEVICES}" \
  -v "$PROJECT_DIR:/workspace" \
  -v "$LLM_CACHE_DIR:/llms" \
  -e "HF_HOME=/llms" \
  -e "HF_HUB_OFFLINE=1" \
  -e "TRANSFORMERS_OFFLINE=1" \
  -e "HF_HUB_DISABLE_TELEMETRY=1" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}" \
  -e "LLAMA_API_KEY=${LLAMA_API_KEY:-local-llamacpp-key}" \
  -e "PROPOSER_MODEL=${PROPOSER_MODEL:-}" \
  -e "PROPOSER_API_BASE=${PROPOSER_API_BASE:-}" \
  -e "PROPOSER_API_KEY=${PROPOSER_API_KEY:-}" \
  -e "PROPOSER_TEMPERATURE=${PROPOSER_TEMPERATURE:-}" \
  -e "PROPOSER_MAX_TOKENS=${PROPOSER_MAX_TOKENS:-}" \
  -e "PYTHONPATH=/workspace/gepa-experiments:${PYTHONPATH:-}" \
  -e "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
  -e "CONFIG_FILE=${CONFIG_FILE:-}" \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -lc "$*" &

docker_pid=$!
wait "$docker_pid"

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

gpu_memory_mib() {
  local device="$1"
  nvidia-smi \
    --id="$device" \
    --query-gpu=memory.free,memory.total \
    --format=csv,noheader,nounits \
    2>/dev/null \
    | head -1 \
    | tr -d ' '
}

gpu_free_memory_mib() {
  local device="$1"
  local values
  values="$(gpu_memory_mib "$device" || true)"
  if [[ -z "$values" ]]; then
    echo "-1"
    return 0
  fi
  echo "${values%,*}"
}

select_highest_free_gpu_except() {
  local excluded_csv="$1"
  shift
  local best_device=""
  local best_free=-1
  local device
  for device in "$@"; do
    if [[ ",${excluded_csv}," == *",${device},"* ]]; then
      continue
    fi
    local free_mib
    free_mib="$(gpu_free_memory_mib "$device")"
    if [[ "$free_mib" -gt "$best_free" ]]; then
      best_free="$free_mib"
      best_device="$device"
    fi
  done
  echo "$best_device"
}

default_judge_min_free_memory_mib() {
  local device="$1"
  local values
  values="$(gpu_memory_mib "$device" || true)"
  if [[ -z "$values" ]]; then
    echo ""
    return 0
  fi
  local total="${values#*,}"
  python3 - "$total" "${GPU_MEMORY_UTILIZATION:-0.90}" <<'PY'
import sys

total = int(float(sys.argv[1]))
util = float(sys.argv[2])
print(int(total * util))
PY
}

wait_for_gpu_free_memory() {
  local label="$1"
  local device="$2"
  local min_free_mib="$3"
  local wait_seconds="${GPU_MEMORY_WAIT_SECONDS:-0}"
  local poll_seconds="${GPU_MEMORY_POLL_SECONDS:-30}"

  if [[ -z "$min_free_mib" || "$min_free_mib" == "0" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi unavailable; cannot verify ${label} GPU memory before startup." >&2
    return 0
  fi

  local deadline=$((SECONDS + wait_seconds))
  while true; do
    local values
    values="$(gpu_memory_mib "$device" || true)"
    if [[ -z "$values" ]]; then
      echo "Could not read GPU memory for ${label} device ${device}." >&2
      return 1
    fi
    local free_mib="${values%,*}"
    local total_mib="${values#*,}"
    if [[ "$free_mib" -ge "$min_free_mib" ]]; then
      echo "${label} GPU memory check passed: device=${device} free=${free_mib}MiB total=${total_mib}MiB min=${min_free_mib}MiB."
      return 0
    fi

    echo "${label} GPU memory check waiting: device=${device} free=${free_mib}MiB total=${total_mib}MiB min=${min_free_mib}MiB."
    if [[ "$wait_seconds" -le 0 || "$SECONDS" -ge "$deadline" ]]; then
      echo "${label} GPU device ${device} does not have enough free memory for startup." >&2
      nvidia-smi >&2 || true
      return 1
    fi
    sleep "$poll_seconds"
  done
}

find_free_tcp_port() {
  local host="$1"
  local start_port="$2"
  local attempts="${3:-100}"
  python3 - "$host" "$start_port" "$attempts" <<'PY'
import socket
import sys

host = sys.argv[1]
start = int(sys.argv[2])
attempts = int(sys.argv[3])

for port in range(start, start + attempts):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    sys.exit(0)

sys.exit(1)
PY
}

start_llamacpp_sidecar() {
  IFS=',' read -r -a allocated_gpus <<< "$VISIBLE_DEVICES"
  if [[ "${#allocated_gpus[@]}" -lt 2 ]]; then
    echo "PROPOSER_BACKEND=llamacpp requires two allocated GPUs; CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}" >&2
    exit 2
  fi

  if [[ -z "${PROPOSER_GPU_DEVICE:-}" ]]; then
    PROPOSER_GPU_DEVICE="$(select_highest_free_gpu_except "${JUDGE_GPU_DEVICE:-}" "${allocated_gpus[@]}")"
  fi
  if [[ -z "${JUDGE_GPU_DEVICE:-}" ]]; then
    JUDGE_GPU_DEVICE="$(select_highest_free_gpu_except "$PROPOSER_GPU_DEVICE" "${allocated_gpus[@]}")"
  fi
  if [[ -z "$JUDGE_GPU_DEVICE" || -z "$PROPOSER_GPU_DEVICE" ]]; then
    echo "Could not select distinct judge/proposer GPUs from CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}" >&2
    exit 2
  fi
  if [[ "$JUDGE_GPU_DEVICE" == "$PROPOSER_GPU_DEVICE" ]]; then
    echo "JUDGE_GPU_DEVICE and PROPOSER_GPU_DEVICE must be distinct; both are ${JUDGE_GPU_DEVICE}." >&2
    exit 2
  fi
  MAIN_VISIBLE_DEVICES="$JUDGE_GPU_DEVICE"
  DOCKER_NETWORK_ARGS=(--network=host)

  LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-llama.cpp:localcuda}"
  LLAMACPP_MODEL_DIR="${LLAMACPP_MODEL_DIR:-$LLM_CACHE_DIR}"
  LLAMACPP_MODEL_NAME="${LLAMACPP_MODEL_NAME:-}"
  LLAMACPP_HF_MODEL="${LLAMACPP_HF_MODEL:-opensota/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M}"
  LLAMACPP_HF_CACHE_DIR="${LLAMACPP_HF_CACHE_DIR:-${LLM_CACHE_DIR}/llamacpp-cache}"
  LLAMACPP_HOST="${LLAMACPP_HOST:-127.0.0.1}"
  PROPOSER_PORT="${PROPOSER_PORT:-8080}"
  PROPOSER_PORT_SEARCH_ATTEMPTS="${PROPOSER_PORT_SEARCH_ATTEMPTS:-100}"
  LLAMA_API_KEY="${LLAMA_API_KEY:-local-llamacpp-key}"
  LLAMACPP_CTX_SIZE="${LLAMACPP_CTX_SIZE:-8192}"
  LLAMACPP_PARALLEL="${LLAMACPP_PARALLEL:-1}"
  LLAMACPP_BATCH_SIZE="${LLAMACPP_BATCH_SIZE:-128}"
  LLAMACPP_N_GPU_LAYERS="${LLAMACPP_N_GPU_LAYERS:-999}"
  LLAMACPP_FLASH_ATTN="${LLAMACPP_FLASH_ATTN:-on}"
  LLAMACPP_READY_ATTEMPTS="${LLAMACPP_READY_ATTEMPTS:-2160}"
  LLAMACPP_READY_SLEEP_SECONDS="${LLAMACPP_READY_SLEEP_SECONDS:-5}"
  PROPOSER_MODEL="${PROPOSER_MODEL:-local-llamacpp}"
  JUDGE_MIN_FREE_MEMORY_MIB="${JUDGE_MIN_FREE_MEMORY_MIB:-0}"
  PROPOSER_MIN_FREE_MEMORY_MIB="${PROPOSER_MIN_FREE_MEMORY_MIB:-0}"
  local proposer_api_base_explicit=0
  if [[ -n "${PROPOSER_API_BASE:-}" ]]; then
    proposer_api_base_explicit=1
  fi

  local configured_proposer_port="$PROPOSER_PORT"
  local free_proposer_port
  if ! free_proposer_port="$(find_free_tcp_port "$LLAMACPP_HOST" "$PROPOSER_PORT" "$PROPOSER_PORT_SEARCH_ATTEMPTS")"; then
    echo "No free llama.cpp proposer port found on ${LLAMACPP_HOST} starting at ${PROPOSER_PORT} for ${PROPOSER_PORT_SEARCH_ATTEMPTS} attempts." >&2
    exit 2
  fi
  if [[ "$free_proposer_port" != "$configured_proposer_port" ]]; then
    if [[ "$proposer_api_base_explicit" == "1" ]]; then
      echo "Configured PROPOSER_PORT=${configured_proposer_port} is busy, but PROPOSER_API_BASE was explicitly set to ${PROPOSER_API_BASE}." >&2
      echo "Choose a free PROPOSER_PORT/PROPOSER_API_BASE pair or unset PROPOSER_API_BASE so the runner can select one automatically." >&2
      exit 2
    fi
    echo "Configured llama.cpp proposer port ${configured_proposer_port} is busy; using free port ${free_proposer_port}."
    PROPOSER_PORT="$free_proposer_port"
  fi

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
  echo "  sidecar endpoint: ${PROPOSER_API_BASE}"
  echo "  sidecar log: ${sidecar_log}"

  wait_for_gpu_free_memory "judge/vLLM" "$JUDGE_GPU_DEVICE" "$JUDGE_MIN_FREE_MEMORY_MIB"
  wait_for_gpu_free_memory "llama.cpp proposer" "$PROPOSER_GPU_DEVICE" "$PROPOSER_MIN_FREE_MEMORY_MIB"

  docker run \
    --rm \
    --detach \
    --name "$SIDECAR_NAME" \
    --init \
    --ipc=host \
    --gpus "device=${PROPOSER_GPU_DEVICE}" \
    -p "${LLAMACPP_HOST}:${PROPOSER_PORT}:8080" \
    -v "${LLAMACPP_MODEL_DIR}:/models:ro" \
    -v "${LLAMACPP_HF_CACHE_DIR}:/root/.cache/llama.cpp" \
    "$LLAMACPP_IMAGE" \
    "${model_args[@]}" \
    --api-key "$LLAMA_API_KEY" \
    --host 0.0.0.0 \
    --port 8080 \
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
  -e "SERVER_PORT=${SERVER_PORT:-}" \
  -e "OUTPUT_DIR=${OUTPUT_DIR:-}" \
  -e "LOG_DIR=${LOG_DIR:-}" \
  -e "EXPERIMENTAL_NLA_TOKEN_STRATEGY=${EXPERIMENTAL_NLA_TOKEN_STRATEGY:-}" \
  -e "NLA_MANIFEST_PATH=${NLA_MANIFEST_PATH:-}" \
  -e "NLA_PRECOMPUTED_PATH=${NLA_PRECOMPUTED_PATH:-}" \
  -e "AUX_JUDGE_MAX_TOKENS=${AUX_JUDGE_MAX_TOKENS:-}" \
  -e "PYTHONUNBUFFERED=1" \
  -e "PYTHONFAULTHANDLER=1" \
  -e "PYTHONPATH=/workspace/gepa-experiments:${PYTHONPATH:-}" \
  -e "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
  -e "CONFIG_FILE=${CONFIG_FILE:-}" \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -lc "$*" &

docker_pid=$!
set +e
wait "$docker_pid"
docker_status=$?
set -e

if [[ "$docker_status" -ne 0 ]]; then
  echo "Main GEPA container exited with status ${docker_status}" >&2
  echo "Last main container log lines:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2 || true
fi

if [[ "${KEEP_MAIN_CONTAINER_ON_FAIL:-0}" != "1" || "$docker_status" -eq 0 ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

exit "$docker_status"

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

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_TOKENS="${MAX_TOKENS:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-gepa-experiments/results/geval_gepa_engaging_qwen25}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

VLLM_LOG="${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
MODEL_CACHE_DIR="/llms/hub/models--${JUDGE_MODEL//\//--}"
VLLM_MODEL_ARG="${JUDGE_MODEL}"

if [[ -d "${MODEL_CACHE_DIR}/snapshots" ]]; then
  MODEL_SNAPSHOT="$(find "${MODEL_CACHE_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -n "$MODEL_SNAPSHOT" ]]; then
    VLLM_MODEL_ARG="$MODEL_SNAPSHOT"
  fi
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

write_dependency_manifest() {
  local manifest_dir="${LOG_DIR}/dependency_manifest_${SLURM_JOB_ID:-local}"
  mkdir -p "$manifest_dir"

  python --version >"${manifest_dir}/python_version.txt" 2>&1
  python -m pip freeze --all >"${manifest_dir}/pip_freeze.txt" 2>&1
  set +e
  python -m pip check >"${manifest_dir}/pip_check.txt" 2>&1
  echo "$?" >"${manifest_dir}/pip_check.exit"
  set -e

  if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W >"${manifest_dir}/apt_packages.txt" 2>&1
  fi
  if command -v gcc >/dev/null 2>&1; then
    gcc --version >"${manifest_dir}/gcc_version.txt" 2>&1
  fi
  if command -v g++ >/dev/null 2>&1; then
    g++ --version >"${manifest_dir}/gxx_version.txt" 2>&1
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"${manifest_dir}/nvidia_smi.txt" 2>&1 || true
  fi

  echo "Dependency manifest: ${manifest_dir}"
}

echo "Starting vLLM judge server"
echo "  config: ${CONFIG_FILE}"
echo "  model: ${JUDGE_MODEL}"
echo "  vLLM model path: ${VLLM_MODEL_ARG}"
echo "  NLA AV checkpoint reserved for next phase: ${NLA_AV_CHECKPOINT}"
echo "  health: ${HEALTH_URL}"
echo "  log: ${VLLM_LOG}"
echo "  max tokens: ${MAX_TOKENS}"

write_dependency_manifest

python -m geval_gepa.preflight \
  --data-source "$DATA_SOURCE" \
  --judge-model "$JUDGE_MODEL" \
  --nla-av-checkpoint "$NLA_AV_CHECKPOINT" \
  --train-contexts "$TRAIN_CONTEXTS" \
  --val-contexts "$VAL_CONTEXTS" \
  --test-contexts "$TEST_CONTEXTS" \
  --seed "$SEED"

vllm serve "$VLLM_MODEL_ARG" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --served-model-name "$JUDGE_MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype auto \
  --download-dir /llms \
  --trust-remote-code \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID="$!"

echo "Waiting for vLLM readiness..."
for _ in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before readiness. Last log lines:" >&2
    tail -100 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "vLLM is ready."
    break
  fi
  sleep 5
done

if ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  echo "Timed out waiting for vLLM. Last log lines:" >&2
  tail -100 "$VLLM_LOG" >&2 || true
  exit 1
fi

BUDGET_ARGS=()
if [[ -n "${GEPA_AUTO:-}" ]]; then
  BUDGET_ARGS=(--gepa-auto "$GEPA_AUTO")
elif [[ -n "${MAX_FULL_EVALS:-}" ]]; then
  BUDGET_ARGS=(--max-full-evals "$MAX_FULL_EVALS")
elif [[ -n "${MAX_METRIC_CALLS:-}" ]]; then
  BUDGET_ARGS=(--max-metric-calls "$MAX_METRIC_CALLS")
else
  echo "Set one of GEPA_AUTO, MAX_FULL_EVALS, or MAX_METRIC_CALLS." >&2
  exit 2
fi

python -m geval_gepa.runner \
  --data-source "$DATA_SOURCE" \
  --label "$LABEL" \
  --train-contexts "$TRAIN_CONTEXTS" \
  --val-contexts "$VAL_CONTEXTS" \
  --test-contexts "$TEST_CONTEXTS" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  --judge-model "$JUDGE_MODEL" \
  --nla-av-checkpoint "$NLA_AV_CHECKPOINT" \
  --nla-extraction-layer "$NLA_EXTRACTION_LAYER" \
  --api-base "http://${SERVER_HOST}:${SERVER_PORT}/v1" \
  --max-tokens "$MAX_TOKENS" \
  --num-threads "$NUM_THREADS" \
  "${BUDGET_ARGS[@]}"

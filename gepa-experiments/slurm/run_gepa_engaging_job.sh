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
DATASET="${DATASET:-topical_chat}"
DIMENSION="${DIMENSION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_TOKENS="${MAX_TOKENS:-512}"
INSTRUCTION_PROPOSER="${INSTRUCTION_PROPOSER:-default}"
PERPLEXITY_FEEDBACK="${PERPLEXITY_FEEDBACK:-0}"
PERPLEXITY_PROMPT_LOGPROBS="${PERPLEXITY_PROMPT_LOGPROBS:-20}"
PROPOSER_MODEL="${PROPOSER_MODEL:-local-llamacpp}"
PROPOSER_API_BASE="${PROPOSER_API_BASE:-}"
PROPOSER_API_KEY="${PROPOSER_API_KEY:-}"
PROPOSER_TEMPERATURE="${PROPOSER_TEMPERATURE:-0.7}"
PROPOSER_MAX_TOKENS="${PROPOSER_MAX_TOKENS:-4096}"
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
echo "  dataset: ${DATASET}"
echo "  dimension: ${DIMENSION:-${LABEL:-legacy-label}}"
echo "  vLLM model path: ${VLLM_MODEL_ARG}"
echo "  NLA AV checkpoint reserved for next phase: ${NLA_AV_CHECKPOINT}"
echo "  health: ${HEALTH_URL}"
echo "  log: ${VLLM_LOG}"
echo "  max tokens: ${MAX_TOKENS}"
echo "  perplexity feedback: ${PERPLEXITY_FEEDBACK}"
if [[ -n "$PROPOSER_API_BASE" ]]; then
  echo "  proposer model: ${PROPOSER_MODEL}"
  echo "  proposer api: ${PROPOSER_API_BASE}"
  echo "  proposer max tokens: ${PROPOSER_MAX_TOKENS}"
  echo "  proposer temperature: ${PROPOSER_TEMPERATURE}"
else
  echo "  proposer model: same as judge"
fi

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

PROPOSER_ARGS=()
if [[ -n "$PROPOSER_API_BASE" ]]; then
  PROPOSER_HEALTH_URL="${PROPOSER_API_BASE%/}/models"
  echo "Checking proposer readiness at ${PROPOSER_HEALTH_URL}..."
  if ! curl -fsS -H "Authorization: Bearer ${PROPOSER_API_KEY}" "$PROPOSER_HEALTH_URL" >/dev/null 2>&1; then
    echo "Proposer endpoint is not ready: ${PROPOSER_HEALTH_URL}" >&2
    exit 1
  fi
  PROPOSER_ARGS=(
    --proposer-model "$PROPOSER_MODEL"
    --proposer-api-base "$PROPOSER_API_BASE"
    --proposer-api-key "$PROPOSER_API_KEY"
    --proposer-temperature "$PROPOSER_TEMPERATURE"
    --proposer-max-tokens "$PROPOSER_MAX_TOKENS"
  )
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

PERPLEXITY_ARGS=()
if [[ "$PERPLEXITY_FEEDBACK" == "1" || "$PERPLEXITY_FEEDBACK" == "true" ]]; then
  PERPLEXITY_ARGS=(
    --perplexity-feedback
    --perplexity-hf-home /llms
    --perplexity-prompt-logprobs "$PERPLEXITY_PROMPT_LOGPROBS"
  )
fi

TASK_ARGS=(--dataset "$DATASET")
if [[ -n "$DIMENSION" ]]; then
  TASK_ARGS+=(--dimension "$DIMENSION")
fi

python -m geval_gepa.runner \
  --data-source "$DATA_SOURCE" \
  "${TASK_ARGS[@]}" \
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
  --instruction-proposer "$INSTRUCTION_PROPOSER" \
  "${PROPOSER_ARGS[@]}" \
  "${PERPLEXITY_ARGS[@]}" \
  "${BUDGET_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/soft_prompt_sipit_topical_chat_engagingness_long.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

EXTRA_ARGS=()
if [[ -n "${SOFT_PROMPT_SIPIT_MAX_SOFT_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--max-soft-tokens "$SOFT_PROMPT_SIPIT_MAX_SOFT_TOKENS")
fi
if [[ -n "${SOFT_PROMPT_SIPIT_CONTROL_MODE:-}" ]]; then
  EXTRA_ARGS+=(--control-mode "$SOFT_PROMPT_SIPIT_CONTROL_MODE")
fi
if [[ -n "${SOFT_PROMPT_SIPIT_CONTROL_TEXT:-}" ]]; then
  EXTRA_ARGS+=(--control-text "$SOFT_PROMPT_SIPIT_CONTROL_TEXT")
fi
if [[ -n "${SOFT_PROMPT_SIPIT_CONTROL_NUM_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--control-num-tokens "$SOFT_PROMPT_SIPIT_CONTROL_NUM_TOKENS")
fi

python gepa-experiments/soft_prompting/sipit_soft_prompt_recover.py \
  --input-dir "${SOFT_PROMPT_SIPIT_INPUT_DIR:-gepa-experiments/results/soft_prompt_topical_chat_engagingness_long}" \
  --output-dir "${SOFT_PROMPT_SIPIT_OUTPUT_DIR:-gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_sipit}" \
  --model-name "${SOFT_PROMPT_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --embedding-file "${SOFT_PROMPT_SIPIT_EMBEDDING_FILE:-soft_prompt_embeddings.pt}" \
  --layer-idx "${SOFT_PROMPT_SIPIT_LAYER_IDX:--1}" \
  --precision "${SOFT_PROMPT_SIPIT_PRECISION:-4}" \
  --top-k "${SOFT_PROMPT_SIPIT_TOP_K:-10}" \
  --seed "${SEED:-42}" \
  --step-size "${SOFT_PROMPT_SIPIT_STEP_SIZE:-1.0}" \
  --projection-iters-base "${SOFT_PROMPT_SIPIT_PROJECTION_ITERS_BASE:-50}" \
  --vocab-scale-factor "${SOFT_PROMPT_SIPIT_VOCAB_SCALE_FACTOR:-25000}" \
  --max-iters-per-token "${SOFT_PROMPT_SIPIT_MAX_ITERS_PER_TOKEN:-500}" \
  "${EXTRA_ARGS[@]}"

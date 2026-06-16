#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/soft_prompt_topical_chat_engagingness_smoke.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

python gepa-experiments/soft_prompting/train_soft_judge.py \
  --model-name "${SOFT_PROMPT_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --dataset "${DATASET:-topical_chat}" \
  --dimension "${DIMENSION:-engagingness}" \
  --data-source "${DATA_SOURCE:-gepa-experiments/cache/tc_usr_data.json}" \
  --train-groups "${TRAIN_CONTEXTS:-4}" \
  --val-groups "${VAL_CONTEXTS:-2}" \
  --test-groups "${TEST_CONTEXTS:-2}" \
  --seed "${SEED:-42}" \
  --num-virtual-tokens "${SOFT_PROMPT_NUM_VIRTUAL_TOKENS:-16}" \
  --max-seq-len "${SOFT_PROMPT_MAX_SEQ_LEN:-1024}" \
  --max-new-tokens "${SOFT_PROMPT_MAX_NEW_TOKENS:-16}" \
  --train-batch-size "${SOFT_PROMPT_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${SOFT_PROMPT_GRADIENT_ACCUMULATION_STEPS:-8}" \
  --eval-batch-size "${SOFT_PROMPT_EVAL_BATCH_SIZE:-2}" \
  --learning-rate "${SOFT_PROMPT_LEARNING_RATE:-0.005}" \
  --epochs "${SOFT_PROMPT_EPOCHS:-3}" \
  --warmup-ratio "${SOFT_PROMPT_WARMUP_RATIO:-0.05}" \
  --output-dir "${SOFT_PROMPT_OUTPUT_DIR:-gepa-experiments/results/soft_prompt_topical_chat_engagingness_smoke}"

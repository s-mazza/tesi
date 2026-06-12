#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/experimental_nla_candidate_content_6_topical_chat_smoke.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

EXPERIMENTAL_NLA_TOKEN_STRATEGY="${EXPERIMENTAL_NLA_TOKEN_STRATEGY:-}"
if [[ -z "$EXPERIMENTAL_NLA_TOKEN_STRATEGY" ]]; then
  echo "Set EXPERIMENTAL_NLA_TOKEN_STRATEGY in ${CONFIG_FILE}." >&2
  exit 2
fi

DATASET="${DATASET:-topical_chat}"
DIMENSION="${DIMENSION:-engagingness}"
OUTPUT_DIR="${OUTPUT_DIR:-gepa-experiments/results/experimental_nla_${EXPERIMENTAL_NLA_TOKEN_STRATEGY}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

NLA_MANIFEST_PATH="${NLA_MANIFEST_PATH:-${OUTPUT_DIR}/experimental_nla_manifest_${EXPERIMENTAL_NLA_TOKEN_STRATEGY}_${SLURM_JOB_ID:-local}.jsonl}"
NLA_PRECOMPUTED_PATH="${NLA_PRECOMPUTED_PATH:-${OUTPUT_DIR}/experimental_nla_precomputed_${EXPERIMENTAL_NLA_TOKEN_STRATEGY}_${SLURM_JOB_ID:-local}.jsonl}"

echo "Building isolated experimental NLA precompute"
echo "  config: ${CONFIG_FILE}"
echo "  strategy: ${EXPERIMENTAL_NLA_TOKEN_STRATEGY}"
echo "  manifest: ${NLA_MANIFEST_PATH}"
echo "  precomputed: ${NLA_PRECOMPUTED_PATH}"

python gepa-experiments/scripts/export_nla_manifest.py \
  --dataset "$DATASET" \
  --dimension "$DIMENSION" \
  --data-source "$DATA_SOURCE" \
  --split "${NLA_PRECOMPUTE_SPLIT:-gepa}" \
  --train-groups "$TRAIN_CONTEXTS" \
  --val-groups "$VAL_CONTEXTS" \
  --test-groups "$TEST_CONTEXTS" \
  --seed "$SEED" \
  --output "$NLA_MANIFEST_PATH"

NLA_PRECOMPUTE_ARGS=()
if [[ -n "${NLA_PRECOMPUTE_LIMIT:-}" ]]; then
  NLA_PRECOMPUTE_ARGS+=(--limit "$NLA_PRECOMPUTE_LIMIT")
fi
if [[ "${NLA_PRECOMPUTE_DRY_RUN:-0}" == "1" || "${NLA_PRECOMPUTE_DRY_RUN:-0}" == "true" ]]; then
  NLA_PRECOMPUTE_ARGS+=(--dry-run)
fi

python gepa-experiments/scripts/experimental_build_nla_precomputed.py \
  --manifest "$NLA_MANIFEST_PATH" \
  --output "$NLA_PRECOMPUTED_PATH" \
  --strategy "$EXPERIMENTAL_NLA_TOKEN_STRATEGY" \
  --activation-model "$JUDGE_MODEL" \
  --nla-checkpoint "$NLA_AV_CHECKPOINT" \
  --layer "$NLA_EXTRACTION_LAYER" \
  --max-new-tokens "${NLA_PRECOMPUTE_MAX_NEW_TOKENS:-160}" \
  --temperature "${NLA_PRECOMPUTE_TEMPERATURE:-0.0}" \
  --activation-dtype "${NLA_ACTIVATION_DTYPE:-float16}" \
  --nla-dtype "${NLA_VERBALIZER_DTYPE:-float16}" \
  --device-map "${NLA_DEVICE_MAP:-auto}" \
  --trust-remote-code \
  "${NLA_PRECOMPUTE_ARGS[@]}"

export NLA_FEEDBACK=1
export NLA_BACKEND=precomputed
export NLA_PRECOMPUTED_AUTO=0
export NLA_PRECOMPUTED_PATH

exec bash gepa-experiments/slurm/run_gepa_engaging_job.sh

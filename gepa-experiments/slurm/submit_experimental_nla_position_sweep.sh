#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${BASE_CONFIG:-gepa-experiments/config/experimental_nla_position_sweep_topical_chat_smoke.env}"
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 2
fi

DEFAULT_STRATEGIES=(
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

if [[ -n "${NLA_POSITION_STRATEGIES:-}" ]]; then
  # shellcheck disable=SC2206
  STRATEGIES=($NLA_POSITION_STRATEGIES)
else
  STRATEGIES=("${DEFAULT_STRATEGIES[@]}")
fi

PORT_BASE="${PORT_BASE:-18220}"
PORT_STEP="${PORT_STEP:-2}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-gepa-experiments/slurm/submit_experimental_nla_strategy.sh}"

for index in "${!STRATEGIES[@]}"; do
  strategy="${STRATEGIES[$index]}"
  server_port=$((PORT_BASE + index * PORT_STEP))
  proposer_port=$((server_port + 1))
  output_dir="gepa-experiments/results/experimental_nla_${strategy}_topical_chat_smoke"
  job_slug="experimental-nla-${strategy}-topical_chat-engagingness"
  echo "Submitting ${strategy}: server_port=${server_port}, proposer_port=${proposer_port}"
  CONFIG_FILE="$BASE_CONFIG" \
    EXPERIMENTAL_NLA_TOKEN_STRATEGY="$strategy" \
    SERVER_PORT="$server_port" \
    PROPOSER_PORT="$proposer_port" \
    OUTPUT_DIR="$output_dir" \
    JOB_SLUG="$job_slug" \
    bash "$SUBMIT_SCRIPT"
done

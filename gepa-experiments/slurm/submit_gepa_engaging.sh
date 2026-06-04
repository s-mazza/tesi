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

GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
SLURM_NODE="${SLURM_NODE:-}"
IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gepa-experiments/results/slurm}"
SLURM_TIME="${SLURM_TIME:-}"
SLURM_MEM="${SLURM_MEM:-}"

mkdir -p "$OUTPUT_ROOT"

NODE_ARGS=()
if [[ -n "$SLURM_NODE" ]]; then
  NODE_ARGS=(-w "$SLURM_NODE")
fi

TIME_ARGS=()
if [[ -n "$SLURM_TIME" ]]; then
  TIME_ARGS=(--time="$SLURM_TIME")
fi

MEM_ARGS=()
if [[ -n "$SLURM_MEM" ]]; then
  MEM_ARGS=(--mem="$SLURM_MEM")
fi

sbatch \
  -N 1 \
  --gpus="$GPU_SPEC" \
  "${NODE_ARGS[@]}" \
  "${TIME_ARGS[@]}" \
  "${MEM_ARGS[@]}" \
  --output="${OUTPUT_ROOT}/slurm-%j-geval-gepa-engaging.out" \
  --export=ALL,IMAGE_NAME="$IMAGE_NAME",CONFIG_FILE="$CONFIG_FILE" \
  gepa-experiments/slurm/run_docker.sh \
  "bash gepa-experiments/slurm/run_gepa_engaging_job.sh"

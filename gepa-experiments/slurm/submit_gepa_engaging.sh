#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/geval_gepa_engaging_qwen25.env}"
GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
SLURM_NODE="${SLURM_NODE:-faretra}"
IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gepa-experiments/results/slurm}"

mkdir -p "$OUTPUT_ROOT"

NODE_ARGS=()
if [[ -n "$SLURM_NODE" ]]; then
  NODE_ARGS=(-w "$SLURM_NODE")
fi

sbatch \
  -N 1 \
  --gpus="$GPU_SPEC" \
  "${NODE_ARGS[@]}" \
  --output="${OUTPUT_ROOT}/slurm-%j-geval-gepa-engaging.out" \
  --export=ALL,IMAGE_NAME="$IMAGE_NAME",CONFIG_FILE="$CONFIG_FILE" \
  gepa-experiments/slurm/run_docker.sh \
  "bash gepa-experiments/slurm/run_gepa_engaging_job.sh"

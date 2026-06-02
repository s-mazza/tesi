#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
LLM_CACHE_DIR="${LLM_CACHE_DIR:-/llms}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: sbatch -N 1 --gpus=<gpu>:1 gepa-experiments/slurm/run_docker.sh '<command>'" >&2
  exit 2
fi

docker run \
  --rm \
  --init \
  --ipc=host \
  --gpus '"device='"${VISIBLE_DEVICES}"'"' \
  -v "$PROJECT_DIR:/workspace" \
  -v "$LLM_CACHE_DIR:/llms" \
  -e "HF_HOME=/llms" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}" \
  -e "PYTHONPATH=/workspace/gepa-experiments:${PYTHONPATH:-}" \
  -e "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -lc "$*"

#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nla_experiments:latest}"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
ARTIFACT_DIR="${NLA_ARTIFACT_ROOT:-$PROJECT_DIR/nla-artifacts}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
DOCKER_HF_HOME="${DOCKER_HF_HOME:-/llms}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: sbatch -N 1 --gpus=<gpu>:1 nla-experiments/slurm/run_docker.sh '<command>'" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR" "$HF_CACHE_DIR"

docker run \
  --rm \
  --ipc=host \
  --gpus '"device='"${CUDA_VISIBLE_DEVICES:-0}"'"' \
  -v "$PROJECT_DIR:/workspace" \
  -v "$ARTIFACT_DIR:/workspace/nla-artifacts" \
  -v "$HF_CACHE_DIR:$DOCKER_HF_HOME" \
  -e "HF_HOME=$DOCKER_HF_HOME" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "NLA_ARTIFACT_ROOT=/workspace/nla-artifacts" \
  -e "PYTHONPATH=/workspace/nla-experiments:/workspace/natural_language_autoencoders:${PYTHONPATH:-}" \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -lc "$*"


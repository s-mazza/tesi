#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
ARTIFACT_DIR="${NLA_ARTIFACT_ROOT:-$PWD/nla-artifacts}"
MANIFEST="${MANIFEST:-nla-artifacts/summeval/task_manifest.jsonl}"
OUTPUT="${OUTPUT:-nla-artifacts/summeval/activations_qwen25_7b_instruct_L20.parquet}"
LIMIT_ARGS=()

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

bash nla-experiments/init_artifact_repo.sh
mkdir -p "$ARTIFACT_DIR/slurm"

sbatch \
  -N 1 \
  --gpus="$GPU_SPEC" \
  --output="$ARTIFACT_DIR/slurm/slurm-%j-extract-qwen.out" \
  nla-experiments/slurm/run_docker.sh \
  "python nla-experiments/summeval/extract_qwen_activations.py --manifest '$MANIFEST' --output '$OUTPUT' ${LIMIT_ARGS[*]}"


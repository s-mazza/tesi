#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
SLURM_NODE="${SLURM_NODE:-}"
ARTIFACT_DIR="${NLA_ARTIFACT_ROOT:-$PWD/nla-artifacts}"
NODE_ARGS=()

if [[ -n "$SLURM_NODE" ]]; then
  NODE_ARGS=(-w "$SLURM_NODE")
fi

bash nla-experiments/init_artifact_repo.sh
mkdir -p "$ARTIFACT_DIR/slurm"

sbatch \
  -N 1 \
  --gpus="$GPU_SPEC" \
  "${NODE_ARGS[@]}" \
  --output="$ARTIFACT_DIR/slurm/slurm-%j-verbalize-nla.out" \
  nla-experiments/slurm/run_docker.sh \
  "bash nla-experiments/slurm/run_verbalize_nla_job.sh"

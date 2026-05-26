#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
ARTIFACT_DIR="${NLA_ARTIFACT_ROOT:-$PWD/nla-artifacts}"

bash nla-experiments/init_artifact_repo.sh
mkdir -p "$ARTIFACT_DIR/slurm"

sbatch \
  -N 1 \
  --gpus="$GPU_SPEC" \
  --output="$ARTIFACT_DIR/slurm/slurm-%j-verbalize-nla.out" \
  nla-experiments/slurm/run_docker.sh \
  "bash nla-experiments/slurm/run_verbalize_nla_job.sh"

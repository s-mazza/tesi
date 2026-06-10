#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-gepa-experiments/results/slurm}"
GPU_SPEC="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
SLURM_NODE="${SLURM_NODE:-moro232}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SLURM_TIME="${SLURM_TIME:-00:05:00}"
SLURM_MEM="${SLURM_MEM:-1G}"
JOB_SLUG="${JOB_SLUG:-slurm-stdout-smoke}"

mkdir -p "$OUTPUT_ROOT"

NODE_ARGS=()
if [[ -n "$SLURM_NODE" && "$SLURM_NODE" != "auto" && "$SLURM_NODE" != "AUTO" ]]; then
  NODE_ARGS=(-w "$SLURM_NODE")
fi

EXCLUDE_ARGS=()
if [[ -n "$SLURM_EXCLUDE" ]]; then
  EXCLUDE_ARGS=(--exclude="$SLURM_EXCLUDE")
fi

SBATCH_OUTPUT="$(sbatch \
  --parsable \
  -N 1 \
  --gpus="$GPU_SPEC" \
  "${NODE_ARGS[@]}" \
  "${EXCLUDE_ARGS[@]}" \
  --time="$SLURM_TIME" \
  --mem="$SLURM_MEM" \
  --job-name="$JOB_SLUG" \
  --output="${OUTPUT_ROOT}/slurm-%j-${JOB_SLUG}.out" \
  --wrap='set -eu; echo "stdout-smoke-start"; date; hostname; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits; echo "stdout-smoke-done"; date')"

JOB_ID="${SBATCH_OUTPUT%%;*}"
echo "Submitted batch job ${JOB_ID}"
echo "Log: ${OUTPUT_ROOT}/slurm-${JOB_ID}-${JOB_SLUG}.out"

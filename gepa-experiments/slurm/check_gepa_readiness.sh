#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-geval_gepa:latest}"
CONFIG_FILE="${CONFIG_FILE:-gepa-experiments/config/geval_gepa_engaging_qwen25.env}"

docker run --rm \
  -v "$PWD:/workspace" \
  -v /llms:/llms \
  -e PYTHONPATH=/workspace/gepa-experiments \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -lc "python -m unittest discover -s gepa-experiments/tests && python -m geval_gepa.deep_preflight --config-file '$CONFIG_FILE'"

#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-llama.cpp:localcuda}"
MODEL_DIR="${MODEL_DIR:-/llms}"
MODEL_NAME="${MODEL_NAME:-}"
HF_MODEL="${HF_MODEL:-opensota/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${MODEL_DIR}/llamacpp-hf-cache}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
GPU_DEVICE="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LLAMA_API_KEY="${LLAMA_API_KEY:-local-llamacpp-key}"

CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"

mkdir -p "$HF_CACHE_DIR"

MODEL_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_ARGS=(-m "/models/${MODEL_NAME}")
else
  MODEL_ARGS=(-hf "$HF_MODEL")
fi

docker run --rm \
  --gpus "device=${GPU_DEVICE}" \
  --network=host \
  --ipc=host \
  -v "${MODEL_DIR}:/models:ro" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${IMAGE}" \
  "${MODEL_ARGS[@]}" \
  --api-key "${LLAMA_API_KEY}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-gpu-layers "${N_GPU_LAYERS}" \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${PARALLEL}" \
  --flash-attn "${FLASH_ATTN}" \
  --batch-size "${BATCH_SIZE}"

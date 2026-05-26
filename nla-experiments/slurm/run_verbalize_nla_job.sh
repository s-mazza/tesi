#!/usr/bin/env bash
set -euo pipefail

NLA_AV_MODEL="${NLA_AV_MODEL:-kitft/nla-qwen2.5-7b-L20-av}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_URL="http://${SGLANG_HOST}:${SGLANG_PORT}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
ACTIVATIONS="${ACTIVATIONS:-nla-artifacts/summeval/activations_qwen25_7b_instruct_L20.parquet}"
OUTPUT="${OUTPUT:-nla-artifacts/summeval/verbalizations.jsonl}"
LIMIT_ARGS=()

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

python -m sglang.launch_server \
  --model-path "$NLA_AV_MODEL" \
  --host "$SGLANG_HOST" \
  --port "$SGLANG_PORT" \
  --disable-radix-cache \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --trust-remote-code \
  ${SGLANG_EXTRA_ARGS:-} &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import os
import time
import urllib.error
import urllib.request

url = f"http://{os.environ.get('SGLANG_HOST', '127.0.0.1')}:{os.environ.get('SGLANG_PORT', '30000')}/health"
deadline = time.time() + int(os.environ.get("SGLANG_STARTUP_TIMEOUT", "900"))
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                raise SystemExit(0)
    except (OSError, urllib.error.URLError):
        time.sleep(5)
raise SystemExit(f"SGLang server did not become healthy before timeout: {url}")
PY

python nla-experiments/summeval/verbalize_nla.py \
  --activations "$ACTIVATIONS" \
  --output "$OUTPUT" \
  --checkpoint "$NLA_AV_MODEL" \
  --sglang-url "$SGLANG_URL" \
  "${LIMIT_ARGS[@]}"


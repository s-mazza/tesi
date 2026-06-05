# llama.cpp proposer handoff

This directory contains the local llama.cpp server used as GEPA's proposer /
reflection LM while the G-EVAL judge still runs through vLLM.

Build the image on the cluster:

```bash
cd gepa-experiments/llamacpp
docker build -t llama.cpp:localcuda .
```

Launch a standalone server:

```bash
HF_MODEL=opensota/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M ./serve_llamacpp.sh
```

The server exposes an OpenAI-compatible endpoint at:

```text
http://127.0.0.1:8080/v1
```

For GEPA jobs this is normally started automatically by
`gepa-experiments/slurm/run_docker.sh` when the selected config sets:

```bash
PROPOSER_BACKEND=llamacpp
GPU_SPEC=nvidia_geforce_rtx_3090:2
```

Use `generate_llamacpp.py` as a tiny standalone endpoint check.

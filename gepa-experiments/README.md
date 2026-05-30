# GEPA Experiments

This folder turns the provided GEPA examples into a single cluster-ready v1
experiment: optimize a G-EVAL-style LLM-as-judge prompt for Topical-Chat/USR.

## V1 Target

- Task: LLM-as-judge scoring, not dialogue generation.
- Dataset: `https://shikib.com/tc_usr_data.json`, cached at
  `cache/tc_usr_data.json` so Slurm jobs do not depend on compute-node network
  access.
- First metric: USR `Engaging`, mapped to G-EVAL `Engagingness`.
- Judge model: `Qwen/Qwen2.5-7B-Instruct`.
- Future NLA hook: `kitft/nla-qwen2.5-7b-L20-av`, layer 20.

The model choice is deliberate: Qwen2.5-7B is the base model for the available
NLA AV checkpoint, so this v1 can become a GEPA+NLA experiment without changing
the judge model family.

## Local Checks

The local tests do not need GPU, DSPy, vLLM, or network access:

```bash
PYTHONPATH=gepa-experiments python3 -m unittest discover -s gepa-experiments/tests
```

## Build Container

On `faretra`, build from the thesis repo root:

```bash
docker build -f gepa-experiments/docker/Dockerfile -t geval_gepa:latest .
```

The exact Torch/CUDA version is not a research constraint for this v1. The
required invariants are:

- vLLM can serve `Qwen/Qwen2.5-7B-Instruct`;
- DSPy/GEPA import and run;
- `/llms` contains the base judge model and the future NLA AV checkpoint;
- no source build of FlashAttention is attempted.

If installing upstream `flash-attn`, pass only a prebuilt wheel URL matching the
resolved container Python/Torch/CUDA stack. Do not allow a source build:

```bash
docker build -f gepa-experiments/docker/Dockerfile -t geval_gepa:latest \
  --build-arg FLASH_ATTN_WHEEL_URL='https://.../flash_attn-...whl' .
```

If no wheel is passed, vLLM's bundled attention backend is used.

Run the preflight before submitting GPU work:

```bash
docker run --rm \
  -v "$PWD:/workspace" -v /llms:/llms \
  -e PYTHONPATH=/workspace/gepa-experiments \
  -w /workspace geval_gepa:latest \
  python -m geval_gepa.preflight
```

## Submit Smoke Run

The default config runs roughly 100 response-level examples: 10 train contexts,
3 validation contexts, and 4 test contexts.

```bash
ssh faretra "cd ~/tesi && IMAGE_NAME=geval_gepa:latest bash gepa-experiments/slurm/submit_gepa_engaging.sh"
```

Outputs are written under `gepa-experiments/results/`, which is intentionally
ignored by the parent repository.

Set `SLURM_NODE=faretra` only when the image/code has not been replicated to
other nodes. Leaving it empty lets Slurm use any compatible GPU node.

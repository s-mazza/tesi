# NLA SummEval Experiment

This folder contains the reproducible code for the first NLA pass requested by
the advisor: run Qwen2.5-7B-Instruct on SummEval evaluation prompts, extract
layer-20 residual-stream activations, and verbalize them with the public Qwen
NLA AV checkpoint.

The parent `tesi` repository tracks code, configs, and Slurm launchers. Runtime
outputs live in a separate local Git repository at `nla-artifacts/`.

## Local CPU Smoke

```bash
bash nla-experiments/init_artifact_repo.sh
python3 nla-experiments/summeval/prepare_summeval.py --download
python3 nla-experiments/summeval/extract_qwen_activations.py --fake --limit 2 \
  --output nla-artifacts/summeval/activations_smoke.parquet
python3 nla-experiments/summeval/verbalize_nla.py --dry-run \
  --activations nla-artifacts/summeval/activations_smoke.parquet \
  --output nla-artifacts/summeval/verbalizations_smoke.jsonl
python3 nla-experiments/summeval/summarize_verbalizations.py \
  --input nla-artifacts/summeval/verbalizations_smoke.jsonl \
  --output nla-artifacts/summeval/report_smoke.md
```

Commit artifact snapshots from inside the artifact repo:

```bash
git -C nla-artifacts status
git -C nla-artifacts add summeval
git -C nla-artifacts commit -m "Record SummEval NLA smoke artifacts"
```

## Cluster GPU Smoke

The Slurm wrappers follow the existing SIPIT pattern: Docker gets the assigned
`CUDA_VISIBLE_DEVICES`, the repo is mounted at `/workspace`, Hugging Face cache
is mounted at `/llms`, and artifacts are written under `/workspace/nla-artifacts`.

```bash
ssh faretra "cd ~/tesi && LIMIT=1 bash nla-experiments/slurm/submit_extract_qwen.sh"
ssh faretra "cd ~/tesi && LIMIT=1 bash nla-experiments/slurm/submit_verbalize_nla.sh"
```

Useful environment overrides:

```bash
GPU_SPEC=nvidia_geforce_rtx_3090:1
IMAGE_NAME=nla_experiments:latest
PYTHON_BIN=python3
NLA_ARTIFACT_ROOT=/path/to/nla-artifacts
HF_CACHE_DIR=$HOME/.cache/huggingface
```

For extraction on the current cluster, the existing `emb-inversion:latest`
image is sufficient:

```bash
IMAGE_NAME=emb-inversion:latest PYTHON_BIN=python3 LIMIT=1 SLURM_NODE=faretra \
  bash nla-experiments/slurm/submit_extract_qwen.sh
```

Build the dedicated Docker image only if the existing images do not provide the
needed NLA/SGLang runtime for verbalization:

```bash
docker build -f nla-experiments/docker/Dockerfile -t nla_experiments:latest .
```

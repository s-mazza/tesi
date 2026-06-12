# GEPA Experiments

This folder turns the provided GEPA examples into a single cluster-ready v1
experiment: optimize a G-EVAL-style LLM-as-judge prompt for Topical-Chat/USR.

## V1 Target

- Task: LLM-as-judge scoring, not dialogue generation.
- Dataset: Topical-Chat/USR from `https://shikib.com/tc_usr_data.json`; set
  `DATA_SOURCE` in the selected config to the dataset copy available on the
  execution machine.
- First metric: USR `Engaging`, mapped to G-EVAL `Engagingness`.
- Judge model: `Qwen/Qwen2.5-7B-Instruct`.
- Future NLA hook: `kitft/nla-qwen2.5-7b-L20-av`, layer 20.

The model choice is deliberate: Qwen2.5-7B is the base model for the available
NLA AV checkpoint, so this v1 can become a GEPA+NLA experiment without changing
the judge model family.

## Folder Map

Only committed implementation files are described here. Generated artifacts,
private planning material, and source handoff material are intentionally
omitted from this GitHub review branch.

```text
gepa-experiments/
├── README.md
├── config/
├── docker/
├── geval_gepa/
├── llamacpp/
├── scripts/
├── slurm/
└── tests/
```

`config/` contains Slurm/job `.env` files passed as `CONFIG_FILE=...` to
`slurm/submit_gepa_engaging.sh`. The most relevant entries are:

- `geval_gepa_experiment_matrix.csv`: dataset/dimension/variant matrix for
  paper-aligned experiments.
- `geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control.env`:
  matched long Topical-Chat PPL-only control.
- `geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b.env`:
  matched long Topical-Chat PPL+fixed-NLA run.
- `geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke.env`:
  Topical-Chat Qwen35B auxiliary-judge smoke.
- `geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke.env`: Topical-Chat Qwen35B
  proposer PPL smoke control.
- `geval_gepa_summeval_consistency_ppl_smoke.env` and
  `geval_gepa_summeval_consistency_ppl_real_nla_smoke.env`: SummEval control
  and NLA smoke configs.
- `geval_gepa_qags_cnn_consistency_ppl_smoke.env`,
  `geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke.env`,
  `geval_gepa_qags_xsum_consistency_ppl_smoke.env`, and
  `geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke.env`: QAGS-CNN and
  QAGS-XSUM control/NLA smoke configs.
- `experimental_nla_candidate_content_6_topical_chat_smoke.env`,
  `experimental_nla_candidate_content_10_topical_chat_smoke.env`, and
  `experimental_nla_hybrid_context_dedup_6_topical_chat_smoke.env`: isolated
  NLA token-selection experiments.

Other `.env` files are legacy, smaller-budget, or intermediate reproducibility
configs.

`docker/` defines the main vLLM/DSPy/GEPA runtime image.

- `Dockerfile`: builds the `geval_gepa:latest` image used by Slurm jobs.
- `requirements.txt`: Python dependencies installed in that image.

`geval_gepa/` is the Python package used by both local tests and cluster jobs.

- `__init__.py`: package marker.
- `data.py`: legacy Topical-Chat/USR loading helpers kept for backward
  compatibility.
- `tasks.py`: dataset registry and loaders for Topical-Chat, SummEval,
  QAGS-CNN, and QAGS-XSUM.
- `prompts.py`: seed G-Eval judge prompts and prompt text utilities.
- `runner.py`: main experiment runner; builds DSPy programs, GEPA optimizer,
  feedback providers, final evaluations, and run artifacts.
- `metrics.py`: parsing and final evaluation metrics, including Pearson,
  Spearman, Kendall tau, MAE, agreement, and coverage.
- `perplexity.py`: response-only perplexity scorer using vLLM prompt logprobs.
- `nla_precompute.py`: extracts Qwen2.5-7B activations and verbalizes them with
  the NLA AV checkpoint.
- `nla_feedback.py`: loads precomputed NLA verbalizations and formats them as
  GEPA feedback.
- `aux_judge.py`: optional Qwen35B auxiliary LLM-as-a-judge feedback provider.
- `proposers.py`: custom instruction proposer logic used by GEPA.
- `trajectory.py`: exports prompt trajectories from GEPA/GEPA-viz artifacts.
- `preflight.py`: fast runtime/data/model-cache sanity checks before GPU jobs.
- `deep_preflight.py`: deeper CPU-side checks for configs, datasets, and model
  snapshots.

`llamacpp/` contains the sidecar proposer service.

- `Dockerfile`: builds the CUDA-enabled llama.cpp server image.
- `generate_llamacpp.py`: helper for llama.cpp model/cache preparation.
- `serve_llamacpp.sh`: local server launch helper.

`scripts/` contains offline artifact builders and diagnostics.

- `export_nla_manifest.py`: exports GEPA train/validation examples as prompts
  for NLA activation extraction.
- `build_nla_precomputed.py`: builds `nla_precomputed_*.jsonl` from a manifest.
- `experimental_build_nla_precomputed.py`: builds alternate experimental NLA
  token-selection artifacts.
- `experimental_nla_token_strategy_analysis.py`: compares candidate/source/
  reference token-selection strategies before running GPU jobs.
- `diagnose_nla_run.py`: pairwise control-vs-NLA diagnostic report generator.
- `analyze_nla_evidence.py`: aggregate cross-run evidence report generator.
- `aggregate_results.py`: compact metric aggregation utility.

`slurm/` contains cluster launch and monitoring helpers.

- `submit_gepa_engaging.sh`: generic Slurm submit wrapper for the `.env`
  configs.
- `run_docker.sh`: Slurm entrypoint that starts Docker, vLLM, and the optional
  llama.cpp sidecar.
- `run_gepa_engaging_job.sh`: inside-container job script; runs preflight,
  optional NLA precompute, vLLM, and `geval_gepa.runner`.
- `check_gepa_readiness.sh`: CPU-side readiness check before GPU submission.
- `cluster_status_snapshot.sh`: single-command queue/log/artifact status
  snapshot to reduce repeated SSH polling.
- `telegram_monitor.py`: Telegram log/state monitor for submitted jobs.
- `submit_experimental_nla_strategy.sh`: submit wrapper for isolated NLA
  strategy experiments.
- `run_experimental_nla_strategy_job.sh`: job body for experimental NLA
  strategies.
- `submit_slurm_stdout_smoke.sh`: small Slurm stdout/debug submission helper.

`tests/` contains local CPU tests.

- `test_data_and_metrics.py`: regression tests for data loading, split logic,
  metric parsing, NLA feedback validation, aux judge feedback plumbing, and
  config behavior.

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

The default image pins `vllm==0.10.2` on the CUDA 12.8/Torch 2.8 backend. This
avoids the observed `vllm 0.22.0` + `torch 2.11.0/cu130` startup crash during
Qwen2 model inspection while still using prebuilt vLLM CUDA binaries.

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

For deeper CPU-only checks before a GPU is available, run:

```bash
ssh faretra "cd ~/tesi && IMAGE_NAME=geval_gepa:latest bash gepa-experiments/slurm/check_gepa_readiness.sh"
```

This validates the runtime pins, config budget, dataset splits, metric parsing,
DSPy/GEPA prompt construction, local Hugging Face snapshots, and vLLM's Qwen2
model import path without starting the vLLM server.

## Split Semantics

The Topical-Chat/USR JSON contains 60 contexts with 6 annotated candidate
responses each, for 360 response-level examples. The runner splits by context
id, not by response row, so responses from the same dialogue never cross split
boundaries.

- `TRAIN_CONTEXTS` becomes GEPA's train set.
- `VAL_CONTEXTS` becomes GEPA's validation/dev set for prompt proposal and
  selection.
- `TEST_CONTEXTS` is the final held-out test set. It is not passed to GEPA and
  is evaluated only after the final prompt has been selected.

The longer configs use all 360 examples as 40 train contexts, 10 validation
contexts, and 10 final-test contexts.

## Submit Smoke Run

The default config runs roughly 100 response-level examples: 10 train contexts,
3 validation contexts, and 4 test contexts.

The longer configs enable the generalizing proposer from the notebook-derived
workflow. It gives GEPA feedback to the proposer while redacting raw
conversation text and filtering labels/metric artifacts so the final prompt does
not overfit by copying validation examples.

```bash
ssh faretra "cd ~/tesi && IMAGE_NAME=geval_gepa:latest bash gepa-experiments/slurm/submit_gepa_engaging.sh"
```

`submit_gepa_engaging.sh` starts a detached Telegram monitor automatically for
each submitted Slurm job when `~/.telegram_credentials` exists on the cluster.
The monitor watches the Slurm state and the matching Slurm log for crash
patterns. Monitor pid/log files are written under the configured output
directory.

Useful overrides:

- `TELEGRAM_MONITOR=0` disables automatic monitoring.
- `TELEGRAM_MONITOR_POLL_SECONDS=30` changes polling frequency.
- `TELEGRAM_MONITOR_LABEL="..."` changes the Telegram message prefix.
- `TELEGRAM_MONITOR_CREDENTIALS=/path/to/credentials` changes credential file.

Run outputs are written under the configured output directory, which should stay
outside the committed code review surface.

Set `SLURM_NODE=faretra` only when the image/code has not been replicated to
other nodes. Leaving it empty lets Slurm use any compatible GPU node.

# Locked-GPU Execution Plan - 2026-06-25

## Objective

Use the two GPUs currently reserved by the supervisor on `faretra` to run the
highest-value GEPA/NLA jobs that have been blocked in the Slurm queue, without
waiting for Slurm to assign the same physical devices.

The scientific priority is to finish the auxiliary-judge branch and the NLA
token-position diagnostics needed for the thesis discussion:

1. verify that `ppl + NLA + aux judge` is wired correctly;
2. run the corresponding long `ppl + NLA + aux judge` experiment;
3. collect matched controls and token-position sweeps that explain whether NLA
   feedback is useful or why it is not helping GEPA in the current setting.

## Supervisor Interpretation

The supervisor's instruction was interpreted as permission to launch Docker
containers directly on `faretra` using:

```bash
--gpus device=0,2
```

Our pipeline uses two separate containers rather than one container consuming
both GPUs:

- GPU `0`: main vLLM judge container with Qwen2.5-7B;
- GPU `2`: llama.cpp sidecar with Qwen35B proposer and, when enabled, auxiliary
  judge feedback.

This is equivalent to the supervisor's instruction because the launcher receives
`CUDA_VISIBLE_DEVICES=0,2`, then pins:

```bash
JUDGE_GPU_DEVICE=0
PROPOSER_GPU_DEVICE=2
```

On the current Docker version, the comma form may require Docker-specific
quoting when used as a single `docker run --gpus` argument. The implemented
runner avoids that ambiguity: it starts one container with `--gpus device=0` and
the other with `--gpus device=2`.

## Why Direct Docker Instead Of Waiting For Slurm

The Slurm queue shows the target jobs as pending even when physical GPUs `0` and
`2` on `faretra` are free. This suggests that the GPUs were reserved manually or
administratively outside the normal Slurm scheduling path. Therefore a normal
`sbatch --gpus=...` job may continue waiting even though the specific devices are
usable through Docker.

Direct Docker execution is acceptable only after checking:

- `faretra` is reachable;
- GPU `0` and GPU `2` are physically free or above the configured memory gates;
- Docker can expose `device=0` and `device=2` separately to the
  `geval_gepa:latest` image;
- the llama.cpp image and the local Qwen35B GGUF are present;
- the Qwen2.5-7B cache exists under `/llms`.

## Output Policy

Direct runs must not overwrite previous Slurm artifacts. The locked-GPU runner
therefore writes all generated config copies, logs, and outputs under a new root:

```text
gepa-experiments/results/locked_gpu_<timestamp>/
```

Each generated config sources the original committed config and then overrides
only operational fields:

- `OUTPUT_DIR`
- `LOG_DIR`
- `SERVER_PORT`
- `PROPOSER_PORT`
- `JOB_SLUG`

This keeps the scientific settings traceable to the original configs while
making direct-run outputs unambiguous.

## Queue Order

The direct queue is intentionally sequential because every active GEPA job in
this branch needs both locked GPUs.

| Order | Label | Original intent | Config / strategy | Reason |
|---:|---|---|---|---|
| 1 | `D1_aux_judge_fixed_smoke_ppl_nla` | smoke | `geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke.env` | Gate for the thesis-critical aux-judge branch. |
| 2 | `D4_aux_judge_fixed_long_ppl_nla` | long | `geval_gepa_topical_chat_engagingness_8h_ppl_nla_auxjudge_llamacpp35b.env` | Main missing result: PPL + fixed NLA + Qwen35B aux judge. |
| 3 | `D2_matched_no_aux_smoke_ppl_nla` | smoke control | `geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke.env` | Confirms matched no-aux path after the same launcher path. |
| 4 | `D3_aux_judge_only_smoke_ppl_aux_no_nla` | smoke ablation | `geval_gepa_topical_chat_engagingness_ppl_auxjudge_llamacpp35b_smoke.env` | Separates aux-judge effect from NLA effect. |
| 5 | `A1_nla_strategy_wiring_probe` | smoke diagnostic | `experimental_nla_candidate_content_6_topical_chat_smoke.env` | Probe for the experimental NLA strategy wiring. |
| 6 | `B_*` | smoke sweep | 12 token-position strategies | Advisor-requested naive NLA token-position comparison. |
| 7 | `D5_matched_no_aux_long_ppl_nla` | long control | `geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b.env` | Useful if time remains; lower priority because a fixed-NLA long already exists. |

If the first smoke fails, the runner stops before launching the long aux-judge
job. If later jobs fail, the runner logs the failure and continues, because at
that point the direct Docker path has already been validated and collecting more
diagnostics is more useful than stopping the whole queue.

## Existing Slurm Jobs

Old Slurm jobs remain useful as a fallback until the direct runner is confirmed
to be alive. After the direct runner starts and passes preflight, duplicate old
pending jobs may be cancelled to avoid duplicated work, port conflicts, and
confusing artifact provenance.

The stale queued IDs are recorded in:

```text
gepa-experiments/status/submitted_followup_jobs_20260617.tsv
```

## SSH Rate-Limit Mitigation

Cluster interaction must be batched:

- one aggregated preflight SSH call;
- one `rsync` for the plan/runner files;
- one launch SSH call;
- infrequent aggregated status checks afterward.

No polling loop should repeatedly open SSH connections from the local Codex
session.

## Stop Conditions

Ask the user or supervisor before continuing if any of these happen:

- GPU `0` or `2` is no longer free before launch;
- Docker cannot expose `device=0,2`;
- the direct run would have to kill unrelated processes;
- required model/cache artifacts are missing and would require a large download;
- the aux smoke fails due to code or parsing problems rather than transient GPU
  contamination.

## Launch Record

Launched on `faretra` at 2026-06-25 21:41 CEST after the runner preflight
passed.

```text
run_id: 20260625T194158Z
remote_pid: 784525
remote_root: gepa-experiments/results/locked_gpu_20260625T194158Z
launch_log: gepa-experiments/results/locked_gpu_20260625T194158Z/locked_gpu_queue_20260625T194158Z.out
gpu_mapping: judge/vLLM=0, proposer/aux-judge llama.cpp=2
```

Startup observations:

- GPU `0` and GPU `2` had about 24.2 GiB free at launch.
- Docker exposed both GPUs separately to the `geval_gepa:latest` image.
- The llama.cpp Qwen35B sidecar started on GPU `2` and became ready at
  `http://127.0.0.1:19011/v1/models`.
- The D1 aux-judge smoke entered NLA precompute successfully after dataset,
  cache, and dependency preflight passed.
- Duplicate stale Slurm jobs from the 2026-06-17 follow-up queue were cancelled
  after the direct runner was confirmed alive.

## 2026-06-25 Runtime Readiness Audit

Additional audit performed while D1 was running, to reduce the chance that later
jobs fail only after reaching their turn:

- local shell syntax check passed for the locked runner, Docker launcher, GEPA
  entrypoint, experimental NLA entrypoint, and all relevant env configs;
- local Python compile check passed for `export_nla_manifest.py`,
  `build_nla_precomputed.py`, and `experimental_build_nla_precomputed.py`;
- remote config existence check passed for all base configs used by the locked
  queue;
- remote strategy validation passed: all 12 queued token-position strategies are
  present in `experimental_build_nla_precomputed.py`;
- future ports are free before use:
  `19020/19021`, `19030/19031`, `19040/19041`, `19050/19051`,
  `19080/19081`, and `19200` through `19223`;
- required remote assets are present: `geval_gepa:latest`,
  `llama.cpp:localcuda`, Qwen2.5-7B cache, NLA AV cache, Qwen35B GGUF, and
  `tc_usr_data.json`;
- the stale Slurm follow-up queue is empty after cancellation.

D1 reached the scientifically important smoke milestones:

```text
NLA manifest rows: 36
NLA activation rows: 210
NLA precomputed feedback rows: 210
NLA coverage: 36/36 examples, useful_rows=210, coverage=1.000
perplexity feedback cache: 36/36 train/validation rows
vLLM judge readiness: passed
llama.cpp proposer readiness: passed
GEPA optimization: started, 72 metric calls / 2.00 full evals
```

This does not prove that every later experiment will finish, but it validates
the shared startup path for the Monday queue: model caches, NLA extraction,
NLA verbalization, perplexity feedback, proposer endpoint, vLLM endpoint, and
GEPA compile entrypoint.

Telegram monitoring for the direct queue is handled by:

```text
gepa-experiments/slurm/telegram_pid_monitor.py
```

It watches the direct-run PID and logs because the standard Telegram monitor is
Slurm-specific and cannot observe a direct Docker queue by job id.

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

## 2026-06-25 D1 Failure And Fix

The first locked-GPU queue `20260625T194158Z` stopped at the D1 smoke gate:

```text
status: failed
elapsed_seconds: 1388
aux_judge_ok: 25/36
aux_judge_success_rate: 0.694
threshold: 0.950
failure: Aux judge feedback success rate is below threshold
artifact: gepa-experiments/results/locked_gpu_20260625T194158Z/D1_aux_judge_fixed_smoke_ppl_nla/aux_judge_feedback_20260625T195512Z.jsonl
```

This was not a GPU, vLLM, GEPA, NLA, or perplexity failure. The failed aux-judge
rows had `content=""`, `finish_reason="length"`, and thousands of characters in
`reasoning_content`. The Qwen35B llama.cpp endpoint was spending the entire
generation budget in hidden reasoning, so the prompt proposer would not receive
visible feedback.

The fix is request-local and keeps the proposer behavior comparable:

- `AuxJudgeFeedbackProvider` now accepts `disable_thinking=True`;
- aux-judge requests then send
  `chat_template_kwargs={"enable_thinking": false}`;
- aux-judge configs set `AUX_JUDGE_DISABLE_THINKING=1`;
- `run_gepa_engaging_job.sh` passes `--aux-judge-disable-thinking` only for
  configs that enable the flag;
- the sidecar itself is not launched with server-wide `--reasoning off`, so the
  proposer path is not globally changed.

Validation before relaunch:

- local GEPA unit suite passed: `49` tests;
- direct llama.cpp endpoint test showed that `/no_think` alone still produced
  empty content, while `chat_template_kwargs.enable_thinking=false` produced
  visible content and no `reasoning_content`;
- remote preflight confirmed the flag in all aux configs and free GPUs `0`/`2`.

Relaunched locked-GPU queue:

```text
run_id: 20260625T201031Z
remote_pid: 856633
remote_root: gepa-experiments/results/locked_gpu_20260625T201031Z
launch_log: gepa-experiments/results/locked_gpu_20260625T201031Z/locked_gpu_queue_20260625T201031Z.out
telegram_monitor_pid: 858342
```

The new D1 log confirms `aux judge disable thinking: 1` inside the container.

Follow-up check:

```text
D1 replacement status: success
D1 elapsed_seconds: 1037
D1 aux_judge rows: 30
D1 aux_judge statuses: ok=30
D1 empty_content_errors: 0
D1 metrics artifact: gepa-experiments/results/locked_gpu_20260625T201031Z/D1_aux_judge_fixed_smoke_ppl_nla/metrics_20260625T202316Z.csv
D1 aux artifact: gepa-experiments/results/locked_gpu_20260625T201031Z/D1_aux_judge_fixed_smoke_ppl_nla/aux_judge_feedback_20260625T202316Z.jsonl
```

D4 long started automatically after D1:

```text
D4 start: 2026-06-25T20:27:52Z
D4 remote output: gepa-experiments/results/locked_gpu_20260625T201031Z/D4_aux_judge_fixed_long_ppl_nla
D4 current phase at 2026-06-25 22:32 CEST: NLA precompute / verbalization
D4 NLA manifest rows: 300
D4 NLA activation rows: 1752
```

The D1 smoke result is not interpreted as a scientific improvement signal: the
optimized prompt was worse on the 12-row smoke final test. Its purpose here was
startup and feedback-path validation, which it satisfied after the no-thinking
fix.

## 2026-06-26 D4 Sidecar Recovery

At 2026-06-26 09:43 CEST the D4 long run was still alive, but Telegram had
started sending many log alerts. The root cause was a llama.cpp proposer sidecar
crash, while the main GEPA/vLLM container kept running and retried proposer
calls:

```text
D4 progress at detection: about 91%
main GEPA container: alive
Qwen35B llama.cpp endpoint: down
main symptom: litellm/OpenAI connection refused during reflection/proposal
sidecar failure: GGML_ASSERT(stat == cudaSuccess) in ggml-cuda.cu
```

This failure was localized to the Qwen35B proposer/aux-judge sidecar. It was not
a vLLM judge crash and did not stop the GEPA process. GEPA skipped several
reflective mutations while the proposer endpoint was unavailable.

Recovery action:

- stopped the stalled first recovery sidecar that had bound the port but did not
  load the model;
- restarted llama.cpp on the same endpoint, `127.0.0.1:19021`, using the local
  GGUF path directly instead of the `-hf` resolver;
- reduced llama.cpp batch size from `512` to `128` for the recovery sidecar;
- confirmed `/v1/models` readiness and GPU `2` memory usage around 21 GiB;
- confirmed that D4 resumed proposer calls by logging new
  `Proposed new text for judge.predict` entries.

Pipeline mitigation for later jobs:

- the first recovery mitigation, `LLAMACPP_BATCH_SIZE=128` with flash attention
  enabled, was not sufficient: the recovery sidecar crashed again near the end
  of D4 with the same `GGML_ASSERT(stat == cudaSuccess)` failure;
- D4 was recovered a second time with `LLAMACPP_BATCH_SIZE=64` and
  `LLAMACPP_FLASH_ATTN=off`; this endpoint became ready and GEPA resumed
  `Proposed new text` entries;
- `run_docker.sh` now defaults `LLAMACPP_BATCH_SIZE` to `64` and
  `LLAMACPP_FLASH_ATTN` to `off`;
- all queued llama.cpp/Qwen35B configs now set `LLAMACPP_BATCH_SIZE=64` and
  `LLAMACPP_FLASH_ATTN=off` explicitly;
- future jobs still use the same model, context size, proposer temperature, and
  feedback settings, but with more conservative llama.cpp execution settings.
- because the D4 recovery sidecar was started manually outside the original
  `run_docker.sh` cleanup scope, a remote watcher now waits for the D4 main
  container to exit and then stops the recovery sidecar. This prevents GPU `2`
  from staying occupied and blocking the next queued job.

Telegram monitor mitigation:

- the direct PID monitor was stopped immediately to end the notification flood;
- `telegram_pid_monitor.py` now starts from the end of existing logs on restart;
- repeated alerts are grouped by error signature;
- repeated signatures have a cooldown, and each poll has a maximum alert count;
- after the first deduplicated restart still produced duplicate alerts from both
  the queue `tee` log and the per-job log, the monitor was restarted to watch
  only per-job logs, with a two-hour cooldown and at most one log alert per poll.

## 2026-06-26 D4 Completion

D4 completed successfully after the second sidecar recovery:

```text
label: D4_aux_judge_fixed_long_ppl_nla
status: END:0
elapsed_seconds: 45992
output: gepa-experiments/results/locked_gpu_20260625T201031Z/D4_aux_judge_fixed_long_ppl_nla
metrics: metrics_20260625T220326Z.csv
optimized_prompt: optimized_prompt_20260625T220326Z.txt
trajectory: prompt_trajectory_20260625T220326Z.jsonl
runtime_manifest: runtime_manifest_20260625T220326Z.json
aux_feedback: aux_judge_feedback_20260625T220326Z.jsonl
```

Final D4 metrics on the 60-row test split:

```text
baseline:  agreement=0.7583 pearson=0.7160 spearman=0.7184 kendall=0.6079 mae=0.4833
optimized: agreement=0.7889 pearson=0.6622 spearman=0.6589 kendall=0.5681 mae=0.4222
```

Interpretation for later analysis: the optimized prompt improved exact agreement
and MAE but worsened all correlation metrics. Because the proposer sidecar was
unavailable for a short interval near the end and several mutation steps were
skipped, the run should be treated as a completed but operationally caveated
aux-judge long result.

The queue then advanced to `D2_matched_no_aux_smoke_ppl_nla`. At the 2026-06-26
11:19 CEST check, D2 was in NLA verbalization, its llama.cpp sidecar was running
with `LLAMACPP_BATCH_SIZE=64` and `LLAMACPP_FLASH_ATTN=off`, and no new startup
error was visible.

## 2026-06-26 Follow-Up Queue To Avoid Idle Locked GPUs

The current locked-GPU queue is not expected to keep the reserved GPUs busy
until Monday 2026-06-29 midday by itself. It still contains useful work
(`D2`, `D3`, `A1`, the NLA token-position sweep, and `D5`), but this is likely
shorter than the available reservation window.

A second direct queue has therefore been prepared in
`gepa-experiments/slurm/run_locked_gpu_followup_queue.sh`. It does not alter or
interrupt the running queue. It is intended to be launched as a waiting wrapper:
it waits for the current queue PID to exit, then starts the follow-up jobs on
the same locked GPU devices.

Follow-up priority order:

1. `F1_clean_aux_long_ppl_nla_seed42`: clean rerun of the aux-judge + PPL + NLA
   long branch after the llama.cpp sidecar hardening. This removes the D4 caveat
   caused by endpoint outages near the end of optimization.
2. `F2_clean_aux_long_ppl_nla_seed43`: same branch with a different seed, to
   give a robustness point for the thesis while keeping the setting unchanged.
3. `F3_summeval_consistency_real_nla_smoke`: Summeval/Consistency smoke for the
   G-Eval matrix extension.
4. `F4_qags_cnn_consistency_real_nla_smoke`: QAGS-CNN/Consistency smoke.
5. `F5_qags_xsum_consistency_real_nla_smoke`: QAGS-XSum/Consistency smoke.
6. `F6_candidate_content_10_strategy_probe`: NLA token-selection probe for the
   candidate-content extraction strategy.
7. `F7_hybrid_context_dedup_6_strategy_probe`: NLA token-selection probe for a
   hybrid/deduplicated extraction strategy.
8. `F8_clean_ppl_only_control_long`: lower-priority PPL-only long control if
   time remains.

All follow-up llama.cpp jobs inherit the safer settings now used by the current
queue:

```text
LLAMACPP_BATCH_SIZE=64
LLAMACPP_FLASH_ATTN=off
GPU_DEVICE=0,2
```

Rationale: F1/F2 are the most useful thesis jobs because they directly address
whether the aux-judge/NLA branch has a stable effect once the proposer sidecar
is not failing. F3-F5 reduce the risk that the dataset-matrix extension remains
untested. F6/F7 preserve the NLA-token-selection diagnostic thread. F8 is last
because a PPL-only control already exists historically, so it is useful but not
as urgent as the NLA/aux-judge branches.

Launch status:

```text
follow-up run id: 20260626Tfollowup02
follow-up root: gepa-experiments/results/locked_gpu_followup_20260626Tfollowup02
waiter pid: 3028011
telegram monitor pid: 3033550
wait condition: current locked queue pid 856633 exits
```

The follow-up waiter and Telegram PID/log monitor were verified alive at
2026-06-26 12:04 CEST. The monitor uses the same conservative settings as the
current queue monitor: 300-second polling, two-hour duplicate-signature
cooldown, and at most one log alert per poll.

Current queue progress after follow-up launch:

```text
D2_matched_no_aux_smoke_ppl_nla: END:0:1891
D3_aux_judge_only_smoke_ppl_aux_no_nla: END:0:1053
A1_nla_strategy_wiring_probe: START at 2026-06-26T10:03:28Z
```

D3 initially looked slow because its main log stayed on `Waiting for vLLM
readiness`, but the wrapper completed normally and advanced to A1. No manual
intervention was needed.

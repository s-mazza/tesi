# GEPA / G-Eval Current Plan Status

Last updated: 2026-06-10

## Objective

Run GEPA experiments on the G-Eval benchmark settings with paper-aligned datasets, dimensions, and metrics, then isolate whether NLA feedback improves GEPA prompt optimization or why it fails to do so.

## Paper-Aligned Targets

The full scientific matrix must cover all G-Eval dimensions, not only one representative metric/dimension:

- SummEval: `fluency`, `coherence`, `consistency`, `relevance`
- Topical-Chat: `naturalness`, `coherence`, `engagingness`, `groundedness`
- QAGS-CNN: `consistency`
- QAGS-XSUM: `consistency`

Final comparison metrics:

- SummEval: `spearman`, `kendall_tau`
- Topical-Chat: `pearson`, `spearman`
- QAGS-CNN/QAGS-XSUM: `pearson`, `spearman`, `kendall_tau`

Internal diagnostics such as `agreement`, `mae`, `coverage`, and `parsed` remain useful for debugging but are not the primary paper comparison metrics.

## Experiment Variants

For each dataset/dimension target, run:

- `base_gepa`: GEPA with metric feedback only
- `ppl`: GEPA with response perplexity feedback from the base Qwen 7B model
- `ppl_nla`: GEPA with perplexity plus NLA verbalization feedback from the base Qwen 7B model
- `ppl_nla_auxjudge`: GEPA with perplexity, NLA, and optional 35B auxiliary judge feedback

The 35B llama.cpp model is used as GEPA proposer. The NLA and perplexity signals are computed on the base Qwen 7B model.

Auxiliary judge meaning:

- The base trained/evaluated judge remains Qwen2.5-7B-Instruct.
- Perplexity is computed on Qwen2.5-7B-Instruct.
- NLA verbalizes activations from Qwen2.5-7B-Instruct.
- The proposer remains Qwen 35B via llama.cpp.
- When enabled, the auxiliary LLM-as-a-judge feedback is also produced by Qwen 35B via llama.cpp and is passed only as extra feedback to GEPA's proposer/reflection loop.
- The auxiliary judge does not replace the base 7B judge and does not directly define final paper metrics.

## NLA Root-Cause Findings

Before scaling the full matrix, the long NLA run was diagnosed against the closest long non-NLA control. The intended control differs only by enabling NLA:

- same dataset and dimension
- same seed
- same train/validation/test context counts
- same proposer model and proposer settings
- same GEPA budget
- same perplexity feedback
- same generalizing instruction proposer

Required artifacts for both runs:

- `metrics_*.csv`
- `run_config_*.json`
- `baseline_predictions_*.jsonl`
- `optimized_predictions_*.jsonl`
- `seed_prompt_*.txt`
- `optimized_prompt_*.txt`
- `prompt_trajectory_*.jsonl`
- `gepa_viz_run_*.json`
- `runtime_manifest_*.json`
- Slurm, vLLM, llama.cpp, GEPA, and NLA precompute logs

Extra required artifact for NLA runs:

- `nla_verbalizations_*.jsonl`

The diagnostic report answered:

- The comparison was 1-to-1 after normalizing legacy artifact keys, except for NLA-specific fields.
- NLA covered the GEPA train/validation examples.
- NLA worsened final-test metrics and prediction-level errors.
- The failure is most likely caused by weak token selection and noisy/repetitive verbalizations, not by a definitive failure of the NLA idea.
- The old selector used first semantic tokens from source/candidate/reference when the budget was small.
- In the long run this produced weak tokens such as `reading`, `recently`, and `from`.
- This means the run tested GEPA with noisy first-token NLA feedback, not a strong NLA condition.

Use `gepa-experiments/scripts/diagnose_nla_run.py` for this comparison.

## Implementation Status

Completed or in progress locally:

- Added paper-needed `kendall_tau` metric support.
- Added `kendall_tau` aggregation support.
- Added NLA precomputed coverage and useful-row validation before GEPA compile.
- Added dry-run NLA guard for scientific runs.
- Added pass-through Slurm variables for NLA minimum coverage and dry-run override.
- Added reusable NLA-vs-control diagnostic script.
- Generated first long-run NLA diagnostic report:
  - `gepa-experiments/results/diagnostics/nla_vs_ppl_long_20260608.md`
  - comparison: long PPL-only llama.cpp proposer run vs long PPL+NLA llama.cpp proposer run
  - normalized config check is 1-to-1 except NLA-specific fields
  - optimized Pearson dropped by 0.121668 with NLA
  - optimized Spearman dropped by 0.129078 with NLA
  - optimized MAE worsened by 0.194444 with NLA
  - prediction-level test movement: 6 examples improved, 18 worsened, 36 unchanged
  - NLA quality signal: 900 rows, 300 covered examples, all emitted parse statuses are `partial_tags`, and 667 rows are duplicate repeated text rows
  - Important clarification: the first diagnostic artifact showed `token_status=unknown` because the runner-emitted NLA artifact did not preserve `token_status`; the raw precompute file had `900/900 token_status=ok`
- Completed root-cause analysis:
  - `gepa-experiments/results/diagnostics/nla_root_cause_20260608.md`
  - root cause: old NLA token selection used weak first semantic tokens from source/candidate/reference when budget was small
  - this produced repeated, generic verbalizations and tested noisy NLA feedback rather than a strong NLA condition
  - implemented candidate-prioritized middle/final token selection
  - preserved `token_status` in emitted NLA artifacts
  - increased real-NLA token and generation budgets
  - removed the old `NLA_PRECOMPUTE_LIMIT=8` from the fixed-NLA smoke config so the coverage gate can validate full GEPA train/validation coverage

Still required:

- Wait for the queued fixed-NLA diagnostics to finish.
- Pull artifacts for `11912918`; artifacts for `11913161` were recovered locally from `faretra`.
- Run `diagnose_nla_run.py` on the PPL-only smoke control `11913161` vs fixed-NLA smoke `11912918`.
- Inspect recovered 1-GPU smoke artifacts for NLA feedback health: coverage, `token_status`, parse status, duplicate verbalizations, and whether the fixed selector is producing candidate-relevant verbalizations.
- Check whether fixed NLA improves verbalization quality before launching another long NLA run.
- If fixed smoke looks better, launch a longer Topical-Chat engagingness PPL+fixed-NLA run.
- If fixed smoke still fails, isolate candidate-only NLA, larger token budgets, lower proposer temperature, and shorter/summarized NLA feedback.
- For any future job that can run on non-`faretra` nodes, check the execution node filesystem or sync artifacts back to `faretra`; `moro232` results are not guaranteed to be visible from `faretra`.
- Queue scientifically useful pending work even when nodes are currently occupied, so jobs accrue scheduling age. Avoid only duplicate jobs or jobs that would answer no useful question.

## Future Step Roadmap

Step 1: finish and analyze smoke diagnostics.

- Completed artifact recovery for `11912914`, `11912915`, `11912916`, `11913111`, `11913112`, and `11913113`; use these artifacts to verify real-NLA precompute and runner behavior on SummEval, QAGS-CNN, and QAGS-XSUM.
- Use queued 1-GPU Topical-Chat jobs `11913130` and `11913131` as an additional matched smoke comparison while waiting for `faretra`.
- Wait for `11912918` to compare Topical-Chat PPL-only vs fixed-NLA on the original engagingness task.
- Generate a diagnostic report for `11913161` vs `11912918`.
- Decision gate: continue with fixed-NLA only if artifacts show healthier NLA feedback than the first long NLA run.

Step 2: launch one longer fixed-NLA Topical-Chat run.

- Dataset/dimension: Topical-Chat engagingness.
- Compare against the closest PPL-only control with same seed, split, proposer, and GEPA budget.
- Use fixed candidate-prioritized NLA token selection.
- Keep Qwen 35B as proposer.
- Do not enable auxiliary judge yet.
- Goal: verify whether fixed-NLA alone can recover or improve over the previous PPL-only result.

Step 3: add Qwen 35B auxiliary LLM-as-a-judge feedback.

- Run a matched Topical-Chat engagingness ablation:
  - `ppl`
  - `ppl_nla`
  - `ppl_nla_auxjudge`
- Keep the same seed, data split, GEPA budget, proposer model, and base 7B judge.
- Only the third variant enables Qwen 35B LLM-as-a-judge feedback.
- Goal: isolate whether auxiliary judge feedback adds useful semantic feedback beyond perplexity and NLA.

Step 4: if Topical-Chat diagnostics are positive, scale to paper-aligned dimensions.

- Start with one dimension per dataset family:
  - SummEval consistency
  - Topical-Chat engagingness
  - QAGS-CNN consistency
  - QAGS-XSUM consistency
- Then expand to all G-Eval dimensions listed above.
- For each target, preserve all prompt artifacts, NLA artifacts, diagnostic reports, and paper-aligned metrics.

Step 5: if NLA remains negative after the fixed selector.

- Keep all creative NLA strategy experiments separate from the current main pipeline.
- Do not change the default NLA selector, runner behavior, or production configs for these experiments.
- Separate experimental scripts/configs are now available for candidate-content token strategies:
  - `gepa-experiments/scripts/experimental_nla_token_strategy_analysis.py`
  - `gepa-experiments/scripts/experimental_build_nla_precomputed.py`
  - `gepa-experiments/slurm/run_experimental_nla_strategy_job.sh`
  - `gepa-experiments/slurm/submit_experimental_nla_strategy.sh`
- Initial CPU-only token strategy analysis is saved in:
  - `gepa-experiments/results/diagnostics/experimental_nla_token_strategies_20260608/token_strategy_report.md`
  - `gepa-experiments/results/diagnostics/experimental_nla_token_strategies_20260608/token_strategy_summary.csv`
- The two best isolated strategies from the CPU analysis are:
  - `candidate_content_6`: candidate-only, weak token rows 0.00%, duplicate row pct 35.58%.
  - `candidate_content_10`: candidate-only, weak token rows 0.00%, duplicate row pct 34.89%.
- `candidate_source_content_8` remains a secondary cross-check, but it is not first priority because it has higher duplicate row pct 51.76%.
- Run candidate-content NLA as separate smoke experiments before considering any merge into the main pipeline.
- Increase `NLA_MAX_TOKENS_PER_EXAMPLE` to 8 or 10 only in separate experimental configs.
- Lower proposer temperature to 0.2 or 0.0 only in matched experimental controls.
- Test summarized/compressed NLA feedback before giving it to the proposer, but keep it isolated from the current runner path unless it clearly wins.
- Test whether different NLA layers/checkpoints are available and compatible with Qwen2.5-7B.
- Merge an experimental NLA strategy into the main pipeline only after it improves the task under a fair 1-to-1 comparison with enough evidence to remove reasonable doubt.
- Only after these ablations decide whether NLA should be reported as negative/diagnostic rather than performance-improving.

Experimental NLA strategy rules:

- Every experimental strategy must keep the same dataset, dimension, seed, split, base 7B judge, proposer 35B, GEPA budget, and final metric computation as its matched control.
- Each experiment must have a paired PPL-only or current fixed-NLA control.
- Each experiment must save a diagnostic report with `diagnose_nla_run.py`.
- Each experiment must report both final metrics and feedback-health metrics: token positions, duplicate verbalizations, parse status, token status, and prediction-level error movement.
- A strategy is not allowed to replace the current pipeline just because it improves one smoke run; it needs at least one longer matched Topical-Chat run and one cross-dataset smoke before merge.

## Current Cluster Queue

The old pinned jobs `11912818`, `11912819`, and `11912820` were cancelled because they had `ReqNodeList=moro232`.

Replacement chain submitted without a node pin, keeping `ExcNodeList=deeplearn2`:

- `11912914`: SummEval consistency smoke, PPL + real NLA
- `11912915`: QAGS-CNN consistency smoke, PPL + real NLA, dependency `afterany:11912914`
- `11912916`: QAGS-XSUM consistency smoke, PPL + real NLA, dependency `afterany:11912915`

Latest status:

- `11912914`, `11912915`, and `11912916` were initially missing from `faretra` checks because they wrote artifacts on `moro232`'s local filesystem. Their artifacts were recovered together with `11913111`, `11913112`, and `11913113`.
- `sacct` is currently unusable because SlurmDB returns `Connection refused`; use node-local artifacts plus `/var/log/slurm/jobcomp.log` when old jobs age out of `scontrol`.
- The recovered 2026-06-08 and 2026-06-09 dataset smoke runs are scientifically usable only as small runner/NLA sanity checks, not as final paper-level comparisons.
- Re-submission on 2026-06-10 01:02 CEST to exploit the idle single-GPU `moro232` node without occupying `faretra`:
  - `11913111`: SummEval consistency smoke, PPL + real NLA, pinned to `moro232`.
  - `11913112`: QAGS-CNN consistency smoke, PPL + real NLA, pinned to `moro232`, dependency `afterany:11913111`.
  - `11913113`: QAGS-XSUM consistency smoke, PPL + real NLA, pinned to `moro232`, dependency `afterany:11913112`.
- This pinning is intentional and limited to 1-GPU dataset smoke jobs. It should not be used for Qwen35B proposer jobs, which need 2 GPUs on the same node.
- Latest check: 2026-06-10 10:38 CEST. Jobs `11913111`, `11913112`, and `11913113` are no longer visible in `squeue` or `scontrol` because `MinJobAge=300 sec` and SlurmDB accounting is down. The text job completion log shows they did run on `moro232` and completed with exit code `0:0`:
  - `11913111`: start 2026-06-10 01:01:56, end 01:27:28.
  - `11913112`: start 2026-06-10 01:27:28, end 01:35:07.
  - `11913113`: start 2026-06-10 01:35:07, end 01:41:47.
- Final diagnosis: the jobs wrote their stdout and result artifacts to the local filesystem visible on `moro232`, not to the filesystem visible from `faretra`. Direct SSH from the local machine to `moro232` works and confirmed the artifacts.
- Recovered and synchronized artifacts from `moro232` to local workspace and then to `faretra`, preserving the expected experiment directories.
- Extra diagnostic jobs `11913124` and `11913129` were cancelled because direct SSH recovery made them unnecessary.
- Important operational consequence: when jobs run on a node whose home/workdir is not shared with `faretra`, result checks must query that node or explicitly sync artifacts back to `faretra`.
- Recovered metrics:
  - `11913111` SummEval consistency smoke: baseline Pearson 0.659360, Spearman 0.615213, MAE 1.375; optimized Pearson 0.618512, Spearman 0.696582, MAE 1.125.
  - `11913112` QAGS-CNN consistency smoke: baseline and optimized both Pearson 1.0, Spearman 1.0, MAE 1.055556 on `n=2`.
  - `11913113` QAGS-XSUM consistency smoke: baseline and optimized both Pearson 0.0, Spearman 0.0, MAE 1.333333 on `n=2`.

Additional Topical-Chat diagnostic chain submitted after the dataset smoke chain:

- `11912917`: failed Topical-Chat engagingness smoke, PPL-only control, llama.cpp proposer
- `11913161`: replacement Topical-Chat engagingness smoke, PPL-only control, llama.cpp proposer, fixed ports
- `11912918`: Topical-Chat engagingness smoke, PPL + fixed NLA, llama.cpp proposer

These are intended to validate whether the NLA token-selection fix improves verbalization quality before launching another long NLA run.

Latest status:

- `11912917`: failed on `faretra` because llama.cpp tried to bind occupied port `127.0.0.1:8080`.
- `11913161`: replacement PPL-only control completed on `faretra`; artifacts were recovered to the local workspace from `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke`.
- `11912918`: dependency on `11913161` is satisfied and the job is pending for 2 x RTX 3090 availability.
- Latest check: 2026-06-10 13:58 CEST.
- Reason: `11913161` and the downstream Qwen35B proposer jobs need 2 x RTX 3090 on the same node. `moro232` has only 1 x RTX 3090, so `faretra` remains the only eligible node.
- `faretra` currently has 3 of 4 GPUs allocated, so only 1 GPU is free and `11912918` cannot start yet. Slurm backfill currently estimates `11912918` around 2026-06-11 13:35 CEST, but this is an unstable estimate.
- No final result artifact for jobs `11912918`, `11912947`, or `11912948` is visible yet.
- Do not submit duplicate matched Qwen35B proposer jobs while `11912918` is pending. Do submit other scientifically useful queued work if it answers a distinct question and can age in the queue.

`11913161` PPL-only smoke result:

- Dataset/dimension: Topical-Chat engagingness.
- Split: 24 GEPA train examples, 12 GEPA validation examples, 12 final-test examples.
- Feedback: perplexity only, no NLA, no auxiliary judge.
- Proposer: Qwen 35B via llama.cpp on `127.0.0.1:18144`.
- Baseline final-test metrics: Pearson 0.603136, Spearman 0.590879, Kendall tau 0.531588, MAE 0.555556, agreement 0.722222.
- Optimized final-test metrics: Pearson 0.536400, Spearman 0.527410, Kendall tau 0.459933, MAE 0.722222, agreement 0.638889.
- Interpretation: this small smoke control validates the fixed-port and artifact path, but the optimized prompt overfit or degraded final-test behavior on the 12-example test slice. It remains useful as the matched control for `11912918`, not as evidence that GEPA improves this setting.

Additional 1-GPU queue-aging jobs submitted on 2026-06-10:

- `11913130`: Topical-Chat engagingness PPL-only single-GPU smoke, pinned to `moro232`, 4 hour limit.
- `11913131`: Topical-Chat engagingness PPL + real-NLA single-GPU smoke, pinned to `moro232`, dependency `afterany:11913130`, 4 hour limit.
- These jobs are not the main Qwen35B proposer comparison, but they are useful while waiting for `faretra`: they provide a matched 1-GPU PPL-only vs real-NLA smoke and accumulate queue priority for `moro232`.
- Because `moro232` has node-local artifacts, check or sync results from `moro232` when these complete.

Failure and recovery action on 2026-06-10:

- `11912917` started on `faretra` at 11:32 CEST and failed after 3 seconds with exit code `125`.
- Cause: llama.cpp sidecar attempted to bind `127.0.0.1:8080`, which was already in use on `faretra`.
- Immediate action: held `11912918`, `11912947`, and `11912948` so the treatment and experimental NLA jobs cannot run without a valid PPL-only control.
- Config fix: changed `geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke.env` from judge/proposer ports `8000/8080` to `18143/18144` and added explicit `SLURM_TIME=04:00:00`.
- Relaunched PPL-only control as `11913161`, then set `11912918` to depend on it with `afterok`.
- Dependency policy changed for the remaining chain: use `afterok` instead of `afterany` so failed controls/treatments do not trigger downstream comparison jobs.
- Prevention fix: `gepa-experiments/slurm/run_docker.sh` now checks the llama.cpp proposer host port before launching the sidecar. If the configured port is busy and `PROPOSER_API_BASE` was not explicitly set, it automatically selects the next free port and passes that endpoint to the main GEPA container.
- Additional config hygiene: old 2-GPU llama.cpp config `geval_gepa_engaging_qwen25_2h_ppl_llamacpp35b_proposer.env` no longer uses risky default ports `8000/8080`; it now uses `18153/18154`.

Additional isolated experimental strategy jobs to queue after `11912918`:

- Topical-Chat engagingness smoke, PPL + experimental `candidate_content_6` NLA, llama.cpp proposer.
- Topical-Chat engagingness smoke, PPL + experimental `candidate_content_10` NLA, llama.cpp proposer.

These jobs must remain outside the main pipeline. They answer only whether alternate NLA token selection makes the feedback condition healthier and whether that translates to better GEPA behavior on a matched smoke setting.

Submission status:

- Local implementation and validation completed.
- Scripts/configs/status and token-strategy diagnostic report/CSV are synced to `faretra`.
- `11912947`: Topical-Chat engagingness smoke, PPL + experimental `candidate_content_6` NLA, llama.cpp proposer, dependency `afterok:11912918`.
- `11912948`: Topical-Chat engagingness smoke, PPL + experimental `candidate_content_10` NLA, llama.cpp proposer, dependency `afterok:11912947`.
- These jobs are intentionally serial and outside the main pipeline.
- Single-GPU Topical-Chat smoke work is now queued as `11913130` and `11913131` to exploit queue aging on `moro232`. It remains a secondary smoke comparison and does not replace the Qwen35B proposer chain.

## Cluster Scheduling Rule

Queue scientifically useful pending work even when the target node is currently occupied, so jobs accrue Slurm priority aging.

Use flexible node selection by default. Pin to `moro232` only when there is a concrete reason, such as:

- the job needs exactly one RTX 3090 and should not compete with the 2-GPU `faretra` Qwen35B proposer chain
- the job is recovering or checking node-local artifacts written on `moro232`
- the experiment is explicitly a 1-GPU smoke/control

Keep:

- `SLURM_EXCLUDE=deeplearn2`
- two GPUs for jobs using llama.cpp proposer sidecar
- serial dependencies for heavy jobs unless explicitly testing independent short smoke runs

Submit-script guard added:

- `gepa-experiments/slurm/submit_gepa_engaging.sh` and `gepa-experiments/slurm/submit_experimental_nla_strategy.sh` now print the scheduling mode before submission.
- `SLURM_NODE=auto` is treated as no pin, so flexible jobs can use any eligible node.
- If a job is pinned, the submit script checks the target node GPU capacity via `scontrol show node`.
- The submit fails immediately if the requested GPU count is larger than the pinned node capacity, preventing mistakes such as a 2-GPU Qwen35B proposer job pinned to single-GPU `moro232`.
- The submit also fails if `PROPOSER_BACKEND=llamacpp` is configured with fewer than 2 GPUs.
- This does not make a 2-GPU job runnable on `moro232`; it only prevents invalid submissions and makes the scheduler constraint explicit at submit time.

## Pre-Submit Checklist

Before submitting any new job, complete this checklist. Do not rely only on the submit script guards; they catch scheduling mistakes, not scientific or artifact mistakes.

Scientific purpose:

- Confirm the job answers a distinct question already in this plan, or update the plan before submitting it.
- Confirm the matched control exists, is queued, or is submitted in the same action.
- Confirm the job is not a duplicate of a queued, running, or completed artifact unless the previous job failed or is explicitly being rerun.
- Confirm whether the job is main-pipeline evidence, secondary smoke evidence, or isolated experimental NLA evidence.
- Confirm the final comparison metrics are paper-aligned for that dataset/dimension.
- Confirm GEPA optimization uses validation examples, not final test examples.
- Confirm final test is used only after optimization.

Config file:

- Confirm `CONFIG_FILE` exists locally and on the target cluster path.
- Confirm `DATASET`, `DIMENSION`, `LABEL`, `SEED`, `TRAIN_CONTEXTS`, `VAL_CONTEXTS`, and `TEST_CONTEXTS` are intentional.
- Confirm split sizes are large enough for the purpose: smoke, diagnostic, long run, or thesis-level comparison.
- Confirm `OUTPUT_DIR` is unique or intentionally reuses a directory with timestamped artifacts.
- Confirm `SLURM_TIME` is set; smoke jobs must not appear as `UNLIMITED`.
- Confirm `SLURM_MEM` is set and consistent with vLLM, NLA, and llama.cpp memory needs.
- Confirm `JOB_SLUG` is unique enough that Slurm stdout files are readable and not ambiguous.
- Run shell syntax validation on changed Slurm scripts before submission.

Scheduling:

- Inspect current queue with `squeue` and current node state with `scontrol show node`.
- Confirm requested GPU count is compatible with the selected backend:
  - llama.cpp proposer sidecar requires 2 GPUs on one node.
  - single-GPU smoke jobs must not pretend to be matched Qwen35B proposer comparisons.
- Confirm node pinning is intentional:
  - use flexible scheduling by default.
  - pin to `moro232` only for 1-GPU smoke/control jobs or node-local artifact work.
  - do not pin 2-GPU Qwen35B proposer jobs to `moro232`.
- Keep `SLURM_EXCLUDE=deeplearn2` unless there is an explicit reason to change it.
- Confirm dependencies use `afterok` for scientific chains where downstream jobs require a valid upstream result.
- Use `afterany` only for cleanup/recovery jobs or cases where downstream execution is valid after failure.
- Confirm held jobs are intentionally held; release only after dependencies and configs are corrected.
- Queue useful pending work even when nodes are occupied, but do not queue duplicate jobs that answer the same question.

Ports and networking:

- Confirm vLLM `SERVER_PORT` is not an old risky default unless intentionally isolated.
- Confirm llama.cpp `PROPOSER_PORT` is not `8080` unless there is a specific reason.
- Confirm `run_docker.sh` port auto-selection is synced on the cluster before submitting llama.cpp proposer jobs.
- If `PROPOSER_API_BASE` is explicitly set, confirm it matches `PROPOSER_PORT`; otherwise the auto-port fallback cannot safely rewrite it.
- Confirm the expected proposer endpoint is printed in the Slurm log at startup.

Model/cache/container readiness:

- Confirm judge model cache exists under `/llms/hub/...` on the execution node.
- Confirm NLA checkpoint cache exists when NLA is enabled.
- Confirm llama.cpp GGUF exists under `/llms/llamacpp-cache` for Qwen35B proposer jobs.
- Confirm the Docker image expected by `IMAGE_NAME` exists on the node or can be pulled/built intentionally.
- Confirm vLLM/Flash Attention compatibility has not changed for the active Docker image.
- Confirm no stale container with the same `CONTAINER_NAME` or `SIDECAR_NAME` can block startup; names include the Slurm job id by default.

Data and split integrity:

- Confirm data source exists on the execution node.
- Confirm preflight checks pass for dataset, split counts, context disjointness, response ids, model cache, and NLA cache.
- For Topical-Chat, confirm the run uses the intended USR data source and dimension.
- For SummEval/QAGS, confirm the run uses the intended paper-aligned dataset/dimension and not a legacy Topical-Chat label path.

NLA/perplexity feedback:

- If `NLA_FEEDBACK=1`, confirm whether this is real NLA or dry-run NLA.
- Dry-run NLA is allowed only for plumbing tests, not scientific comparison.
- If real NLA is enabled, confirm `NLA_BACKEND=precomputed`, `NLA_PRECOMPUTED_AUTO=1`, token budget, layer, dtypes, and checkpoint.
- Confirm `NLA_PRECOMPUTE_LIMIT` is intentional; do not accidentally cap scientific runs below train/validation coverage.
- Confirm `NLA_MIN_COVERAGE` is strict enough for the run purpose.
- Confirm perplexity feedback is computed on the base Qwen2.5-7B judge model, not on the 35B proposer.

Artifacts and monitoring:

- Confirm Slurm stdout path is under `gepa-experiments/results/slurm`.
- Confirm Telegram monitor credentials exist if monitor is enabled.
- Confirm monitor log and pid paths will be unique for the job id.
- If the job can run on `moro232`, plan artifact retrieval from `moro232` because results may be node-local.
- After submission, record the job id, dependency chain, config path, output dir, node pin/exclude, and expected comparison in this file.
- After completion, sync artifacts from the execution node to local workspace and to `faretra` if needed.
- After completion, verify required artifacts: metrics, run config, baseline and optimized predictions, seed and optimized prompts, prompt trajectory, GEPA viz, runtime manifest, Slurm log, vLLM log, llama.cpp log when applicable, and NLA verbalizations when applicable.

Post-submit sanity:

- Immediately inspect `scontrol show job` for the submitted job.
- Confirm `TimeLimit`, `ReqNodeList`, `ExcNodeList`, `TRES`, `Dependency`, `StdOut`, and `WorkDir`.
- If any field is wrong, fix it with `scontrol update` before the job starts, or cancel and resubmit.
- If the job starts and fails before model loading, inspect Slurm stdout first; common startup failures include occupied ports, missing image, missing model cache, wrong node filesystem, or bad dependency/config path.
- If a control job fails, hold or rewire downstream treatment jobs before they start.

## SSH / IPS Mitigation

The UniBo network can quarantine a client when it detects an IPS-like pattern. In this project the risky pattern is many short-lived SSH/SCP/rsync connections from Codex, especially parallel checks and aborted retries. The mitigation is to keep command throughput high while reducing new TCP/SSH handshakes.

Current local SSH configuration for `faretra` and `moro232` uses:

- `ControlMaster auto`
- `ControlPersist 30m`
- `ControlPath ~/.ssh/cm-%C`
- public-key-only batch authentication
- short connect timeout and one connection attempt

Operational rule:

- Prefer one persistent master connection to `faretra`.
- Use `faretra` as the main Slurm entrypoint.
- Direct SSH to `moro232` is appropriate for node-local artifact inspection and sync, because `moro232` result files may not be visible from `faretra`.
- Do not run parallel local SSH status checks.
- Use `gepa-experiments/slurm/cluster_status_snapshot.sh` for status: it collects queue, node state, job state, log tails, artifacts, and monitor logs in one remote command.
- When a job seems to disappear without artifacts, first check the execution node filesystem directly or sync from that node. Use `gepa-experiments/slurm/submit_slurm_stdout_smoke.sh` only if direct node checks do not explain the missing artifacts.

## Acceptance Criteria

NLA can be considered thesis-ready only if at least one of these is true:

- It improves paper-aligned metrics under a fair 1-to-1 comparison.
- Or, if it does not improve, we have a reproducible diagnostic report explaining the bottleneck and next implementation step.

Before launching another long fixed-NLA run, the fixed-NLA smoke comparison must show that the feedback condition itself is healthier:

- NLA emitted artifacts preserve `token_status=ok`.
- Verbalizations are not dominated by first-token source/reference activations.
- Duplicate verbalization rows are substantially lower than the first long NLA run.
- `partial_tags` are reduced, or at least the unclosed tag behavior is confirmed not to truncate semantic content.
- The PPL-only smoke and fixed-NLA smoke are compared with `diagnose_nla_run.py`.

Before launching `ppl_nla_auxjudge`, the fixed-NLA run should have a valid diagnostic report. The auxiliary judge experiment should answer a separate question: whether Qwen 35B feedback helps the proposer use NLA/perplexity signals better, not whether the base judge model itself changes.

The full experiment matrix is considered scientifically usable only when every dataset/dimension target has:

- baseline/seed evaluation
- optimized evaluation
- paper-aligned metrics
- config artifact
- prompt artifacts
- runtime artifact
- prompt trajectory artifact

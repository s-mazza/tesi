# GEPA / G-Eval Current Plan Status

Last updated: 2026-06-08

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
- Pull artifacts for `11912917` and `11912918`.
- Run `diagnose_nla_run.py` on the PPL-only smoke control vs fixed-NLA smoke.
- Check whether fixed NLA improves verbalization quality before launching another long NLA run.
- If fixed smoke looks better, launch a longer Topical-Chat engagingness PPL+fixed-NLA run.
- If fixed smoke still fails, isolate candidate-only NLA, larger token budgets, lower proposer temperature, and shorter/summarized NLA feedback.

## Future Step Roadmap

Step 1: finish currently queued smoke diagnostics.

- Wait for `11912914`, `11912915`, and `11912916` to verify real-NLA precompute and runner behavior on SummEval, QAGS-CNN, and QAGS-XSUM.
- Wait for `11912917` and `11912918` to compare Topical-Chat PPL-only vs fixed-NLA on the original engagingness task.
- Generate a diagnostic report for `11912917` vs `11912918`.
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

- `11912914`, `11912915`, and `11912916` are no longer visible in `squeue` or `scontrol`.
- `sacct` is currently unusable because SlurmDB returns `Connection refused`.
- No Slurm stdout files or new result artifacts with job ids `11912914`, `11912915`, or `11912916` are visible under `gepa-experiments/results`.
- Therefore these three dataset smoke jobs must not be counted as scientifically completed yet.
- Next action: once SlurmDB/logging is stable, either recover their status from accounting or re-submit equivalent smoke jobs with confirmed stdout/artifact creation.

Additional Topical-Chat diagnostic chain submitted after the dataset smoke chain:

- `11912917`: Topical-Chat engagingness smoke, PPL-only control, llama.cpp proposer
- `11912918`: Topical-Chat engagingness smoke, PPL + fixed NLA, llama.cpp proposer

These are intended to validate whether the NLA token-selection fix improves verbalization quality before launching another long NLA run.

Latest status:

- `11912917`: pending, dependency satisfied, but blocked by `ReqNodeNotAvail,_UnavailableNodes:faretra`.
- `11912918`: pending on dependency `afterany:11912917`.
- Latest check: 2026-06-09 09:50 CEST.
- Reason: `11912917` needs 2 x RTX 3090 for vLLM judge plus llama.cpp Qwen35B proposer. `faretra` has 4 x RTX 3090 but all 4 are currently allocated; `moro232` has only 1 x RTX 3090 and is currently allocated, so it cannot run this 2-GPU job.
- No new Slurm stdout or result artifact for jobs `11912917`, `11912918`, `11912947`, or `11912948` is visible yet.
- Do not submit more matched Qwen35B proposer jobs until `11912917` starts or `faretra` frees GPUs, because they would queue behind the same 2-GPU bottleneck.

Additional isolated experimental strategy jobs to queue after `11912918`:

- Topical-Chat engagingness smoke, PPL + experimental `candidate_content_6` NLA, llama.cpp proposer.
- Topical-Chat engagingness smoke, PPL + experimental `candidate_content_10` NLA, llama.cpp proposer.

These jobs must remain outside the main pipeline. They answer only whether alternate NLA token selection makes the feedback condition healthier and whether that translates to better GEPA behavior on a matched smoke setting.

Submission status:

- Local implementation and validation completed.
- Scripts/configs/status and token-strategy diagnostic report/CSV are synced to `faretra`.
- `11912947`: Topical-Chat engagingness smoke, PPL + experimental `candidate_content_6` NLA, llama.cpp proposer, dependency `afterany:11912918`.
- `11912948`: Topical-Chat engagingness smoke, PPL + experimental `candidate_content_10` NLA, llama.cpp proposer, dependency `afterany:11912947`.
- These jobs are intentionally serial and outside the main pipeline.
- A single-GPU smoke config exists, but it is not a matched scientific comparison for Qwen35B proposer runs. It was not launched because it would not answer the current NLA-vs-control question and `moro232` has only one GPU and low real free memory.
- Latest check: `moro232` is now allocated, so even non-comparable single-GPU smoke work should not be queued opportunistically right now.

## Cluster Scheduling Rule

Do not pin jobs to `moro232` unless there is a concrete node-specific reason. Prefer allowing Slurm to choose either `faretra` or `moro232` to exploit the first available compatible GPU.

Keep:

- `SLURM_EXCLUDE=deeplearn2`
- two GPUs for jobs using llama.cpp proposer sidecar
- serial dependencies for heavy jobs unless explicitly testing independent short smoke runs

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

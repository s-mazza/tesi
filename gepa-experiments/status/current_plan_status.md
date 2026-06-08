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

## NLA Root-Cause Plan

Before scaling the full matrix, diagnose the long NLA run against the closest long non-NLA control. The intended control differs only by enabling NLA:

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

The diagnostic report must answer:

- Was the comparison truly 1-to-1?
- Did NLA cover all GEPA train/validation examples?
- Were verbalizations non-empty, parsed, non-placeholder, and semantically useful?
- Which token positions were verbalized?
- Did NLA improve or worsen prediction-level absolute error?
- Did GEPA incorporate NLA feedback into prompt changes?
- Did NLA cause validation overfit or longer less-general prompts?
- Is the failure likely due to token selection, layer/checkpoint, proposer stochasticity, feedback length, or noisy verbalizations?

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
  - NLA quality signal: 900 rows, 300 covered examples, but all parse statuses are `partial_tags`, token status is `unknown`, and 667 rows are duplicate repeated text rows
- Completed root-cause analysis:
  - `gepa-experiments/results/diagnostics/nla_root_cause_20260608.md`
  - root cause: old NLA token selection used weak first semantic tokens from source/candidate/reference when budget was small
  - this produced repeated, generic verbalizations and tested noisy NLA feedback rather than a strong NLA condition
  - implemented candidate-prioritized middle/final token selection
  - preserved `token_status` in emitted NLA artifacts
  - increased real-NLA token and generation budgets

Still required:

- Run the updated tests locally.
- Commit changes in small descriptive commits.
- Sync updated files to the cluster.
- Pull complete artifacts for any future long NLA and non-NLA reruns.
- After queued fixed-NLA diagnostics finish, run `diagnose_nla_run.py` on their outputs.

## Current Cluster Queue

The old pinned jobs `11912818`, `11912819`, and `11912820` were cancelled because they had `ReqNodeList=moro232`.

Replacement chain submitted without a node pin, keeping `ExcNodeList=deeplearn2`:

- `11912914`: SummEval consistency smoke, PPL + real NLA
- `11912915`: QAGS-CNN consistency smoke, PPL + real NLA, dependency `afterany:11912914`
- `11912916`: QAGS-XSUM consistency smoke, PPL + real NLA, dependency `afterany:11912915`

As of the latest check, `11912914` is pending for resources, and the other two are pending on dependencies.

Additional Topical-Chat diagnostic chain submitted after the dataset smoke chain:

- `11912917`: Topical-Chat engagingness smoke, PPL-only control, llama.cpp proposer
- `11912918`: Topical-Chat engagingness smoke, PPL + fixed NLA, llama.cpp proposer

These are intended to validate whether the NLA token-selection fix improves verbalization quality before launching another long NLA run.

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

The full experiment matrix is considered scientifically usable only when every dataset/dimension target has:

- baseline/seed evaluation
- optimized evaluation
- paper-aligned metrics
- config artifact
- prompt artifacts
- runtime artifact
- prompt trajectory artifact

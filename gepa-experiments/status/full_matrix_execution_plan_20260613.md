# Full G-Eval / GEPA Matrix Execution Plan

Last updated: 2026-06-13

## Purpose

This document extends the current GEPA / G-Eval plan. It does not replace the
existing single-dimension pipeline or the NLA root-cause work. The goal is to
define the complete job matrix needed to produce thesis-ready evidence across
all G-Eval datasets, dimensions, and feedback variants, plus a separate
multi-dimension prompt pipeline where one prompt scores all dimensions of one
dataset in a single pass.

The planning assumption is intentionally exhaustive: first define everything
that could be useful, with expected runtime and artifacts, then decide what to
launch.

## Non-Negotiable Invariants

- Keep the current single-dimension runner reproducible. Existing configs,
  outputs, and analysis scripts must keep working.
- Implement the new multi-dimension prompt pipeline as an independent path.
  It may reuse loaders, metric utilities, feedback providers, Slurm/Docker
  helpers, and artifact writers, but it must have separate entrypoints,
  configs, output directories, and result tables.
- Preserve final-test separation. GEPA may use train and validation/dev rows
  only. Final-test rows are evaluated only after prompt selection.
- Store enough artifacts so long jobs do not need to be repeated just because a
  later analysis needs a missing field.
- Treat paper-aligned single-dimension jobs and multi-dimension joint-prompt
  jobs as separate experimental families. The joint-prompt results can report
  the same final metrics, but they are not a strict 1-to-1 reproduction of
  G-Eval's per-dimension prompt setup.

## Current Evidence Used For Runtime Estimates

Runtime estimates are based on local recovered `runtime_manifest_*.json`
artifacts, not on Slurm accounting.

Observed long Qwen35B proposer Topical-Chat engagingness jobs:

- PPL current-code control: 30,049 s, about 8.35 h.
- Fixed-NLA long: 30,519 s, about 8.48 h.
- Old NLA long: 28,356 s, about 7.88 h.

Observed smoke jobs:

- Topical-Chat Qwen35B PPL smoke, 4/2/2 groups: about 7.2 min.
- Topical-Chat Qwen35B PPL+NLA smokes, 4/2/2 groups: about 4.5-4.8 min.
- SummEval 7B proposer PPL/PPL+NLA consistency smokes, 4/2/2 groups:
  about 9.5-11.1 min.
- QAGS-CNN 7B proposer PPL/PPL+NLA smokes, 4/2/2 groups:
  about 1.2-1.4 min.
- QAGS-XSUM 7B proposer PPL/PPL+NLA smokes, 4/2/2 groups:
  about 0.9-1.0 min.

Uncertainty:

- Topical-Chat long estimates are medium confidence because they are based on
  multiple matched long runs.
- SummEval and QAGS long estimates are lower confidence because current
  completed jobs are small 7B-proposer smokes, not Qwen35B long runs.
- `ppl_nla_auxjudge` estimates are lower confidence because the auxiliary judge
  adds Qwen35B calls during feedback generation and we do not yet have a
  completed long aux-judge run.

## Required Artifact Contract For All Future Long Jobs

Each job should preserve:

- `run_config_*.json` with dataset, dimensions, split sizes, seed, feedback
  flags, proposer, judge, GEPA budget, and output schema version.
- `runtime_manifest_*.json` with start/end time, elapsed seconds, Slurm id,
  node, CUDA device mapping, and output directory.
- New required `stage_timing_*.json` with at least:
  data loading/splitting, preflight, NLA manifest export, NLA precompute,
  vLLM startup, llama.cpp startup, perplexity precompute, GEPA compile,
  baseline final-test evaluation, optimized final-test evaluation, artifact
  export, and total wall time.
- `dependency_manifest_*` with Python version, `pip freeze`, `pip check`,
  system package list when available, compiler versions, and `nvidia-smi`.
- Slurm stdout/stderr, vLLM logs, llama.cpp logs, and Telegram monitor logs.
- `split_manifest_*.json` with exact train/validation/final-test example ids.
- `metrics_*.csv` with all final G-Eval metrics plus diagnostics.
- `baseline_predictions_*.jsonl` and `optimized_predictions_*.jsonl` with raw
  model text, parsed score(s), parse status, targets, source text, reference or
  fact, candidate output, group id, and example id.
- `seed_prompt_*.txt`, `optimized_prompt_*.txt`,
  `prompt_trajectory_*.jsonl`, and `gepa_viz_run_*.json`.
- If PPL is enabled: per-example response NLL, perplexity, token count, and
  any failure status, not only aggregate feedback text.
- If NLA is enabled: NLA manifest, precomputed rows, emitted feedback rows,
  token metadata, parse status, activation summary stats, duplicate statistics,
  coverage, and useful-row counts.
- If auxiliary judge is enabled: raw auxiliary judge response, parsed
  rubric-level lesson, whether NLA text was included, and failure status.
- A post-run diagnostic summary comparing the job to its closest control when a
  matched control exists.

Timing metrics already exist only at total-runtime level. Before launching the
complete long matrix, add per-stage timing so later thesis analysis can explain
where compute time was spent.

## Metrics To Save

For every single-dimension job:

- Primary paper metrics:
  - SummEval: Spearman and Kendall tau.
  - Topical-Chat: Pearson and Spearman.
  - QAGS-CNN / QAGS-XSUM: Pearson, Spearman, and Kendall tau.
- Diagnostic metrics:
  - MAE, normalized agreement, parsed count, coverage, total rows, prediction
    distribution, prompt changed yes/no, trajectory rows, unique prompt count.

For every multi-dimension joint-prompt job:

- The same per-dimension metrics as above, one row per dimension.
- Macro-average across dimensions for each metric.
- Per-example wide predictions with one parsed score per dimension.
- Parse coverage per dimension and whole-response parse coverage.
- A separate flag marking these as `joint_prompt` so they are not mixed with
  paper-aligned single-dimension results.

## Existing Single-Dimension Pipeline

The existing pipeline optimizes one prompt for one dataset dimension. This is
the paper-aligned path and remains the primary result family.

Variants:

- `base_gepa`: GEPA with metric feedback only.
- `ppl`: GEPA with metric feedback plus response-only perplexity from
  Qwen2.5-7B.
- `ppl_nla`: GEPA with metric feedback, perplexity, and NLA verbalizations from
  Qwen2.5-7B.
- `ppl_nla_auxjudge`: GEPA with metric feedback, perplexity, NLA, and Qwen35B
  auxiliary judge feedback.

The proposer remains Qwen35B via llama.cpp for thesis-level jobs.

## New Independent Multi-Dimension Pipeline

Add a separate joint-prompt pipeline with its own entrypoint and configs, for
example:

- `geval_gepa.multidim_runner`
- `config/multidim_geval_gepa_*.env`
- `slurm/submit_gepa_multidim.sh` or a runner-module switch that defaults to
  the current single-dimension runner unless explicitly set
- output directories under `gepa-experiments/results/multidim_*`

The joint prompt should ask the judge to score all dimensions for one dataset in
one response.

Topical-Chat joint dimensions:

- naturalness
- coherence
- engagingness
- groundedness

SummEval joint dimensions:

- fluency
- coherence
- consistency
- relevance

QAGS-CNN and QAGS-XSUM have only consistency. Joint-prompt jobs for these are
therefore mostly symmetry/control jobs, not a new multi-dimension capability.

Proposed output format:

```text
Rationale:
<brief dimension-aware rationale>

Scores:
naturalness: <1-3>
coherence: <1-3>
engagingness: <1-3>
groundedness: <1-3>
```

For SummEval and QAGS, use the dataset score scale, typically 1-5.

GEPA needs one scalar optimization score, so the joint-prompt metric should
optimize a macro objective over dimensions. Recommended default:

- parse all dimension scores;
- compute normalized agreement per dimension;
- average normalized agreement across dimensions for the GEPA feedback score;
- include per-dimension errors in feedback text so the proposer can see which
  dimensions are causing failures.

Final evaluation must still compute paper metrics separately per dimension.

## Complete Single-Dimension Job Matrix

Point estimates are wall-clock job hours, assuming Qwen35B proposer via
llama.cpp and the current long-style budget. `done` means an equivalent
thesis-level long job already exists locally.

| id | priority | dataset | dimension | variant | status | estimate |
|---|---:|---|---|---|---|---:|
| SD-01 | 1 | topical_chat | engagingness | base_gepa | needed | 7.5 h |
| SD-02 | 1 | topical_chat | engagingness | ppl | done: current-code long control | 8.4 h |
| SD-03 | 1 | topical_chat | engagingness | ppl_nla | done: fixed-NLA long | 8.5 h |
| SD-04 | 1 | topical_chat | engagingness | ppl_nla_auxjudge | needed | 12.0 h |
| SD-05 | 2 | topical_chat | naturalness | base_gepa | needed | 7.5 h |
| SD-06 | 2 | topical_chat | naturalness | ppl | needed | 8.5 h |
| SD-07 | 2 | topical_chat | naturalness | ppl_nla | needed | 9.0 h |
| SD-08 | 3 | topical_chat | naturalness | ppl_nla_auxjudge | needed | 12.0 h |
| SD-09 | 2 | topical_chat | coherence | base_gepa | needed | 7.5 h |
| SD-10 | 2 | topical_chat | coherence | ppl | needed | 8.5 h |
| SD-11 | 2 | topical_chat | coherence | ppl_nla | needed | 9.0 h |
| SD-12 | 3 | topical_chat | coherence | ppl_nla_auxjudge | needed | 12.0 h |
| SD-13 | 2 | topical_chat | groundedness | base_gepa | needed | 7.5 h |
| SD-14 | 2 | topical_chat | groundedness | ppl | needed | 8.5 h |
| SD-15 | 2 | topical_chat | groundedness | ppl_nla | needed | 9.0 h |
| SD-16 | 3 | topical_chat | groundedness | ppl_nla_auxjudge | needed | 12.0 h |
| SD-17 | 2 | summeval | fluency | base_gepa | needed | 11.0 h |
| SD-18 | 2 | summeval | fluency | ppl | needed | 12.0 h |
| SD-19 | 2 | summeval | fluency | ppl_nla | needed | 13.0 h |
| SD-20 | 3 | summeval | fluency | ppl_nla_auxjudge | needed | 17.0 h |
| SD-21 | 2 | summeval | coherence | base_gepa | needed | 11.0 h |
| SD-22 | 2 | summeval | coherence | ppl | needed | 12.0 h |
| SD-23 | 2 | summeval | coherence | ppl_nla | needed | 13.0 h |
| SD-24 | 3 | summeval | coherence | ppl_nla_auxjudge | needed | 17.0 h |
| SD-25 | 1 | summeval | consistency | base_gepa | needed | 11.0 h |
| SD-26 | 1 | summeval | consistency | ppl | needed | 12.0 h |
| SD-27 | 1 | summeval | consistency | ppl_nla | needed | 13.0 h |
| SD-28 | 2 | summeval | consistency | ppl_nla_auxjudge | needed | 17.0 h |
| SD-29 | 2 | summeval | relevance | base_gepa | needed | 11.0 h |
| SD-30 | 2 | summeval | relevance | ppl | needed | 12.0 h |
| SD-31 | 2 | summeval | relevance | ppl_nla | needed | 13.0 h |
| SD-32 | 3 | summeval | relevance | ppl_nla_auxjudge | needed | 17.0 h |
| SD-33 | 2 | qags_cnn | consistency | base_gepa | needed | 4.0 h |
| SD-34 | 2 | qags_cnn | consistency | ppl | needed | 4.5 h |
| SD-35 | 2 | qags_cnn | consistency | ppl_nla | needed | 5.0 h |
| SD-36 | 3 | qags_cnn | consistency | ppl_nla_auxjudge | needed | 7.0 h |
| SD-37 | 2 | qags_xsum | consistency | base_gepa | needed | 4.0 h |
| SD-38 | 2 | qags_xsum | consistency | ppl | needed | 4.5 h |
| SD-39 | 2 | qags_xsum | consistency | ppl_nla | needed | 5.0 h |
| SD-40 | 3 | qags_xsum | consistency | ppl_nla_auxjudge | needed | 7.0 h |

Estimated total single-dimension matrix:

- All 40 jobs including already completed equivalents: about 401 h.
- Remaining after counting the two completed Topical-Chat engagingness long
  jobs: about 384 h.
- Most of these are 2-GPU jobs if Qwen35B proposer is used. Approximate GPU
  time is roughly double the wall-clock job time.

## Complete Multi-Dimension Joint-Prompt Matrix

These jobs are separate from the paper-aligned single-dimension pipeline. They
answer whether one optimized prompt can score all dimensions of a dataset at
once.

| id | priority | dataset | dimensions scored together | variant | status | estimate |
|---|---:|---|---|---|---|---:|
| MD-01 | 4 | topical_chat | all 4 | base_gepa | needed | 14.0 h |
| MD-02 | 4 | topical_chat | all 4 | ppl | needed | 15.0 h |
| MD-03 | 4 | topical_chat | all 4 | ppl_nla | needed | 16.0 h |
| MD-04 | 5 | topical_chat | all 4 | ppl_nla_auxjudge | needed | 22.0 h |
| MD-05 | 4 | summeval | all 4 | base_gepa | needed | 20.0 h |
| MD-06 | 4 | summeval | all 4 | ppl | needed | 21.0 h |
| MD-07 | 4 | summeval | all 4 | ppl_nla | needed | 22.0 h |
| MD-08 | 5 | summeval | all 4 | ppl_nla_auxjudge | needed | 28.0 h |
| MD-09 | 5 | qags_cnn | consistency only | base_gepa | needed | 4.0 h |
| MD-10 | 5 | qags_cnn | consistency only | ppl | needed | 4.5 h |
| MD-11 | 5 | qags_cnn | consistency only | ppl_nla | needed | 5.0 h |
| MD-12 | 5 | qags_cnn | consistency only | ppl_nla_auxjudge | needed | 7.0 h |
| MD-13 | 5 | qags_xsum | consistency only | base_gepa | needed | 4.0 h |
| MD-14 | 5 | qags_xsum | consistency only | ppl | needed | 4.5 h |
| MD-15 | 5 | qags_xsum | consistency only | ppl_nla | needed | 5.0 h |
| MD-16 | 5 | qags_xsum | consistency only | ppl_nla_auxjudge | needed | 7.0 h |

Estimated total joint-prompt matrix: about 199 h.

## Combined Runtime Estimate

Full exhaustive matrix:

- Single-dimension matrix: about 401 h job wall-clock.
- Joint-prompt matrix: about 199 h job wall-clock.
- Combined: about 600 h job wall-clock.
- Remaining after already completed Topical-Chat engagingness long PPL and
  fixed-NLA equivalents: about 583 h job wall-clock.

If `faretra` can run two 2-GPU jobs concurrently on four RTX 3090 GPUs, the
absolute lower bound is about 292-300 h calendar time. In practice queueing,
other users, GPU memory fragmentation, failed startups, and node availability
make the calendar time substantially longer.

## Launch Priority

Priority 1: thesis-core missing ablations.

- SD-04 Topical-Chat engagingness `ppl_nla_auxjudge`.
- SD-01 Topical-Chat engagingness `base_gepa`.
- SD-25 to SD-27 SummEval consistency `base_gepa`, `ppl`, `ppl_nla`.

Reason: these give the most direct context for the existing NLA story and for
the dataset family already partially tested.

Priority 2: complete paper dimensions for the two main multi-dimension
datasets.

- Topical-Chat naturalness, coherence, groundedness: SD-05 to SD-15.
- SummEval fluency, coherence, relevance: SD-17 to SD-24 and SD-29 to SD-31.
- QAGS-CNN and QAGS-XSUM consistency: SD-33 to SD-39.

Reason: these produce the table coverage needed to say whether current findings
are specific to engagingness/consistency or general across paper dimensions.

Priority 3: auxiliary judge breadth.

- SD-08, SD-12, SD-16, SD-20, SD-24, SD-28, SD-32, SD-36, SD-40.

Reason: aux-judge is important for the NLA thesis angle, but expensive. If the
first aux-judge Topical-Chat run is clearly negative, these should be reviewed
before launch rather than blindly executed.

Priority 4: joint-prompt multi-dimension jobs for Topical-Chat and SummEval.

- MD-01 to MD-08.

Reason: these are scientifically useful but not a direct paper reproduction.
They should start after the single-dimension paper-aligned table is underway or
when there is spare GPU capacity.

Priority 5: joint-prompt QAGS symmetry jobs.

- MD-09 to MD-16.

Reason: QAGS has only one dimension, so these mostly test that the independent
multi-dimension pipeline behaves consistently on all dataset families.

## Implementation Decisions For Multi-Dimension Pipeline

- Use a new result namespace and a new runner entrypoint.
- Reuse `TaskSpec`, loaders, split logic, `compute_regression_metrics`,
  perplexity, NLA precompute, and aux-judge providers where possible.
- Add a multi-dimension dataset view that groups labels for the same
  source/candidate pair into one example with a target score per dimension.
- Keep single-dimension configs unchanged.
- Add parser tests for full, partial, malformed, and out-of-scale
  multi-dimension score blocks.
- Add final metrics tests that verify per-dimension and macro rows are both
  written.
- Add artifact schema versioning so joint-prompt prediction files cannot be
  confused with single-dimension prediction files.

## Ambiguities To Clarify Before Launching The Full Matrix

- Whether the final thesis table should include all four variants for every
  dataset/dimension, or whether `ppl_nla_auxjudge` should be launched only if
  the first aux-judge runs are promising.
- Whether full SummEval/QAGS long jobs should use the same 40/10/10 group split
  shape as Topical-Chat or a dataset-specific split that better matches their
  original validation/test structure.
- Whether joint-prompt jobs should use `ppl_nla` or `ppl_nla_auxjudge` as the
  first variant if only one joint run per dataset is launched initially.
- Whether QAGS joint-prompt jobs should be included in the first wave despite
  having only one dimension.
- Whether bootstrap confidence intervals should be added to final metrics
  before long jobs, so statistical uncertainty is archived alongside point
  estimates.


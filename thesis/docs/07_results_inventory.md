# Chapter 5 Results Plan

Status date: 2026-06-15

This document defines the intended structure of Chapter 5. It replaces the old
artifact-only inventory with a results-chapter plan, while keeping the artifact
state needed to trace each claim back to concrete files.

Chapter 5 should report what was obtained, what can be claimed from it, and what
must remain diagnostic or future work. It should not introduce new method or
setup details except when a short reminder is needed to interpret a table.

## Goal

Present the thesis results in the clearest form: baseline comparisons,
ablations, failure-mode analysis, and efficiency numbers.

The chapter must let a reader answer:

- which experiments produced usable evidence;
- which results are paper-aligned and which are smoke/diagnostic only;
- which baselines each result should be compared against;
- whether NLA improved GEPA, failed to improve it, or only provided diagnostic
  signal;
- which artifact supports each table or claim.

Target length: enough to cover the full thesis progression without becoming a
log dump. Detailed artifact paths and long prompt text should stay in appendix
or in the artifact ledger at the end of this planning document.

## Scope Boundary

Include:

- canonical dataset validation and coverage results;
- embedding-inversion reproduction diagnostics;
- SIPIT reproduction/interim evidence, collision checks, and logical/random
  prefix results if final outputs exist;
- standalone NLA activation extraction and verbalization evidence;
- GEPA/G-EVAL prompt-optimization results;
- PPL, NLA, candidate-only NLA, fixed-NLA, and auxiliary-judge ablations;
- runtime and efficiency numbers;
- failure modes that affect scientific interpretation;
- prompt-change evidence when discussing GEPA.

Exclude:

- theoretical explanations of metrics or models; those belong to Chapter 1;
- literature comparisons; those belong to Chapter 2;
- method diagrams and algorithm descriptions; those belong to Chapter 3;
- cluster/container setup details; those belong to Chapter 4;
- raw logs, full prompt dumps, and long JSONL artifacts.

## Writing Guidelines For Results

The advisor guideline for Chapter 5 is: include all obtained results in the
clearest format, including baseline comparisons, ablations, and efficiency
numbers.

Required result-writing rules:

- Every result table must state the denominator or `n`.
- Every table must state the baseline or control used for comparison.
- Paper-aligned results and diagnostic/smoke results must be visually and
  textually separated.
- If a run is a smoke test, call it a smoke test and do not use it as final
  performance evidence.
- If a prompt did not change, say so before interpreting metric changes.
- Efficiency results should include elapsed time and, where available, runtime
  components or throughput.
- Negative results should be kept when they explain a design decision or rule
  out a hypothesis.

Reference-thesis style to preserve:

- introduce each result subsection with the question being answered;
- show the result in a compact table;
- bold best values only when the comparison is scientifically valid;
- write one short paragraph after the table explaining the main takeaway;
- include runtime/efficiency in a separate table when it materially matters.

## Proposed Section Structure

### 5.1 Results Overview And Evidence Levels

Purpose: orient the reader before the detailed result sections.

This section should introduce an evidence-level table:

| Result family | Evidence level | Main claim status | Main artifact |
|---|---|---|---|
| Canonical semantic-fidelity dataset | Complete validation | Thesis dataset contribution | `thesis-datasets/reports/` |
| Embedding inversion | Diagnostic / negative reproduction | Useful failure-mode evidence, not paper-level reproduction | `embedding-inversion-demo/RESULTS.md` |
| SIPIT | Interim reproduction + collision evidence | Strong interim GPT-2 signal; final CSV/JSON still missing locally | `spit/SIPIT/notes/` |
| Standalone NLA | Plumbing validation | NLA extraction/verbalization works, not semantic-fidelity proof | `nla-artifacts/summeval/` |
| GEPA + PPL | Positive long-run evidence | GEPA+PPL can improve Topical-Chat engagingness in observed run | GEPA result dirs |
| GEPA + raw/fixed NLA | Mixed diagnostic evidence | Raw NLA is not reliably helpful; fixed-NLA is weak-positive but not a clean prompt-improvement win | GEPA diagnostics |
| GEPA + auxiliary judge | Planned / pending in current docs | Needed to test NLA semantic compression | GEPA status docs |

The section should explicitly state that the thesis cannot currently claim
"NLA robustly improves GEPA" unless later auxiliary-judge or replicated long
runs provide stronger evidence.

### 5.2 Canonical Semantic-Fidelity Dataset Validation

Purpose: report the dataset contribution before using it in later experiments.

Current validated counts:

| Block | Phenomenon | Rows | Role |
|---|---|---:|---|
| A | Controlled standard sentences | 40 | Control text |
| B | Negation | 720 | Logical polarity stress test |
| C | Commonsense/counterfactual | 1320 | Plausibility and counterfactual stress test |
| Total | All blocks | 2080 | Canonical thesis corpus |

Split counts:

| Split | Rows |
|---|---:|
| Train | 962 |
| Validation | 290 |
| Test | 828 |

Interpretation:

- This is a thesis result/contribution, not only preprocessing.
- It operationalizes the original semantic-fidelity question.
- It should be reported as validated because it feeds SIPIT and NLA exports.

Artifacts:

- `thesis-datasets/reports/build_report.md`
- `thesis-datasets/reports/validation_report.md`
- `thesis-datasets/reports/sipit_export_report.md`
- `thesis-datasets/reports/nla_export_report.md`

Open result gap:

- If the thesis makes strong claims about semantic flips, add final metrics for
  negation preservation, polarity flip rate, and counterfactual preservation.

### 5.3 Embedding-Inversion Diagnostics

Purpose: report the early embedding-inversion work as diagnostic evidence.

Expected table:

| Run / branch | Architecture or change | Main metric | Outcome | Interpretation |
|---|---|---|---|---|

Current quantitative anchors:

- Long Jina-v3 run `11108134`: not paper-level; full-mask token accuracy about
  0.127 at 97.5K steps.
- Corrected v3 10K probe `11108177`: weak full-mask behavior.
- Tiny full-mask overfit `11108213`: positive control reached 1.000 train
  accuracy with the paper-equation v2 architecture and Eq. 4-style loss.
- Full-data probes `11108216` and `11108226`: did not recover strong full-mask
  behavior.
- Qwen3-Embedding tiny-overfit probe learned more easily than Jina-v3,
  suggesting encoder/tokenizer choice materially affects difficulty.

Interpretation:

- This branch should be framed as a reproduction/diagnostic branch.
- It supports the thesis theme that metric fidelity and artifact-level
  reproducibility matter.
- Do not present it as a successful Jina paper reproduction unless a later
  clean run changes the evidence.

Artifacts:

- `embedding-inversion-demo/RESULTS.md`
- `embedding-inversion-demo/EXPERIMENTS.md`
- `embedding-inversion-demo/FAILURE_MODES.md`
- `embedding-inversion-demo/PROVENANCE_LEDGER.md`

### 5.4 SIPIT Reproduction And Extensions

Purpose: report exact hidden-state inversion evidence and connect it to the
semantic-fidelity track.

Subsections to write:

- GPT-2 collision check.
- GPT-2 Table 5 reproduction/interim metrics.
- BruteForce and HardPrompts baseline status.
- Logical20 dataset result status.
- Random-prefix analysis if final result tables are available.

Expected collision table:

| Layer | Prompts checked | Minimum L2 distance | `torch.allclose` collisions |
|---|---:|---:|---:|
| First | 100 | 6.1603 | 0 |
| Middle | 100 | 27.1380 | 0 |
| Last | 100 | 110.2129 | 0 |

Expected interim reproduction table:

| Method | Completed prompts with usable trace | Recovery signal | Search-cost signal | Status |
|---|---:|---|---|---|
| SIPIT | 47 observed prompts | Found every token before vocabulary exhaustion | Mean vocabulary explored about 4.41% | Interim/log-derived |
| BruteForce | Partial | Exhausted-token failures in completed prompts | About full-vocabulary scans | Cancelled before final report |
| HardPrompts | Partial | No final exact-match metric | Long optimization trace only | Cancelled before final report |

Interpretation:

- The collision check is usable as a completed result.
- The Table 5 evidence is strong but must be called interim/log-derived unless
  final CSV/JSON outputs are recovered.
- Runtime should be diagnostic only because the paper used A100-SXM 64GB while
  local runs used RTX-class GPUs.
- `known-prefix-control` and `full-sequence` random-prefix results must be kept
  separate because they answer different questions.

Artifacts:

- `spit/SIPIT/notes/REPRODUCTION.md`
- `spit/SIPIT/notes/REPRODUCTION_RUN_2026-05-20.md`
- `spit/SIPIT/notes/INTERIM_METRICS_2026-05-22.md`
- `spit/SIPIT/notes/METRICS_SNAPSHOT_2026-05-22_1110.md`
- `spit/SIPIT/data/reproduce/reports/table5/gpt2_collision_check_cpu.json`
- `spit/SIPIT/data/reproduce/logical20_gpt2_clean/`
- `spit/SIPIT/scripts/random_prefix/README.md`

### 5.5 Standalone NLA Validation

Purpose: show that NLA activation extraction and AV verbalization work before
the GEPA integration.

Expected table:

| Run | Dataset/sample | Activation source | Rows | Parse/injection status | Claim |
|---|---|---|---:|---|---|

Current evidence:

- Qwen2.5-7B-Instruct layer-20 activations can be extracted.
- The compatible Qwen layer-20 AV checkpoint can verbalize these activations.
- The sample12/full streamed verbalization produced 24 verbalization rows.
- Standalone artifacts validate plumbing, not semantic-fidelity behavior on the
  canonical logical dataset.

Interpretation:

- Use this section to distinguish "NLA works technically" from "NLA provides
  useful semantic feedback for GEPA".
- Include one short verbalization example only if it helps the reader
  understand what AV output looks like.

Artifacts:

- `nla-experiments/README.md`
- `nla-artifacts/summeval/RUNS.md`
- `nla-artifacts/summeval/activations_qwen25_7b_instruct_L20_sample12.parquet`
- `nla-artifacts/summeval/verbalizations_qwen25_7b_L20_sample12_transformers_stream.jsonl`
- `nla-artifacts/summeval/report_qwen25_7b_L20_sample12_transformers_stream.md`

### 5.6 GEPA/G-EVAL Prompt Optimization

Purpose: report the main current experimental branch.

This section should be organized around comparisons, not chronology.

#### 5.6.1 GEPA + PPL Long-Run Evidence

Question:

Does GEPA with perplexity feedback improve a G-EVAL-style Topical-Chat
engagingness judge prompt?

Expected table columns:

| Run | Variant | n | Split | Proposer | Prompt changed? | Pearson | Spearman | Kendall | Agreement | MAE | Runtime | Artifact |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|

Current status:

- The first PPL long run is the first clear positive GEPA+PPL result.
- It is useful for historical motivation.
- It is not always the cleanest control for later NLA claims because the code
  and setup evolved afterward.

Artifact:

- `gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer`

#### 5.6.2 Raw NLA Long Run

Question:

Does adding raw NLA verbalizations directly to proposer feedback improve GEPA?

Current status:

- The first raw-NLA long run is negative diagnostic evidence.
- It motivates later fixes: better token selection, fixed NLA artifact schema,
  duplicate analysis, and auxiliary-judge compression.

Artifact:

- `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b`

#### 5.6.3 Fixed-NLA Versus Current-Code PPL Control

Question:

Under the current code path, does fixed-NLA improve over a matched PPL-only
long control?

Current clean comparison:

| Comparison | n | Main result | Interpretation |
|---|---:|---|---|
| Fixed-NLA optimized minus current-code PPL optimized | 60 | Pearson +0.022940, Spearman +0.018872, Kendall +0.015841, agreement +0.011111, MAE -0.022222 | Weak-positive diagnostic signal |

Additional interpretation:

- Both runs selected the byte-identical seed prompt.
- Only two final-test predictions improved, zero worsened, and 58 were
  unchanged.
- This is not evidence that GEPA discovered a better prompt because of NLA.
- It is a small positive final-test delta under a matched setup and should be
  treated as weak evidence unless replicated or strengthened by auxiliary-judge
  results.
- The matched current-code PPL control runtime was 30049.160 seconds, about
  8h20m49s.

Artifacts:

- `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control`
- `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b`
- `gepa-experiments/results/diagnostics/nla_fixed_long_vs_current_ppl_long_20260612.md`

#### 5.6.4 Candidate-Only NLA And Duplicate Feedback Diagnostics

Question:

Was the NLA problem mainly caused by repeated source/reference feedback, and can
candidate-only NLA solve it?

Current status:

- Candidate-only NLA reduced duplicate feedback but worsened metrics.
- Candidate-content-10 was less bad than candidate-content-6, but still
  negative on paper-primary correlations and worse on MAE.
- This rules out the simple "duplicates are the only problem" hypothesis.

Artifacts:

- `gepa-experiments/results/experimental_nla_candidate_content_6_topical_chat_smoke`
- `gepa-experiments/results/experimental_nla_candidate_content_10_topical_chat_smoke`
- `gepa-experiments/results/diagnostics/nla_candidate_content_6_vs_ppl_smoke_20260610.md`
- `gepa-experiments/results/diagnostics/nla_candidate_content_10_vs_ppl_smoke_20260611.md`

#### 5.6.5 Auxiliary-Judge Feedback

Question:

Can Qwen35B compress raw NLA/PPL/example feedback into rubric-level feedback
that is more useful to GEPA?

Current status from local docs:

- The first auxiliary-judge smoke exposed an empty-feedback bug and should not
  be interpreted as an improvement claim.
- A replacement aux-judge smoke and dependent long run are tracked in
  `gepa-experiments/status/relatore_results_index_20260614.md`.
- This branch is thesis-critical if the final claim is about making NLA useful
  for GEPA, because raw/fixed NLA alone is not yet convincing.

Artifacts:

- `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke`
- `gepa-experiments/status/relatore_results_index_20260614.md`

Interpretation rule:

- If an aux-judge run succeeds, compare it against both PPL-only and fixed-NLA
  matched controls.
- If it fails, inspect `aux_judge_feedback_*.jsonl` before deciding whether the
  failure is due to NLA content, auxiliary-judge compression, or GEPA overfit.

#### 5.6.6 Dataset And Dimension Matrix

Question:

How far do the results generalize beyond Topical-Chat engagingness?

Current status:

- The full paper-aligned matrix is planned across Topical-Chat, SummEval,
  QAGS-CNN, and QAGS-XSUM.
- Some dataset smokes and controls exist, including SummEval and QAGS plumbing
  checks.
- Small `n` smokes, especially QAGS with very small test sets, should not be
  reported as scientific performance claims.

Expected table:

| Dataset | Dimension | Variant | n | Paper-aligned metrics | Status | Artifact |
|---|---|---|---:|---|---|---|

Artifacts:

- `gepa-experiments/status/full_matrix_execution_plan_20260613.md`
- `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_real_nla_smoke`
- `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_smoke`
- `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke`
- `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_smoke`
- `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke`
- `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_smoke`
- `gepa-experiments/results/diagnostics/nla_summeval_consistency_vs_ppl_smoke_20260611.md`

### 5.7 Runtime And Efficiency Results

Purpose: report cost, not only quality.

Expected table:

| Branch | Run | GPU setting | Elapsed time | Stage timings available? | Notes |
|---|---|---|---:|---|---|

Current available metrics:

- GEPA runs save total elapsed time in `runtime_manifest_*.json`.
- SIPIT records inversion time and per-token time in its CSV/JSON outputs or
  log-derived summaries.
- Embedding-inversion logs record elapsed time and sample rate.
- The current GEPA runner does not yet provide full per-stage timing for data
  loading, vLLM startup, llama.cpp startup, PPL precompute, NLA precompute,
  GEPA compile, baseline eval, optimized eval, and artifact export.

Interpretation:

- Include total elapsed time for every thesis-grade run.
- Report per-stage timings when available.
- Do not compare runtime across methods as a pass/fail scientific metric when
  hardware differs materially.

### 5.8 Failure Modes And Negative Evidence

Purpose: explain why some branches were changed or deprioritized.

Failure/negative evidence to include:

| Failure mode | Evidence | Consequence |
|---|---|---|
| Embedding-inversion full-data runs stayed far below paper-level behavior | Jina-v3 probes and ablations | Treat as diagnostic reproduction branch |
| SIPIT final CSV/JSON missing locally | SIPIT notes and current tree | Label Table 5 evidence interim/log-derived unless recovered |
| Raw NLA worsened GEPA | First raw-NLA long run | Do not claim raw NLA helps |
| Fixed-NLA weak-positive but no prompt change | Matched long comparison | Treat as weak diagnostic evidence |
| Candidate-only NLA negative | Candidate-content smokes | Duplicate-only hypothesis insufficient |
| Aux-judge empty feedback bug | Aux-judge smoke | Do not use early aux metrics as improvement evidence |
| Node-local artifacts on `moro232` | Current plan notes | Check execution node or sync artifacts explicitly |
| GPU memory occupied by other users | Failed replacement controls | Failure not caused by thesis code, but affects scheduling/retry policy |

This section should be short but explicit. It prevents the thesis from looking
like a sequence of unrelated failed runs and instead explains which hypotheses
were ruled out.

### 5.9 Final Claim Matrix

Purpose: close the results chapter with a clear claim/status table.

Expected table:

| Claim | Supported? | Evidence | Caveat |
|---|---|---|---|
| The canonical dataset was built and validated | Yes | Dataset reports | Semantic flip metrics still needed for some claims |
| Embedding inversion reproduction was diagnostic, not paper-level | Yes | Jina-v3 probes | Not a successful paper reproduction |
| SIPIT exact inversion behaves as expected in GPT-2 interim evidence | Partially | Collision check and log-derived recovery | Final CSV/JSON missing locally |
| Standalone NLA plumbing works | Yes | NLA artifacts | Not semantic-fidelity proof |
| GEPA+PPL can improve Topical-Chat engagingness | Yes for observed long run | First PPL long run | Historical setup evolved |
| Raw NLA robustly improves GEPA | No | Raw-NLA and candidate-only diagnostics | Needs semantic compression or better use |
| Fixed-NLA improves GEPA | Weak / not yet conclusive | Matched long control | Prompt unchanged; small prediction movement |
| Auxiliary judge makes NLA useful | Pending in current docs | Aux-judge branch | Needs valid smoke/long result |

## Tables And Figures To Add

Main tables:

- Evidence-level overview table.
- Canonical dataset validation table.
- Embedding-inversion diagnostic run table.
- SIPIT collision and interim reproduction tables.
- Standalone NLA validation table.
- GEPA long-run comparison table.
- NLA ablation/failure-mode table.
- Dataset/dimension matrix status table.
- Runtime and efficiency table.
- Final claim matrix.

Figures:

- Prefer tables for most Chapter 5 content.
- Add a plot only if it genuinely clarifies a trajectory, such as embedding
  inversion validation loss or GEPA prompt-search trajectory.
- Every figure used in the thesis must be mentioned explicitly in the text.

Prompt examples:

- Include short prompt excerpts only if they explain the result.
- Full prompts should be moved to appendix or linked artifacts:
  seed prompt, old PPL optimized prompt, old raw-NLA optimized prompt,
  fixed-NLA final prompt, and one failed candidate-only prompt.

## Artifact Ledger

This ledger preserves the old inventory state. It is not the structure of the
final Results chapter; it is the support map for writing it.

### Cross-Cutting Docs

| Artifact | Status | Planned use |
|---|---|---|
| `thesis/docs/09_prior_work_census.md` | Available | Broader local/cluster work inventory |
| `gepa-experiments/status/current_plan_status.md` | Available | Current GEPA plan, job state, and interpretation notes |
| `gepa-experiments/status/full_matrix_execution_plan_20260613.md` | Available | Matrix plan across datasets/dimensions/variants |
| `gepa-experiments/status/relatore_call_brief_20260611.md` | Available | Advisor-call explanation of GEPA evolution |
| `gepa-experiments/status/relatore_results_index_20260614.md` | Available | Advisor-facing GEPA artifact index |

### Embedding Inversion Artifacts

| Artifact | Status | Planned use |
|---|---|---|
| `embedding-inversion-demo/RESULTS.md` | Available | Main curated result anchors |
| `embedding-inversion-demo/EXPERIMENTS.md` | Available | Decision tree and branch status |
| `embedding-inversion-demo/FAILURE_MODES.md` | Available | Failure-mode discussion |
| `embedding-inversion-demo/PROVENANCE_LEDGER.md` | Available | Data/model provenance |

### Dataset Artifacts

| Artifact | Status | Planned use |
|---|---|---|
| `thesis-datasets/reports/build_report.md` | Available | Dataset count and construction table |
| `thesis-datasets/reports/validation_report.md` | Available | Validation status |
| `thesis-datasets/reports/sipit_export_report.md` | Available | SIPIT export evidence |
| `thesis-datasets/reports/nla_export_report.md` | Available | NLA export evidence |

### SIPIT Artifacts

| Artifact | Status | Planned use |
|---|---|---|
| `spit/SIPIT/notes/REPRODUCTION.md` | Available | Reproduction protocol |
| `spit/SIPIT/notes/REPRODUCTION_RUN_2026-05-20.md` | Available | Run setup and caveats |
| `spit/SIPIT/notes/INTERIM_METRICS_2026-05-22.md` | Available | Interim metric table |
| `spit/SIPIT/notes/METRICS_SNAPSHOT_2026-05-22_1110.md` | Available | Later interim snapshot |
| `spit/SIPIT/data/reproduce/reports/table5/gpt2_collision_check_cpu.json` | Available | Collision check table |
| `spit/SIPIT/data/reproduce/logical20_gpt2_clean/` | Available | Logical dataset evidence |
| Final Table 5 CSV/JSON reports | Not currently available locally | Recover if possible; otherwise label interim |
| Final logical/random-prefix result tables | Not currently available in this inventory | Recover/run if they are thesis-critical |

### Standalone NLA Artifacts

| Artifact | Status | Planned use |
|---|---|---|
| `nla-experiments/README.md` | Available | Pipeline context |
| `nla-artifacts/summeval/RUNS.md` | Available | Run history |
| `nla-artifacts/summeval/activations_qwen25_7b_instruct_L20_sample12.parquet` | Available | Activation extraction evidence |
| `nla-artifacts/summeval/verbalizations_qwen25_7b_L20_sample12_transformers_stream.jsonl` | Available | AV verbalization evidence |
| `nla-artifacts/summeval/report_qwen25_7b_L20_sample12_transformers_stream.md` | Available | Standalone NLA summary |
| NLA semantic-fidelity evaluation on canonical logical dataset | Not currently available | Needed for standalone semantic-fidelity claim |

### GEPA Result Directories

| Artifact | Status | Planned use |
|---|---|---|
| `gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer` | Available | First positive GEPA+PPL long run |
| `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control` | Available | Clean current-code PPL long control |
| `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b` | Available | Matched fixed-NLA long run |
| `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b` | Available | Negative raw-NLA long run |
| `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke` | Available | Debug artifact for empty-feedback bug, not improvement evidence |
| `gepa-experiments/results/experimental_nla_candidate_content_6_topical_chat_smoke` | Available | Negative candidate-only NLA diagnostic |
| `gepa-experiments/results/experimental_nla_candidate_content_10_topical_chat_smoke` | Available | Negative candidate-only NLA diagnostic |
| `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_real_nla_smoke` | Available | Dataset smoke / diagnostic |
| `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_smoke` | Available | Matched smoke control |
| `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke` | Available | Dataset smoke / diagnostic |
| `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_smoke` | Available | Matched smoke control |
| `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke` | Available | Dataset smoke / diagnostic |
| `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_smoke` | Available | Matched smoke control |

### GEPA Diagnostic Reports

| Artifact | Status | Planned use |
|---|---|---|
| `gepa-experiments/results/diagnostics/nla_evidence_deep_dive_20260612.md` | Available | Aggregate NLA evidence and root-cause analysis |
| `gepa-experiments/results/diagnostics/nla_fixed_long_vs_current_ppl_long_20260612.md` | Available | Cleanest long fixed-NLA vs PPL comparison |
| `gepa-experiments/results/diagnostics/nla_vs_ppl_fixed_smoke_20260610.md` | Available | Positive fixed-NLA smoke comparison |
| `gepa-experiments/results/diagnostics/nla_candidate_content_6_vs_ppl_smoke_20260610.md` | Available | Negative candidate-only comparison |
| `gepa-experiments/results/diagnostics/nla_candidate_content_10_vs_ppl_smoke_20260611.md` | Available | Negative candidate-only comparison |
| `gepa-experiments/results/diagnostics/nla_summeval_consistency_vs_ppl_smoke_20260611.md` | Available | SummEval smoke comparison |
| `gepa-experiments/results/diagnostics/nla_activation_verbalization_quality_20260610.md` | Available | NLA artifact quality discussion |

### External/Shared Result Snapshot

| Artifact | Status | Planned use |
|---|---|---|
| `https://github.com/s-mazza/tesi/tree/gepa-experiments-code-review/gepa-experiments/results` | Available as remote snapshot | Advisor review link when direct attachments are inconvenient |

## Not Planned For Main Text Or Not Currently Available

Not planned for main text:

- raw Slurm logs, except short excerpts for failure explanations;
- full JSONL prediction dumps;
- full prompt trajectories;
- full prompts, unless moved to appendix;
- early failed jobs whose root cause was purely infrastructure and not a
  scientific condition;
- duplicated smoke runs that answer no distinct question.

Not currently available or incomplete:

- final SIPIT Table 5 CSV/JSON result reports in the checked local tree;
- final SIPIT logical/random-prefix tables, unless recovered or rerun;
- standalone NLA semantic-fidelity evaluation on the canonical logical dataset;
- final auxiliary-judge long result in the current local docs;
- full paper-aligned GEPA matrix across all dimensions;
- GEPA per-stage timing and peak GPU memory for final long runs;
- final prompt appendix collecting all important seed/optimized prompts.

## Current Claim Status

The thesis can currently claim:

- the semantic-fidelity dataset was built and validated;
- embedding-inversion reproduction produced useful diagnostic/negative evidence;
- SIPIT GPT-2 collision checks passed and interim exact-recovery evidence is
  strong, but final local CSV/JSON reports are missing;
- standalone Qwen2.5-7B NLA extraction and verbalization are operational;
- GEPA+PPL can improve the observed Topical-Chat engagingness task;
- raw NLA feedback is not reliably helpful in the current GEPA setup;
- fixed-NLA is technically healthier and weak-positive against a matched
  current-code PPL control, but it did not produce a new optimized prompt;
- auxiliary-judge compression is the next critical test if the thesis wants to
  argue that NLA can improve GEPA rather than only diagnose it.

The thesis cannot yet claim:

- NLA robustly improves GEPA;
- standalone NLA preserves negation or counterfactual meaning on the canonical
  dataset;
- QAGS or SummEval smoke numbers are final scientific performance results;
- exact reproduction of all G-EVAL paper results, because the current
  experiments use a different model setting and only partial matrix coverage.

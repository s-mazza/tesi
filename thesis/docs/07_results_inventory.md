# Results Inventory

This file tracks which results are available for the thesis and how they should
be interpreted.

## Prior-Work Census

The broader inventory of local and cluster work is tracked in:

`thesis/docs/09_prior_work_census.md`

Use that document first when deciding whether a result belongs to the thesis,
because this file currently summarizes the main evidence rather than every
artifact in every folder.

## Embedding Inversion Diagnostics

Path:

`embedding-inversion-demo/`

Important docs:

- `embedding-inversion-demo/RESULTS.md`
- `embedding-inversion-demo/EXPERIMENTS.md`
- `embedding-inversion-demo/FAILURE_MODES.md`
- `embedding-inversion-demo/PROVENANCE_LEDGER.md`

Interpretation:

- This branch is a diagnostic/negative reproduction track for Jina-style
  embedding inversion.
- It did not reach paper-level full-mask reconstruction in the documented
  Jina-v3 probes.
- It produced useful evidence on architecture/loss choices, tiny-overfit
  positive controls, gradient clipping, embedding provenance, and encoder
  differences.
- It should be included as earlier thesis work, but claims must be phrased as
  reproduction diagnostics unless a clean paper-level run is later obtained.

## Canonical Dataset Results

Path:

`thesis-datasets/`

Current validated counts:

- Total rows: 2080.
- Block A standard: 40.
- Block B negation: 720.
- Block C commonsense/counterfactual: 1320.
- Splits: train 962, validation 290, test 828.
- Validation status: PASS.

Interpretation:

- This is a thesis contribution, not just preprocessing.
- It operationalizes the semantic-fidelity question and feeds SIPIT/NLA
  exports.

## SIPIT Results

Path:

`spit/SIPIT/`

Important docs/artifacts:

- `spit/SIPIT/notes/REPRODUCTION.md`
- `spit/SIPIT/notes/REPRODUCTION_RUN_2026-05-20.md`
- `spit/SIPIT/notes/INTERIM_METRICS_2026-05-22.md`
- `spit/SIPIT/notes/METRICS_SNAPSHOT_2026-05-22_1110.md`
- `spit/SIPIT/data/reproduce/reports/table5/gpt2_collision_check_cpu.json`
- `spit/SIPIT/data/reproduce/logical20_gpt2_clean/`
- `spit/SIPIT/scripts/random_prefix/README.md`

Interpretation:

- GPT-2 collision check completed successfully on 100 prompts and found zero
  `torch.allclose` collisions.
- Log-derived Table 5 evidence showed SIPIT recovering every token before
  vocabulary exhaustion on the first 47 completed prompts.
- BruteForce and HardPrompts were cancelled before complete official reports.
- Local final Table 5 CSV/JSON outputs are not currently present in the checked
  tree; recover them from the older remote workspace if they exist, otherwise
  label these metrics as interim/log-derived.
- The logical20 dataset and random-prefix extension are important for the
  semantic-fidelity thesis, even if final random-prefix result tables still
  need to be recovered or run.

## Standalone NLA Results

Paths:

`nla-experiments/`

`nla-artifacts/`

Important docs/artifacts:

- `nla-experiments/README.md`
- `nla-artifacts/summeval/RUNS.md`
- `nla-artifacts/summeval/activations_qwen25_7b_instruct_L20_sample12.parquet`
- `nla-artifacts/summeval/verbalizations_qwen25_7b_L20_sample12_transformers_stream.jsonl`
- `nla-artifacts/summeval/report_qwen25_7b_L20_sample12_transformers_stream.md`

Interpretation:

- Qwen2.5-7B layer-20 activation extraction and NLA AV verbalization are
  operational.
- These runs validate the NLA plumbing independently from GEPA.
- They are not yet a full semantic-fidelity evaluation of NLA on the canonical
  logical dataset.

## GitHub Results Snapshot

Relevant result artifacts were uploaded to:

`https://github.com/s-mazza/tesi/tree/gepa-experiments-code-review/gepa-experiments/results`

Use this link for advisor review when direct attachments are inconvenient.

## Core Long Runs

### First PPL Long Run

Path:

`gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer`

Interpretation:

- Shows the first clear positive GEPA+PPL result.
- Useful for historical motivation.
- Not always the cleanest control for later NLA claims because code and setup
  evolved afterward.

### Current-Code PPL Long Control

Path:

`gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control`

Interpretation:

- Cleanest current PPL-only control for fixed-NLA.
- Same main setup as fixed-NLA except NLA is disabled.

### Fixed-NLA Long Run

Path:

`gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b`

Interpretation:

- Slightly improves metrics over current-code PPL.
- Weak evidence only, because the optimized prompt stayed identical to the
  seed prompt.

### First Raw-NLA Long Run

Path:

`gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b`

Interpretation:

- Negative diagnostic condition.
- Useful to explain why raw NLA feedback and weak token selection are not
  enough.

### Auxiliary Judge Smoke

Path:

`gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke`

Interpretation:

- Exposed the empty-feedback bug.
- Its metrics should not be used as evidence that auxiliary judge improves
  GEPA.
- Useful as a debugging artifact and motivation for the non-empty feedback
  guard.

## Diagnostic Reports

Main report:

`gepa-experiments/results/diagnostics/nla_evidence_deep_dive_20260612.md`

Other useful reports:

- `nla_fixed_long_vs_current_ppl_long_20260612.md`
- `nla_vs_ppl_fixed_smoke_20260610.md`
- `nla_candidate_content_10_vs_ppl_smoke_20260611.md`
- `nla_summeval_consistency_vs_ppl_smoke_20260611.md`
- `nla_activation_verbalization_quality_20260610.md`

## Current Claim Status

The thesis cannot yet claim that NLA robustly improves GEPA. It can currently
claim that:

- GEPA+PPL can improve the task in the observed long run.
- Raw NLA can hurt when token selection and feedback semantics are weak.
- Fixed-NLA is technically healthier but only weakly positive in the matched
  long run.
- Candidate-only token selection does not solve the issue by itself.
- Auxiliary judge feedback is the next necessary experiment for transforming
  NLA into a more useful proposer signal.

# Results Inventory

This file tracks which results are available for the thesis and how they should
be interpreted.

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

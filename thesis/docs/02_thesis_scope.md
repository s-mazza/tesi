# Thesis Scope

## Working Goal

Study whether latent-to-text methods preserve semantic content when the input is
logically difficult, negated, counterfactual, or contrary to commonsense.

The thesis includes two connected tracks:

- analysis of embedding/activation inversion methods to identify their
  potential and their semantic failure modes;
- Latent-GEPA track: use latent-representation analysis, including perplexity
  and activation verbalization signals, to improve G-Eval-style judge prompts.

GEPA is the latest experimental branch, not the only thesis topic. It tests
whether NLA-style activation information can become useful feedback for prompt
optimization.

## Core Research Questions

1. What do inversion and verbalization methods reveal about semantic content in
   latent representations, especially under negation, polarity changes, and
   counterfactual inputs?
2. Can latent-representation analysis be turned into useful guidance for
   Latent-GEPA, the prompt-optimization pipeline proposed in this thesis for
   improving G-EVAL-style judge prompts?

## Main Contribution Candidates

- A canonical semantic-fidelity dataset covering standard sentences, negation,
  and commonsense/counterfactual pairs.
- A reproduction/diagnostic study of embedding inversion with Jina-style
  conditional masked diffusion.
- A SIPIT reproduction path plus logical-dataset and random-prefix extensions.
- A standalone NLA extraction/verbalization pipeline for Qwen2.5-7B layer-20
  activations.
- A reproducible GEPA pipeline for G-EVAL-style prompt optimization.
- A study of perplexity feedback as an additional signal for prompt proposal.
- A diagnostic study of raw NLA feedback in this setting.
- An auxiliary-judge feedback path that tests whether raw NLA can be compressed
  into more useful rubric-level feedback.

## Current Evidence Position

- The embedding-inversion branch has strong diagnostic evidence but not a clean
  paper-level Jina reproduction.
- The canonical dataset build is reproducible and currently passes validation.
- SIPIT has a completed collision check and strong log-derived interim GPT-2
  Table 5 behavior, but final local CSV/JSON reports still need recovery or
  explicit caveating.
- Standalone NLA on SummEval has completed Qwen2.5-7B layer-20 extraction and
  verbalization smokes.
- PPL feedback produced a clearly positive early long run.
- The first raw-NLA long run was negative and is best treated as a diagnostic
  condition.
- Fixed-NLA is technically healthier, but current long-run evidence is weak:
  metrics improved slightly against a matched PPL control while the optimized
  prompt remained byte-identical to the seed.
- Candidate-only NLA reduced duplicate feedback but did not improve the task.
- Auxiliary judge feedback is the next required test before making a stronger
  claim about NLA.

## Out Of Scope Unless Time Allows

- Claiming exact reproduction of the G-EVAL paper across all original model
  choices.
- Claiming exact reproduction of the Jina embedding-inversion paper unless the
  remaining paper-level gaps are explicitly resolved.
- Training a new embedding inversion model from scratch beyond the already
  documented reproduction diagnostics.
- Training a new NLA checkpoint.
- Treating smoke-test metrics as final scientific evidence.
- Replacing the base judge with Qwen35B. Qwen35B is proposer and optional
  auxiliary judge, not the primary evaluated model.

## Success Criteria

The thesis should be able to defend one of two outcomes:

- Positive outcome: NLA-derived feedback, after suitable transformation,
  improves GEPA over matched PPL controls on paper-aligned metrics.
- Diagnostic outcome: raw NLA does not reliably improve GEPA in this setting,
  and the thesis explains why using artifact-level evidence, inversion
  diagnostics, and ablation runs.

Either outcome must be anchored in the broader semantic-fidelity question, not
only in one prompt-optimization result.

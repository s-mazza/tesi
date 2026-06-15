# Thesis Scope

## Working Goal

Study whether latent-to-text methods preserve semantic content when the input is
logically difficult, negated, counterfactual, or contrary to commonsense.

The thesis includes two connected tracks:

- inversion/verbalization track: embedding inversion, SIPIT hidden-state
  inversion, and NLA activation verbalization;
- GEPA/G-Eval track: use perplexity, NLA verbalizations, and optional
  auxiliary-judge feedback to improve G-Eval-style judge prompts.

GEPA is the latest experimental branch, not the only thesis topic. It tests
whether NLA-style activation information can become useful feedback for prompt
optimization.

## Core Research Questions

1. Do embedding/activation inversion methods preserve logical semantic content,
   or do they reconstruct more plausible but semantically altered text?
2. Do standard reconstruction metrics capture errors such as lost negation,
   inverted polarity, or commonsense normalization?
3. Can SIPIT recover prompts on standard and logical stress-test settings, and
   what happens when continuous random prefixes are introduced?
4. Do NLA verbalizations preserve semantically important activation content, or
   do they produce generic/completion-like descriptions?
5. Can GEPA improve LLM-as-a-judge prompts on G-EVAL-style evaluation tasks?
6. Does adding perplexity feedback help GEPA propose better prompts?
7. Does raw NLA feedback help GEPA, or does it need to be transformed into a
   more rubric-aligned signal?
8. Can a stronger auxiliary judge/proposer help convert NLA feedback into
   useful prompt-level feedback?

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

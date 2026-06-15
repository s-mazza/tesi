# Thesis Scope

## Working Goal

Study whether GEPA can improve G-EVAL-style LLM-as-a-judge prompts, and whether
feedback derived from the base model's internal behavior, especially
perplexity and NLA verbalizations, can make this optimization more effective.

## Core Research Questions

1. Can GEPA improve LLM-as-a-judge prompts on G-EVAL-style evaluation tasks?
2. Does adding perplexity feedback help GEPA propose better prompts?
3. Does raw NLA feedback help GEPA, or does it need to be transformed into a
   more rubric-aligned signal?
4. Can a stronger auxiliary judge/proposer help convert NLA feedback into
   useful prompt-level feedback?

## Main Contribution Candidates

- A reproducible GEPA pipeline for G-EVAL-style prompt optimization.
- A study of perplexity feedback as an additional signal for prompt proposal.
- A diagnostic study of raw NLA feedback in this setting.
- An auxiliary-judge feedback path that tests whether raw NLA can be compressed
  into more useful rubric-level feedback.

## Current Evidence Position

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
- Training a new NLA checkpoint.
- Treating smoke-test metrics as final scientific evidence.
- Replacing the base judge with Qwen35B. Qwen35B is proposer and optional
  auxiliary judge, not the primary evaluated model.

## Success Criteria

The thesis should be able to defend one of two outcomes:

- Positive outcome: GEPA with transformed NLA/auxiliary feedback improves over
  a matched PPL control on paper-aligned metrics.
- Diagnostic outcome: raw NLA does not reliably improve GEPA in this setting,
  and the thesis explains why using artifact-level evidence and ablation runs.

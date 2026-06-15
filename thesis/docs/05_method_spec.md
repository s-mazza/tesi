# Method Specification

## Core Pipeline

Each example contains an input context, a candidate output, and one or more
human scores. The base judge receives the example and a prompt, then produces a
rationale and a discrete score. A metric function compares this score to the
human target and returns both a scalar GEPA optimization score and textual
feedback.

GEPA uses this feedback on train/validation examples to propose and select new
prompts. The selected prompt is evaluated only afterward on the final-test
split.

## Model Roles

- Base judge: `Qwen/Qwen2.5-7B-Instruct`.
- Perplexity model: same base Qwen2.5-7B model.
- NLA model: same base Qwen2.5-7B model and compatible NLA checkpoint.
- Proposer: Qwen35B through llama.cpp.
- Auxiliary judge: Qwen35B through llama.cpp when enabled.

The auxiliary judge does not replace the base judge and does not define final
metrics. It only produces extra proposer feedback.

## Feedback Variants

- `base_gepa`: metric feedback only.
- `ppl`: metric feedback plus response-only perplexity.
- `ppl_nla`: metric feedback plus perplexity plus NLA verbalizations.
- `ppl_nla_auxjudge`: metric feedback plus perplexity plus NLA, compressed or
  interpreted by the auxiliary 35B judge.

## Perplexity Feedback

Perplexity is computed on the candidate response under the base judge model.
The feedback should be treated as an internal diagnostic signal, not a final
evaluation metric.

## NLA Feedback

NLA verbalizes activation vectors for selected tokens. Tokens can come from:

- source/context text;
- reference/fact text;
- candidate output text.

Current evidence shows that candidate-only selection removes duplicates but
does not automatically improve GEPA. The method chapter must explain why token
selection and feedback compression are part of the research question.

## Auxiliary Judge Feedback

The auxiliary judge receives the example, human target, base judge prediction,
agreement score, base rationale, and optional NLA feedback. It should return a
short prompt-level lesson for the proposer, not a replacement final score.

Auxiliary feedback must be non-empty. Empty output is treated as an error and
invalidates the run.

## Artifact Policy

Every long run should preserve:

- metrics;
- seed and optimized prompts;
- prompt trajectory;
- per-example predictions;
- split manifest;
- run configuration;
- runtime manifest;
- NLA artifacts when enabled;
- auxiliary judge artifacts when enabled;
- Slurm/vLLM/llama.cpp logs.

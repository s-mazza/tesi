# Method Specification

## Overall Method Structure

The thesis method has multiple linked pipelines. They should be presented as a
progression, not as unrelated scripts:

1. Build a canonical semantic-fidelity dataset with standard, negation, and
   commonsense/counterfactual blocks.
2. Test latent-to-text reconstruction/verbalization methods on this problem
   family: embedding inversion diagnostics, SIPIT hidden-state inversion, and
   NLA activation verbalization.
3. Use NLA and other base-model internal signals as feedback for GEPA in a
   G-Eval-style LLM-as-a-judge optimization task.

## Canonical Dataset Pipeline

Each canonical row has stable ids, source metadata, split, phenomenon, label,
input text, paired text when applicable, and pair id. The dataset is split into:

- Block A: controlled standard sentences.
- Block B: negation pairs.
- Block C: commonsense/counterfactual pairs.

SIPIT exports tokenize canonical rows with the target model tokenizer. NLA
exports keep canonical ids and paired text so verbalizations can be analyzed
for semantic flips.

## SIPIT Pipeline

SIPIT receives decoder-only LM hidden states and attempts to recover the
original prompt tokens. The thesis implementation includes:

- paper reproduction support for GPT-2 Table 5 and related baselines;
- collision checks;
- a logical dataset built from canonical Blocks B and C;
- a random-prefix extension with separate `full-sequence` and
  `known-prefix-control` modes.

The `known-prefix-control` condition fixes the continuous random prefix and
recovers only the real prompt. It must be kept separate from `full-sequence`,
where SIPIT tries to discretize random continuous prefix vectors into
vocabulary tokens.

## Standalone NLA Pipeline

For Qwen2.5-7B-Instruct, activations are extracted from layer 20 residual stream
positions and verbalized with the compatible Qwen layer-20 AV checkpoint. These
standalone runs validate the NLA plumbing before the same idea is used inside
GEPA.

## GEPA Core Pipeline

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

## GEPA NLA Feedback

In the GEPA branch, NLA verbalizes activation vectors for selected prompt tokens
under the base Qwen2.5-7B judge model. Tokens can come from:

- source/context text;
- reference/fact text;
- candidate output text.

Current evidence shows that candidate-only selection removes duplicates but
does not automatically improve GEPA. The method chapter must explain why token
selection and feedback compression are part of the research question.

This GEPA use of NLA is different from a direct semantic-fidelity evaluation of
NLA. Here, verbalizations are weak proposer feedback, not final labels.

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

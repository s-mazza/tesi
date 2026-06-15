# Chapter 3 Method Plan

This document defines the intended structure of Chapter 3. It should explain
what was built and how the pieces connect, without turning into an environment
or result chapter.

Chapter 3 should be written as a coherent method, not as a list of scripts. The
reader should understand the full pipeline before seeing exact hyperparameters
and numerical results.

## Goal

Describe the thesis method for testing semantic fidelity in latent-to-text
methods and for using internal model signals, especially perplexity and NLA
verbalizations, as feedback in GEPA prompt optimization.

Target length: enough to make the implementation reproducible at the conceptual
level. Exact command lines, machine details, and long tables belong to Chapter
4 or appendices.

## Scope Boundary

Include:

- the canonical semantic-fidelity dataset design;
- the latent-to-text evaluation pipeline;
- embedding-inversion diagnostic branch;
- SIPIT reproduction and thesis-specific extensions;
- standalone NLA extraction and verbalization;
- G-EVAL-style judge task definition;
- GEPA optimization loop;
- perplexity feedback;
- NLA feedback generation for GEPA;
- auxiliary Qwen35B judge feedback;
- artifact policy for reproducibility.

Exclude from Chapter 3:

- exact cluster setup;
- Docker image details;
- vLLM and llama.cpp deployment commands;
- Slurm scheduling;
- full hyperparameter tables;
- final metric values;
- detailed failure analysis of individual runs.

Those topics belong to Chapter 4 for setup and Chapter 5 for results. Chapter 3
should define the method independently of one specific run.

## Proposed Section Structure

### 3.1 Method Overview

Purpose: present the thesis as two connected tracks.

Track 1: evaluate latent-to-text methods under semantic stress.

- embedding inversion diagnostics;
- SIPIT hidden-state inversion;
- NLA activation verbalization.

Track 2: test whether internal signals from the base model can improve
G-EVAL-style prompt optimization.

- base judge predicts scores;
- GEPA optimizes the judge prompt;
- perplexity, NLA, and auxiliary-judge feedback are added as ablations.

The section should explain that GEPA is the application branch where NLA-derived
feedback is tested, not the only thesis topic.

### 3.2 Canonical Semantic-Fidelity Dataset

Purpose: define the dataset used to stress latent-to-text methods.

Each canonical row should include stable ids, source metadata, split,
phenomenon, label, input text, optional paired text, and pair id.

Dataset blocks:

- Block A: controlled standard sentences;
- Block B: negation pairs;
- Block C: commonsense/counterfactual pairs.

Expected discussion:

- why standard sentences are needed as controls;
- why negation and counterfactual examples are semantically hard;
- how paired rows make semantic flips visible;
- how stable ids allow SIPIT, NLA, and later analyses to be joined.

Value-adding example:

- Best format: a small table with one row per block.
- Structure: `block`, `phenomenon`, `input text`, `paired text`, `expected
  semantic issue`.
- Reason: this is the clearest way to show why lexical similarity can be high
  while logical meaning changes. Avoid a long dataset dump in the method
  chapter.

### 3.3 Embedding-Inversion Diagnostic Branch

Purpose: describe the early embedding-inversion reproduction and diagnostic
work without overclaiming it as a clean paper-level reproduction unless the
remaining gaps are resolved.

Expected discussion:

- embedding inversion takes a fixed-size text embedding and tries to recover
  text;
- the Jina-style conditional masked diffusion branch was the main reproduction
  target;
- local diagnostics tested architecture, masking, loss, and checkpoint
  behavior;
- the branch contributes failure-mode evidence and context for the broader
  semantic-fidelity framing.

Value-adding example:

- Best format: a compact method table, not pseudocode.
- Structure: `variant`, `changed component`, `intended diagnostic`, `claim
  status`.
- Reason: this branch involved several diagnostic attempts. A table prevents
  the narrative from sounding like one clean final method if the evidence is
  actually diagnostic.

### 3.4 SIPIT Hidden-State Inversion

Purpose: define how SIPIT is used and extended.

SIPIT receives decoder-only LM hidden states and attempts to recover the
original prompt tokens.

The thesis implementation includes:

- paper reproduction support for GPT-2 Table 5 and related baselines;
- collision checks;
- a logical dataset built from canonical Blocks B and C;
- a random-prefix extension with separate `full-sequence` and
  `known-prefix-control` modes.

Important distinction:

- `known-prefix-control` fixes the continuous random prefix and recovers only
  the real prompt;
- `full-sequence` tries to discretize random continuous prefix vectors into
  vocabulary tokens.

These two settings answer different questions and must not be collapsed into
one result.

Value-adding example:

- Best format: a two-row comparison table.
- Structure: `condition`, `input to SIPIT`, `what is recovered`, `valid claim`.
- Reason: this prevents confusion between exact prompt recovery with a fixed
  continuous prefix and impossible/off-vocabulary recovery of the prefix itself.

### 3.5 Standalone NLA Activation Verbalization

Purpose: explain how NLA is used before it is integrated into GEPA.

For Qwen2.5-7B-Instruct, activations are extracted from layer-20 residual stream
positions and verbalized with the compatible Qwen layer-20 AV checkpoint.

Expected discussion:

- the base model provides the activation vectors;
- the AV produces natural-language descriptions of selected vectors;
- standalone runs validate extraction, checkpoint compatibility, and artifact
  format;
- standalone NLA is not yet a GEPA feedback method.

Value-adding example:

- Best format: a short boxed example with one token and one verbalization.
- Structure: `source text`, `selected token`, `layer`, `AV verbalization`,
  `interpretation caveat`.
- Reason: a concrete example makes the AV concept intuitive, while the caveat
  reminds the reader that readable text is not automatically faithful.

### 3.6 G-EVAL-Style Judge Task

Purpose: define the task optimized by GEPA.

Each example contains:

- an input context;
- a candidate output;
- optional reference, fact, or source document depending on dataset;
- one or more human scores.

The base judge receives the example and a dimension-specific prompt, then
produces a rationale and a discrete score. A metric function compares this
prediction to the human target.

Expected discussion:

- the task is LLM-as-a-judge scoring, not free-form generation;
- the final target is agreement with human judgments;
- train/validation examples are used during GEPA optimization;
- final-test examples are used only after prompt selection.

Value-adding example:

- Best format: a minimal input/output schema block.
- Structure: one JSON-like example showing `context`, `candidate`, `dimension`,
  `human_score`, `judge_score`, and `parsed_score_status`.
- Reason: this is more useful than a prose-only description because later
  prediction artifacts are JSONL.

### 3.7 GEPA Optimization Loop

Purpose: explain the prompt optimizer used in the GEPA branch.

GEPA workflow:

1. start from a seed judge prompt;
2. evaluate candidate prompts on train/validation examples;
3. collect metric feedback and optional auxiliary feedback;
4. ask the proposer model to generate revised prompts;
5. keep candidates that improve validation behavior;
6. evaluate the selected prompt on the held-out final-test split.

Model roles:

- base judge: `Qwen/Qwen2.5-7B-Instruct`;
- proposer: Qwen35B through llama.cpp;
- final metrics: computed from base-judge predictions versus human scores.

The proposer does not replace the base judge. It only proposes prompt edits.

Value-adding example:

- Best format: pseudocode.
- Structure: `seed_prompt`, `trainset`, `valset`, feedback providers, proposer,
  prompt selection, final-test evaluation.
- Reason: GEPA is an algorithmic loop. Pseudocode will be clearer than a
  diagram alone and avoids binding the method to a specific Python file.

### 3.8 Feedback Variants

Purpose: define the experimental method variants as ablations.

Variants:

- `base_gepa`: metric feedback only;
- `ppl`: metric feedback plus response-only perplexity;
- `ppl_nla`: metric feedback plus perplexity plus NLA verbalizations;
- `ppl_nla_auxjudge`: metric feedback plus perplexity plus NLA, compressed or
  interpreted by the auxiliary Qwen35B judge.

Expected discussion:

- each variant adds one feedback source or transformation;
- comparisons must be made against matched controls;
- these are method variants, not final conclusions.

Value-adding example:

- Best format: an ablation table.
- Structure: rows are variants, columns are `metric feedback`, `perplexity`,
  `NLA`, `auxiliary judge`, `scientific question`.
- Reason: this table will be reused by Chapter 4 and makes the experimental
  matrix easier to read.

### 3.9 Perplexity Feedback

Purpose: explain how response-only perplexity is used.

Perplexity is computed on the candidate response under the base
Qwen2.5-7B judge model.

Expected discussion:

- lower perplexity means the model assigned higher probability to the response;
- the signal describes model surprise/confidence;
- perplexity is included in proposer feedback;
- perplexity is not a final evaluation metric and does not replace human-score
  agreement.

Value-adding example:

- Best format: a one-line formula plus a short numeric toy row.
- Structure: define average negative log-likelihood and `ppl = exp(avg_nll)`;
  then show `tokens`, `avg_nll`, `ppl`, and feedback phrase.
- Reason: this is enough to explain the quantity without turning the chapter
  into a probability tutorial.

### 3.10 NLA Feedback For GEPA

Purpose: define how activation verbalizations become proposer feedback.

In the GEPA branch, NLA verbalizes activation vectors for selected prompt/example
tokens under the base Qwen2.5-7B judge model.

Token sources:

- source/context text;
- reference/fact text;
- candidate output text.

Expected discussion:

- selected token positions are converted to activation vectors;
- the AV turns those vectors into text;
- verbalizations are attached to GEPA feedback;
- token selection is part of the method because weak tokens can produce weak
  or repetitive feedback;
- current evidence shows candidate-only selection reduces duplicates but does
  not automatically improve GEPA.

This GEPA use of NLA is different from direct semantic-fidelity evaluation of
NLA. Here, verbalizations are weak proposer feedback, not final labels.

Value-adding example:

- Best format: a compact before/after feedback snippet.
- Structure: `metric feedback only` versus `metric + PPL + NLA feedback` for
  the same example.
- Reason: the reader needs to see what the proposer actually receives, but the
  snippet must be short to avoid turning the method chapter into a log dump.

### 3.11 Auxiliary Judge Feedback

Purpose: explain the Qwen35B auxiliary judge as a feedback compressor.

The auxiliary judge receives:

- the example;
- the human target;
- the base judge prediction;
- the agreement/error signal;
- the base judge rationale;
- optional NLA feedback.

It returns a short prompt-level lesson for the proposer. It does not return the
final score used in paper metrics.

Validity rule:

- auxiliary feedback must be non-empty;
- empty output is an error and invalidates the run;
- the auxiliary judge does not replace the base Qwen2.5-7B judge.

Value-adding example:

- Best format: a three-column table.
- Structure: `raw evidence`, `auxiliary judge lesson`, `prompt-level use`.
- Reason: this makes clear that the auxiliary judge compresses feedback rather
  than becoming the evaluated judge.

### 3.12 Artifact And Reproducibility Policy

Purpose: define what every long run must preserve so Chapter 5 can analyze it
without rerunning expensive jobs.

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

Expected discussion:

- prompt trajectories are needed to inspect what GEPA proposed;
- split manifests are needed to prove final-test separation;
- feedback artifacts are needed to diagnose whether PPL, NLA, or auxiliary
  judge signals were actually present;
- logs and runtime manifests support reproducibility and efficiency analysis.
  Mention them here only as required artifacts; explain deployment details in
  Chapter 4.

## Figures To Add

### Figure 3.1: End-To-End Thesis Method

Required original figure. Best format: draw.io diagram exported as PDF.

Structure:

1. canonical dataset example;
2. latent representation extraction;
3. three latent-to-text branches: embedding inversion, SIPIT, NLA;
4. semantic-fidelity evaluation;
5. GEPA/G-EVAL branch:
   - base judge;
   - metric feedback;
   - optional PPL/NLA/auxiliary feedback;
   - Qwen35B proposer;
   - validation prompt selection;
   - final-test evaluation.

Reason: this is the main bridge between the older inversion work and the newer
GEPA/NLA feedback work.

### Figure 3.2: GEPA Feedback Flow

Optional figure. Add it only if Figure 3.1 becomes too dense or if the GEPA
branch needs to be visually separated.

Best format: small pipeline diagram or algorithm box, also exported as PDF if
drawn.

Structure:

`example -> base judge -> metric feedback -> optional PPL/NLA -> optional
auxiliary judge -> proposer -> revised prompt -> validation selection`.

Reason: this figure can clarify that the auxiliary judge affects feedback to
the proposer but does not define final metrics.

## Examples To Include

Use examples sparingly. The method chapter should include only examples that
clarify a transformation or prevent a likely misunderstanding.

Recommended examples:

- dataset example table in Section 3.2;
- SIPIT `full-sequence` versus `known-prefix-control` comparison table in
  Section 3.4;
- one NLA token/verbalization boxed example in Section 3.5;
- JSON-like G-EVAL example schema in Section 3.6;
- GEPA pseudocode in Section 3.7;
- feedback-variant ablation table in Section 3.8;
- short PPL formula and toy row in Section 3.9;
- short feedback snippet for NLA/auxiliary judge in Sections 3.10 and 3.11.

Avoid full prompts and large log excerpts in Chapter 3. Put full prompts,
complete artifacts, and long examples in an appendix if they are needed.

## Transition To Chapter 4

Close Chapter 3 by stating that the next chapter fixes the experimental
instantiation of this method: datasets and dimensions, model checkpoints,
cluster environment, hyperparameters, metrics, baselines, and comparison
protocol.

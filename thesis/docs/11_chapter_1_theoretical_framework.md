# Chapter 1 Theoretical Framework

This document defines the intended content of Chapter 1. The chapter should
teach only the theory needed to understand the method and experiments that
follow. Implementation details belong to Chapter 4.

## Goal

Give the reader the conceptual tools needed to understand why this thesis cares
about semantic fidelity in latent-to-text methods, and why activation-derived
signals and prompt optimization are relevant later in the thesis.

Target length: about ten pages.

Chapter 1 must avoid presenting named methods or papers as if it were already a
related-work chapter. Names such as SIPIT, NLA, GEPA, and G-EVAL belong to
Chapter 2 and to the method/experimental chapters. Chapter 1 should introduce
only the theoretical concepts needed to understand them.

## Scope Boundary

Include:

- representation types: text embeddings, hidden states, logits, residual
  streams, soft prompts;
- embedding inversion and activation inversion;
- semantic fidelity versus surface-form reconstruction;
- negation, logical polarity, counterfactuals, and commonsense violations;
- exact hidden-state prompt recovery;
- activation verbalization;
- perplexity as a model confidence or surprisal signal;
- LLM-as-a-judge evaluation;
- prompt optimization at a conceptual level;
- human-agreement metrics used later in judge evaluation.

Exclude from Chapter 1:

- cluster setup;
- Docker image details;
- vLLM and llama.cpp deployment details;
- quantization choices;
- flash-attention compatibility;
- Slurm scheduling and runtime engineering.

Those engineering details should be introduced in Chapter 4, where models,
hardware, libraries, and reproducibility constraints are discussed.

## Proposed Section Structure

### 1.1 Neural Text Representations

Explain the representations that appear later in the thesis:

- embeddings as fixed-size vectors used for retrieval and semantic similarity;
- hidden states and residual streams as internal activations of decoder-only
  language models;
- logits as pre-softmax output distributions;
- soft prompts as continuous prompt-like objects.

The goal is not to survey every representation, but to make clear what each
method consumes and produces.

Suggested subsections:

- From Tokens to Internal States.
- Output-Side and Prompt-Side Representations.

### 1.2 Latent-To-Text Inversion

Introduce the general question: given a latent representation, what information
about the original text can be recovered?

Cover the distinction between:

- embedding inversion, where the input is usually a sentence/text embedding;
- hidden-state inversion, where the input is an internal state of an LM;
- activation verbalization, where the output is not necessarily the original
  text, but a natural-language description of what the activation represents.

This section should prepare the reader for later hidden-state inversion and
activation-verbalization methods without becoming a related-work survey.

Suggested subsections:

- The General Recovery Question.
- Main Inversion Targets.
- Different Notions of Success.

### 1.3 Semantic Fidelity

Define why standard reconstruction metrics can be insufficient. A reconstruction
can look close by lexical overlap or embedding similarity while changing:

- negation;
- logical polarity;
- causal direction;
- counterfactual content;
- commonsense plausibility.

Use one or two minimal examples. Do not overload the chapter with all dataset
examples; dataset construction belongs to Chapters 3 and 4.

Suggested subsections:

- Surface Similarity Is Not Meaning.
- Logical Stress Cases.
- Consequences for Evaluation.

### 1.4 Exact Hidden-State Recovery

Explain the underlying concept:

- decoder-only hidden states can contain enough information to recover the
  prompt;
- exact recovery is different from generating a plausible paraphrase;
- exact recovery is a stronger target than generating a plausible continuation;
- exact recovery matters because it gives a clearer upper bound for semantic
  fidelity.

Do not discuss implementation controls such as collision checks, known-prefix
controls, or random-prefix experiments in Chapter 1. Those belong to the method
and experimental chapters, where the reader has enough context.

Suggested subsections:

- Input Recovery as a Stronger Target.
- Why It Matters for Semantic Fidelity.

### 1.5 Activation Verbalization

Introduce the general activation-to-language framing:

- define activation as an intermediate numerical state produced inside a model
  while processing an input;
- define verbalization as translating that internal state into a
  natural-language description of what it appears to encode;
- verbalizers map activation vectors to text;
- reconstructors can map text back toward activations, if relevant later;
- activation verbalizations can be possible semantic evidence;
- verbalizations are useful but not automatically faithful or task-aligned.

This section should make later activation-verbalization experiments
understandable without introducing the named related-work system in Chapter 1.

Suggested subsections:

- What Is Being Verbalized.
- Description Rather Than Reconstruction.
- Faithfulness Caveats.

### 1.6 Perplexity and Model Confidence

Explain perplexity as a surprisal/confidence signal computed from the evaluated
model. Keep it conceptual:

- lower perplexity means the model assigned higher probability to the observed
  output;
- response-only perplexity can be used as auxiliary feedback;
- perplexity is not the same as human-agreement metrics.

Suggested subsections:

- Definition.
- Use as an Auxiliary Signal.

### 1.7 LLM-as-a-Judge Evaluation

Introduce LLM-as-a-judge evaluation:

- introduce the technique at a high level before rubric details: a language
  model receives context, candidate output, and evaluation instructions, then
  returns a score, label, preference, or explanation;
- mention that the paradigm is used in general assistant evaluation,
  rubric-guided NLG evaluation, open evaluator models, and domain-specific
  medical/legal settings;
- define a rubric as the written criteria used to evaluate an output;
- separate the evaluation task from the method used to perform scoring;
- explain that an LLM can be used as the scoring procedure for a rubric;
- distinguish dataset/benchmark, judge prompt, model, and scoring method.

Avoid listing benchmark-specific dimensions in Chapter 1. Use at most one
generic example such as factual consistency to clarify what a criterion is.
Dataset-specific dimensions and exact metrics belong to Chapter 4.

Suggested subsections:

- Rubrics and Scoring Tasks.
- The LLM as the Scoring Procedure.
- The Judge Prompt as Part of the Measurement.

### 1.8 Agreement Metrics

Define how agreement with human judgments is read at a high level:

- correlation metrics measure whether model scores move with human scores;
- higher correlation is better;
- metric choice affects what kind of improvement can be claimed.
- agreement with human annotations is not absolute truth;
- recent judge-alignment work motivates looking beyond raw correlation when
  case-level agreement or human-like judging behavior matters.

The exact metric list, ranges, and paper-aligned comparisons belong to Chapter 4.
Chapter 1 should include compact formulas and citations for Pearson, Spearman,
and Kendall, while Chapter 4 explains the paper-aligned interpretation ranges
and which metrics are used in each experiment.

Suggested subsections:

- Agreement Rather than Absolute Truth.
- Correlation Metrics.
- Interpreting Improvements.

### 1.9 Prompt Optimization

Explain prompt optimization as the process of searching for prompts that improve
task performance:

- evaluate a candidate prompt on examples;
- collect feedback from failures or trajectories;
- use a proposer model to generate revised prompts;
- keep better candidates according to validation performance.

In the LLM-as-a-judge setting, make clear that optimization changes the judge
prompt, not the underlying evaluation task or human reference scores.

The full pipeline, feedback variants, prompt logging, and implementation details
belong to Chapter 3 and Chapter 4.

Suggested subsections:

- Prompts as Search Objects.
- Optimization in Judge Tasks.

## Transition To Chapter 2

Close the chapter by stating that the next chapter positions these concepts in
the literature: embedding inversion, exact hidden-state recovery, activation
verbalization, prompt interpretability, LLM-as-a-judge evaluation, and prompt
optimization.

## Text Structure And Formatting

Use subsections whenever a section combines more than one conceptual block, for
example definition, examples, caveats, and evaluation consequences. Prefer
short, readable subsections over long uninterrupted sections.

Use light typographic emphasis in the final LaTeX:

- `\textbf{...}` for key terms when first introduced or contrasted;
- italics only when helpful for emphasis or terminology;
- no excessive bolding, because Chapter 1 should read like prose rather than
  lecture notes.

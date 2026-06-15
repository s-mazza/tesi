# Chapter 1 Theoretical Framework

This document defines the intended content of Chapter 1. The chapter should
teach only the theory needed to understand the method and experiments that
follow. Implementation details belong to Chapter 4.

## Goal

Give the reader the conceptual tools needed to understand why this thesis cares
about semantic fidelity in latent-to-text methods, and why GEPA/G-EVAL is a
useful later testbed for NLA-derived feedback.

Target length: about ten pages.

## Scope Boundary

Include:

- representation types: text embeddings, hidden states, logits, residual
  streams, soft prompts;
- embedding inversion and activation inversion;
- semantic fidelity versus surface-form reconstruction;
- negation, logical polarity, counterfactuals, and commonsense violations;
- SIPIT as hidden-state prompt recovery;
- Natural Language Activations as activation verbalization;
- perplexity as a model confidence or surprisal signal;
- LLM-as-a-judge evaluation;
- prompt optimization and GEPA at a conceptual level;
- human-agreement metrics used later in G-EVAL-style evaluation.

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

### 1.2 Latent-To-Text Inversion

Introduce the general question: given a latent representation, what information
about the original text can be recovered?

Cover the distinction between:

- embedding inversion, where the input is usually a sentence/text embedding;
- hidden-state inversion, where the input is an internal state of an LM;
- activation verbalization, where the output is not necessarily the original
  text, but a natural-language description of what the activation represents.

This section should prepare the reader for SIPIT and NLA without becoming a
related-work survey.

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

### 1.4 SIPIT And Exact Hidden-State Recovery

Explain SIPIT conceptually:

- decoder-only hidden states can contain enough information to recover the
  prompt;
- exact recovery is different from generating a plausible paraphrase;
- collision checks and known-prefix controls matter because they decide what
  can actually be claimed.

The detailed SIPIT reproduction and random-prefix experiments belong to later
chapters.

### 1.5 Natural Language Activations

Introduce the NLA framing:

- AV maps activation vectors to text;
- AR maps text back to activation vectors;
- the thesis uses AV-style verbalizations as possible semantic evidence;
- verbalizations are useful but not automatically faithful or task-aligned.

This section should make the later GEPA/NLA experiments understandable without
claiming that NLA is already a validated feedback signal.

### 1.6 Perplexity And Model Confidence

Explain perplexity as a surprisal/confidence signal computed from the evaluated
model. Keep it conceptual:

- lower perplexity means the model assigned higher probability to the observed
  output;
- response-only perplexity can be used as auxiliary feedback;
- perplexity is not the same as human-agreement metrics.

### 1.7 LLM-As-A-Judge And G-EVAL

Introduce LLM-as-a-judge evaluation and the G-EVAL task family:

- a model scores generated text according to a rubric;
- final quality is measured by agreement with human scores;
- dimensions such as coherence, consistency, fluency, relevance, and
  engagingness are dataset/task dependent.

Dataset-specific details and exact metrics belong to Chapter 4.

### 1.8 Prompt Optimization And GEPA

Explain prompt optimization as the process of searching for prompts that improve
task performance. Then introduce GEPA only at the conceptual level:

- evaluate a candidate prompt on examples;
- collect feedback from failures or trajectories;
- use a proposer model to generate revised prompts;
- keep better candidates according to validation performance.

The full pipeline, feedback variants, prompt logging, and implementation details
belong to Chapter 3 and Chapter 4.

### 1.9 Agreement Metrics

Define how agreement with human judgments is read at a high level:

- correlation metrics measure whether model scores move with human scores;
- higher correlation is better;
- metric choice affects what kind of improvement can be claimed.

The exact metric list, ranges, and paper-aligned comparisons belong to Chapter 4.

## Transition To Chapter 2

Close the chapter by stating that the next chapter positions these concepts in
the literature: embedding inversion, SIPIT, NLA, prompt interpretability,
G-EVAL, and GEPA.

# Related Work Map

This document tracks the literature to cover before writing Chapter 2.

## Bibliography Policy

- Primary BibTeX source: DBLP.
- Secondary source: Google Scholar when DBLP is missing.
- Avoid URLs in bibliography entries. Use footnotes for software/project links
  when needed.
- For each cited paper, record the specific reason it matters to this thesis.

## Required Work Groups

### Embedding Inversion

Purpose: introduce the latent-to-text problem for embedding vectors and explain
why high lexical or embedding similarity is not enough for the thesis question.

Expected discussion:

- What is inverted: sentence/text embedding vectors.
- Which methods perform iterative autoregressive reconstruction versus
  diffusion-style denoising.
- Why Jina-style conditional masked diffusion became the early reproduction
  target.
- Which standard metrics are usually reported and why they can miss logical
  semantic flips.

### Hidden-State Inversion And SIPIT

Purpose: define exact prompt recovery from decoder-only hidden states and place
SIPIT as the strongest exact-recovery method in the thesis.

Expected discussion:

- SIPIT input/output contract: hidden states of a decoder-only LM to prompt
  tokens.
- Injectivity and collision checks.
- Baselines such as BruteForce and HardPrompts.
- Why logical stress tests and random continuous prefixes are relevant to the
  thesis.

### Prompt Waywardness And Soft-Prompt Interpretability

Purpose: cover the risk of treating continuous prompt representations as if
they were ordinary discrete text.

Expected discussion:

- Discretized interpretation of continuous prompts.
- Prompt length/model-size effects when continuous prompts are projected into
  token space.
- Relation to the SIPIT random-prefix distinction between full-sequence
  recovery and known-prefix controls.

### Natural Language Activations

Purpose: justify why internal activation verbalizations might provide useful
semantic evidence and why they may also be noisy.

Expected discussion:

- AV versus AR: vector-to-text and text-to-vector directions.
- Which checkpoint is compatible with the current base model
  Qwen2.5-7B-Instruct.
- What token/layer is verbalized.
- Why token choice matters.
- Why raw verbalizations may be completion-like or not rubric-aligned.

### G-EVAL And LLM-As-A-Judge

Purpose: define the benchmark family and the evaluation setting this thesis
uses.

Expected discussion:

- What G-EVAL evaluates.
- Which datasets/dimensions are used.
- Which final metrics matter.
- Why model choice is separate from dataset/metric comparability.

### GEPA

Purpose: describe the optimizer used to improve prompts.

Expected discussion:

- Reflective prompt proposal.
- Train/validation use during optimization.
- Final-test separation.
- Prompt trajectory and candidate selection.

### Prompt Optimization

Purpose: position GEPA relative to other prompt search and prompt engineering
methods.

Expected discussion:

- Manual prompt engineering versus automated search.
- Feedback-driven prompt updates.
- Risk of overfitting validation examples.

### Perplexity And Confidence Signals

Purpose: explain perplexity feedback as a lightweight model-internal signal.

Expected discussion:

- Response-only perplexity.
- How it differs from final evaluation metrics.
- How it can help describe model uncertainty or surprise.

### Efficient LLM Inference

Purpose: justify engineering choices needed to run the experiments.

Expected discussion:

- vLLM for serving the base judge.
- llama.cpp for Qwen35B proposer/auxiliary judge.
- Flash attention and prebuilt compatibility constraints.

## Open Bibliography Tasks

- Extract exact BibTeX entries for embedding inversion, Jina conditional masked
  diffusion, SIPIT, Prompt Waywardness, interpretable soft prompts, G-EVAL,
  GEPA, NLA, vLLM, llama.cpp if citable, Qwen2.5, Qwen3/35B source if needed,
  and dataset papers.
- Decide whether software tools should be cited in bibliography or referenced
  in footnotes.

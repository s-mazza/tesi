# Related Work Map

This document tracks the literature to cover before writing Chapter 2.

## Bibliography Policy

- Primary BibTeX source: DBLP.
- Secondary source: Google Scholar when DBLP is missing.
- Avoid URLs in bibliography entries. Use footnotes for software/project links
  when needed.
- For each cited paper, record the specific reason it matters to this thesis.

## Required Work Groups

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

### Natural Language Activations

Purpose: justify why internal activation verbalizations might provide useful
feedback.

Expected discussion:

- What is verbalized.
- Why token choice matters.
- Why raw verbalizations may be noisy or not rubric-aligned.

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

- Extract exact BibTeX entries for G-EVAL, GEPA, NLA, vLLM, llama.cpp if
  citable, Qwen2.5, Qwen3/35B source if needed, and dataset papers.
- Decide whether software tools should be cited in bibliography or referenced
  in footnotes.

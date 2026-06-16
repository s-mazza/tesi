# Chapter 2 Related Work Plan

This document defines the intended structure of Chapter 2. It maps the paper
inventory in `10_paper_reading_list.md` into sections that can later be written
as LaTeX source.

Chapter 2 should not be a generic survey. Each section must explain what prior
work did, why it matters for this thesis, and what gap remains for our work.

## Goal

Position the thesis against prior work on latent-to-text inversion,
activation verbalization, semantic faithfulness, LLM-as-a-judge evaluation, and
prompt optimization.

Target length: about five pages.

## Scope Boundary

Include:

- embedding inversion and text reconstruction from embeddings;
- hidden-state inversion, especially SIPIT;
- semantic/logical stress tests such as negation and commonsense violations;
- prompt waywardness and soft-prompt interpretability;
- Natural Language Activations and activation verbalization;
- LLM-as-a-judge evaluation and G-EVAL;
- prompt optimization, with GEPA as the main optimizer.

Auxiliary feedback signals such as perplexity, NLA verbalizations, and
auxiliary-judge summaries are thesis-specific method variants. They should be
mentioned only as a bridge from GEPA to Chapter 3 unless a directly relevant
related-work citation is added.

Exclude from Chapter 2:

- cluster setup;
- Docker image details;
- vLLM and llama.cpp deployment details;
- quantization choices;
- flash-attention compatibility;
- Slurm scheduling and runtime engineering.

Those topics are implementation constraints, not related-work contributions for
this thesis. They belong in Chapter 4 under experimental setup, infrastructure,
and reproducibility. If a software tool must be cited, cite or footnote it
there, not as a standalone related-work section.

## Bibliography Policy

- Primary BibTeX source: DBLP.
- Secondary source: ACL Anthology or arXiv when DBLP is missing.
- Use Google Scholar only when the above sources do not provide a usable entry.
- Avoid URLs in bibliography entries. Use footnotes for software/project links
  when needed.
- For each cited paper, record the specific reason it matters to this thesis.
- The concrete paper inventory lives in `10_paper_reading_list.md`. This file
  keeps the chapter structure; the reading list keeps canonical URLs, local PDF
  paths, duplicate local copies, and priority notes.

## Figure Policy

Do not add paper figures to Chapter 2 unless they clearly reduce explanatory
text and are explicitly referenced in the surrounding paragraph. The current
draft intentionally avoids paper figures because the related-work claims are
short enough to explain in prose, while copied figures would require careful
copyright handling and source captions.

Preferred figures for the thesis:

- Chapter 3 original method diagram showing the inversion/verbalization branch
  and the GEPA feedback branch;
- Chapter 4 or Chapter 5 tables for datasets, metrics, models, and
  experimental variants;
- Chapter 5 result plots produced from our artifacts.

Possible Chapter 2 paper figures, only if later needed:

- a GEPA optimization-loop figure, if the method text becomes too long;
- an NLA AV/AR schematic, if the distinction between activation verbalization
  and reconstruction remains unclear after Chapter 3 is drafted.

If any paper figure is included, the caption must explicitly state that it is
adapted or reproduced from the cited work and the figure must be mentioned in
the text.

## Proposed Section Structure

### 2.1 Embedding Inversion

Purpose: introduce the literature on reconstructing text from sentence or text
embeddings, and explain why this is the original latent-to-text setting behind
the thesis.

Papers to cover:

- generative embedding inversion attacks, such as GEIA;
- zero-shot or training-free methods, such as ZSinvert and Zero2Text;
- diffusion-style inversion, especially the Jina conditional masked diffusion
  work used in the local reproduction branch;
- Vec2Text or equivalent baselines when needed to explain the field.

Expected discussion:

- what representation is inverted;
- whether the method requires target-encoder access, training pairs, online
  queries, or a trained decoder;
- which metrics are normally reported;
- why high reconstruction or embedding-similarity scores can still miss
  semantic/logical errors.

Thesis contrast:

The thesis is not only asking whether text can be reconstructed. It asks
whether logically important meaning is preserved when the input is negated,
counterfactual, or contrary to commonsense.

### 2.2 Hidden-State Inversion And SIPIT

Purpose: position SIPIT as the strongest exact-recovery method for decoder-only
hidden states and separate hidden-state inversion from embedding inversion.

Papers/resources to cover:

- SIPIT paper;
- SIPIT repository only as implementation reference, not as a separate paper;
- SIPIT baselines such as BruteForce and HardPrompts if they are needed to
  explain the comparison.

Expected discussion:

- SIPIT input/output contract: decoder-only LM hidden states to prompt tokens;
- injectivity claim and what it allows one to recover;
- why exact recovery differs from plausible paraphrase generation;
- why collision checks and known-prefix controls matter;
- how logical stress-test prompts extend the reproduction beyond standard
  benchmark text.

Thesis contrast:

The thesis uses SIPIT not only as a reproduction target, but also as a way to
test whether exact recovery remains meaningful under semantic stress and
continuous/random-prefix variants.

### 2.3 Semantic Stress Tests And Negation

Purpose: motivate why semantic fidelity requires more than standard surface
reconstruction metrics.

Papers/datasets to cover:

- negation benchmarks such as `This is not a Dataset`;
- papers on negation neglect or related failures if they remain relevant after
  the final bibliography pass;
- any commonsense/counterfactual benchmark source used to justify the canonical
  logical dataset.

Expected discussion:

- negation can flip truth conditions while preserving lexical overlap;
- commonsense-plausible rewrites can be semantically wrong;
- standard metrics such as BLEU, ROUGE, BERTScore, or cosine similarity can
  miss polarity and factuality changes;
- this motivates the thesis canonical semantic-fidelity dataset.

Thesis contrast:

The thesis applies this semantic-stress perspective to inversion and
activation-verbalization methods, not only to ordinary LLM classification or
generation.

### 2.4 Prompt Waywardness And Soft-Prompt Interpretability

Purpose: cover the risk of treating continuous prompt-like representations as
if their nearest discrete tokens were faithful natural-language explanations.

Papers/resources to cover:

- Prompt Waywardness;
- Towards Interpretable Soft Prompts;
- the T5 soft-prompt notebook as implementation context only when discussing
  local design inspiration.

Expected discussion:

- continuous prompts can project to discrete strings that are misleading,
  arbitrary, or contradictory;
- prompt interpretability requires faithfulness, not just readable text;
- interpretability and task performance can trade off;
- this is relevant to both SIPIT random-prefix analysis and NLA feedback.

Thesis contrast:

The thesis treats textual explanations of continuous or latent objects as
hypotheses to validate, not as automatically faithful interpretations.

### 2.5 Natural Language Activations

Purpose: introduce NLA as the main prior work for converting activations into
natural-language descriptions.

Papers/resources to cover:

- Natural Language Autoencoders;
- related caveats on activation geometry or concept manifolds if useful for
  explaining why raw verbalizations can be noisy.

Expected discussion:

- AV maps activation vectors to text;
- AR maps text back to activation vectors;
- released checkpoints and model/layer compatibility;
- verbalizations may capture useful semantic information but can also be
  completion-like, generic, repetitive, or not aligned with the downstream
  rubric;
- token and layer choice affect what is verbalized.

Thesis contrast:

The thesis tests whether NLA verbalizations can become useful feedback for GEPA,
not merely whether they are readable in isolation.

### 2.6 LLM-As-A-Judge And G-EVAL

Purpose: introduce the evaluation setting used by the GEPA branch and define
why agreement with human judgments is the relevant target.

Papers to cover:

- G-EVAL as the core benchmark/reference;
- other LLM-as-a-judge work only if needed to contextualize G-EVAL.

Expected discussion:

- G-EVAL evaluates generated text with rubric-like prompts;
- datasets and dimensions include SummEval, Topical-Chat, and QAGS;
- final metrics measure correlation or agreement with human scores;
- model choice for our experiments is separate from dataset/metric alignment
  with the paper.

Thesis contrast:

The thesis uses G-EVAL-style tasks as a controlled testbed for prompt
optimization and NLA-derived feedback, not as a claim of exact model-level
reproduction of the original paper.

### 2.7 Prompt Optimization And GEPA

Purpose: position GEPA among methods that improve prompts using feedback from
task executions.

Papers to cover:

- GEPA as the central prompt optimizer;
- other prompt-search or prompt-optimization papers only if they clarify what
  GEPA does differently.

Expected discussion:

- manual prompt engineering versus automated prompt search;
- reflective prompt proposal from examples, trajectories, and feedback;
- validation-based prompt selection;
- risk of overfitting validation examples;
- why prompt trajectories and intermediate prompts must be stored.

Thesis contrast:

The thesis does not propose a new general prompt optimizer. It studies whether
perplexity, NLA, and auxiliary-judge feedback can make GEPA improve a
G-EVAL-style judge prompt.

### Transition Bridge: Auxiliary Feedback Signals

Do not keep this as a standalone related-work section unless a directly
relevant paper is added. In the current structure, perplexity, NLA
verbalizations, and auxiliary-judge summaries are introduced as thesis-specific
method variants after the GEPA discussion and then defined operationally in
Chapter 3.

Expected transition:

- perplexity was introduced theoretically in Chapter 1;
- NLA is introduced as related work in Section 2.5;
- GEPA is introduced as related work in Section 2.7;
- Chapter 3 combines them as proposal aids, not as final evaluation metrics;
- matched controls are required before claiming that any auxiliary signal
  improves GEPA.

## Transition To Chapter 3

Close Chapter 2 by stating the gap:

Prior work studies inversion, activation verbalization, prompt
interpretability, LLM-as-a-judge evaluation, and prompt optimization mostly as
separate problems. This thesis connects them by asking whether latent semantic
signals, especially NLA verbalizations and perplexity, can improve
G-EVAL-style prompt optimization while preserving a clear train/validation/test
separation.

## Open Bibliography Tasks

- Extract exact BibTeX entries for the core set: embedding inversion papers,
  Jina conditional masked diffusion, SIPIT, Prompt Waywardness, interpretable
  soft prompts, NLA, G-EVAL, GEPA, Qwen model papers if needed, and dataset
  papers.
- Decide which low-priority local PDFs from `10_paper_reading_list.md` should
  become citations and which should remain background reading.
- Decide whether vLLM, llama.cpp, FlashAttention, Docker, and Slurm need
  footnotes or software citations in Chapter 4. They should not be a Chapter 2
  related-work section unless the thesis claim changes toward systems work.

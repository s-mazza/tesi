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
- If DBLP lists both a peer-reviewed venue entry and a `CoRR` entry for the
  same paper, cite the peer-reviewed venue entry.
- Use Google Scholar only when the above sources do not provide a usable entry.
- Avoid URLs in bibliography entries. Use footnotes for software/project links
  when needed.
- For each cited paper, record the specific reason it matters to this thesis.
- The concrete paper inventory lives in `10_paper_reading_list.md`. This file
  keeps the chapter structure; the reading list keeps canonical URLs, local PDF
  paths, duplicate local copies, and priority notes.

## Figure Policy

Use original-paper figures in Chapter 2 when they reduce ambiguity or make a
method easier to understand than prose alone. A figure should not be added only
because the cited work has one: it must be explicitly referenced in the
surrounding paragraph and the caption must state the original source.

For copied or lightly cropped figures:

- keep only the part that helps the reader understand the mechanism;
- remove duplicated paper captions from the image when possible;
- state `Reproduced from ...` or `Adapted from ...` in the thesis caption, but
  do not name the original paper figure number;
- cite the corresponding bibliographic entry in the caption;
- avoid screenshots of tables, logs, or dense algorithm blocks when a normal
  thesis table or pseudocode would be clearer.

Preferred figures for the thesis:

- Chapter 3 original method diagram showing the inversion/verbalization branch
  and the GEPA feedback branch;
- Chapter 4 or Chapter 5 tables for datasets, metrics, models, and
  experimental variants;
- Chapter 5 result plots produced from our artifacts.

Chapter 2 paper figures currently included:

- embedding inversion conditional masked diffusion diagram from Xiao
  \cite{DBLP:journals/corr/abs-2602-11047};
- Prompt Waywardness continuous-prompt/discrete-projection diagram from
  Khashabi et al. \cite{DBLP:conf/naacl/KhashabiLMQ0WHK22};
- NLA activation verbalizer/reconstructor schematic from Fraser-Taliente et al.
  \cite{frasertaliente2026nla};
- G-EVAL framework diagram from Liu et al.
  \cite{DBLP:conf/emnlp/LiuIXWXZ23};
- GEPA candidate-pool and prompt-evolution loop from Agrawal et al.
  \cite{DBLP:journals/corr/abs-2507-19457}.

Figures intentionally not included:

- SIPIT: the locally inspected paper figure was more theoretical than
  explanatory for the reader at this point of the thesis;
- semantic stress tests: the thesis-specific examples explain the issue more
  directly than adding another benchmark figure.

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

### 2.2 Hidden-State Inversion and SIPIT

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

### 2.3 Semantic Stress Tests and Negation

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

### 2.4 Prompt Optimization

Purpose: discuss prompt optimization as a standalone related-work area. The
section should first cover latent/continuous prompt optimization, then hard or
natural-language prompt optimization, with GEPA as the concrete optimizer used
in the thesis.

#### 2.4.1 Latent and Continuous Prompt Optimization

Purpose: cover soft prompting, prefix tuning, and the risk of treating
continuous prompt-like representations as if their nearest discrete tokens were
faithful natural-language explanations.

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

#### 2.4.2 Hard and Natural-Language Prompt Optimization

Purpose: explain prompt optimization methods that search over readable
instructions rather than continuous vectors.

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

#### 2.4.3 Why GEPA Fits This Thesis

Purpose: make clear that GEPA is not the whole prompt-optimization field, but
the natural hard-prompt optimizer for this thesis because it can consume textual
feedback.

Thesis contrast:

The thesis does not propose a new general prompt optimizer. It studies whether
perplexity, NLA, and auxiliary-judge feedback can make GEPA improve a
G-EVAL-style judge prompt.

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

### 2.6 LLM-as-a-Judge and G-EVAL

Purpose: introduce the evaluation setting used by the GEPA branch and define
why agreement with human judgments is the relevant target.

Papers to cover:

- G-EVAL as the core benchmark/reference;
- MT-Bench / Chatbot Arena as general LLM-as-a-judge evaluation context;
- Prometheus as an open evaluator model for fine-grained rubric scoring;
- JUDGE-BENCH as evidence that judge validity varies across tasks;
- agreement-focused judge work, including the Gemma-3-27B-IT result if the
  exact claim is verified;
- medical and legal LLM-as-a-judge examples to show the technique is used
  beyond generic NLG.

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

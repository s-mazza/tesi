# Chapter Outline

## Abstract

Summarize context, problem, method, setup, and main results in about 250 words.
No citations.

## Introduction

Expanded motivation for semantic fidelity in embedding/activation inversion:
surface similarity can hide lost negation, polarity flips, or normalization
toward commonsense. Introduce GEPA/G-Eval as the later branch where activation
verbalizations are tested as feedback for improving LLM-as-a-judge prompts.
Close with the thesis outline.

## Chapter 1: Theoretical Framework

Planned sections:

- Text embeddings, hidden states, logits, residual streams, and soft prompts.
- Embedding inversion and activation inversion.
- Semantic fidelity versus surface-form reconstruction.
- Negation, logical polarity, counterfactuals, and commonsense violations.
- SIPIT and exact hidden-state prompt recovery.
- Perplexity as a model confidence/surprisal signal.
- Natural Language Activation verbalization.
- LLMs as judges and automatic evaluation.
- Prompting and prompt optimization.
- Metrics for agreement with human judgments.

## Chapter 2: Related Work

Planned sections:

- Embedding inversion, including Jina-style conditional masked diffusion.
- SIPIT and hidden-state inversion baselines.
- Prompt waywardness and interpretable soft prompts.
- NLA or activation verbalization work.
- G-EVAL and LLM-as-a-judge evaluation.
- GEPA and reflective prompt optimization.
- Prior work on prompt optimization and prompt search.
- Perplexity/confidence signals in evaluation or feedback.
- Efficient local inference for LLM experiments.

Each section must end by clarifying what this thesis does differently.

## Chapter 3: Method

Planned sections:

- Overall task definition: semantic fidelity of latent-to-text methods.
- Canonical logical/semantic-stress dataset.
- Embedding-inversion reproduction diagnostics.
- SIPIT reproduction, logical dataset export, and random-prefix extension.
- Standalone NLA extraction and verbalization.
- G-Eval task definition and baseline judge prompt.
- GEPA optimization loop.
- Perplexity feedback.
- NLA feedback generation for GEPA.
- Auxiliary 35B judge feedback.
- Prompt trajectory and artifact logging.

Required original figure:

- End-to-end thesis diagram showing dataset example -> latent representation
  extraction -> inversion/verbalization -> semantic-fidelity evaluation, plus
  the GEPA branch: base judge -> metric feedback -> optional PPL/NLA/aux
  feedback -> proposer -> validation selection -> final-test evaluation.

## Chapter 4: Experimental Setup

Planned sections:

- Canonical semantic-fidelity dataset and logical splits.
- SIPIT datasets and prompt-window construction.
- NLA activation extraction targets.
- Datasets and dimensions.
- Models and roles.
- Cluster environment.
- Docker/vLLM/llama.cpp setup.
- Metrics.
- Splits and data budget.
- Hyperparameters and ablations.
- Baselines and comparison protocol.

## Chapter 5: Results

Planned sections:

- Embedding-inversion diagnostics and failure-mode summary.
- SIPIT collision check, Table 5 interim/recovered results, logical dataset,
  and random-prefix results if available.
- Standalone NLA activation-verbalization smokes.
- PPL long-run result.
- Raw-NLA negative long-run result.
- Fixed-NLA smoke and long comparison.
- Candidate-only NLA diagnostic.
- Dataset smoke results.
- Auxiliary judge results once available.
- Efficiency and runtime analysis.

## Chapter 6: Conclusions and Future Work

Summarize what was learned, what can be claimed, and what remains open. Future
work should include stronger NLA transformation strategies, direct NLA semantic
fidelity scoring on the canonical logical dataset, larger GEPA matrix coverage,
and cleaner multi-dimension prompting if developed.

## Acknowledgments

Thank Prof. Moro first, then other collaborators and personal acknowledgments.

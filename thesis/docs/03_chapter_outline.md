# Chapter Outline

## Abstract

Summarize context, problem, method, setup, and main results in about 250 words.
No citations.

## Introduction

Expanded motivation for LLM-as-a-judge evaluation, prompt optimization, and the
need for better feedback signals. Close with the thesis outline.

## Chapter 1: Theoretical Framework

Planned sections:

- Large Language Models and instruction following.
- LLMs as judges and automatic evaluation.
- Prompting and prompt optimization.
- Metrics for agreement with human judgments.
- Perplexity as a model confidence/surprisal signal.
- Natural Language Activation verbalization.
- Efficient inference with vLLM, llama.cpp, quantization, and flash attention.

## Chapter 2: Related Work

Planned sections:

- G-EVAL and LLM-as-a-judge evaluation.
- GEPA and reflective prompt optimization.
- Prior work on prompt optimization and prompt search.
- NLA or activation verbalization work.
- Perplexity/confidence signals in evaluation or feedback.
- Efficient local inference for LLM experiments.

Each section must end by clarifying what this thesis does differently.

## Chapter 3: Method

Planned sections:

- Task definition and notation.
- Baseline G-EVAL prompt.
- GEPA optimization loop.
- Perplexity feedback.
- NLA feedback generation.
- Auxiliary 35B judge feedback.
- Prompt trajectory and artifact logging.

Required original figure:

- End-to-end pipeline diagram showing dataset example -> base judge ->
  metric feedback -> optional PPL/NLA/aux feedback -> GEPA proposer ->
  validation selection -> final-test evaluation.

## Chapter 4: Experimental Setup

Planned sections:

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

- PPL long-run result.
- Raw-NLA negative long-run result.
- Fixed-NLA smoke and long comparison.
- Candidate-only NLA diagnostic.
- Dataset smoke results.
- Auxiliary judge results once available.
- Efficiency and runtime analysis.

## Chapter 6: Conclusions and Future Work

Summarize what was learned, what can be claimed, and what remains open. Future
work should include stronger NLA transformation strategies, larger matrix
coverage, and cleaner multi-dimension prompting if developed.

## Acknowledgments

Thank Prof. Moro first, then other collaborators and personal acknowledgments.

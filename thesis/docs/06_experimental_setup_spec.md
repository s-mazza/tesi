# Experimental Setup Specification

## Datasets And Dimensions

Paper-aligned target matrix:

- Topical-Chat: naturalness, coherence, engagingness, groundedness.
- SummEval: fluency, coherence, consistency, relevance.
- QAGS-CNN: consistency.
- QAGS-XSUM: consistency.

Topical-Chat engagingness is the current main development target because most
long-run evidence exists there.

## Metrics

- Topical-Chat: Pearson and Spearman are the primary paper-aligned metrics.
- SummEval: Spearman and Kendall tau are the primary paper-aligned metrics.
- QAGS-CNN and QAGS-XSUM: Pearson, Spearman, and Kendall tau are primary.
- MAE, normalized agreement, parsed count, and coverage are diagnostics.

Metric descriptions must include directionality and score boundaries.

## Models

- Base judge: Qwen2.5-7B-Instruct.
- Proposer: Qwen35B via llama.cpp.
- Auxiliary judge: Qwen35B via llama.cpp when enabled.
- NLA checkpoint: compatible with the base Qwen2.5-7B model.

The setup chapter must explain that Qwen35B is not used as the final evaluated
judge in the current pipeline.

## Infrastructure

Experiments run on the university Slurm cluster using Docker containers. vLLM
serves the base judge. llama.cpp serves the Qwen35B proposer and optional
auxiliary judge. Flash attention compatibility must be controlled through the
prebuilt environment used by the container.

## Splits

GEPA optimization uses train and validation rows only. Final-test rows are used
only after prompt selection. The exact split manifest must be saved for every
run used in the thesis.

## Hyperparameters To Track

- GEPA budget: `max_full_evals` or equivalent.
- Number of train/validation/test groups.
- Number of threads.
- vLLM max model length.
- vLLM max number of sequences.
- Proposer temperature and max tokens.
- NLA tokens per example.
- NLA extraction layer.
- Auxiliary judge max tokens and success threshold.

The final thesis table should list values tried and mark final values with an
asterisk.

## Baselines

The most important baseline for NLA claims is a matched PPL-only run with the
same dataset, dimension, seed, split sizes, proposer, GEPA budget, and
instruction proposer. Older runs can be reported as historical context but not
as the cleanest ablation.

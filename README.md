# Thesis Workspace

This repository tracks lightweight thesis code, reproducible scripts, and README
documentation. Large artifacts such as paper PDFs, datasets, checkpoints,
generated reports, generated results, advisor messages, and cluster logs are
intentionally ignored.

## Current Direction

The project studies embedding and activation inversion with a focus on semantic
fidelity rather than only surface-form reconstruction. The central question is
whether inversion methods preserve negation, logical polarity, contradiction,
and counterfactual or commonsense-violating content, or whether they normalize
inputs toward more plausible text.

## Active Components

- `embedding-inversion-demo`: prior embedding-inversion reproduction work and
  diagnostics.
- `spit/SIPIT`: prompt recovery method and baselines from the SIPIT paper.
- `prompt-waywardness`: related prompt-inversion / prompt-optimization work.
- `towards_interpretable_softprompts`: related soft-prompt interpretability
  material.

The code repositories above are tracked as submodule references so this parent
repository stays small and suitable for advisor-facing progress updates.

## Immediate Work

The current implemented step is dataset standardization. The pipeline in
`thesis-datasets/` builds a canonical corpus for three blocks:

- Block A: controlled standard short sentences, used as a sanity check.
- Block B: negation pairs from `jinaai/negation-dataset` and
  `HiTZ/This-is-not-a-dataset`.
- Block C: commonsense/counterfactual pairs from SemEval 2020 Task 4 and a
  controlled synthetic set.

Generated reports and local call notes are intentionally kept out of Git. The
README files and scripts should be enough to reproduce them.

## Git Note

In the current Codex sandbox, `.git` is a read-only mountpoint. The parent repo
metadata is stored in `.git-real` locally, so commands in this environment use:

```bash
git --git-dir=.git-real --work-tree=. status
```

Outside that sandbox, the repository can be cloned normally from GitHub.

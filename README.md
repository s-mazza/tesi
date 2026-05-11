# Thesis Workspace

This repository tracks lightweight thesis progress, planning documents, and
reproducible scripts. Large artifacts such as paper PDFs, datasets, checkpoints,
generated results, and cluster logs are intentionally ignored.

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

See `NEXT_STEPS.md` for the current advisor-driven plan: standardize datasets
first, then run small method-specific smoke experiments on SIPIT and NLA AV.

## Git Note

In the current Codex sandbox, `.git` is a read-only mountpoint. The parent repo
metadata is stored in `.git-real` locally, so commands in this environment use:

```bash
git --git-dir=.git-real --work-tree=. status
```

Outside that sandbox, the repository can be cloned normally from GitHub.

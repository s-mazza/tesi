# Progress Update For Advisor Call - 2026-05-11

## Framing

The project focus is now semantic fidelity in embedding / activation inversion:
we are not only measuring whether text is reconstructed, but whether logical
content is preserved. The key phenomenon is that a reconstruction can remain
lexically or embedding-similar while deleting negation, reversing polarity, or
normalizing a counterfactual statement into a commonsense one.

Working research questions:

- RQ1: Do inversion methods reconstruct standard in-distribution text well?
- RQ2: Do standard reconstruction metrics capture logical semantic fidelity?
- RQ3: Is negation preserved or systematically removed / attenuated?
- RQ4: Are counterfactual or commonsense-violating inputs preserved, or corrected
  toward plausible statements?
- RQ5: Is commonsense-normalization bias consistent across inversion families?

## What Is Ready

I implemented a reproducible dataset-standardization pipeline in
`thesis-datasets/`.

Commands:

```bash
python3 thesis-datasets/scripts/build_canonical.py
python3 thesis-datasets/scripts/validate_corpus.py
python3 thesis-datasets/scripts/export_for_sipit.py
python3 thesis-datasets/scripts/export_for_nla.py
```

One-command runner:

```bash
thesis-datasets/scripts/run_all.sh
```

Validation status: `PASS`.

## Corpus Built

Canonical file:

```text
thesis-datasets/processed/canonical.jsonl
```

This file is generated and ignored by Git; the scripts and compact reports are
tracked.

Current build:

| Block | Purpose | Rows |
|---|---:|---:|
| A | Standard short text sanity set | 40 |
| B | Negation | 720 |
| C | Commonsense violations / counterfactuals | 1320 |

Total:

- Rows: `2080`
- Pairs/groups: `1060`
- Validation warnings: `0`
- Validation errors: `0`

Sources:

| Source | Pairs/groups | Rows |
|---|---:|---:|
| Manual controlled standard sentences | 40 | 40 |
| `jinaai/negation-dataset` | 240 | 480 |
| `HiTZ/This-is-not-a-dataset` | 120 | 240 |
| SemEval 2020 Task 4 ComVE | 360 | 720 |
| Controlled synthetic commonsense violations | 300 | 600 |

Synthetic commonsense set:

- Exactly `300` controlled pairs.
- Balanced categories: physics, biology, causality, time, quantity.
- Each pair contains an implausible/counterfactual original and a commonsense
  corrected version.

## Canonical Schema

Each row has:

```json
{
  "id": "...",
  "block": "A|B|C",
  "source": "...",
  "split": "train|validation|test",
  "input_text": "...",
  "paired_text": "...",
  "label": "standard|positive|negative|counterfactual|commonsense_corrected",
  "phenomenon": "standard|negation|commonsense_violation",
  "category": "physics|biology|causality|time|quantity|null",
  "pair_id": "...",
  "metadata": {}
}
```

This makes pairwise semantic-flip evaluation possible: for each reconstruction
or verbalization, we can compare whether it stays closer to the original row or
drifts toward the paired alternative.

## Method Exports Ready

SIPIT smoke export:

```text
thesis-datasets/processed/sipit/
```

NLA Activation Verbalizer manifest:

```text
thesis-datasets/processed/nla/activation_manifest.jsonl
```

These are generated and ignored by Git. The reports describing them are tracked:

- `thesis-datasets/reports/sipit_export_report.md`
- `thesis-datasets/reports/nla_export_report.md`

## Important Method Distinction

SIPIT and NLA AV must not be described as solving the same exact task.

- SIPIT: hidden-state to exact prompt recovery for decoder-only LMs.
- HardPrompts: baseline from SIPIT; approximate and not guaranteed exact.
- NLA AV: activation/residual-stream to text description/verbalization, not
  necessarily exact prompt reconstruction.

The common evaluation layer is semantic fidelity: whether the output preserves
negation, polarity, and counterfactual content.

## Suggested Call Agenda

1. Confirm that the canonical schema is acceptable.
2. Confirm whether Block A should stay as controlled short sentences or also add
   NQ/MS MARCO samples in the first experimental batch.
3. Confirm whether the synthetic Block C categories are sufficient for the first
   300 examples.
4. Decide the first method smoke run:
   - SIPIT on GPT-2 with a small per-block sample.
   - NLA AV on Qwen2.5-7B layer-20 activations.
5. Decide semantic-fidelity metric stack:
   - standard metrics for Block A;
   - pairwise original-vs-paired classification for Blocks B/C;
   - explicit negation-preservation and commonsense-normalization rates.

## Source References

- Advisor message: `messaggio_relatore.txt`.
- SIPIT local paper text: `spit/2510.15511v4.txt`.
- Anthropic NLA: https://transformer-circuits.pub/2026/nla/index.html
- NLA code: https://github.com/kitft/natural_language_autoencoders
- Jina negation dataset: https://huggingface.co/datasets/jinaai/negation-dataset
- This-is-not-a-dataset: https://huggingface.co/datasets/HiTZ/This-is-not-a-dataset
- SemEval 2020 Task 4 ComVE: https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation

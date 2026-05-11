# Dataset Build Report

Canonical path: `/home/mazza/Documents/tesi/thesis-datasets/processed/canonical.jsonl`
Total rows: `2080`
Total pairs/groups: `1060`

## Counts

- By block: `{'A': 40, 'B': 720, 'C': 1320}`
- By phenomenon: `{'commonsense_violation': 1320, 'negation': 720, 'standard': 40}`
- By source: `{'jina_negation': 480, 'manual': 40, 'semeval2020_task4': 720, 'synthetic_commonsense': 600, 'this_is_not': 240}`
- By split: `{'test': 828, 'train': 962, 'validation': 290}`
- By label: `{'commonsense_corrected': 660, 'counterfactual': 660, 'negative': 360, 'positive': 360, 'standard': 40}`

## Pair Counts By Source

- `jina_negation`: `240` pairs/groups
- `manual`: `40` pairs/groups
- `semeval2020_task4`: `360` pairs/groups
- `synthetic_commonsense`: `300` pairs/groups
- `this_is_not`: `120` pairs/groups

## Sources

- Manual controlled standard sentences for Block A.
- Hugging Face `jinaai/negation-dataset` for Block B.
- Hugging Face `HiTZ/This-is-not-a-dataset` for Block B.
- Official SemEval 2020 Task 4 ComVE CSVs from the public GitHub repository for Block C.
- Controlled synthetic commonsense violations for Block C.

## Warnings

- None.

# Semantic-Fidelity Corpus Stats - 2026-06-26

Source artifact:

- `thesis-datasets/processed/canonical.jsonl`

Validation reports checked:

- `thesis-datasets/reports/build_report.md`
- `thesis-datasets/reports/validation_report.md`

The recomputed values match the existing validation reports.

## Global Counts

| Quantity | Value |
|---|---:|
| Total rows | 2080 |
| Pairs/groups | 1060 |
| Validation status | PASS |

## Counts By Block

| Block | Phenomenon | Rows |
|---|---|---:|
| A | Standard controlled sentences | 40 |
| B | Negation and polarity | 720 |
| C | Commonsense/counterfactual violation | 1320 |

## Counts By Source

| Source | Rows |
|---|---:|
| manual | 40 |
| jina_negation | 480 |
| this_is_not | 240 |
| semeval2020_task4 | 720 |
| synthetic_commonsense | 600 |

## Counts By Split

| Split | Rows |
|---|---:|
| train | 962 |
| validation | 290 |
| test | 828 |

## Counts By Label

| Label | Rows |
|---|---:|
| standard | 40 |
| positive | 360 |
| negative | 360 |
| counterfactual | 660 |
| commonsense_corrected | 660 |

## Input Lengths

Whitespace-token statistics over `input_text`.

| Group | n | Mean | Median | p90 | Max |
|---|---:|---:|---:|---:|---:|
| all rows | 2080 | 13.17 | 9 | 26 | 33 |
| Block A | 40 | 7.47 | 8 | 9 | 10 |
| Block B | 720 | 9.43 | 8 | 16 | 33 |
| Block C | 1320 | 15.39 | 11 | 27 | 29 |
| train split | 962 | 15.51 | 10.5 | 26 | 29 |
| validation split | 290 | 10.01 | 8 | 23 | 26 |
| test split | 828 | 11.57 | 9 | 26 | 33 |

## Representative Examples

| Block | Label | Input text | Paired text |
|---|---|---|---|
| A | standard | The meeting starts at nine in the morning. | - |
| B | negative | The church is empty of song. | The church is filled with song. |
| C | commonsense_corrected | when it rains humidity forms | when it is hot humidity forms |

## Thesis Use

Use these values when expanding the Chapter 3 dataset section. Keep exact
artifact paths in planning documents and reports; in the final thesis text,
describe sources and statistics without implementation-level filenames unless
the detail is required for reproducibility.

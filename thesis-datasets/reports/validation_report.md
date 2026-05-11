# Corpus Validation Report

Canonical path: `/home/mazza/Documents/tesi/thesis-datasets/processed/canonical.jsonl`
Rows: `2080`
Pairs/groups: `1060`
Validation status: `PASS`

## Counts By Block

- `A`: `40`
- `B`: `720`
- `C`: `1320`

## Counts By Source

- `jina_negation`: `480`
- `manual`: `40`
- `semeval2020_task4`: `720`
- `synthetic_commonsense`: `600`
- `this_is_not`: `240`

## Counts By Split

- `test`: `828`
- `train`: `962`
- `validation`: `290`

## Counts By Label

- `commonsense_corrected`: `660`
- `counterfactual`: `660`
- `negative`: `360`
- `positive`: `360`
- `standard`: `40`

## Token Lengths

- mean whitespace tokens: `9.14`
- p50: `9`
- p90: `14`
- max: `33`

## Sample Rows

```json
[
  {
    "id": "A_manual_test_0000_standard",
    "block": "A",
    "source": "manual",
    "label": "standard",
    "input_text": "The meeting starts at nine in the morning.",
    "paired_text": null
  },
  {
    "id": "A_manual_test_0005_standard",
    "block": "A",
    "source": "manual",
    "label": "standard",
    "input_text": "A small dog slept under the wooden table.",
    "paired_text": null
  },
  {
    "id": "A_manual_test_0010_standard",
    "block": "A",
    "source": "manual",
    "label": "standard",
    "input_text": "The river flows through the valley.",
    "paired_text": null
  },
  {
    "id": "A_manual_test_0015_standard",
    "block": "A",
    "source": "manual",
    "label": "standard",
    "input_text": "The laptop was placed next to the window.",
    "paired_text": null
  },
  {
    "id": "A_manual_test_0020_standard",
    "block": "A",
    "source": "manual",
    "label": "standard",
    "input_text": "The airplane landed during heavy rain.",
    "paired_text": null
  }
]
```

## Warnings

- None.

## Errors

- None.

# Thesis Datasets

This folder contains the reproducible dataset pipeline for the thesis work on
semantic fidelity in embedding and activation inversion.

Generated dataset payloads are intentionally ignored by Git:

- `raw/`: downloaded source files.
- `processed/`: canonical JSONL and method-specific exports.
- `cache/`: local runtime cache.

Tracked files are limited to scripts, documentation, and compact reports that
can be sent to the advisor.

## Build

```bash
python3 thesis-datasets/scripts/build_canonical.py
python3 thesis-datasets/scripts/validate_corpus.py
python3 thesis-datasets/scripts/export_for_sipit.py
python3 thesis-datasets/scripts/export_for_nla.py
```

The canonical corpus is written to:

```text
thesis-datasets/processed/canonical.jsonl
```

Advisor-facing summaries are written to:

```text
thesis-datasets/reports/
```

## Canonical Schema

Each row has a stable schema:

```json
{
  "id": "block_source_split_index_variant",
  "block": "A|B|C",
  "source": "manual|jina_negation|this_is_not|semeval2020_task4|synthetic_commonsense",
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

## Current Sources

- Manual controlled standard sentences for Block A.
- `jinaai/negation-dataset` for Block B.
- `HiTZ/This-is-not-a-dataset` for Block B.
- Official SemEval 2020 Task 4 ComVE CSV files for Block C.
- 300 controlled synthetic commonsense-violation pairs for Block C.

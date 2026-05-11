# Next Steps: Dataset Standardization

## Goal

Standardize the datasets before running more inversion experiments. The thesis
should test whether inversion methods preserve logical and semantic content,
especially negation and commonsense violations, not only whether reconstructed
text is lexically similar to the input.

## Research Questions

- RQ1: Do SOTA inversion methods reconstruct standard in-distribution text well?
- RQ2: Do standard reconstruction metrics capture logical semantic fidelity?
- RQ3: Is negation preserved or systematically removed/attenuated?
- RQ4: Are counterfactual or commonsense-violating inputs preserved, or corrected
  toward plausible statements?
- RQ5: Is commonsense-normalization bias consistent across inversion families?

## Dataset Blocks

### Block A: Standard Text

Purpose: verify that the pipeline works on ordinary short text.

Sources:
- Short controlled prompts.
- Natural Questions or MS MARCO style short examples.
- Manual/template-generated sentences.

Primary metrics: token accuracy, exact match, ROUGE, BERTScore.

### Block B: Negation

Purpose: test whether logical polarity survives inversion.

Sources:
- `HiTZ/This-is-not-a-dataset`.
- `jinaai/negation-dataset`.

Required fields:
- Positive sentence.
- Negative sentence.
- Pair id.
- Distractor flag when available.
- Negation category/scope when available.

Primary metrics: negation preservation rate, pairwise flip rate, and standard
surface metrics for comparison.

### Block C: Counterfactual / Commonsense Violations

Purpose: test whether methods preserve implausible inputs or normalize them.

Sources:
- SemEval 2020 Task 4 ComVE.
- A synthetic corpus of 300 controlled examples.

Synthetic example schema:

```json
{
  "original": "The fire cooled the water.",
  "corrected": "The fire heated the water.",
  "category": "physics",
  "phenomenon": "commonsense_violation"
}
```

Initial synthetic categories:
- physics
- biology
- causality
- time
- quantity

## Canonical Corpus Schema

Each normalized row should include:

```json
{
  "id": "block_source_split_index_variant",
  "block": "A|B|C",
  "source": "manual|nq|msmarco|this_is_not|jina_negation|semeval2020_task4|synthetic_commonsense",
  "split": "train|validation|test",
  "input_text": "...",
  "paired_text": "...",
  "label": "positive|negative|counterfactual|commonsense_corrected|standard",
  "phenomenon": "standard|negation|commonsense_violation",
  "category": "physics|biology|causality|time|quantity|null",
  "pair_id": "...",
  "metadata": {}
}
```

## Planned Pipeline

- `download_sources.py`: download or cache source datasets with provenance.
- `build_canonical.py`: normalize all blocks into JSONL/Parquet.
- `validate_corpus.py`: check schema, pair integrity, duplicates, counts, and
  tokenizer lengths.
- `export_for_sipit.py`: convert canonical rows into SIPIT `DatasetCollection`
  format plus an id sidecar.
- `export_for_nla.py`: produce activation-extraction manifests for NLA AV.

## Method Scope

Start with training-free or released methods:
- SIPIT as the central prompt-recovery method.
- SIPIT baselines, especially HardPrompts, reported separately.
- Anthropic NLA Activation Verbalizer for activation-to-text verbalization.

Pause new Jina encoder reproduction work unless it becomes necessary later.

## Acceptance Criteria

- Source datasets can be downloaded or their unavailability is documented.
- Canonical corpus builds reproducibly from one command.
- Validation report includes counts by block/source/split/category, duplicate
  rate, sample rows, token length summaries, and source provenance.
- Synthetic Block C contains exactly 300 controlled examples balanced across the
  five initial categories.
- SIPIT and NLA exports are generated from the same canonical rows.

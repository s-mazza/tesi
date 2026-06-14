# NLA Activation Verbalization Quality - 2026-06-10

## Scope

Run analyzed:

- `11913262`: Topical-Chat engagingness smoke, PPL + fixed real NLA, Qwen2.5-7B base judge, Qwen 35B llama.cpp proposer.
- Matched control: `11913161`, same smoke setting with PPL only.
- Main diagnostic report: `gepa-experiments/results/diagnostics/nla_vs_ppl_fixed_smoke_20260610.md`.

Important artifact limitation:

- The current runner does not persist raw activation vectors.
- The available artifacts store token metadata and natural-language verbalizations generated from NLA activation vectors.
- Therefore this analysis is about activation verbalizations, token positions, parser status, duplication, and downstream GEPA behavior. It is not a direct numeric vector-space analysis.

## Quantitative Summary

Input NLA precompute:

- Precomputed rows: 210.
- Covered examples: 36/36.
- Useful rows after parser/useful-row fix: 210/210.
- Token status: `ok` for all 210 rows.
- Parse status: `partial_tags` for all 210 rows after closing-tag-only output is normalized.
- Token position mix:
  - candidate rows: 108.
  - source rows: 36.
  - reference rows: 66.
- Rows per example:
  - 30 examples have 6 rows.
  - 6 examples have 5 rows.
- Verbalization length:
  - min: 7 words.
  - median: 11 words.
  - max: 15 words.
  - mean: about 10.7 words.

Repeated verbalizations:

- Unique precomputed verbalization strings: 123/210.
- Duplicate rows by exact string in direct inspection: 87/210.
- Duplicate rows under the diagnostic script normalization: 107/210.
- The diagnostic value is stricter and should be used for reports.

Metric movement versus the PPL-only control:

- Pearson: 0.536400 -> 0.674979, +25.84%.
- Spearman: 0.527410 -> 0.674693, +27.93%.
- Kendall tau: 0.459933 -> 0.606407, +31.85%.
- Agreement: 0.638889 -> 0.763889, +19.57%.
- MAE: 0.722222 -> 0.472222, 34.62% lower error.
- Prediction movement: 4 examples improved, 1 worsened, 7 unchanged.

## Qualitative Patterns

The fixed selector changed the signal profile in the intended direction:

- It no longer relies mostly on weak first source/reference tokens.
- Candidate activations are now the largest part of the feedback.
- Token status is preserved and all rows are usable.
- Closing-tag-only verbalizer outputs no longer cause false `useful_rows=0` failures.

The signal is still noisy:

- Many verbalizations look like likely text continuations rather than high-level rubric concepts.
- Several repeated verbalizations come from source/reference rows that are identical across all candidate responses in the same context.
- Example repeated source/reference-style outputs include:
  - `genomes" or "brain complexity" or "traits have been identified."`
  - `or "database" or "model organism," completing the scientific claim about fish genetics.`
  - `expecting "month" or "time" or "day for violation."`
- Candidate rows are more response-specific and are likely the more useful part of the feedback for this task.

How GEPA used the signal:

- The PPL-only optimized prompt became stricter and emphasized generic high/medium/low definitions.
- The fixed-NLA optimized prompt introduced a more useful threshold rule:
  - relevant responses that keep the conversation flowing should usually receive at least 2.
  - score 1 should be reserved for off-topic, dismissive, repetitive, or conversation-ending responses.
  - score 3 requires a stronger conversational contribution such as a new angle, follow-up, specific detail, anecdote, or enthusiasm.
- This matches the observed error pattern where the PPL-only run underrated several high-human-score examples.

## Problems And Ambiguities

1. Raw activation vectors are not persisted.

This prevents direct vector diagnostics such as norm distributions, cosine clustering, layer comparisons, or checking whether duplicate verbalizations correspond to duplicate vectors or only to a repetitive verbalizer.

2. Source/reference rows repeat across responses from the same context.

This can inflate feedback length while adding little example-specific information. It may still help by reminding the proposer about topic grounding, but it is probably inefficient and may cause repeated prompt-edit pressure.

3. Verbalizer semantics are often low-level.

The verbalizer frequently describes likely token continuations or topical fragments. It only indirectly maps to rubric-level engagingness. The smoke improvement suggests the proposer can still extract useful direction when combined with metric feedback, but this should not be overclaimed.

4. The smoke test is small.

The final-test slice has 12 examples. The positive movement is strong but not sufficient for thesis-level conclusions.

5. The parser fix is necessary but changes quality accounting.

Before the fix, closing-tag-only outputs were counted as `missing_tags` and rejected. After the fix, they are `partial_tags` with stripped semantic content. This should be documented so old and new NLA quality reports are not compared naively on parse status alone.

## Next Actions

Main pipeline:

- Run the long fixed-NLA Topical-Chat engagingness job and compare it against the closest long PPL-only control.
- If the long result stays positive, treat fixed-NLA as the current main NLA implementation for scaling.
- If the long result reverses, analyze whether the smoke improvement was small-split variance or NLA-induced overfitting.

Separate NLA strategy experiments:

- Finish `candidate_content_6` and `candidate_content_10` smoke jobs.
- Compare their final metrics and feedback-health metrics against `11913262`.
- Candidate-only can be considered for merge only if it improves or matches final metrics while reducing duplicate rows and preserving coverage.

Recommended diagnostics to add next:

- Persist activation-vector summary statistics, not necessarily full vectors:
  - L2 norm.
  - mean and standard deviation.
  - token position.
  - layer.
  - example id.
  - optional per-example cosine similarity between selected activations.
- Track duplicate verbalizations separately for candidate, source, and reference rows.
- Add a per-context duplicate-rate metric to expose repeated source/reference feedback.
- Save a compact per-candidate NLA feedback block as seen by GEPA, so prompt changes can be traced to exact examples.

Decision:

- The fixed-NLA smoke is strong enough to justify the long fixed-NLA run.
- It is not strong enough to merge candidate-only changes or scale all paper dimensions yet.
- The current candidate-only experiments should remain isolated until they prove that they reduce duplicate/noisy feedback without hurting final metrics.

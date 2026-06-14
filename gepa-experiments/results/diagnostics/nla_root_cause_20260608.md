# NLA Root-Cause Analysis - 2026-06-08

## Conclusion

The first long GEPA run with NLA did not fail because NLA is necessarily useless for the thesis idea. It failed because the NLA signal passed to GEPA was low quality for prompt optimization.

The strongest root cause is token selection: with `NLA_MAX_TOKENS_PER_EXAMPLE=3`, the previous selector took one token from source, candidate, and reference, and for each field it selected the first semantic token. In the actual long run this produced weak tokens such as `reading`, `recently`, and `from`. These tokens are not good evidence for deciding Topical-Chat engagingness, so the proposer received long but mostly generic verbalizations.

This means the run tested "GEPA with noisy/weak NLA feedback", not a clean NLA-enhanced GEPA setting.

## Evidence

Control run:

- `gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer`

NLA run:

- `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b`

Diagnostic report:

- `gepa-experiments/results/diagnostics/nla_vs_ppl_long_20260608.md`

The comparison is normalized 1-to-1 except for NLA-specific fields:

- dataset: Topical-Chat
- dimension: engagingness
- seed: 42
- train/validation/test groups: 40/10/10
- judge model: Qwen2.5-7B-Instruct
- proposer: llama.cpp Qwen 35B
- perplexity feedback: enabled in both
- instruction proposer: `generalizing`

Metric movement on final test:

- optimized Pearson: `0.632812 -> 0.511144`, delta `-0.121668`
- optimized Spearman: `0.619893 -> 0.490815`, delta `-0.129078`
- optimized MAE: `0.422222 -> 0.616667`, relative worsening `46.05%`

Prediction-level movement:

- joined final-test examples: 60
- NLA improved absolute error: 6
- NLA worsened absolute error: 18
- unchanged: 36

NLA verbalization quality:

- precomputed rows: 900
- covered GEPA train/validation examples: 300
- rows per covered example: 3
- precompute token status: 900/900 `ok`
- emitted NLA artifact originally lost `token_status`, so the first diagnostic showed `unknown`
- parse status: 900/900 `partial_tags`
- raw generations started with `<explanation>` but never closed `</explanation>`
- source rows: only 33 unique texts over 300 rows
- reference rows: only 55 unique texts over 300 rows
- candidate rows: 259 unique texts over 300 rows
- repeated text rows: 667

Why repetition happened:

- source and reference are shared across multiple responses in the same context, so their first-token activations repeat heavily
- reference first tokens are often weak words such as `from` or `that`
- candidate first tokens are often discourse openers rather than semantically decisive response content

GEPA behavior:

- control best validation score from logs: `0.7666666666666667`
- NLA best validation score from logs: `0.7055555555555557`
- therefore this is not only final-test overfit; NLA also underperformed during validation search
- NLA final prompt became a different rubric that over-penalized several high-human-score responses as score 1
- final-test prediction distribution shifted from control `{2: 36, 1: 14, 3: 10}` to NLA `{1: 30, 2: 21, 3: 9}`

## Fixes Applied

Implemented in commit `93265a1`:

- NLA token selector now prioritizes candidate-output activations.
- With small token budgets, it samples middle/final semantic tokens instead of first tokens.
- With 3 tokens, the selector now chooses 2 candidate tokens and 1 source token.
- NLA emitted artifacts now preserve `token_status`.
- Real-NLA configs now use `NLA_MAX_TOKENS_PER_EXAMPLE=6`.
- Real-NLA configs now use `NLA_PRECOMPUTE_MAX_NEW_TOKENS=160` to reduce partial-tag truncation.
- Removed the fixed-NLA smoke limit that would have failed the new coverage gate.

The pending smoke jobs were synchronized before starting, so they should use the fixed selector/configs.

## Jobs Queued

Existing dataset smoke chain without node pin:

- `11912914`: SummEval consistency, PPL + real NLA
- `11912915`: QAGS-CNN consistency, PPL + real NLA
- `11912916`: QAGS-XSUM consistency, PPL + real NLA

Additional Topical-Chat diagnostic chain:

- `11912917`: Topical-Chat engagingness, PPL-only control, llama.cpp proposer
- `11912918`: Topical-Chat engagingness, PPL + fixed NLA, llama.cpp proposer

Both diagnostic jobs are queued after `11912916` and are not pinned to a specific node. They exclude `deeplearn2`.

## Next Actions

1. Wait for `11912917` and `11912918`.
2. Run `diagnose_nla_run.py` on those two smoke outputs.
3. Check whether fixed NLA improves at least verbalization quality:
   - no first-token-only policy
   - `token_status=ok` visible in emitted artifact
   - lower duplicate rows
   - fewer `partial_tags`
   - NLA feedback mentions candidate content more often
4. If fixed smoke looks better, launch a longer Topical-Chat engagingness PPL+fixed-NLA run.
5. If fixed smoke still fails, next experiments should isolate:
   - NLA without source/reference, candidate-only
   - `NLA_MAX_TOKENS_PER_EXAMPLE=8` or `10`
   - proposer temperature `0.2` or `0.0`
   - different NLA extraction layers if compatible checkpoints exist
   - shorter/summarized NLA feedback before passing it to the proposer

## Current Answer

The current answer is: NLA did not improve because the implementation fed GEPA weak, repetitive, first-token verbalizations. The immediate fix is not to abandon NLA, but to repair token selection and rerun a controlled diagnostic comparison. That fix has been implemented and the controlled jobs are queued.

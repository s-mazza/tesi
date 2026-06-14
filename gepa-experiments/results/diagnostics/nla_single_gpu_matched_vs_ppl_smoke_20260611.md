# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_smoke`
- nla: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_matched_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_single_gpu_matched_vs_ppl_smoke_20260611_predictions.csv`

## Config Check
- OK `dataset`: control=`topical_chat` nla=`topical_chat`
- OK `dimension`: control=`engagingness` nla=`engagingness`
- OK `seed`: control=`42` nla=`42`
- OK `train_groups`: control=`4` nla=`4`
- OK `val_groups`: control=`2` nla=`2`
- OK `test_groups`: control=`2` nla=`2`
- OK `judge_model`: control=`Qwen/Qwen2.5-7B-Instruct` nla=`Qwen/Qwen2.5-7B-Instruct`
- OK `proposer_model`: control=`Qwen/Qwen2.5-7B-Instruct` nla=`Qwen/Qwen2.5-7B-Instruct`
- OK `proposer_temperature`: control=`0.0` nla=`0.0`
- OK `proposer_max_tokens`: control=`4096` nla=`4096`
- OK `instruction_proposer`: control=`generalizing` nla=`generalizing`
- OK `perplexity_feedback`: control=`True` nla=`True`
- DIFF `nla_feedback`: control=`False` nla=`True`
- DIFF `nla_backend`: control=`` nla=`precomputed`
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_matched_smoke/nla_precomputed_11913388.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`6`

## Metric Deltas
- baseline `agreement` (higher better): control=0.722222, nla=0.722222, delta=0.000000, rel_pct=0.000000
- baseline `pearson` (higher better): control=0.603136, nla=0.603136, delta=0.000000, rel_pct=0.000000
- baseline `spearman` (higher better): control=0.590879, nla=0.590879, delta=0.000000, rel_pct=0.000000
- baseline `kendall_tau` (higher better): control=nan, nla=nan, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=0.555556, nla=0.555556, delta=0.000000, rel_pct=0.000000
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.625000, nla=0.722222, delta=0.097222, rel_pct=15.555556
- optimized `pearson` (higher better): control=0.314271, nla=0.603136, delta=0.288864, rel_pct=91.915682
- optimized `spearman` (higher better): control=0.345082, nla=0.590879, delta=0.245797, rel_pct=71.228389
- optimized `kendall_tau` (higher better): control=nan, nla=nan, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=0.750000, nla=0.555556, delta=-0.194444, rel_pct=-25.925926
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 12
- NLA improved abs error: 3
- NLA worsened abs error: 0
- unchanged: 9

## NLA Feedback Quality
- verbalization rows: 216
- covered examples: 36
- avg rows per covered example: 6.000000
- avg verbalization words: 11.523148
- suspicious rows: 0
- duplicate text rows: 114
- duplicate text rows by category: `{'candidate': 7, 'source': 107}`
- parse status: `{'missing_tags': 216}`
- token status: `{'ok': 216}`
- token position prefixes: `{'source': 108, 'candidate': 108}`
- token categories: `{'source': 108, 'candidate': 108}`
- rows with activation summary stats: 0

## Trajectory
- control: `{'candidates': 7, 'accepted_candidates': 4, 'avg_prompt_words': 214.57142857142858, 'max_prompt_words': 241}`
- nla: `{'candidates': 11, 'accepted_candidates': 1, 'avg_prompt_words': 197.0, 'max_prompt_words': 228}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=6: or "the conversation is about..." or "a sentence about weather."
</explanation>
- count=6: like "the Astros" or "the Rangers last year."
</explanation>
- count=6: " or "in the lineup" or "both" or "great hitters."
</explanation>
- count=6: day" or "I am a customer" — a greeting or prompt phrase.
</explanation>
- count=6: or "version of the character is more likely to be" — a specific topic about creativity.
</explanation>

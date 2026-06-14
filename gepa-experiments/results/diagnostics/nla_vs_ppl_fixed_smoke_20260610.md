# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke`
- nla: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_vs_ppl_fixed_smoke_20260610_predictions.csv`

## Config Check
- OK `dataset`: control=`topical_chat` nla=`topical_chat`
- OK `dimension`: control=`engagingness` nla=`engagingness`
- OK `seed`: control=`42` nla=`42`
- OK `train_groups`: control=`4` nla=`4`
- OK `val_groups`: control=`2` nla=`2`
- OK `test_groups`: control=`2` nla=`2`
- OK `judge_model`: control=`Qwen/Qwen2.5-7B-Instruct` nla=`Qwen/Qwen2.5-7B-Instruct`
- OK `proposer_model`: control=`local-llamacpp` nla=`local-llamacpp`
- OK `proposer_temperature`: control=`0.7` nla=`0.7`
- OK `proposer_max_tokens`: control=`4096` nla=`4096`
- OK `instruction_proposer`: control=`generalizing` nla=`generalizing`
- OK `perplexity_feedback`: control=`True` nla=`True`
- DIFF `nla_feedback`: control=`False` nla=`True`
- DIFF `nla_backend`: control=`` nla=`precomputed`
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke/nla_precomputed_11913262.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`6`

## Metric Deltas
- baseline `agreement` (higher better): control=0.722222, nla=0.722222, delta=0.000000, rel_pct=0.000000
- baseline `pearson` (higher better): control=0.603136, nla=0.603136, delta=0.000000, rel_pct=0.000000
- baseline `spearman` (higher better): control=0.590879, nla=0.590879, delta=0.000000, rel_pct=0.000000
- baseline `kendall_tau` (higher better): control=0.531588, nla=0.531588, delta=0.000000, rel_pct=0.000000
- baseline `mae` (lower better): control=0.555556, nla=0.555556, delta=0.000000, rel_pct=0.000000
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.638889, nla=0.763889, delta=0.125000, rel_pct=19.565217
- optimized `pearson` (higher better): control=0.536400, nla=0.674979, delta=0.138580, rel_pct=25.835188
- optimized `spearman` (higher better): control=0.527410, nla=0.674693, delta=0.147283, rel_pct=27.925814
- optimized `kendall_tau` (higher better): control=0.459933, nla=0.606407, delta=0.146474, rel_pct=31.846851
- optimized `mae` (lower better): control=0.722222, nla=0.472222, delta=-0.250000, rel_pct=-34.615385
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 12
- NLA improved abs error: 4
- NLA worsened abs error: 1
- unchanged: 7

## NLA Feedback Quality
- verbalization rows: 210
- covered examples: 36
- avg rows per covered example: 5.833333
- avg verbalization words: 10.776190
- suspicious rows: 0
- duplicate text rows: 107
- parse status: `{'partial_tags': 210}`
- token status: `{'ok': 210}`
- token position prefixes: `{'candidate': 108, 'source': 36, 'reference': 66}`

## Trajectory
- control: `{'candidates': 7, 'accepted_candidates': 3, 'avg_prompt_words': 222.42857142857142, 'max_prompt_words': 295}`
- nla: `{'candidates': 9, 'accepted_candidates': 2, 'avg_prompt_words': 240.66666666666666, 'max_prompt_words': 292}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=6: like "the Astros" or "the Rangers last year."
- count=6: "uniformed officials" or "employees of the team" completing the historical context.
- count=6: or "version of the character is more likely to be" — a specific topic about creativity.
- count=6: " or "of overproduction" or "and revenue failed to..."
- count=6: " or "ticket" or "record" to close the noun phrase describing failure.

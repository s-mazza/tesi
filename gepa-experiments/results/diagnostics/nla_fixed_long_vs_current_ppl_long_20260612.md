# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control`
- nla: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_fixed_long_vs_current_ppl_long_20260612.prediction_errors.csv`

## Config Check
- OK `dataset`: control=`topical_chat` nla=`topical_chat`
- OK `dimension`: control=`engagingness` nla=`engagingness`
- OK `seed`: control=`42` nla=`42`
- OK `train_groups`: control=`40` nla=`40`
- OK `val_groups`: control=`10` nla=`10`
- OK `test_groups`: control=`10` nla=`10`
- OK `judge_model`: control=`Qwen/Qwen2.5-7B-Instruct` nla=`Qwen/Qwen2.5-7B-Instruct`
- OK `proposer_model`: control=`local-llamacpp` nla=`local-llamacpp`
- OK `proposer_temperature`: control=`0.7` nla=`0.7`
- OK `proposer_max_tokens`: control=`4096` nla=`4096`
- OK `instruction_proposer`: control=`generalizing` nla=`generalizing`
- OK `perplexity_feedback`: control=`True` nla=`True`
- DIFF `nla_feedback`: control=`False` nla=`True`
- DIFF `nla_backend`: control=`` nla=`precomputed`
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b/nla_precomputed_11913284.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`6`

## Metric Deltas
- baseline `agreement` (higher better): control=0.741667, nla=0.752778, delta=0.011111, rel_pct=1.498127
- baseline `pearson` (higher better): control=0.658218, nla=0.681158, delta=0.022940, rel_pct=3.485195
- baseline `spearman` (higher better): control=0.658203, nla=0.677076, delta=0.018872, rel_pct=2.867246
- baseline `kendall_tau` (higher better): control=0.555309, nla=0.571150, delta=0.015841, rel_pct=2.852694
- baseline `mae` (lower better): control=0.516667, nla=0.494444, delta=-0.022222, rel_pct=-4.301075
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.741667, nla=0.752778, delta=0.011111, rel_pct=1.498127
- optimized `pearson` (higher better): control=0.658218, nla=0.681158, delta=0.022940, rel_pct=3.485195
- optimized `spearman` (higher better): control=0.658203, nla=0.677076, delta=0.018872, rel_pct=2.867246
- optimized `kendall_tau` (higher better): control=0.555309, nla=0.571150, delta=0.015841, rel_pct=2.852694
- optimized `mae` (lower better): control=0.516667, nla=0.494444, delta=-0.022222, rel_pct=-4.301075
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 60
- NLA improved abs error: 2
- NLA worsened abs error: 0
- unchanged: 58

## NLA Feedback Quality
- verbalization rows: 1752
- covered examples: 300
- avg rows per covered example: 5.840000
- avg verbalization words: 10.549087
- suspicious rows: 0
- duplicate text rows: 925
- duplicate text rows by category: `{'candidate': 77, 'reference': 550, 'source': 298}`
- parse status: `{'partial_tags': 1752}`
- token status: `{'ok': 1752}`
- token position prefixes: `{'candidate': 888, 'source': 300, 'reference': 564}`
- token categories: `{'candidate': 888, 'source': 300, 'reference': 564}`
- rows with activation summary stats: 0

## Trajectory
- control: `{'candidates': 799, 'accepted_candidates': 130, 'avg_prompt_words': 282.1188986232791, 'max_prompt_words': 449}`
- nla: `{'candidates': 740, 'accepted_candidates': 136, 'avg_prompt_words': 293.3810810810811, 'max_prompt_words': 481}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=6: candy bar surprise" or "the first day of a sports revolution."
- count=6: "the first" or "a famous name" or "before his NFL career."
- count=6: waste time" or "that's a good point about laziness."
- count=6: came from trees" or "they stored things," completing the quirky fact.
- count=6: time" or "a while ago," completing the personal confession about sports decline.

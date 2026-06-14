# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer`
- nla: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_fixed_long_vs_ppl_long_20260611_predictions.csv`

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
- DIFF `nla_feedback`: control=`` nla=`True`
- DIFF `nla_backend`: control=`` nla=`precomputed`
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b/nla_precomputed_11913284.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`` nla=`6`

## Metric Deltas
- baseline `agreement` (higher better): control=0.761111, nla=0.752778, delta=-0.008333, rel_pct=-1.094891
- baseline `pearson` (higher better): control=0.551831, nla=0.681158, delta=0.129326, rel_pct=23.435866
- baseline `spearman` (higher better): control=0.547848, nla=0.677076, delta=0.129228, rel_pct=23.588350
- baseline `kendall_tau` (higher better): control=nan, nla=0.571150, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=0.477778, nla=0.494444, delta=0.016667, rel_pct=3.488372
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.788889, nla=0.752778, delta=-0.036111, rel_pct=-4.577465
- optimized `pearson` (higher better): control=0.632812, nla=0.681158, delta=0.048346, rel_pct=7.639919
- optimized `spearman` (higher better): control=0.619893, nla=0.677076, delta=0.057183, rel_pct=9.224616
- optimized `kendall_tau` (higher better): control=nan, nla=0.571150, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=0.422222, nla=0.494444, delta=0.072222, rel_pct=17.105263
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 60
- NLA improved abs error: 8
- NLA worsened abs error: 15
- unchanged: 37

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
- control: `{'candidates': 0, 'accepted_candidates': 0, 'avg_prompt_words': 0.0, 'max_prompt_words': 0}`
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

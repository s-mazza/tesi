# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_smoke`
- nla: `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_real_nla_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_summeval_consistency_vs_ppl_smoke_20260611_predictions.csv`

## Config Check
- OK `dataset`: control=`summeval` nla=`summeval`
- OK `dimension`: control=`consistency` nla=`consistency`
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
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_summeval_consistency_ppl_real_nla_smoke/nla_precomputed_11913111.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`2`

## Metric Deltas
- baseline `agreement` (higher better): control=0.656250, nla=0.656250, delta=0.000000, rel_pct=0.000000
- baseline `pearson` (higher better): control=0.635545, nla=0.659360, delta=0.023814, rel_pct=3.747073
- baseline `spearman` (higher better): control=0.505233, nla=0.615213, delta=0.109980, rel_pct=21.768140
- baseline `kendall_tau` (higher better): control=0.469183, nla=nan, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=1.375000, nla=1.375000, delta=0.000000, rel_pct=0.000000
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.723958, nla=0.718750, delta=-0.005208, rel_pct=-0.719424
- optimized `pearson` (higher better): control=0.701281, nla=0.618512, delta=-0.082769, rel_pct=-11.802605
- optimized `spearman` (higher better): control=0.790860, nla=0.696582, delta=-0.094278, rel_pct=-11.920966
- optimized `kendall_tau` (higher better): control=0.716853, nla=nan, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=1.104167, nla=1.125000, delta=0.020833, rel_pct=1.886792
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 32
- NLA improved abs error: 3
- NLA worsened abs error: 3
- unchanged: 26

## NLA Feedback Quality
- verbalization rows: 100
- covered examples: 50
- avg rows per covered example: 2.000000
- avg verbalization words: 48.460000
- suspicious rows: 0
- duplicate text rows: 86
- duplicate text rows by category: `{'candidate': 37, 'source': 49}`
- parse status: `{'partial_tags': 100}`
- token status: `{'unknown': 100}`
- token position prefixes: `{'source': 50, 'candidate': 50}`
- token categories: `{'source': 50, 'candidate': 50}`
- rows with activation summary stats: 0

## Trajectory
- control: `{'candidates': 15, 'accepted_candidates': 4, 'avg_prompt_words': 206.33333333333334, 'max_prompt_words': 241}`
- nla: `{'candidates': 7, 'accepted_candidates': 5, 'avg_prompt_words': 180.28571428571428, 'max_prompt_words': 214}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=14: Structured sports article format with numbered questions and answers pattern, presenting football player statistics in a template format with a specific player's name and performance details.

The ope
- count=11: Wiki-style article format with "Did you know?" header suggesting a trivia or historical context about a figure, likely a tech or financial term.

The text "A famous quote about Serge" strongly implies
- count=9: Historical/encyclopedic article format with a quoted passage structure ("The mysterious term 'The glowing"), suggesting a literary or cultural context about a scientific term or phenomenon.

The phras
- count=7: Historical/encyclopedic article format with a quoted passage structure ("The mysterious term 'The glowing"), suggesting a literary or cultural reference about a scientific concept or historical event.
- count=5: Wiki-style article format with "Did you know?" header suggesting a trivia or historical context about a figure, likely a tech or cultural event.

The text "Serge" is the opening of a proper noun name,

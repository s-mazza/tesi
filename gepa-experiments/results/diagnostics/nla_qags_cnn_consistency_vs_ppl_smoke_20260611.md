# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_smoke`
- nla: `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_qags_cnn_consistency_vs_ppl_smoke_20260611_predictions.csv`

## Config Check
- OK `dataset`: control=`qags_cnn` nla=`qags_cnn`
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
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_real_nla_smoke/nla_precomputed_11913112.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`2`

## Metric Deltas
- baseline `agreement` (higher better): control=0.736111, nla=0.736111, delta=0.000000, rel_pct=0.000000
- baseline `pearson` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- baseline `spearman` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- baseline `kendall_tau` (higher better): control=1.000000, nla=nan, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=1.055556, nla=1.055556, delta=0.000000, rel_pct=0.000000
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.847222, nla=0.736111, delta=-0.111111, rel_pct=-13.114754
- optimized `pearson` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `spearman` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `kendall_tau` (higher better): control=1.000000, nla=nan, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=0.611111, nla=1.055556, delta=0.444444, rel_pct=72.727273
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 2
- NLA improved abs error: 1
- NLA worsened abs error: 1
- unchanged: 0

## NLA Feedback Quality
- verbalization rows: 12
- covered examples: 6
- avg rows per covered example: 2.000000
- avg verbalization words: 49.333333
- suspicious rows: 0
- duplicate text rows: 0
- duplicate text rows by category: `{'candidate': 0, 'source': 0}`
- parse status: `{'partial_tags': 12}`
- token status: `{'unknown': 12}`
- token position prefixes: `{'source': 6, 'candidate': 6}`
- token categories: `{'source': 6, 'candidate': 6}`
- rows with activation summary stats: 0

## Trajectory
- control: `{'candidates': 3, 'accepted_candidates': 2, 'avg_prompt_words': 185.33333333333334, 'max_prompt_words': 232}`
- nla: `{'candidates': 3, 'accepted_candidates': 2, 'avg_prompt_words': 185.66666666666666, 'max_prompt_words': 210}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=1: Arabic-language article format with historical/educational tone, presenting a quote about a politician's death, likely a news headline or blog post structure.

The sentence "This was one of the greate
- count=1: Structured article format with numbered questions and bold headers continuing a pattern of movie title analysis with data visualization, presenting a specific movie prediction question about "The Grea
- count=1: Formal article structure with "Definition" header signals a list or article format, likely a quiz or informational content about a word or concept.

The phrase "The phrase 'Dogs" opens a specific exam
- count=1: Structured article format with numbered claims presenting a headline claim about AI news article data, alternating with a specific study result about dogs' ability to detect cancer.

The paragraph ope
- count=1: Formal news article structure with headline format ("Australian English language news headline generator") suggests a list or article context about climate change or political event.

The phrase "Sydn

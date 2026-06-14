# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_smoke`
- nla: `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_qags_xsum_consistency_vs_ppl_smoke_20260611_predictions.csv`

## Config Check
- OK `dataset`: control=`qags_xsum` nla=`qags_xsum`
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
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_real_nla_smoke/nla_precomputed_11913113.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`2`

## Metric Deltas
- baseline `agreement` (higher better): control=0.708333, nla=0.666667, delta=-0.041667, rel_pct=-5.882353
- baseline `pearson` (higher better): control=1.000000, nla=0.000000, delta=-1.000000, rel_pct=-100.000000
- baseline `spearman` (higher better): control=1.000000, nla=0.000000, delta=-1.000000, rel_pct=-100.000000
- baseline `kendall_tau` (higher better): control=1.000000, nla=nan, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=1.166667, nla=1.333333, delta=0.166667, rel_pct=14.285714
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.708333, nla=0.666667, delta=-0.041667, rel_pct=-5.882353
- optimized `pearson` (higher better): control=1.000000, nla=0.000000, delta=-1.000000, rel_pct=-100.000000
- optimized `spearman` (higher better): control=1.000000, nla=0.000000, delta=-1.000000, rel_pct=-100.000000
- optimized `kendall_tau` (higher better): control=1.000000, nla=nan, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=1.166667, nla=1.333333, delta=0.166667, rel_pct=14.285714
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 2
- NLA improved abs error: 0
- NLA worsened abs error: 1
- unchanged: 1

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
- control: `{'candidates': 3, 'accepted_candidates': 1, 'avg_prompt_words': 175.66666666666666, 'max_prompt_words': 189}`
- nla: `{'candidates': 3, 'accepted_candidates': 2, 'avg_prompt_words': 167.33333333333334, 'max_prompt_words': 196}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=1: British media headline format with informal tone and social media-style posts, suggesting a news article or tweet listing celebrity gossip items.

The opening sentence "the bbc has confirmed" signals 
- count=1: Structured news headline format with alternating quotes and bullet points presenting a mock Twitter-style headline pattern about a cryptocurrency's stock price announcement.

The paragraph opening "Th
- count=1: Wiki article format with "Star Wars" context and "Trivia" section structure suggests a quoted or named example is being introduced, likely a character or concept from a game or media.

The phrase "Dar
- count=1: British social media format with numbered list structure presenting a meme headline analysis, mixing sports event details with a specific UK political award format.

The paragraph "a new meme about a 
- count=1: Academic/encyclopedic tone with a headline format suggesting a news article or blog post about economics, likely referencing a specific study or phenomenon.

The text "The term 'Cost of living' " begi

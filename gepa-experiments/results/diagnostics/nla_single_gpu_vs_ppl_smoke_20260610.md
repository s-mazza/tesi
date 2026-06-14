# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_smoke`
- nla: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_single_gpu_vs_ppl_smoke_20260610_predictions.csv`

## Config Check
- OK `dataset`: control=`topical_chat` nla=`topical_chat`
- OK `dimension`: control=`engagingness` nla=`engagingness`
- OK `seed`: control=`42` nla=`42`
- DIFF `train_groups`: control=`4` nla=`2`
- DIFF `val_groups`: control=`2` nla=`1`
- DIFF `test_groups`: control=`2` nla=`1`
- OK `judge_model`: control=`Qwen/Qwen2.5-7B-Instruct` nla=`Qwen/Qwen2.5-7B-Instruct`
- OK `proposer_model`: control=`Qwen/Qwen2.5-7B-Instruct` nla=`Qwen/Qwen2.5-7B-Instruct`
- OK `proposer_temperature`: control=`0.0` nla=`0.0`
- DIFF `proposer_max_tokens`: control=`4096` nla=`2048`
- OK `instruction_proposer`: control=`generalizing` nla=`generalizing`
- OK `perplexity_feedback`: control=`True` nla=`True`
- DIFF `nla_feedback`: control=`False` nla=`True`
- DIFF `nla_backend`: control=`` nla=`precomputed`
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_smoke/nla_precomputed_11913131.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`2`

## Metric Deltas
- baseline `agreement` (higher better): control=0.722222, nla=0.694444, delta=-0.027778, rel_pct=-3.846154
- baseline `pearson` (higher better): control=0.603136, nla=0.643921, delta=0.040785, rel_pct=6.762228
- baseline `spearman` (higher better): control=0.590879, nla=0.651533, delta=0.060654, rel_pct=10.265016
- baseline `kendall_tau` (higher better): control=nan, nla=nan, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=0.555556, nla=0.611111, delta=0.055556, rel_pct=10.000000
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.625000, nla=0.694444, delta=0.069444, rel_pct=11.111111
- optimized `pearson` (higher better): control=0.314271, nla=0.643921, delta=0.329650, rel_pct=104.893457
- optimized `spearman` (higher better): control=0.345082, nla=0.651533, delta=0.306450, rel_pct=88.805011
- optimized `kendall_tau` (higher better): control=nan, nla=nan, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=0.750000, nla=0.611111, delta=-0.138889, rel_pct=-18.518519
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 0
- NLA improved abs error: 0
- NLA worsened abs error: 0
- unchanged: 0

## NLA Feedback Quality
- verbalization rows: 24
- covered examples: 12
- avg rows per covered example: 2.000000
- avg verbalization words: 29.500000
- suspicious rows: 18
- duplicate text rows: 21
- duplicate text rows by category: `{'candidate': 0, 'source': 21}`
- parse status: `{'dry_run': 18, 'partial_tags': 6}`
- token status: `{'unknown': 24}`
- token position prefixes: `{'source': 21, 'candidate': 3}`
- token categories: `{'source': 21, 'candidate': 3}`
- rows with activation summary stats: 0

## Trajectory
- control: `{'candidates': 7, 'accepted_candidates': 4, 'avg_prompt_words': 214.57142857142858, 'max_prompt_words': 241}`
- nla: `{'candidates': 3, 'accepted_candidates': 1, 'avg_prompt_words': 197.66666666666666, 'max_prompt_words': 209}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=18: DRY RUN ONLY: placeholder for a real NLA activation verbalization at this semantic token. Do not use this backend for scientific results.
- count=3: Formal chat format with numbered questions and a prompt structure suggesting a language learning platform or AI conversation context about English language.

The opening question "Do you like" signals
- count=1: Formal sports discussion thread with multiple responses listing opinions about a celebrity athlete's popularity and future prospects, now transitioning to a casual comment section with numbered questi
- count=1: Blog post format with sports commentary threads listing celebrity gossip facts, mixing humor and ranking topics around a female singer's social media presence.

The pattern "one of my friends said he 
- count=1: Chat format with informal discussion posts listing sports topics, alternating between student responses about a celebrity's business and personal opinions.

The final entry "yeah" begins a direct quot

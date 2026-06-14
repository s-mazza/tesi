# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke`
- nla: `gepa-experiments/results/experimental_nla_candidate_content_10_topical_chat_smoke`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_candidate_content_10_vs_ppl_smoke_20260611_predictions.csv`

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
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/experimental_nla_candidate_content_10_topical_chat_smoke/experimental_nla_precomputed_candidate_content_10_11912948.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`0` nla=`10`

## Metric Deltas
- baseline `agreement` (higher better): control=0.722222, nla=0.722222, delta=0.000000, rel_pct=0.000000
- baseline `pearson` (higher better): control=0.603136, nla=0.603136, delta=0.000000, rel_pct=0.000000
- baseline `spearman` (higher better): control=0.590879, nla=0.590879, delta=0.000000, rel_pct=0.000000
- baseline `kendall_tau` (higher better): control=0.531588, nla=0.531588, delta=0.000000, rel_pct=0.000000
- baseline `mae` (lower better): control=0.555556, nla=0.555556, delta=0.000000, rel_pct=0.000000
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.638889, nla=0.625000, delta=-0.013889, rel_pct=-2.173913
- optimized `pearson` (higher better): control=0.536400, nla=0.402090, delta=-0.134309, rel_pct=-25.039032
- optimized `spearman` (higher better): control=0.527410, nla=0.371727, delta=-0.155682, rel_pct=-29.518279
- optimized `kendall_tau` (higher better): control=0.459933, nla=0.310087, delta=-0.149846, rel_pct=-32.580014
- optimized `mae` (lower better): control=0.722222, nla=0.750000, delta=0.027778, rel_pct=3.846154
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 12
- NLA improved abs error: 1
- NLA worsened abs error: 2
- unchanged: 9

## NLA Feedback Quality
- verbalization rows: 213
- covered examples: 30
- avg rows per covered example: 7.100000
- avg verbalization words: 10.657277
- suspicious rows: 0
- duplicate text rows: 0
- duplicate text rows by category: `{'candidate': 0}`
- parse status: `{'partial_tags': 213}`
- token status: `{'ok': 213}`
- token position prefixes: `{'experimental': 213}`
- token categories: `{'candidate': 213}`
- rows with activation summary stats: 0

## Trajectory
- control: `{'candidates': 7, 'accepted_candidates': 3, 'avg_prompt_words': 222.42857142857142, 'max_prompt_words': 295}`
- nla: `{'candidates': 7, 'accepted_candidates': 3, 'avg_prompt_words': 223.14285714285714, 'max_prompt_words': 262}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=1: team" or "and basketball" or "but my favorite sports team."
- count=1: or "as a baseball team" or "and the beer company" completing the mix.
- count=1: cultural reference, likely continuing a personal opinion or unrelated trivia question.
- count=1: "apple" (referring to the famous California tech culture), completing the joke.
- count=1: "all the basketball" or "watching and eating pizza."

# NLA Run Diagnostic Report

## Runs
- control: `gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer`
- nla: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b`
- prediction comparison csv: `gepa-experiments/results/diagnostics/nla_vs_ppl_long_20260608.prediction_errors.csv`

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
- DIFF `nla_precomputed_path`: control=`` nla=`gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b/nla_precomputed_11912657.jsonl`
- DIFF `nla_max_tokens_per_example`: control=`` nla=`3`

## Metric Deltas
- baseline `agreement` (higher better): control=0.761111, nla=0.769444, delta=0.008333, rel_pct=1.094891
- baseline `pearson` (higher better): control=0.551831, nla=0.723364, delta=0.171533, rel_pct=31.084306
- baseline `spearman` (higher better): control=0.547848, nla=0.721559, delta=0.173711, rel_pct=31.707970
- baseline `kendall_tau` (higher better): control=nan, nla=nan, delta=nan, rel_pct=nan
- baseline `mae` (lower better): control=0.477778, nla=0.461111, delta=-0.016667, rel_pct=-3.488372
- baseline `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000
- optimized `agreement` (higher better): control=0.788889, nla=0.691667, delta=-0.097222, rel_pct=-12.323944
- optimized `pearson` (higher better): control=0.632812, nla=0.511144, delta=-0.121668, rel_pct=-19.226569
- optimized `spearman` (higher better): control=0.619893, nla=0.490815, delta=-0.129078, rel_pct=-20.822637
- optimized `kendall_tau` (higher better): control=nan, nla=nan, delta=nan, rel_pct=nan
- optimized `mae` (lower better): control=0.422222, nla=0.616667, delta=0.194444, rel_pct=46.052632
- optimized `coverage` (higher better): control=1.000000, nla=1.000000, delta=0.000000, rel_pct=0.000000

## Prediction-Level Error Movement
- joined examples: 60
- NLA improved abs error: 6
- NLA worsened abs error: 18
- unchanged: 36

## NLA Feedback Quality
- verbalization rows: 900
- covered examples: 300
- avg rows per covered example: 3.000000
- avg verbalization words: 73.544444
- suspicious rows: 4
- duplicate text rows: 667
- parse status: `{'partial_tags': 900}`
- token status: `{'unknown': 900}`
- token position prefixes: `{'source': 300, 'candidate': 300, 'reference': 300}`

## Trajectory
- control: `{'candidates': 0, 'accepted_candidates': 0, 'avg_prompt_words': 0.0, 'max_prompt_words': 0}`
- nla: `{'candidates': 631, 'accepted_candidates': 147, 'avg_prompt_words': 291.33280507131536, 'max_prompt_words': 454}`

## Next Checks
- If config rows other than NLA differ, rerun a stricter 1-to-1 control.
- If NLA coverage or useful rows are low, fix precompute before interpreting metrics.
- If suspicious or duplicate verbalizations are high, inspect token selection and AV checkpoint output.
- If validation improves but test worsens, treat this as NLA-induced overfit and reduce proposer temperature or feedback length.
- If NLA text is good but ignored in prompt diffs, inspect proposer sanitization/reflection inputs.

## Top Repeated NLA Text
- count=33: Formal AI chat context with numbered prompts about "machine learning" or "weather" topic, suggesting a chat interface or prompt list format with user input expected.

The text "hello" appears to be th
- count=24: Formal chat format with numbered questions and a prompt structure suggesting a language learning platform or AI conversation context about English language.

The opening question "Do you like" signals
- count=21: Formal AI chat context with numbered prompts about "machine learning" or "weather" topic, suggesting a chat interface or prompt list format with user input expected.

The text "hello" appears to be th
- count=18: Chatbot format with numbered questions and responses about a language, mixing informal conversational prompts with a specific task about "dog" emoji.

The phrase "did you know" signals a new question 
- count=18: Chat format with "AI" response pattern showing emoji context and a user's dialogue about "happy" or "sad" mood, suggesting a transcript or list context.

The phrase "my recent conversation is : " sign

# NLA Evidence Deep Dive

## Run Metrics
| run | kind | program | n | pearson | spearman | kendall | agreement | mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppl_long_old | long | baseline | 60.000000 | 0.551831 | 0.547848 | nan | 0.761111 | 0.477778 |
| ppl_long_old | long | optimized | 60.000000 | 0.632812 | 0.619893 | nan | 0.788889 | 0.422222 |
| old_nla_long | long | baseline | 60.000000 | 0.723364 | 0.721559 | nan | 0.769444 | 0.461111 |
| old_nla_long | long | optimized | 60.000000 | 0.511144 | 0.490815 | nan | 0.691667 | 0.616667 |
| fixed_nla_long | long | baseline | 60.000000 | 0.681158 | 0.677076 | 0.571150 | 0.752778 | 0.494444 |
| fixed_nla_long | long | optimized | 60.000000 | 0.681158 | 0.677076 | 0.571150 | 0.752778 | 0.494444 |
| ppl_smoke_q35 | smoke | baseline | 12.000000 | 0.603136 | 0.590879 | 0.531588 | 0.722222 | 0.555556 |
| ppl_smoke_q35 | smoke | optimized | 12.000000 | 0.536400 | 0.527410 | 0.459933 | 0.638889 | 0.722222 |
| fixed_nla_smoke_q35 | smoke | baseline | 12.000000 | 0.603136 | 0.590879 | 0.531588 | 0.722222 | 0.555556 |
| fixed_nla_smoke_q35 | smoke | optimized | 12.000000 | 0.674979 | 0.674693 | 0.606407 | 0.763889 | 0.472222 |
| candidate6_smoke | smoke | baseline | 12.000000 | 0.603136 | 0.590879 | 0.531588 | 0.722222 | 0.555556 |
| candidate6_smoke | smoke | optimized | 12.000000 | 0.330901 | 0.301374 | 0.259889 | 0.583333 | 0.833333 |
| candidate10_smoke | smoke | baseline | 12.000000 | 0.603136 | 0.590879 | 0.531588 | 0.722222 | 0.555556 |
| candidate10_smoke | smoke | optimized | 12.000000 | 0.402090 | 0.371727 | 0.310087 | 0.625000 | 0.750000 |
| single_gpu_ppl | single_gpu | baseline | 12.000000 | 0.603136 | 0.590879 | nan | 0.722222 | 0.555556 |
| single_gpu_ppl | single_gpu | optimized | 12.000000 | 0.314271 | 0.345082 | nan | 0.625000 | 0.750000 |
| single_gpu_nla_matched | single_gpu | baseline | 12.000000 | 0.603136 | 0.590879 | nan | 0.722222 | 0.555556 |
| single_gpu_nla_matched | single_gpu | optimized | 12.000000 | 0.603136 | 0.590879 | nan | 0.722222 | 0.555556 |
| summeval_ppl | dataset_smoke | baseline | 32.000000 | 0.635545 | 0.505233 | 0.469183 | 0.656250 | 1.375000 |
| summeval_ppl | dataset_smoke | optimized | 32.000000 | 0.701281 | 0.790860 | 0.716853 | 0.723958 | 1.104167 |
| summeval_nla | dataset_smoke | baseline | 32.000000 | 0.659360 | 0.615213 | nan | 0.656250 | 1.375000 |
| summeval_nla | dataset_smoke | optimized | 32.000000 | 0.618512 | 0.696582 | nan | 0.718750 | 1.125000 |
| qags_cnn_ppl | dataset_smoke | baseline | 2.000000 | 1.000000 | 1.000000 | 1.000000 | 0.736111 | 1.055556 |
| qags_cnn_ppl | dataset_smoke | optimized | 2.000000 | 1.000000 | 1.000000 | 1.000000 | 0.847222 | 0.611111 |
| qags_cnn_nla | dataset_smoke | baseline | 2.000000 | 1.000000 | 1.000000 | nan | 0.736111 | 1.055556 |
| qags_cnn_nla | dataset_smoke | optimized | 2.000000 | 1.000000 | 1.000000 | nan | 0.736111 | 1.055556 |
| qags_xsum_ppl | dataset_smoke | baseline | 2.000000 | 1.000000 | 1.000000 | 1.000000 | 0.708333 | 1.166667 |
| qags_xsum_ppl | dataset_smoke | optimized | 2.000000 | 1.000000 | 1.000000 | 1.000000 | 0.708333 | 1.166667 |
| qags_xsum_nla | dataset_smoke | baseline | 2.000000 | 0.000000 | 0.000000 | nan | 0.666667 | 1.333333 |
| qags_xsum_nla | dataset_smoke | optimized | 2.000000 | 0.000000 | 0.000000 | nan | 0.666667 | 1.333333 |

## Prompt And Search Behavior
| run | seed=opt | seed words | opt words | candidates | accepted | unique prompts | seed score | opt score | best score | cand > seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppl_long_old | False | 0 | 337 | 0 | 0 | 0 | nan | nan | nan | 0 |
| old_nla_long | False | 192 | 339 | 631 | 147 | 629 | 0.691667 | 0.705556 | 0.705556 | 7 |
| fixed_nla_long | True | 192 | 192 | 740 | 136 | 590 | 0.697222 | 0.697222 | 0.697222 | 0 |
| ppl_smoke_q35 | False | 192 | 219 | 7 | 3 | 6 | 0.597222 | 0.666667 | 0.666667 | 2 |
| fixed_nla_smoke_q35 | False | 192 | 244 | 9 | 2 | 5 | 0.597222 | 0.694444 | 0.694444 | 1 |
| candidate6_smoke | False | 192 | 263 | 9 | 2 | 4 | 0.597222 | 0.708333 | 0.708333 | 1 |
| candidate10_smoke | False | 192 | 242 | 7 | 3 | 4 | 0.597222 | 0.680556 | 0.680556 | 2 |
| single_gpu_ppl | False | 192 | 211 | 7 | 4 | 7 | 0.597222 | 0.638889 | 0.638889 | 3 |
| single_gpu_nla_matched | True | 192 | 192 | 11 | 1 | 7 | 0.597222 | 0.597222 | 0.597222 | 0 |
| summeval_ppl | False | 151 | 186 | 15 | 4 | 14 | 0.718750 | 0.812500 | 0.812500 | 3 |
| summeval_nla | False | 151 | 201 | 7 | 5 | 7 | 0.718750 | 0.796875 | 0.796875 | 1 |
| qags_cnn_ppl | False | 151 | 173 | 3 | 2 | 3 | 0.805556 | 0.930556 | 0.930556 | 1 |
| qags_cnn_nla | True | 151 | 151 | 3 | 2 | 3 | 0.805556 | 0.805556 | 0.805556 | 0 |
| qags_xsum_ppl | True | 151 | 151 | 3 | 1 | 3 | 0.833333 | 0.833333 | 0.833333 | 0 |
| qags_xsum_nla | True | 151 | 151 | 3 | 2 | 3 | 0.833333 | 0.833333 | 0.833333 | 0 |

## NLA Feedback Health
| run | rows | examples | dupe % | avg words | completion-like % | rubric-like % | token status | top categories | activation stats rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_nla_long | 900 | 300 | 61.44 | 73.54 | 97.11 | 91.67 | unknown:900 | source:300, candidate:300, reference:300 | 0 |
| fixed_nla_long | 1752 | 300 | 42.18 | 10.55 | 89.73 | 8.45 | ok:1752 | candidate:888, reference:564, source:300 | 0 |
| fixed_nla_smoke_q35 | 210 | 36 | 41.43 | 10.78 | 87.62 | 8.10 | ok:210 | candidate:108, reference:66, source:36 | 0 |
| candidate6_smoke | 187 | 36 | 0.00 | 10.74 | 92.51 | 9.09 | ok:187 | experimental:187 | 0 |
| candidate10_smoke | 213 | 30 | 0.00 | 10.66 | 90.14 | 8.45 | ok:213 | experimental:213 | 0 |
| single_gpu_nla_matched | 216 | 36 | 43.06 | 11.52 | 87.96 | 13.43 | ok:216 | source:108, candidate:108 | 0 |
| summeval_nla | 100 | 50 | 68.00 | 48.46 | 77.00 | 95.00 | unknown:100 | source:50, candidate:50 | 0 |
| qags_cnn_nla | 12 | 6 | 0.00 | 49.33 | 91.67 | 100.00 | unknown:12 | source:6, candidate:6 | 0 |
| qags_xsum_nla | 12 | 6 | 0.00 | 49.33 | 83.33 | 83.33 | unknown:12 | source:6, candidate:6 | 0 |

## Optimized Prediction Distributions
| run | n | target buckets | prediction distribution | pred MAE |
| --- | --- | --- | --- | --- |
| ppl_long_old | 60 | {"high": 21, "low": 21, "mid": 18} | {"1": 14, "2": 36, "3": 10} | 0.422222 |
| old_nla_long | 60 | {"high": 21, "low": 21, "mid": 18} | {"1": 30, "2": 21, "3": 9} | 0.616667 |
| fixed_nla_long | 60 | {"high": 21, "low": 21, "mid": 18} | {"1": 23, "2": 19, "3": 18} | 0.494444 |
| ppl_smoke_q35 | 12 | {"high": 6, "low": 5, "mid": 1} | {"1": 6, "2": 4, "3": 2} | 0.722222 |
| fixed_nla_smoke_q35 | 12 | {"high": 6, "low": 5, "mid": 1} | {"1": 2, "2": 7, "3": 3} | 0.472222 |
| candidate6_smoke | 12 | {"high": 6, "low": 5, "mid": 1} | {"1": 5, "2": 6, "3": 1} | 0.833333 |
| candidate10_smoke | 12 | {"high": 6, "low": 5, "mid": 1} | {"1": 5, "2": 5, "3": 2} | 0.750000 |
| single_gpu_ppl | 12 | {"high": 6, "low": 5, "mid": 1} | {"1": 3, "2": 5, "3": 4} | 0.750000 |
| single_gpu_nla_matched | 12 | {"high": 6, "low": 5, "mid": 1} | {"1": 2, "2": 8, "3": 2} | 0.555556 |
| summeval_ppl | 32 | {"high": 28, "low": 2, "mid": 2} | {"2": 4, "3": 4, "4": 24} | 1.104167 |
| summeval_nla | 32 | {"high": 28, "low": 2, "mid": 2} | {"1": 3, "2": 1, "3": 2, "4": 25, "5": 1} | 1.125000 |
| qags_cnn_ppl | 2 | {"high": 1, "low": 1} | {"2": 1, "3": 1} | 0.611111 |
| qags_cnn_nla | 2 | {"high": 1, "low": 1} | {"1": 1, "2": 1} | 1.055556 |
| qags_xsum_ppl | 2 | {"high": 1, "mid": 1} | {"2": 1, "3": 1} | 1.166667 |
| qags_xsum_nla | 2 | {"high": 1, "mid": 1} | {"3": 2} | 1.333333 |

## Pairwise Treatment vs Control
| treatment | control | metric deltas | joined | improved | worsened | unchanged | mean abs error delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old_nla_long | ppl_long_old | pearson:-0.1217, spearman:-0.1291, agreement:-0.0972, mae:+0.1944 | 60 | 6 | 18 | 36 | 0.194444 |
| fixed_nla_long | ppl_long_old | pearson:+0.0483, spearman:+0.0572, agreement:-0.0361, mae:+0.0722 | 60 | 8 | 15 | 37 | 0.072222 |
| fixed_nla_smoke_q35 | ppl_smoke_q35 | pearson:+0.1386, spearman:+0.1473, kendall_tau:+0.1465, agreement:+0.1250, mae:-0.2500 | 12 | 4 | 1 | 7 | -0.250000 |
| candidate6_smoke | ppl_smoke_q35 | pearson:-0.2055, spearman:-0.2260, kendall_tau:-0.2000, agreement:-0.0556, mae:+0.1111 | 12 | 2 | 4 | 6 | 0.111111 |
| candidate10_smoke | ppl_smoke_q35 | pearson:-0.1343, spearman:-0.1557, kendall_tau:-0.1498, agreement:-0.0139, mae:+0.0278 | 12 | 1 | 2 | 9 | 0.027778 |
| single_gpu_nla_matched | single_gpu_ppl | pearson:+0.2889, spearman:+0.2458, agreement:+0.0972, mae:-0.1944 | 12 | 3 | 0 | 9 | -0.194444 |
| summeval_nla | summeval_ppl | pearson:-0.0828, spearman:-0.0943, agreement:-0.0052, mae:+0.0208 | 32 | 3 | 3 | 26 | 0.020833 |
| qags_cnn_nla | qags_cnn_ppl | pearson:+0.0000, spearman:+0.0000, agreement:-0.1111, mae:+0.4444 | 2 | 1 | 1 | 0 | 0.444444 |
| qags_xsum_nla | qags_xsum_ppl | pearson:-1.0000, spearman:-1.0000, agreement:-0.0417, mae:+0.1667 | 2 | 0 | 1 | 1 | 0.166667 |

## Evidence-Based Observations
- `fixed_nla_long` selected the seed prompt unchanged: seed_equals_optimized=True, while exploring 740 trajectory rows.
- `fixed_nla_long` NLA health improved over the first old NLA run on token status and length, but still has 42.18% duplicate verbalization rows and 89.73% completion-like text.
- `old_nla_long` is the clearest negative control: duplicate verbalization rows are 61.44% and optimized metrics drop versus PPL-only.
- Candidate-only NLA is not sufficient: `candidate6_smoke` and `candidate10_smoke` remove most source/reference repetition, but both still degrade optimized correlations versus the PPL-only smoke control.
- `candidate10_smoke` has 0.00% duplicate NLA rows, so its failure argues against the simple hypothesis that duplicate rows alone explain NLA underperformance.
- SummEval consistency smoke is directionally negative for NLA versus PPL-only; QAGS smokes are too small to support a claim.
- Working hypothesis: raw NLA verbalizations mostly describe token continuations or latent associations, not metric-aligned reasons for why the judge should raise/lower a G-Eval score. The proposer can overfit this text into stricter or more dispersed rubrics.
- Next experiment should transform NLA into short, rubric-conditioned error feedback before GEPA reflection, preferably with the 35B auxiliary judge/proposer summarizing NLA together with target, prediction, and error direction.

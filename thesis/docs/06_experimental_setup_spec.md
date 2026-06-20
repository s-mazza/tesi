# Chapter 4 Experimental Setup Plan

This document defines the intended content of Chapter 4. The chapter should make
the experiments reproducible and make clear which metrics support each claim.
It should follow the practical style of the reference thesis: environment,
model variants, hyperparameters, metrics, and run conditions are explicit before
the results are discussed.

Chapter 4 should not contain final result interpretation. That belongs to
Chapter 5. Chapter 4 defines what was measured, how it was measured, and under
which constraints.

## Goal

Describe the datasets, models, hardware, software stack, hyperparameters,
metrics, baselines, and artifact policy used in the thesis experiments.

The chapter must let a reader answer:

- which data was used in each experiment;
- which model played which role;
- which metrics are paper-aligned, diagnostic, feedback-only, or missing;
- which hardware/software constraints affected reproducibility;
- which artifacts make a run auditable after it finishes.

Target length: about five to seven pages, plus tables.

## Scope Boundary

Include:

- dataset sources, dimensions, split policy, and dataset sizes;
- model roles and checkpoint compatibility;
- university cluster hardware, Slurm, Docker, vLLM, llama.cpp, and NLA
  dependencies;
- exact metrics used for embedding inversion, SIPIT, standalone NLA, and
  GEPA/G-EVAL;
- time and efficiency metrics;
- hyperparameters and values tried;
- baselines and ablation controls;
- artifact and versioning policy.

Exclude:

- theoretical explanations of embeddings, hidden states, perplexity, NLA, GEPA,
  and LLM-as-a-judge evaluation; these belong to Chapter 1;
- literature positioning; this belongs to Chapter 2;
- algorithmic method descriptions; these belong to Chapter 3;
- final result interpretation; this belongs to Chapter 5;
- operational cluster troubleshooting unless it affects reproducibility.

## Advisor Guideline Coverage

The final LaTeX chapter must explicitly satisfy the advisor's experimental
setup checklist. The current plan maps those requirements as follows:

| Advisor requirement | Where this plan covers it | Still to verify before writing |
|---|---|---|
| Implementation details | Sections 4.3, 4.4, 4.6 | Add exact final container image, dependency versions, and launch command snippets for thesis-grade runs |
| Libraries used | Sections 4.3, 4.4, 4.6 | Record exact versions for Python, PyTorch, vLLM, llama.cpp, transformers, DSPy/GEPA, NLA code, and flash-attention |
| Models used | Section 4.2 | Confirm final checkpoints and local cache paths for every reported run |
| Why those models and not others | Sections 4.2, 4.7 | Write explicit justifications: Qwen2.5-7B for NLA compatibility, Qwen35B for stronger proposer/aux feedback, GPT-2 for clean SIPIT reproduction |
| Optional snippets or pseudocode | Sections 4.4, 4.8 | Include only compact snippets if they clarify split semantics, scoring, or run configuration |
| Datasets used and why | Section 4.1 | Add final source citations and exact dataset row counts for every dataset/dimension in the result tables |
| Dataset structure | Section 4.1 | Include fields such as source/context, candidate output, reference/fact, human score, group id, and score scale |
| Experiment environment | Section 4.3 | Re-check hardware/driver/CUDA snapshot for final runs |
| Metrics and boundaries | Section 4.5 | Ensure every metric table states range, direction, and interpretation |
| Hyperparameters with tried values and final asterisk | Section 4.6 | Fill values from final run configs and mark final values with `*` in LaTeX |
| Baselines and comparisons | Section 4.7 | Keep paper-aligned baselines separate from diagnostic/smoke controls |
| Prompts if useful | Sections 3.7--3.10 and 5.5 | Keep prompt material where it explains the method or result: exact seed prompt and proposer-feedback record in Chapter 3; optimized-prompt comparison and full final prompt in Chapter 5 |

Decision:

- This checklist should remain in the planning document.
- The final thesis chapter should not contain this checklist verbatim. It should
  be used as a pre-writing and pre-submission coverage check.

## Lessons From The Reference Thesis

The previous student thesis is useful for structure, not for content. The
following elements should be mirrored and adapted:

- an `Environment` section that states cluster hardware and Docker execution;
- a `Model Variants` section that explains why each model was selected;
- hyperparameter tables with final values marked by an asterisk;
- compact metric tables with directionality and denominator/coverage;
- efficiency metrics such as latency, total time, throughput, or average time
  when they matter for the system.

For this thesis, the equivalent of the reference thesis performance table is
not a chatbot-latency table only. We need runtime and processing metrics for
long GEPA runs, NLA precomputation, SIPIT inversion, and embedding-inversion
training/probes.

## Proposed Section Structure

### 4.1 Datasets And Splits

Purpose: define all datasets before any result table appears.

Datasets to cover:

| Experiment branch | Dataset | Current status | Include in Chapter 4 |
|---|---|---|---|
| Semantic-fidelity dataset | Canonical thesis corpus with standard, negation, and commonsense/counterfactual blocks | Built and validated | Yes |
| Embedding inversion | Jina/Qwen encoder cached datasets and probes | Diagnostic branch | Yes, compactly |
| SIPIT | Paper-style GPT-2 Table 5 dataset plus logical 20-token dataset | Reproduction and extension | Yes |
| Standalone NLA | SummEval activation/verbalization samples | Plumbing validation | Yes, compactly |
| GEPA/G-EVAL | Topical-Chat, SummEval, QAGS-CNN, QAGS-XSUM | Main current branch | Yes |
| Soft-prompt tuning | Topical-Chat engagingness | Frozen-model prompt-tuning diagnostic | Yes, as GEPA-adjacent explainability branch |

Dataset table to add:

| Dataset | Source | Task | Size used | Split policy | Thesis role |
|---|---|---|---:|---|---|

Required details:

- Semantic-fidelity canonical dataset:
  - total rows: 2080;
  - Block A controlled standard sentences: 40 rows;
  - Block B negation: 720 rows;
  - Block C commonsense/counterfactual: 1320 rows;
  - stable split counts: train 962, validation 290, test 828.
- SIPIT logical dataset:
  - built from canonical Blocks B and C;
  - balanced labels: positive, negative, counterfactual,
    commonsense-corrected;
  - clean GPT-2 20-token variant:
    `spit/SIPIT/data/reproduce/logical20_gpt2_clean/`;
  - current documented size: 40 prompts, 10 per logical category.
- GEPA/G-EVAL datasets and dimensions:
  - Topical-Chat: naturalness, coherence, engagingness, groundedness;
  - SummEval: fluency, coherence, consistency, relevance;
  - QAGS-CNN: consistency;
  - QAGS-XSUM: consistency.
- GEPA splits:
  - split by `group_id`, not by row, to avoid related candidates leaking across
    train/validation/final-test;
  - GEPA train rows are used during prompt search;
  - GEPA validation rows are used for candidate prompt validation and selection;
  - final-test rows are never passed to GEPA and are evaluated only after prompt
    selection;
  - every thesis run must save the exact split manifest.
- Soft-prompt Topical-Chat split:
  - same Topical-Chat engagingness task family as the GEPA branch;
  - main random-init runs use 40 train groups, 10 validation groups, and 10
    final-test groups, corresponding to 240 / 60 / 60 rows;
  - the 2048-token context setting keeps all 240 training rows, while the
    1024-token setting tokenizes 234/240 training rows;
  - seed sweeps must be reported as robustness evidence, not as independent
    datasets.

Important wording:

- Do not call the GEPA final-test split "validation".
- Do not claim G-EVAL paper-level reproduction unless the dataset, dimensions,
  split policy, prompt setting, and metric set are all aligned with the paper.

### 4.2 Model Roles And Checkpoints

Purpose: avoid ambiguity around "base model", "judge", "proposer", and NLA.

Model-role table to add:

| Role | Model/checkpoint | Used for | Included in final metrics? |
|---|---|---|---|
| Embedding encoder | Jina-v3 / Qwen3-Embedding diagnostic branches | Embedding-inversion inputs | No, unless branch is reported |
| SIPIT target model | GPT-2 | Hidden-state inversion target | Yes for SIPIT results |
| Standalone NLA base model | Qwen/Qwen2.5-7B-Instruct | Activation extraction | No direct final score |
| NLA AV checkpoint | Qwen2.5-7B layer-20 AV checkpoint | Activation verbalization | Feedback/diagnostic |
| GEPA base judge | Qwen/Qwen2.5-7B-Instruct | Scores G-EVAL examples | Yes |
| Perplexity model | Same Qwen/Qwen2.5-7B-Instruct | Response-only PPL feedback | No, feedback-only |
| Proposer | Qwen35B via llama.cpp | Proposes prompt edits | No |
| Auxiliary judge | Qwen35B via llama.cpp | Produces extra proposer feedback when enabled | No |
| Soft-prompt base model | Qwen/Qwen2.5-7B-Instruct | Frozen model whose prompt embeddings are tuned to create learned readout targets | Yes as sanity/task-learning evidence |
| SIPIT readout target for soft prompts | Qwen/Qwen2.5-7B-Instruct embeddings/hidden states | Tests what SIPIT can invert from learned virtual tokens | No, primary diagnostic/readout object |

Required explanation:

- The "trained" or optimized task model in the GEPA branch is the 7B base judge
  plus its prompt. We are not fine-tuning model weights.
- The 35B model is not the final evaluated judge. It is used as proposer, and
  optionally as an auxiliary feedback model.
- Perplexity and NLA are computed on the 7B base model because this is the
  model whose behavior the prompt is trying to improve.
- The NLA checkpoint is model/layer-specific, so NLA verbalizations should not
  be moved to a different base model without a compatible checkpoint.
- Soft-prompt runs do not fine-tune Qwen2.5-7B weights. They train only virtual
  prompt embeddings, so they answer a different question from GEPA prompt
  search.

### 4.3 Hardware And Cluster Environment

Purpose: document enough hardware and deployment detail to make runtimes and
failures interpretable.

Observed cluster snapshot: 2026-06-15.

Primary Slurm nodes:

| Node | Address | GPUs | GPU memory | CPU | RAM | Driver/CUDA observed | Notes |
|---|---|---:|---:|---|---:|---|---|
| `faretra` | `137.204.107.40` | 4 x NVIDIA GeForce RTX 3090 | 24 GiB each | AMD EPYC 7443, 24 cores / 48 threads | about 124 GB | 550.54.14 / CUDA 12.4 | Primary node for multi-GPU GEPA jobs |
| `moro232` | `137.204.107.232` | 1 x NVIDIA GeForce RTX 3090 | 24 GiB | Intel i5-6400, 4 cores | about 31 GB | 555.42.02 / CUDA 12.5 | Single-GPU fallback node |

Cluster details observed from Slurm and host checks:

- Slurm version observed on `faretra`: 21.08.6.
- `faretra` kernel: Linux 5.4.0-216 on Ubuntu; Slurm reports 48 logical CPUs
  and 124000 MB real memory.
- `moro232` kernel: Linux 5.15.0-177 on Ubuntu; Slurm reports 4 CPUs and 31989
  MB real memory.
- Exact driver/CUDA values should be re-recorded for the final thesis runs,
  because they can change independently of the code.

Additional visible nodes:

| Node | GPUs | Role in current thesis runs |
|---|---|---|
| `deeplearn2` | 1 x Titan XP, 1 x RTX 3090 | Not a primary target for current vLLM/flash-attention GEPA setup |
| `moro43` | 1 x RTX 5090 | Potential future node, but not part of the stable current setup |
| `moro49`, `moro157` | CPU-only in Slurm view | Not used for GPU runs |

Software/deployment details to report:

- Slurm manages GPU allocation and job scheduling.
- Experiments run inside Docker containers to keep dependencies stable.
- vLLM serves the 7B base judge and exposes OpenAI-compatible endpoints.
- llama.cpp serves Qwen35B for proposer and optional auxiliary-judge feedback.
- Flash-attention compatibility must match the container's Python, Torch, CUDA,
  and GPU stack. The final thesis should report the exact container image or
  dependency manifest used by each final run.
- Hugging Face model files are expected under the cluster cache, typically
  `/llms`, when the model is already available locally.
- Some runs use one GPU; Qwen35B proposer or auxiliary-judge configurations can
  require an additional GPU and therefore cannot always run on `moro232`.

Hardware table to add:

| Hardware item | Value | Why it matters |
|---|---|---|
| GPU model and VRAM | RTX 3090, 24 GiB | Determines vLLM memory limits and batch settings |
| Number of GPUs | 1 on `moro232`, 4 on `faretra` | Determines whether proposer sidecar can run |
| CPU/RAM | Node-specific | Affects preprocessing and container startup |
| CUDA/driver | Record per run | Affects vLLM and flash-attention compatibility |
| Slurm job id/node | Saved in runtime manifest | Links artifacts to execution environment |

### 4.4 Software, Versioning, And Artifacts

Purpose: define what must be saved for scientific reproducibility.

Required artifact checklist for every thesis-grade GEPA run:

- `metrics_*.csv`;
- `baseline_predictions_*.jsonl`;
- `optimized_predictions_*.jsonl`;
- `seed_prompt_*.txt`;
- `optimized_prompt_*.txt`;
- `prompt_trajectory_*.jsonl`;
- `gepa_viz_run.json` or fallback trajectory;
- `split_manifest_*.json`;
- `run_config_*.json`;
- `runtime_manifest_*.json`;
- Slurm stdout/stderr logs;
- NLA verbalization artifact when NLA is enabled;
- auxiliary-judge feedback artifact when auxiliary judge is enabled;
- git commit hash and clean/dirty status at launch.

Required artifact checklist for SIPIT:

- dataset JSON/metadata;
- collision check JSON;
- inversion CSV/JSON with `match`, `inversion_time`, `timesteps`, and
  per-token `times`;
- summary JSON/Markdown;
- Slurm logs when official CSV/JSON is unavailable.

Required artifact checklist for soft-prompt tuning and SIPIT readout:

- `metrics.json` with baseline and soft-prompt metrics on validation and
  final-test splits;
- adapter metadata and train configuration;
- `nearest_tokens.jsonl` with nearest-token L2/cosine diagnostics;
- `sipit_soft_prompt_manifest.json`;
- `soft_prompt_embeddings.pt` or a reproducible pointer to the embedding
  artifact;
- SIPIT readout `sipit_recovery.json` and `summary.md`;
- control readouts for random hard tokens, initialization prompt embeddings,
  and random continuous vectors.

Required artifact checklist for embedding inversion:

- config YAML;
- checkpoint metadata;
- training logs with step, train loss, validation loss, token accuracy, learning
  rate, elapsed time;
- fixed-mask and sequential decoding evaluations;
- embedding ablation/provenance reports.

### 4.5 Metrics

Purpose: list every metric we have, decide whether it belongs in the thesis,
and identify missing metrics that should be added before final runs.

Direction conventions:

- Pearson, Spearman, Kendall tau, agreement, accuracy, exact match, coverage,
  token accuracy, and throughput: higher is better.
- MAE, loss, NLL, perplexity, runtime, latency, and failure count: lower is
  better.
- Diagnostic metrics must not be confused with final task metrics.

Metric boundaries to define in the final thesis:

| Metric | Boundary/range | How to read it |
|---|---|---|
| Pearson correlation | `[-1, 1]` | Higher is better; `1` is perfect linear agreement, `0` is no linear correlation, negative values indicate inverse relation |
| Spearman correlation | `[-1, 1]` | Higher is better; rank-based agreement with human scores |
| Kendall tau | `[-1, 1]` | Higher is better; pairwise rank agreement, more conservative under ties |
| MAE | `[0, max_score - min_score]` | Lower is better; average absolute distance from human mean score |
| Normalized agreement | `[0, 1]` | Higher is better; `1` means exact agreement, `0` means largest possible scale error |
| Coverage | `[0, 1]` | Higher is better; parsed predictions divided by total predictions |
| Exact match / accuracy | `[0, 1]` or percent | Higher is better; fraction of exactly recovered prompts/tokens/examples |
| Loss / cross-entropy / NLL | `[0, +inf)` | Lower is better; optimization or surprisal signal |
| Perplexity | `[1, +inf)` in normal cases | Lower is better; exponentiated mean NLL over response tokens |
| Runtime / latency | `[0, +inf)` seconds | Lower is better for efficiency, but only comparable under similar hardware and run settings |
| Throughput | `[0, +inf)` tokens/s or examples/s | Higher is better; only comparable under similar hardware and batching |

G-EVAL score scales to state:

| Dataset/dimension | Score boundary in current runner |
|---|---|
| Topical-Chat naturalness | 1 to 3 |
| Topical-Chat coherence / Maintains Context | 1 to 3 |
| Topical-Chat engagingness | 1 to 3 |
| Topical-Chat groundedness / Uses Knowledge | 0 to 1 |
| SummEval dimensions | 1 to 5 |
| QAGS-CNN consistency | 1 to 5 |
| QAGS-XSUM consistency | 1 to 5 |

#### 4.5.1 Semantic-Fidelity And Embedding-Inversion Metrics

| Metric | Available now | Include? | Use |
|---|---|---|---|
| Token accuracy | Yes, from training/eval logs | Yes | Main reconstruction signal for inversion probes |
| Validation loss / CE | Yes | Yes, diagnostic | Training trajectory and paper-reference comparison |
| Full-mask token accuracy | Yes | Yes | Key stress setting for embedding inversion |
| Fixed-mask sweep accuracy | Yes for selected runs | Yes | Shows behavior at 10%, 50%, 100% mask |
| Sequential decoding accuracy | Yes for selected runs | Yes | Tests whether Eq. 11-style decoding helps |
| Embedding ablation accuracy | Yes | Yes | Checks whether output depends on the embedding |
| Gradient norm / clipping diagnostics | Yes for probes | Include in appendix or failure analysis | Explains training failure modes |
| ROUGE-L | Implemented in diagnostic script | Maybe | Useful only if clean generated text outputs are available |
| BLEU | Implemented in diagnostic script | Maybe | Surface-overlap diagnostic, not semantic proof |
| STS/cosine similarity | Implemented in diagnostic script | Maybe | Semantic-overlap diagnostic, can miss negation |
| BERTScore | Not implemented | Should add if inversion text outputs become central | Standard reconstruction metric, but not enough alone |
| Negation preservation rate | Not implemented | Should add | Required for the thesis semantic-fidelity claim |
| Polarity/semantic flip rate | Not implemented | Should add | Required for logical-fidelity conclusions |
| Counterfactual preservation rate | Not implemented | Should add | Required for commonsense/counterfactual block claims |

Chapter placement:

- Put token accuracy, full-mask accuracy, validation loss, and ablation metrics
  in the main text if the embedding-inversion branch is discussed as a result.
- Put ROUGE/BLEU/STS/BERTScore only as secondary metrics because they can miss
  logical flips.
- Add negation/polarity/counterfactual metrics before making any strong
  semantic-fidelity claim.

#### 4.5.2 SIPIT Metrics

| Metric | Available now | Include? | Use |
|---|---|---|---|
| Exact prompt match | Available in code; final reports partially missing | Yes | Primary SIPIT success metric |
| Token accuracy | Available in random-prefix extension and summaries | Yes | Secondary recovery metric |
| `torch.allclose` collision count | Yes | Yes | Validates practical injectivity/collision assumption |
| Minimum hidden-state L2 distance | Yes | Yes | Collision-check margin |
| Timesteps per token | Yes | Yes | Search-cost metric |
| Vocabulary explored percent | Yes in summaries | Yes | Efficiency and comparison to BruteForce |
| Inversion time | Yes | Yes | Runtime metric |
| Per-token time list | Yes in code outputs | Appendix/diagnostic | Useful for outlier analysis |
| Random-prefix nearest L2 rank | Yes in random-prefix diagnostics | Yes if random-prefix results are used | Explains off-vocabulary prefix behavior |
| Prefix top-k match flags | Yes in random-prefix diagnostics | Diagnostic | Supports full-sequence vs known-prefix interpretation |
| Failure count | Yes | Yes | Needed for incomplete/cancelled baselines |

Important caveat:

- If final SIPIT CSV/JSON outputs remain unavailable, Table 5 results must be
  labeled as interim/log-derived, not full official reproduction.
- Runtime is diagnostic because paper hardware differs from the available RTX
  3090 cluster.

#### 4.5.3 Soft-Prompt And SIPIT-Readout Metrics

| Metric | Available now | Include? | Use |
|---|---|---|---|
| Validation Pearson/Spearman/Kendall | Yes | Yes | Measures whether soft prompt improves the held-out validation split used for model selection |
| Final-test Pearson/Spearman/Kendall | Yes | Yes | Sanity check that the learned soft prompt changes useful task behavior |
| Validation/final-test MAE and normalized agreement | Yes | Yes, secondary | Shows absolute score error and exact-scale movement |
| Parse coverage | Yes | Yes | Confirms scores were parsed for all examples |
| Tokenized train rows | Yes | Yes | Explains the 1024 vs 2048 context difference |
| Number of virtual tokens | Yes | Yes | Capacity/overfit sweep variable |
| Soft-prompt initialization | Yes | Yes | Separates random-init evidence from text-init controls |
| Nearest-token L2 | Yes | Yes | Main geometric distance to the discrete vocabulary manifold |
| Nearest-token cosine | Yes for random-init readouts | Yes, qualitative | Secondary geometric similarity to nearest token |
| Nearest cosine variance | Yes for random-init readouts | Yes, qualitative | Variance across virtual tokens of cosine to each top-1 L2 nearest token |
| SIPIT `all_positions_verified` | Yes | Yes | Exact discrete recovery success/failure |
| SIPIT recovered text | Yes | Yes, qualitative/appendix | Useful only with the verification flag and nearest distances |
| SIPIT elapsed time and per-token timesteps | Yes | Yes, diagnostic | Indicates whether the bounded search exhausted its budget |

Interpretation rules:

- Random-init soft-prompt metrics are mainly used to verify that the learned
  virtual tokens are task-relevant before interpreting their SIPIT readout.
- SIPIT readout metrics are interpretability diagnostics. They should not be
  used to claim that a learned continuous prompt is natural language unless the
  recovery verifies exactly or nearest-token distances are plausibly small.
- Text-init soft prompts are controls because nearest-token projection is
  biased by the initialization sentence.

Soft-prompt SIPIT-readout control modes:

| Mode | Target vectors | Role | How to read the result |
|---|---|---|---|
| `soft_prompt` | PEFT virtual-token embeddings saved after training | Main diagnostic target | Tests whether trained continuous prompt vectors can be mapped back to faithful discrete text |
| `random_hard_tokens` | Exact embeddings of sampled vocabulary ids | Positive control | Should verify exactly; failure would indicate a pipeline or recovery-budget problem even for discrete targets |
| `init_prompt` | Exact embeddings of the tokenized seed instruction | Text-initialization control | Nearest tokens should reconstruct the seed text; failed full verification mainly indicates bounded recovery exhaustion, not off-manifold embeddings |
| `random_continuous` | Gaussian continuous vectors, norm-matched to the soft prompt when available | Negative control | Expected to fail exact verification because the target vectors are not vocabulary embeddings |

This table should be included because the control labels are otherwise
ambiguous. `random_hard_tokens` is the clean positive control that verified
end-to-end. `init_prompt` is also made of real token embeddings, but it is
included to diagnose text-initialization bias and recovery-budget behavior:
nearest-token projection recovers the seed text, while iterative verification
can still stop after a prefix. `soft_prompt` and `random_continuous` are
continuous-target readouts, so exact SIPIT verification is not expected unless
the vectors happen to lie very close to the discrete embedding manifold.

#### 4.5.4 Standalone NLA Metrics

| Metric | Available now | Include? | Use |
|---|---|---|---|
| Activation rows extracted | Yes | Yes | Confirms extraction coverage |
| Token positions verbalized | Yes | Yes | Shows what activation is being described |
| Layer and checkpoint | Yes | Yes | Required for compatibility |
| Parse status | Yes | Yes | NLA output health |
| Injection check status | Yes in standalone artifacts | Yes | Verifies artifact integrity |
| Verbalization count | Yes | Yes | Coverage/scale |
| Verbalization examples | Yes | Maybe | Include a small qualitative table or appendix |
| Semantic correctness of verbalizations | Not fully measured | Should add if standalone NLA is a result claim | Needed to claim semantic fidelity |
| AR reconstruction score | Not used | No unless AR is introduced | Current thesis uses AV, not AR |

Chapter placement:

- Use standalone NLA primarily as a setup/plumbing validation unless a dedicated
  semantic-fidelity evaluation is run.
- Do not claim that NLA preserves negation or counterfactuality from readability
  alone.

#### 4.5.5 GEPA/G-EVAL Final Metrics

Paper-aligned primary metrics:

| Dataset | Dimension(s) | Paper-aligned metrics | Include as main metrics |
|---|---|---|---|
| Topical-Chat | Naturalness, coherence, engagingness, groundedness | Pearson, Spearman | Yes |
| SummEval | Fluency, coherence, consistency, relevance | Spearman, Kendall tau | Yes |
| QAGS-CNN | Consistency | Pearson, Spearman, Kendall tau | Yes |
| QAGS-XSUM | Consistency | Pearson, Spearman, Kendall tau | Yes |

Metrics computed by the current GEPA runner:

| Metric | Available now | Include? | Use |
|---|---|---|---|
| `n` | Yes | Yes | Number of parsed predictions used in correlations |
| `total` | Yes | Yes | Final-test denominator |
| `parsed` | Yes | Yes | Parsing success count |
| `coverage` | Yes | Yes | Validity of run, parsed / total |
| `pearson` | Yes | Yes where paper-aligned; otherwise appendix | Linear agreement with human scores |
| `spearman` | Yes | Yes where paper-aligned | Rank agreement |
| `kendall_tau` | Yes | Yes where paper-aligned; diagnostic otherwise | Rank agreement with tie handling |
| `mae` | Yes | Yes, secondary | Absolute scoring error |
| `agreement` | Yes | Yes, secondary | Normalized absolute agreement |
| Optimized-vs-baseline delta | Yes via aggregation scripts | Yes | Shows GEPA improvement within a run |
| Relative improvement percent | Yes via aggregation scripts | Yes | Useful in tables, especially for advisor updates |
| Prompt trajectory candidate count | Yes | Yes, diagnostic | Shows whether GEPA actually searched |
| Seed vs optimized prompt equality | Derivable | Yes | Prevents false claims when prompt did not change |
| Prediction-level abs-error movement | Available via diagnostics | Yes for ablations | Shows which examples improved or worsened |

Important caveats:

- The final metrics are computed on the held-out final-test split, not on GEPA
  train or validation examples.
- Perplexity and NLA are not final G-EVAL metrics. They are feedback signals
  given to the proposer.
- A run where the optimized prompt is byte-identical to the seed prompt can
  still show metric noise, but it is not evidence that GEPA learned a better
  prompt.

#### 4.5.6 GEPA Feedback Metrics

Perplexity feedback:

| Metric | Available now | Include? | Use |
|---|---|---|---|
| `response_mean_nll` | Yes | Diagnostic | Base model surprisal over candidate response |
| `response_perplexity` | Yes | Diagnostic | Human-readable surprisal scale |
| `response_token_count` | Yes | Diagnostic | Normalizes response length |
| Perplexity request failure count | Partially visible in logs/errors | Should include | Ensures feedback was available |

NLA feedback:

| Metric | Available now | Include? | Use |
|---|---|---|---|
| NLA precomputed coverage | Yes | Yes | Gate before optimization |
| NLA rows | Yes | Yes | Amount of feedback available |
| Useful NLA rows | Yes | Yes | Filters empty/bad verbalizations |
| Missing example ids sample | Yes | Diagnostic | Debug coverage gaps |
| Token position | Yes | Yes | Explains which tokens were verbalized |
| Token text | Yes | Appendix/diagnostic | Helps inspect artifacts |
| Parse status | Yes | Yes | NLA output health |
| Token status | Yes | Yes | Extraction health |
| Verbalization length/duplication | Available through diagnostics, not always in summary | Should include | Helps explain weak NLA results |
| Token-selection strategy | Available in run config / precompute outputs | Yes | Needed for NLA ablation comparisons |

Auxiliary-judge feedback:

| Metric | Available now | Include? | Use |
|---|---|---|---|
| Aux feedback row count | Yes | Yes | Coverage |
| Aux status counts | Yes | Yes | Success/error rate |
| Aux success rate | Yes, printed/validated | Yes | Run validity gate |
| Raw auxiliary feedback | Yes | Appendix/diagnostic | Qualitative explanation |
| Non-empty feedback rate | Derivable | Should include | Detects silent failures |
| Auxiliary response length | Not summarized | Optional | Detects truncation or verbosity |

#### 4.5.7 Time And Efficiency Metrics

Metrics already available:

| Metric | Available now | Include? | Source |
|---|---|---|---|
| Run started timestamp | Yes | Yes | `runtime_manifest_*.json` |
| Run finished timestamp | Yes | Yes | `runtime_manifest_*.json` |
| Total elapsed seconds | Yes | Yes | `runtime_manifest_*.json` |
| Slurm job id | Yes | Yes | `runtime_manifest_*.json` |
| Slurm job name | Yes | Yes | `runtime_manifest_*.json` |
| Slurm node list | Yes | Yes | `runtime_manifest_*.json` |
| CUDA visible devices | Yes | Yes | `runtime_manifest_*.json` |
| SIPIT inversion time | Yes | Yes | SIPIT CSV/JSON/log summaries |
| SIPIT per-token times | Yes | Diagnostic/appendix | SIPIT output |
| Embedding-inversion elapsed time | Yes in logs | Yes | Training logs |
| Embedding-inversion sample rate | Yes in logs | Diagnostic | Training logs |

Metrics we do not yet have but should add:

| Missing metric | Priority | Why |
|---|---|---|
| Per-stage GEPA timing | High | Separates data loading, model startup, PPL, NLA, GEPA compile, baseline eval, optimized eval |
| vLLM startup time | Medium | Explains queue-to-useful-work overhead |
| llama.cpp startup time | Medium | Important for Qwen35B proposer runs |
| NLA precompute time | High | Needed to evaluate cost of NLA feedback |
| Aux-judge feedback generation time | High | Needed because auxiliary judge can dominate runtime |
| Tokens/sec for judge generation | Medium | Efficiency comparison across settings |
| Tokens/sec for proposer generation | Medium | Explains 35B proposer cost |
| Peak GPU memory | High | Prevents undocumented OOM/fit assumptions |
| CPU RAM and disk usage at launch | Medium | Useful because Docker/cache pressure has caused failures |
| Queue wait time | Optional | Useful operationally, but not a scientific runtime metric |

Decision:

- Include total elapsed time for every thesis-grade run now.
- Add per-stage timing before the final long matrix if possible.
- Treat Slurm accounting as secondary because earlier runs showed that artifact
  timestamps and runtime manifests are more reliable than scheduler summaries
  alone.

### 4.6 Hyperparameters To Report

Purpose: make long-run choices auditable and avoid relying on run names.

Hyperparameter tables to add:

GEPA table:

| Hyperparameter | Values tried | Final value | Why |
|---|---|---|---|
| Dataset/dimension | Per matrix | Mark final | Defines task |
| Train groups | Multiple | Mark final | GEPA prompt-search budget/data |
| Validation groups | Multiple | Mark final | Prompt selection |
| Final-test groups | Multiple | Mark final | Evaluation denominator |
| Random seed | Multiple/recorded | Mark final | Split reproducibility |
| GEPA budget | Smoke/long variants | Mark final | Convergence time |
| Number of threads | Tried values | Mark final | Parallel evaluation speed |
| vLLM max model length | Tried values | Mark final | Prevents context/OOM failures |
| vLLM max number of sequences | Tried values | Mark final | Throughput/memory tradeoff |
| Proposer temperature | Tried values | Mark final | Prompt diversity |
| Proposer max tokens | Tried values | Mark final | Allows long prompts without truncation |
| Instruction proposer mode | Current variants | Mark final | Anti-overfit behavior |

Feedback table:

| Hyperparameter | Values tried | Final value | Why |
|---|---|---|---|
| PPL prompt logprobs | Current default 20 unless changed | Mark final | Must cover response tokens |
| NLA layer | 20 | 20 | Compatible Qwen NLA checkpoint |
| NLA max tokens per example | Candidate values | Mark final | Token-selection strategy |
| NLA min coverage | Current gate | Mark final | Prevents partial feedback |
| Aux judge max tokens | Candidate values | Mark final | Feedback completeness |
| Aux judge min success rate | Current gate | Mark final | Prevents silent aux failures |

Soft-prompt table:

| Hyperparameter | Values tried | Final/reporting value | Why |
|---|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | Mark final | Same model family as GEPA base judge and NLA source |
| Dataset/dimension | Topical-Chat engagingness | Mark final | Same task used in the main GEPA pilot |
| Split groups | 40/10/10 | Mark final | Gives 240/60/60 row split |
| Soft-prompt initialization | text, random | random for main; text as control | Avoids biasing nearest-token interpretation toward the seed prompt |
| Virtual tokens | 8, 16, 32 | Mark final per run | Capacity/overfit and interpretability sweep |
| Max sequence length | 1024, 2048 | Mark final per run | 2048 prevents dropping long training rows |
| Learning rate | 0.005 in completed random-init runs | Mark final | Prompt-tuning optimization setting |
| Epochs | 5.0 in completed random-init runs | Mark final | Prompt-tuning budget |
| Batch / accumulation | train batch 1, accumulation 8 | Mark final | Fits Qwen2.5-7B prompt tuning on RTX 3090 |
| Quantization | 4-bit load for training | Mark final | Memory constraint on available GPUs |
| Random seed | 42, 43, 44 | Mark final per run | Robustness check |

SIPIT readout table for soft prompts:

| Hyperparameter | Values tried | Final/reporting value | Why |
|---|---|---|---|
| Target layer | 28 for Qwen2.5-7B readout runs | Mark final | Layer used by the bounded recovery script |
| Precision | 4, 16 for selected checks | Mark final | Tests whether recovery failure is precision/budget-related |
| Max iterations per token | 500 | Mark final | Bounded search budget |
| Control mode | soft prompt, random hard tokens, init prompt, random continuous | Mark final | Separates discrete-token recovery from continuous off-manifold targets |
| Nearest-token ranking | L2 primary, cosine diagnostic | Mark final | L2 selects nearest vocabulary embedding; cosine aids qualitative reading |

Infrastructure table:

| Hyperparameter | Values tried | Final value | Why |
|---|---|---|---|
| Docker image | Record exact image/hash | Mark final | Reproducibility |
| Torch/CUDA/flash-attention versions | Record exact versions | Mark final | vLLM compatibility |
| GPU memory utilization | Tried values | Mark final | OOM prevention |
| llama.cpp context size | Tried values | Mark final | Proposer prompt capacity |
| llama.cpp GPU layers / tensor split | Tried values | Mark final | 35B fit and speed |

Use an asterisk in final thesis tables for the final selected values, following
the reference thesis style.

### 4.7 Baselines And Comparisons

Purpose: make each comparison scientifically interpretable.

Embedding inversion:

- Compare architecture/loss/checkpoint variants only when they share data,
  tokenizer, schedule, and evaluation protocol.
- Compare against paper target values only as reproduction context if the setup
  matches enough details.

SIPIT:

- Primary paper baselines: BruteForce and HardPrompts.
- Exact-match accuracy and vocabulary explored are the key comparison metrics.
- Mistral FP4 and cancelled runs must be labeled as incomplete.

Soft prompts:

- Compare soft-prompt performance against the same frozen-model baseline on the
  same split and seed.
- Compare virtual-token lengths only when dataset, split, seed, max sequence
  length, and training budget are otherwise matched.
- Treat text initialization as a control, not as the main interpretability
  condition, because nearest-token projection is strongly influenced by the
  initialization text.
- Compare SIPIT readout of soft prompts against random hard-token,
  initialization-prompt, and random-continuous controls before interpreting any
  recovered text.

GEPA/G-EVAL:

- `base_gepa`: metric feedback only.
- `ppl`: metric feedback plus response-only perplexity.
- `ppl_nla`: metric feedback plus perplexity plus NLA verbalizations.
- `ppl_nla_auxjudge`: metric feedback plus perplexity plus NLA, optionally
  compressed by a Qwen35B auxiliary judge.

Clean NLA claim requires:

- same dataset;
- same dimension;
- same seed;
- same train/validation/final-test split sizes;
- same proposer;
- same GEPA budget;
- same instruction proposer;
- PPL control when comparing PPL+NLA.

Smokes:

- Smoke runs are valid for crash detection and artifact validation.
- Smoke runs are not sufficient to make final scientific claims about GEPA
  convergence.

### 4.8 Figures And Tables To Add

Tables:

- Dataset inventory and split table.
- Cluster hardware table.
- Model-role table.
- Metric inventory table with directionality.
- Hyperparameter table with values tried and final values marked.
- Artifact checklist table.
- Runtime/efficiency table.

Figures:

- Chapter 4 does not need many figures, but one compact experiment setup figure
  can help:
  `dataset -> split -> base judge/vLLM -> GEPA -> proposer/llama.cpp ->
  final-test metrics`.
- If Chapter 3 already contains the full method diagram, Chapter 4 can instead
  use only tables and keep the figure reference there.

Value-adding examples:

- A short example of a `run_config_*.json` field group can be included if it
  clarifies split semantics or model roles.
- A compact example metric row can show how to read `coverage`, `n`, and
  correlation metrics.
- Prompt and proposer-feedback examples are embedded directly in the
  single-file LaTeX source, but not as a standalone appendix: Chapter 3 contains
  method-facing plain-text/Python-like examples close to the real artifacts,
  while Chapter 5 contains prompt changes and the full final optimized prompt
  that explain the reported results. Large logs should remain summarized or
  linked, not embedded.

### 4.9 Missing Items Before Final Thesis Writing

High priority:

- Add or extract per-stage timing for GEPA runs.
- Add peak GPU memory or at least `nvidia-smi` snapshots for final runs.
- Add negation preservation, polarity flip, and counterfactual preservation
  metrics if the semantic-fidelity dataset is used for final claims.
- Recover or rerun final SIPIT CSV/JSON outputs if possible; otherwise label
  the existing SIPIT evidence as interim/log-derived.
- Summarize NLA feedback health with coverage, useful rows, duplicate rate,
  token-selection strategy, and non-empty verbalization rate.
- Summarize auxiliary-judge feedback health with success rate and non-empty
  feedback rate.

Medium priority:

- Add BERTScore for text reconstruction outputs if the embedding-inversion
  branch is kept in main results.
- Add calibration/error-bucket summaries for GEPA judge predictions:
  over-rating rate, under-rating rate, exact-score rate, and score distribution.
- Add prompt-length and prompt-token-count trajectory metrics for GEPA.
- Add Docker image hash and exact dependency manifest for thesis-grade runs.

Low priority:

- Queue wait time.
- Full throughput profiling for every model server.
- Detailed CPU/RAM/disk telemetry for every run, unless it explains a failure.

## Transition To Chapter 5

Close Chapter 4 by stating that the next chapter uses these datasets, metrics,
and artifacts to report:

- embedding-inversion reproduction diagnostics;
- SIPIT reproduction and logical/random-prefix experiments;
- standalone NLA validation;
- GEPA/G-EVAL results across PPL, NLA, and auxiliary-judge feedback variants;
- runtime and failure-mode analysis.

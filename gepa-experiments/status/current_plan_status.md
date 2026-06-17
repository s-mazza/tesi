# GEPA / G-Eval Current Plan Status

Last updated: 2026-06-17

## Objective

Run GEPA experiments on the G-Eval benchmark settings with paper-aligned datasets, dimensions, and metrics, then isolate whether NLA feedback improves GEPA prompt optimization or why it fails to do so.

## Paper-Aligned Targets

The full scientific matrix must cover all G-Eval dimensions, not only one representative metric/dimension:

- SummEval: `fluency`, `coherence`, `consistency`, `relevance`
- Topical-Chat: `naturalness`, `coherence`, `engagingness`, `groundedness`
- QAGS-CNN: `consistency`
- QAGS-XSUM: `consistency`

Final comparison metrics:

- SummEval: `spearman`, `kendall_tau`
- Topical-Chat: `pearson`, `spearman`
- QAGS-CNN/QAGS-XSUM: `pearson`, `spearman`, `kendall_tau`

Internal diagnostics such as `agreement`, `mae`, `coverage`, and `parsed` remain useful for debugging but are not the primary paper comparison metrics.

## Experiment Variants

For each dataset/dimension target, run:

- `base_gepa`: GEPA with metric feedback only
- `ppl`: GEPA with response perplexity feedback from the base Qwen 7B model
- `ppl_nla`: GEPA with perplexity plus NLA verbalization feedback from the base Qwen 7B model
- `ppl_nla_auxjudge`: GEPA with perplexity, NLA, and optional 35B auxiliary judge feedback

The 35B llama.cpp model is used as GEPA proposer. The NLA and perplexity signals are computed on the base Qwen 7B model.

## 2026-06-16 Multi-Node Utilization Update

Current scheduling rule:

- Two-GPU Qwen35B proposer jobs remain flexible but effectively require `faretra`, because `moro232` has only one RTX 3090 and cannot host the llama.cpp proposer sidecar plus the main vLLM judge job.
- Single-GPU jobs that answer a distinct secondary question should be pinned to `moro232` when it is idle, so they do not compete with the two-GPU Qwen35B queue.
- `moro43` has an RTX 5090 but is not part of the current stable path; use it only after a compatibility smoke for the active CUDA/vLLM/container stack.

Submitted single-GPU soft-prompt jobs for the advisor-requested soft-prompt/SIPIT follow-up:

- `11914224`: first soft-prompt smoke on `moro232`; failed immediately because `train_soft_judge.py` imported the obsolete type name `EvalRow` instead of `EvalExample`.
- Fix applied: `gepa-experiments/soft_prompting/train_soft_judge.py` now imports and annotates `EvalExample`.
- `11914225`: replacement soft-prompt smoke on `moro232`; completed and produced adapter, metrics, predictions, `soft_prompt_embeddings.pt`, `nearest_tokens.jsonl`, and `sipit_soft_prompt_manifest.json` under `gepa-experiments/results/soft_prompt_topical_chat_engagingness_smoke`.
- Smoke split: 4/2/2 Topical-Chat groups, 24 train rows, 12 validation rows, 12 test rows.
- Smoke result: baseline and soft-prompt metrics were identical on validation and test, so this is only a plumbing/artifact validation, not scientific evidence that soft prompting helps.
- `11914226`: longer soft-prompt run on `moro232`, using all 60 Topical-Chat groups as 40/10/10 train/validation/test. It is intended to produce a more meaningful learned soft prompt for later SIPIT verbalization.
- `11914232`: follow-up soft-prompt full-context run on `moro232`, dependency `afterok:11914226`, same 40/10/10 split but `SOFT_PROMPT_MAX_SEQ_LEN=2048` and eval batch 1. It was queued because `11914226` skipped 6 training rows at max length 1024.
- `11914226` completed: 240 train rows, 234 tokenized; validation Pearson improved from 0.5371 to 0.5483, but test Pearson dropped from 0.7568 to 0.7235. Treat as useful soft-prompt artifact generation, not as a robust task improvement.
- `11914232` completed: 240 train rows, 240 tokenized; validation Pearson improved from 0.5371 to 0.5996, but test Pearson dropped from 0.7568 to 0.7085. Treat as a stronger validation signal but still not a final scientific improvement claim because final-test degradation remains.
- `11914237`: SIPIT-style bounded recovery for the 2048 soft prompt, dependency `afterok:11914232`, pinned to `moro232` because it consumes node-local 2048 artifacts produced there.
- `11914238`: older pinned SIPIT recovery for the 1024 soft prompt was cancelled to avoid unnecessary node pinning.
- `11914239`: SIPIT-style bounded recovery for the 1024 soft prompt, submitted without `SLURM_NODE` after syncing 1024 artifacts to both `moro232` and `faretra`; this job can run on the first compatible single-GPU node.

Advisor-requested NLA token-position sweep queued on `faretra`:

- Jobs `11914211` through `11914222` cover `candidate_first_1`, `candidate_middle_1`, `candidate_last_1`, `candidate_fml_3`, `candidate_quintile_5`, `candidate_even_8`, `source_fml_3`, `reference_fml_3`, `balanced_fml_9`, `prompt_tail_6`, `evaluation_tail_3`, and `hybrid_context_dedup_8`.
- These jobs intentionally keep the Qwen35B proposer setting and therefore need two RTX 3090 GPUs on the same node; they cannot run on single-GPU `moro232`.
- They remain separate from the main pipeline and should be compared as an isolated NLA-token-selection diagnostic.

Operational fixes from this update:

- `gepa-experiments/config/soft_prompt_topical_chat_engagingness_smoke.env` now requests `SLURM_MEM=28G`, because `moro232` cannot satisfy the previous 64G request.
- Added `gepa-experiments/config/soft_prompt_topical_chat_engagingness_long.env`.
- Added `gepa-experiments/config/soft_prompt_topical_chat_engagingness_long_2048.env`.
- Added `gepa-experiments/soft_prompting/sipit_soft_prompt_recover.py` and `gepa-experiments/slurm/run_soft_prompt_sipit_job.sh` for separate SIPIT recovery diagnostics on learned soft-prompt embeddings.
- Added SIPIT recovery configs for the 1024 and 2048 soft-prompt artifacts.
- `gepa-experiments/slurm/submit_soft_prompt.sh` now supports `SLURM_DEPENDENCY` and starts the Telegram monitor like the GEPA submitters.
- `gepa-experiments/slurm/telegram_monitor.py` now retries Telegram sends with an insecure SSL context only after certificate verification fails, and logs HTTP error bodies for diagnosis.
- Cluster Telegram monitoring currently has a CA-chain issue on both `faretra` and `moro232`; the monitor fallback prevents SSL-only failures, but any persistent HTTP 403 still indicates a token/chat/API problem to debug separately.
- Scheduling rule refinement: independent single-GPU jobs should not be pinned. Artifact-dependent jobs may be pinned only when the consumed artifact is node-local and has not yet been synchronized to all eligible nodes.

## 2026-06-17 Soft-Prompt Random-Init Follow-Up

Advisor-requested soft-prompt fix has been implemented and submitted. The old
soft-prompt runs used PEFT `PromptTuningInit.TEXT`, initialized from the seed
judge instruction. That made the nearest-token projection heavily biased toward
the initialization sentence, so the SIPIT/nearest-token diagnostics were not a
clean test of what the learned soft tokens encode. The new runs use
`PromptTuningInit.RANDOM` by default, while keeping `--soft-prompt-init text`
available only as an explicit control.

Implementation changes:

- `train_soft_judge.py` now supports `--soft-prompt-init {random,text}`;
  `random` is the default used by Slurm.
- nearest-token decoding now ranks candidate tokens by L2 distance in embedding
  space.
- cosine similarity is still saved as a qualitative metric for every nearest
  token.
- aggregate diagnostics now include `nearest_mean_l2`,
  `nearest_mean_cosine`, and `nearest_cosine_variance`.
- `nearest_cosine_variance` is interpreted as the variance, across virtual
  tokens, of the cosine similarity between each learned soft token and its
  top-1 L2 nearest discrete token. High variance means some soft tokens are much
  closer to the token embedding manifold than others.
- `submit_soft_prompt.sh` now sets the Slurm job name to `JOB_SLUG`, so new
  soft-prompt jobs are identifiable in `squeue`.

Validation before submission:

- local unit tests: `PYTHONPATH=gepa-experiments python3 -m unittest discover -s gepa-experiments/tests` passed, 45 tests.
- `faretra` and `moro232` Docker preflight passed: image
  `geval_gepa_softprompt:latest` is present, PEFT exposes `RANDOM`, and both
  soft-prompt scripts compile inside the container.

Submitted random-init jobs:

```text
11930497  smoke random-init, 16 virtual tokens, 4/2/2 groups, node moro232
11930498  long random-init, max_seq_len=1024, 16 virtual tokens, node faretra
11930499  SIPIT recovery for 11930498, dependency afterok:11930498, node faretra
11930500  long random-init, max_seq_len=2048, 16 virtual tokens, node moro232
11930501  SIPIT recovery for 11930500, dependency afterok:11930500, node moro232
11930502  SIPIT precision16 recovery for 11930500, dependency afterok:11930500, node moro232
11930503  long random-init, max_seq_len=2048, 8 virtual tokens, node faretra
11930504  SIPIT recovery for 11930503, dependency afterok:11930503, node faretra
11930505  long random-init, max_seq_len=2048, 32 virtual tokens, node moro232
11930506  SIPIT recovery for 11930505, dependency afterok:11930505, node moro232
11930507  long random-init, max_seq_len=2048, 16 virtual tokens, seed 43, flexible node
11930508  long random-init, max_seq_len=2048, 16 virtual tokens, seed 44, flexible node
```

Design decisions:

- The 8/16/32 virtual-token sweep is the first controlled length sweep. The
  16-token setting reproduces the previous length; 8 tests whether a shorter
  prompt reduces overfit and improves interpretability; 32 tests whether more
  capacity improves validation/test behavior or makes the soft prompt more
  off-manifold.
- Training jobs with dependent SIPIT recovery are pinned in pairs to the same
  node to avoid node-local artifact misses. Seed-only robustness jobs do not
  have dependent recovery and can run on the first compatible node.
- The smoke job is for plumbing only. Scientific conclusions should use the
  long 40/10/10 group runs and the matched SIPIT diagnostics.

## 2026-06-13 Matrix Expansion

The plan now includes an exhaustive job matrix and a new independent
multi-dimension prompt family. The full details, job ordering, artifact
requirements, and runtime estimates are archived in:

- `gepa-experiments/status/full_matrix_execution_plan_20260613.md`

This addition does not replace the current single-dimension pipeline. The
existing runner/configs remain the paper-aligned primary path. The new
multi-dimension path must be implemented separately so future readers can
reproduce both experimental families independently.

New multi-dimension requirement:

- For each dataset family, run a joint-prompt setting where one prompt scores
  every available dimension for that dataset in one response.
- Topical-Chat joint prompt scores naturalness, coherence, engagingness, and
  groundedness.
- SummEval joint prompt scores fluency, coherence, consistency, and relevance.
- QAGS-CNN and QAGS-XSUM have only consistency, so their joint-prompt jobs are
  symmetry/control jobs rather than true multi-dimension jobs.
- Joint-prompt results must be stored under separate output namespaces and
  marked as `joint_prompt` in result tables. They must not be mixed with
  paper-aligned single-dimension G-Eval comparisons.

Complete exhaustive matrix:

- Single-dimension paper-aligned jobs: 10 dataset/dimension targets x 4
  variants = 40 jobs.
- Joint-prompt multi-dimension jobs: 4 dataset families x 4 variants = 16 jobs.
- Total planned matrix: 56 jobs.
- Estimated total wall-clock job time: about 600 hours before queueing and
  failures; about 583 hours remain after counting the already completed
  Topical-Chat engagingness long PPL and fixed-NLA equivalents.

Required timing improvement before launching the full long matrix:

- Current artifacts store total elapsed time in `runtime_manifest_*.json`.
- Future long jobs must also write per-stage timing, at minimum for data
  loading/splitting, preflight, NLA manifest export, NLA precompute, vLLM
  startup, llama.cpp startup, perplexity precompute, GEPA compile, baseline
  final-test evaluation, optimized final-test evaluation, and artifact export.

Deadline and launch priority from the expanded plan:

- Target completion date: before 2026-06-29, meaning by the end of
  2026-06-28 CEST.
- `base_gepa` jobs are now lowest priority because they are not currently
  needed for the main thesis decision.
- Priority 1: Topical-Chat engagingness `ppl_nla_auxjudge`, SummEval
  consistency `ppl`/`ppl_nla`, and one joint-prompt benchmark:
  Topical-Chat all-dimensions `ppl`.
- Priority 2: remaining non-base paper dimensions for Topical-Chat, SummEval,
  QAGS-CNN, and QAGS-XSUM.
- Priority 3: broader auxiliary-judge variants after the first aux-judge result
  is known.
- Priority 4: remaining Topical-Chat/SummEval joint-prompt jobs.
- Priority 5: QAGS joint-prompt symmetry jobs.
- Priority 6: all `base_gepa` jobs, launched only if spare GPU capacity would
  otherwise be unused or after the non-base matrix is queued.

2026-06-14 launch update:

- Added a matrix-wide smoke gate policy in
  `gepa-experiments/status/full_matrix_execution_plan_20260613.md`. Smokes are
  evaluated for every job class, not only for auxiliary-judge jobs.
- Added long config
  `gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_nla_auxjudge_llamacpp35b.env`
  for `SD-04`.
- Submitted fresh `SD-04` smoke after the auxiliary-judge fix:
  job `11913885`.
- Submitted `SD-04` long run with Slurm dependency `afterok:11913885`:
  job `11913886`.
- Rationale: the previous aux-judge smoke was not scientifically valid because
  Qwen35B feedback was empty but marked as `ok`. The new smoke must verify the
  fixed aux-judge success-rate guard before the long job can consume GPU time.

2026-06-15 launch update:

- Smoke job `11913885` started on `faretra` and failed after 23 seconds before
  GEPA began.
- Root cause: llama.cpp tried to load the Qwen35B GGUF sidecar on a GPU with
  only about 1.7 GiB free. The model load then failed with CUDA OOM while
  allocating about 20.6 GiB. This was a cluster/GPU allocation contamination
  issue, not a GEPA metric/proposer error.
- The dependent long job `11913886` was left in `DependencyNeverSatisfied` and
  was cancelled.
- Launcher fix: for llama.cpp proposer jobs, `run_docker.sh` now selects the
  proposer GPU from the allocated Slurm devices by highest free memory when no
  explicit `PROPOSER_GPU_DEVICE` is set, keeps judge/proposer devices distinct,
  and uses config-level startup memory gates.
- Aux-judge smoke/long configs now require:
  - `JUDGE_MIN_FREE_MEMORY_MIB=22500`
  - `PROPOSER_MIN_FREE_MEMORY_MIB=22500`
  - `GPU_MEMORY_WAIT_SECONDS=300`
- The five-minute memory wait is intentional for these Qwen35B aux-judge runs:
  if Slurm assigns a GPU that is briefly polluted by unrelated processes, the
  job waits for the GPU to become usable instead of failing immediately. If the
  GPU is still below the required free-memory threshold after the wait, the job
  fails cleanly and releases the allocation.
- Replacement smoke submitted: job `11913922`.
- Replacement long submitted with `afterok:11913922`: job `11913923`.

2026-06-16 replacement update:

- Smoke job `11913922` failed before GEPA began. Root cause: the llama.cpp
  Qwen35B sidecar saw only 7578 MiB free on the selected RTX 3090 and failed
  while allocating about 20583 MiB.
- Additional discrepancy found: the remote launcher on `faretra` still used a
  stale GPU-selection path that assigned fixed first/second devices instead of
  the local highest-free-memory selection. The local `run_docker.sh` was synced
  to `faretra` before resubmission.
- Config hardening at submission time: both aux-judge smoke and long configs
  set `GPU_MEMORY_WAIT_SECONDS=300` with 18000/22500 MiB memory thresholds.
- Follow-up hardening: `JUDGE_MIN_FREE_MEMORY_MIB` was raised from 18000 to
  22500 before the replacement jobs started. With `GPU_MEMORY_UTILIZATION=0.90`
  on a 24 GiB RTX 3090, an 18 GiB gate can still pass while vLLM later needs
  more than 21 GiB and crashes during startup.
- Pre-submit readiness check passed for the smoke config after the sync.
- Cancelled stale dependent long job `11913923`
  (`DependencyNeverSatisfied`).
- Replacement smoke submitted: job `11914197`.
- Replacement long submitted with `afterok:11914197`: job `11914198`.

Auxiliary judge meaning:

- The base trained/evaluated judge remains Qwen2.5-7B-Instruct.
- Perplexity is computed on Qwen2.5-7B-Instruct.
- NLA verbalizes activations from Qwen2.5-7B-Instruct.
- The proposer remains Qwen 35B via llama.cpp.
- When enabled, the auxiliary LLM-as-a-judge feedback is also produced by Qwen 35B via llama.cpp and is passed only as extra feedback to GEPA's proposer/reflection loop.
- The auxiliary judge does not replace the base 7B judge and does not directly define final paper metrics.

## NLA Root-Cause Findings

Before scaling the full matrix, the long NLA run was diagnosed against the closest long non-NLA control. The intended control differs only by enabling NLA:

- same dataset and dimension
- same seed
- same train/validation/test context counts
- same proposer model and proposer settings
- same GEPA budget
- same perplexity feedback
- same generalizing instruction proposer

Required artifacts for both runs:

- `metrics_*.csv`
- `run_config_*.json`
- `baseline_predictions_*.jsonl`
- `optimized_predictions_*.jsonl`
- `seed_prompt_*.txt`
- `optimized_prompt_*.txt`
- `prompt_trajectory_*.jsonl`
- `gepa_viz_run_*.json`
- `runtime_manifest_*.json`
- Slurm, vLLM, llama.cpp, GEPA, and NLA precompute logs

Extra required artifact for NLA runs:

- `nla_verbalizations_*.jsonl`

The diagnostic report answered:

- The comparison was 1-to-1 after normalizing legacy artifact keys, except for NLA-specific fields.
- NLA covered the GEPA train/validation examples.
- NLA worsened final-test metrics and prediction-level errors.
- The failure is most likely caused by weak token selection and noisy/repetitive verbalizations, not by a definitive failure of the NLA idea.
- The old selector used first semantic tokens from source/candidate/reference when the budget was small.
- In the long run this produced weak tokens such as `reading`, `recently`, and `from`.
- This means the run tested GEPA with noisy first-token NLA feedback, not a strong NLA condition.

Use `gepa-experiments/scripts/diagnose_nla_run.py` for this comparison.

Additional cross-run evidence report:

- `gepa-experiments/scripts/analyze_nla_evidence.py`
- latest report: `gepa-experiments/results/diagnostics/nla_evidence_deep_dive_20260612.md`
- latest machine-readable artifact: `gepa-experiments/results/diagnostics/nla_evidence_deep_dive_20260612.json`

This report aggregates the long runs, Qwen35B smoke runs, single-GPU audit runs, dataset smoke controls, prompt trajectories, prediction distributions, and NLA feedback-health statistics.

Current root-cause conclusion from the aggregate evidence:

- The first old NLA long run was a clearly bad NLA condition: weak first-token selection and high duplicate feedback led to a large metric drop.
- The fixed-NLA long run is much healthier technically, but GEPA selected the seed prompt unchanged after 740 trajectory rows.
- Fixed-NLA beats the old PPL long run on Topical-Chat Pearson/Spearman but loses on agreement/MAE, so the old comparison is not a clean NLA improvement claim.
- The matched current-code PPL long control `11913587` has now completed. Against this stricter control, fixed-NLA is slightly better on all recorded final-test metrics, but both runs selected the byte-identical seed prompt. The observed gain is only two improved final-test predictions out of 60, so it is weak evidence and not proof that GEPA found a better prompt because of NLA.
- Candidate-only NLA is not the answer by itself. `candidate_content_6` and `candidate_content_10` remove duplicate source/reference repetition, but both worsen the matched Qwen35B smoke results.
- Therefore the most likely bottleneck is not just duplicate text. The raw NLA verbalizations are still mostly completion/association-style text, not metric-aligned explanations of why the score should move up or down.
- The next useful intervention is to transform NLA into short rubric-conditioned error feedback before GEPA's reflection/proposer step, instead of handing raw token verbalizations directly to the proposer.

## Fixed-NLA Smoke Findings

The first fixed-NLA smoke that completed successfully is `11913262`, replacing failed job `11912918`.

Matched comparison:

- Control: `11913161`, Topical-Chat engagingness PPL-only smoke.
- Treatment: `11913262`, same setting plus fixed real NLA feedback.
- Same dataset, dimension, seed, split sizes, proposer model/settings, perplexity feedback, and GEPA budget.
- Only intended differences are `nla_feedback`, `nla_backend`, `nla_precomputed_path`, and `nla_max_tokens_per_example`.

Final-test metric movement on the 12-example smoke slice:

- Optimized Pearson: 0.536400 -> 0.674979, delta +0.138580, +25.84%.
- Optimized Spearman: 0.527410 -> 0.674693, delta +0.147283, +27.93%.
- Optimized Kendall tau: 0.459933 -> 0.606407, delta +0.146474, +31.85%.
- Optimized agreement: 0.638889 -> 0.763889, delta +0.125000, +19.57%.
- Optimized MAE: 0.722222 -> 0.472222, delta -0.250000, 34.62% lower error.
- Prediction-level movement: 4 improved examples, 1 worsened example, 7 unchanged.

NLA feedback health for `11913262`:

- Raw activation vectors are not persisted in the current artifact format. The stored artifacts contain token metadata and natural-language verbalizations of activation vectors.
- Precomputed rows: 210.
- Covered examples: 36/36.
- Useful rows after the parser/validator fix: 210/210.
- Emitted verbalization rows: 210.
- Token status: `ok` for all 210 rows.
- Parse status: `partial_tags` for all 210 rows after stripping closing-tag-only verbalizer output.
- Token position mix: 108 candidate rows, 36 source rows, 66 reference rows.
- Rows per example: 30 examples have 6 rows, 6 examples have 5 rows.
- Verbalization length: mean about 10.7 words, median 11 words.
- Duplicate/repeated verbalizations remain non-trivial: the diagnostic report finds 107 duplicate text rows. This is much better than the first long NLA run, but still a core quality issue.

Manual qualitative pattern:

- The fixed selector no longer collapses to weak first tokens only; candidate tokens now dominate the feedback rows.
- Many source/reference verbalizations are repeated across the six candidate responses from the same context, because source and reference text are context-level signals. This likely explains a large fraction of duplicate rows.
- The verbalizations often describe likely continuations or topical completions, not direct high-level rubric concepts. Example pattern: `"or database or model organism"` for a zebrafish-related activation.
- Despite imperfect verbalization semantics, the proposer used the extra feedback to move the prompt toward a useful scoring rule: relevant responses that keep the conversation flowing should usually be at least 2, while score 1 should be reserved for genuinely disengaged, irrelevant, or conversation-ending responses.
- This prompt change directly matches the error pattern where the PPL-only run underrated several high-scoring human examples.

Current interpretation:

- Fixed-NLA is now a plausible positive signal, not just a plumbing test.
- The smoke result is scientifically encouraging but not sufficient for thesis claims because the final test has only 12 examples.
- The next main-pipeline action is a longer fixed-NLA run with the same long setting as the prior PPL-only long control.
- `candidate_content_10` has now finished and is negative, so the next diagnostic action is no longer candidate-only token selection. The next isolated strategy is semantic compression of NLA feedback, starting with the auxiliary-judge smoke.
- Detailed activation-verbalization analysis is saved in `gepa-experiments/results/diagnostics/nla_activation_verbalization_quality_20260610.md`.

Extra activation-verbalization reading from the current artifacts:

- The current artifacts do not contain raw activation vectors. They contain selected-token metadata and natural-language verbalizations generated from those activation vectors.
- Fixed-NLA `11913262` has 210 rows across 36 examples: 108 candidate rows, 36 source rows, and 66 reference rows.
- Fixed-NLA has 123/210 unique normalized verbalizations, with 87 exact duplicate normalized rows in direct inspection and 107 duplicate rows under the stricter diagnostic normalization.
- Fixed-NLA still has many completion-style verbalizations: 192/210 rows look like quoted continuations or topical completions rather than direct rubric concepts.
- Candidate-only precompute for completed job `11912947` had 187 rows across the same 36 examples, all candidate rows, all `token_status=ok`, all non-empty `partial_tags`, and 187/187 unique normalized verbalizations.
- Candidate-only `11912947` has fewer weak token rows than fixed-NLA in direct inspection: 21/187, 11.2%, versus 40/210, 19.0%, for fixed-NLA.
- Candidate-only `11912947` now has final GEPA metrics and should be treated as negative evidence for a candidate-only replacement, despite the cleaner feedback artifact.
- Candidate-only optimized metrics are much worse than fixed-NLA on the same 12-example final-test slice:
  - Pearson: fixed-NLA 0.674979 vs candidate-only 0.330901.
  - Spearman: fixed-NLA 0.674693 vs candidate-only 0.301374.
  - Kendall tau: fixed-NLA 0.606407 vs candidate-only 0.259889.
  - Agreement: fixed-NLA 0.763889 vs candidate-only 0.583333.
  - MAE: fixed-NLA 0.472222 vs candidate-only 0.833333.
- Against fixed-NLA, candidate-only lowers five final-test predictions and worsens all five of those cases or leaves them no better. It especially underrates high-human-score responses:
  - `context_055_response_00`: target 3.0, fixed-NLA 2, candidate-only 1.
  - `context_055_response_04`: target 2.67, fixed-NLA 2, candidate-only 1.
  - `context_055_response_05`: target 3.0, fixed-NLA 3, candidate-only 2.
  - `context_058_response_05`: target 3.0, fixed-NLA 3, candidate-only 2.
- Qualitative read: candidate-only made the feedback artifact cleaner, but GEPA produced a stricter, more aesthetic prompt that over-penalizes responses for not having enough flair or direct social integration. Fixed-NLA produced a better dataset-aligned rule: relevant responses that keep the conversation moving should usually get at least 2, and score 1 should be reserved for genuinely disengaged or irrelevant replies.
- Updated hypothesis: duplicate reduction alone is not the right objective. Source/reference rows are repetitive, but they may provide useful grounding and may help GEPA avoid an overly strict candidate-style rubric. The next NLA strategy should deduplicate or compress context-level rows rather than removing source/reference feedback entirely.
- Implementation decision: the hybrid deduplicated strategy is implemented only in the isolated experimental path, not in the main fixed-NLA runner path. The provider now supports optional `__group__:<group_id>` rows, so one context-level source/reference verbalization can be reused for all responses in the same context without storing repeated source/reference rows for every candidate.
- `hybrid_context_dedup_6` is the first ready hybrid smoke config. It keeps the same total NLA feedback budget as fixed-NLA: up to 4 candidate rows per response plus 1 source and 1 reference row shared per context.
- CPU token-selection check on the current 36-example smoke split:
  - `current_fixed_6`: 210 selected token rows, 108 candidate, 36 source, 66 reference, weak token pct 17.14.
  - `candidate_content_6`: 187 rows, all candidate, weak token pct 0.00.
  - `hybrid_context_dedup_6`: 141 rows, 129 candidate, 6 source, 6 reference, weak token pct 0.00.
  - `hybrid_context_dedup_8`: 199 rows, 187 candidate, 6 source, 6 reference, weak token pct 0.00.
- Decision: do not launch or merge `hybrid_context_dedup_6` ahead of the current long control and auxiliary-judge smoke unless explicitly needed. Candidate10 already confirms that pure candidate-only feedback is weak; hybrid dedup remains a backup test for "keep grounding but remove repeated context rows."

## Single-GPU Topical-Chat Smoke Audit

Jobs `11913130` and `11913131` completed on `moro232` and their node-local artifacts were recovered to the local workspace.

Artifact locations:

- PPL-only single-GPU smoke: `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_smoke`
- PPL + NLA single-GPU smoke: `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_smoke`
- Diagnostic report: `gepa-experiments/results/diagnostics/nla_single_gpu_vs_ppl_smoke_20260610.md`

Result status:

- These two runs are not a clean scientific ablation.
- `11913130` used 4 train groups, 2 validation groups, and 2 test groups; `11913131` used 2 train groups, 1 validation group, and 1 test group.
- Both runs used `Qwen/Qwen2.5-7B-Instruct` as proposer, not Qwen 35B via llama.cpp.
- `11913131` was contaminated by incomplete NLA precompute: the config had `NLA_PRECOMPUTE_LIMIT=6`, while GEPA used 18 train+validation rows. The emitted NLA feedback artifact contains 18 `DRY RUN ONLY` placeholder rows and only 6 real `partial_tags` rows.
- Therefore, the apparent metric delta from this comparison is not scientifically interpretable and must not be used as NLA evidence.

Observed metrics, recorded only for audit:

- `11913130` PPL-only final test, optimized: Pearson 0.314271, Spearman 0.345082, MAE 0.750000, agreement 0.625000 on 12 test examples.
- `11913131` PPL+NLA final test, optimized: Pearson 0.643921, Spearman 0.651533, MAE 0.611111, agreement 0.694444 on 6 different test examples.
- Prediction-level comparison joined 0 examples because the final-test slices differ.

Fix applied after the audit:

- `NlaFeedbackProvider` no longer falls back to dry-run placeholder verbalizations when `backend=precomputed` and rows are missing.
- Added a regression test that verifies missing precomputed rows produce no dry-run artifact.
- Removed `NLA_PRECOMPUTE_LIMIT=6` from the single-GPU real-NLA smoke config and added explicit `NLA_MIN_COVERAGE=0.95`.
- Local validation: `PYTHONPATH=gepa-experiments python3 -m pytest gepa-experiments/tests/test_data_and_metrics.py` passes with 38 tests.

Decision:

- Do not relaunch the single-GPU NLA smoke immediately; it would answer a secondary question and may compete with more important queued work.
- If a future single-GPU smoke is needed, rerun it only after the current code/config fix is synced, and treat it as plumbing evidence only unless the split/proposer/budget are made 1-to-1.

Matched follow-up `11913388`:

- `11913388` completed on `moro232` and artifacts were recovered locally from `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_single_gpu_matched_smoke`.
- Matched control: `11913130`, PPL-only single-GPU smoke.
- Diagnostic report: `gepa-experiments/results/diagnostics/nla_single_gpu_matched_vs_ppl_smoke_20260611.md`.
- Config match is clean except intended NLA fields: same dataset, dimension, seed, train/validation/test groups, judge/proposer model, proposer settings, instruction proposer, and perplexity feedback.
- Split is exactly matched: 24 GEPA train rows, 12 GEPA validation rows, and 12 final-test rows.
- NLA precompute is complete and real: 216 emitted rows across 36 covered train+validation examples, all `token_status=ok`, no dry-run placeholders, no suspicious rows.
- NLA artifact caveat: emitted rows have `parse_status=missing_tags` and no compact activation summary stats. The missing tags contain usable text and were accepted by the validator, but future NLA runs on `moro232` should sync the full latest code path, not only the provider/config files, so the activation-stat fields are preserved.
- Duplicate feedback remains high: 114 duplicate rows, dominated by source rows (107 source duplicates, 7 candidate duplicates). This is the same qualitative issue seen in prior fixed-NLA smoke runs.

Matched single-GPU metrics:

- PPL-only `11913130` optimized: Pearson 0.314271, Spearman 0.345082, MAE 0.750000, agreement 0.625000.
- PPL+NLA `11913388` optimized: Pearson 0.603136, Spearman 0.590879, MAE 0.555556, agreement 0.722222.
- Delta NLA minus PPL optimized: Pearson +0.288864 (+91.92%), Spearman +0.245797 (+71.23%), MAE -0.194444 (-25.93%), agreement +0.097222 (+15.56%).
- Prediction-level movement: 3 improved final-test examples, 0 worsened, 9 unchanged.

Interpretation:

- The result is technically positive against the PPL-only optimized prompt, but it is not evidence that NLA improved over the seed baseline.
- `11913388` optimized prompt is byte-identical to its seed prompt, so the "improvement" comes from NLA preventing the bad PPL-only prompt update rather than producing a better prompt.
- This is useful debugging evidence: under the 7B proposer single-GPU setup, NLA changed GEPA's search/selection behavior enough to avoid final-test degradation. It remains secondary because the thesis-relevant proposer setting is Qwen35B via llama.cpp.

## Implementation Status

Completed or in progress locally:

- Added paper-needed `kendall_tau` metric support.
- Added `kendall_tau` aggregation support.
- Added NLA precomputed coverage and useful-row validation before GEPA compile.
- Added dry-run NLA guard for scientific runs.
- Added pass-through Slurm variables for NLA minimum coverage and dry-run override.
- Added reusable NLA-vs-control diagnostic script.
- Generated first long-run NLA diagnostic report:
  - `gepa-experiments/results/diagnostics/nla_vs_ppl_long_20260608.md`
  - comparison: long PPL-only llama.cpp proposer run vs long PPL+NLA llama.cpp proposer run
  - normalized config check is 1-to-1 except NLA-specific fields
  - optimized Pearson dropped by 0.121668 with NLA
  - optimized Spearman dropped by 0.129078 with NLA
  - optimized MAE worsened by 0.194444 with NLA
  - prediction-level test movement: 6 examples improved, 18 worsened, 36 unchanged
  - NLA quality signal: 900 rows, 300 covered examples, all emitted parse statuses are `partial_tags`, and 667 rows are duplicate repeated text rows
  - Important clarification: the first diagnostic artifact showed `token_status=unknown` because the runner-emitted NLA artifact did not preserve `token_status`; the raw precompute file had `900/900 token_status=ok`
- Completed root-cause analysis:
  - `gepa-experiments/results/diagnostics/nla_root_cause_20260608.md`
  - root cause: old NLA token selection used weak first semantic tokens from source/candidate/reference when budget was small
  - this produced repeated, generic verbalizations and tested noisy NLA feedback rather than a strong NLA condition
  - implemented candidate-prioritized middle/final token selection
  - preserved `token_status` in emitted NLA artifacts
  - increased real-NLA token and generation budgets
  - removed the old `NLA_PRECOMPUTE_LIMIT=8` from the fixed-NLA smoke config so the coverage gate can validate full GEPA train/validation coverage
- Audited completed single-GPU Topical-Chat jobs `11913130` and `11913131`; marked `11913131` as non-scientific because incomplete NLA precompute caused dry-run placeholder feedback to enter the run.
- Hardened the NLA feedback provider so precomputed NLA can no longer silently fall back to dry-run placeholders when rows are missing.
- Removed the dangerous `NLA_PRECOMPUTE_LIMIT=6` from the single-GPU real-NLA smoke config and made `NLA_MIN_COVERAGE=0.95` explicit.

Latest completed long-control update:

- `11912948` completed and is negative. Larger candidate-only 10-token feedback did not recover the fixed-NLA smoke gain and should not be promoted into the main pipeline.
- `11913284`, the longer Topical-Chat engagingness PPL+fixed-NLA run, completed successfully and all required final artifacts were recovered locally.
- Current-code matched control `11913415` was submitted on 2026-06-11 with the same split, seed, GEPA budget, Qwen35B proposer, and PPL feedback as `11913284`, but with `NLA_FEEDBACK=0`.
- `11913415` failed after 53 seconds before GEPA started. Root cause: vLLM startup saw only 7.22 GiB free on the assigned judge GPU while `GPU_MEMORY_UTILIZATION=0.90` required about 21.32 GiB. `nvidia-smi` showed the memory was occupied by other users' processes on `faretra`, not by our stale containers.
- Replacement attempts `11913482` and `11913557` also failed before GEPA started because llama.cpp Qwen35B was assigned a proposer GPU with only about 7.4 GiB free and could not allocate its about 20.6 GiB CUDA buffer.
- Replacement current-code matched control `11913587` completed successfully on `faretra`; artifacts were synced locally to `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control`.
- Diagnostic report: `gepa-experiments/results/diagnostics/nla_fixed_long_vs_current_ppl_long_20260612.md`.
- Runtime: 30,049.160 seconds, about 8h20m49s.
- Baseline and optimized metrics are identical because GEPA also selected the unchanged seed prompt in the PPL current-code control.
- PPL current-code optimized, n=60: Pearson 0.658218, Spearman 0.658203, Kendall tau 0.555309, agreement 0.741667, MAE 0.516667.
- Fixed-NLA optimized minus PPL current-code optimized: Pearson +0.022940, Spearman +0.018872, Kendall tau +0.015841, agreement +0.011111, MAE -0.022222.
- Prediction movement: 2 improved examples, 0 worsened examples, 58 unchanged.
- Important interpretation: because seed and optimized prompts are byte-identical across both runs, this is not evidence that GEPA discovered a better prompt under NLA. It is a small positive final-test delta under otherwise matched runs, likely within judge/runtime stochasticity unless replicated.
- Do not use `11913131` as evidence for NLA effectiveness; it is an audit-only failure case because dry-run placeholders entered the feedback artifact.
- Since the matched current-code long control shows only a weak positive delta and no prompt improvement, prioritize auxiliary-judge summarized NLA feedback and hybrid context deduplication. Do not prioritize pure candidate-only replacement because both candidate-only smokes are now negative.
- Non-invasive diagnostic artifact support is now implemented for future runs:
  - prediction JSONL rows now include `source_text`, `fact`, `reference`, and `candidate_output`, so final-test qualitative errors can be inspected without rejoining the dataset manually.
  - NLA precomputed/verbalization rows now include compact activation-vector statistics: dimension, L2 norm, mean, std, min, max, and absolute mean.
  - Raw activation vectors are still intentionally not persisted by default.
- Caveat: activation summary stats are still missing from the recovered long-run NLA artifact, so the next NLA-heavy run must verify that the latest artifact schema is synced on the execution node before startup.
- For any future job that can run on non-`faretra` nodes, check the execution node filesystem or sync artifacts back to `faretra`; `moro232` results are not guaranteed to be visible from `faretra`.
- Queue scientifically useful pending work even when nodes are currently occupied, so jobs accrue scheduling age. Avoid only duplicate jobs or jobs that would answer no useful question.
- Completed three single-GPU PPL-only controls for already completed real-NLA dataset smoke runs:
  - `gepa-experiments/config/geval_gepa_summeval_consistency_ppl_smoke.env`
  - `gepa-experiments/config/geval_gepa_qags_cnn_consistency_ppl_smoke.env`
  - `gepa-experiments/config/geval_gepa_qags_xsum_consistency_ppl_smoke.env`
- These controls are matched to the corresponding real-NLA smoke split/budget and were pinned to `moro232`. They are technical ablations only, not final Qwen35B proposer results.

## Dataset Single-GPU Control Findings

Jobs `11913404`, `11913405`, and `11913406` completed on `moro232`; artifacts were recovered locally and diagnostic reports were generated.

Artifact locations:

- SummEval consistency PPL-only control: `gepa-experiments/results/geval_gepa_summeval_consistency_ppl_smoke`
- QAGS-CNN consistency PPL-only control: `gepa-experiments/results/geval_gepa_qags_cnn_consistency_ppl_smoke`
- QAGS-XSUM consistency PPL-only control: `gepa-experiments/results/geval_gepa_qags_xsum_consistency_ppl_smoke`
- Diagnostic reports:
  - `gepa-experiments/results/diagnostics/nla_summeval_consistency_vs_ppl_smoke_20260611.md`
  - `gepa-experiments/results/diagnostics/nla_qags_cnn_consistency_vs_ppl_smoke_20260611.md`
  - `gepa-experiments/results/diagnostics/nla_qags_xsum_consistency_vs_ppl_smoke_20260611.md`

SummEval consistency is the only one of these three smoke comparisons with enough final-test rows to read directionally:

- PPL-only optimized on 32 final-test examples: Pearson 0.701281, Spearman 0.790860, Kendall tau 0.716853, agreement 0.723958, MAE 1.104167.
- PPL+real-NLA optimized on the matched split: Pearson 0.618512, Spearman 0.696582, agreement 0.718750, MAE 1.125000. Kendall tau is unavailable in the older real-NLA artifact because that run predates the metric addition.
- NLA minus PPL-only: Pearson -0.082769 (-11.80%), Spearman -0.094278 (-11.92%), agreement -0.005208 (-0.72%), MAE +0.020833 (+1.89%, worse).
- Prediction movement: 3 improved, 3 worsened, 26 unchanged.
- Feedback artifact issue: 100 NLA rows across 50 examples, but 86 duplicate text rows and no activation-summary stats because the NLA run used an older artifact schema.

QAGS-CNN and QAGS-XSUM controls are valid plumbing checks but not scientifically interpretable:

- Both final-test slices have only `n=2`, so Pearson/Spearman/Kendall can be unstable or degenerate.
- QAGS-CNN PPL-only optimized improves agreement/MAE over the matched NLA run, but with only 2 examples this is not enough for a claim.
- QAGS-XSUM PPL-only optimized is better than the matched NLA run on the reported metrics, but the tiny final-test slice and older NLA artifact make this diagnostic-only.

Interpretation:

- These single-GPU dataset controls do not support a claim that the current NLA feedback improves GEPA across datasets.
- They do support the root-cause hypothesis that NLA feedback quality and compression matter: SummEval NLA has high duplicate text rows, and the older dataset NLA artifacts lack the richer token/activation stats now required for diagnosis.
- Do not scale dataset-level NLA from these 7B-proposer smoke runs alone. Use them as technical controls, not thesis-level performance claims.
- Next action from these results is not to launch duplicates; the matched current-code PPL-only long control has completed, and the next decisive experiment is an aux-judge or compressed-NLA smoke that tests whether raw NLA can be transformed into rubric-conditioned feedback.

## Fixed-NLA Long Run Findings

Job `11913284` completed on `faretra` and artifacts were recovered locally from `gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b`.

Runtime and artifacts:

- Runtime: 30,519.012 seconds, about 8h28m39s.
- Split: 40 train groups, 10 validation groups, 10 final-test groups; rows are 240 GEPA train, 60 GEPA validation, and 60 final test.
- Proposer: Qwen 35B via llama.cpp, `PROPOSER_TEMPERATURE=0.7`, `PROPOSER_MAX_TOKENS=4096`.
- Judge/PPL/NLA base model: `Qwen/Qwen2.5-7B-Instruct`.
- `prompt_trajectory_20260610T163111Z.jsonl` has 740 candidate rows.
- Final `optimized_prompt_20260610T163111Z.txt` is byte-identical to `seed_prompt_20260610T163111Z.txt`.

Final-test metrics for `11913284`:

- Baseline and optimized are identical because GEPA selected the unchanged seed prompt.
- Pearson 0.681158, Spearman 0.677076, Kendall tau 0.571150, agreement 0.752778, MAE 0.494444 on 60 examples.

Comparison against the old long PPL-only control `geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer`:

- Diagnostic report: `gepa-experiments/results/diagnostics/nla_fixed_long_vs_ppl_long_20260611.md`.
- Config comparison is mostly matched after legacy normalization: same dataset/dimension/seed/split/proposer/PPL feedback, NLA-specific fields differ.
- Fixed-NLA optimized vs old PPL-only optimized:
  - Pearson: 0.632812 -> 0.681158, delta +0.048346 (+7.64%).
  - Spearman: 0.619893 -> 0.677076, delta +0.057183 (+9.22%).
  - Agreement: 0.788889 -> 0.752778, delta -0.036111 (-4.58%).
  - MAE: 0.422222 -> 0.494444, delta +0.072222 (+17.11%, worse).
- Prediction-level movement against old PPL-only: 8 improved, 15 worsened, 37 unchanged.
- Prediction distribution shifted from PPL-only `{1: 14, 2: 36, 3: 10}` to fixed-NLA `{1: 23, 2: 19, 3: 18}`. This increases rank correlation by spreading scores more, but worsens absolute error on several human-score-2 examples.

NLA feedback quality for `11913284`:

- 1752 real NLA rows across 300 covered GEPA train/validation examples.
- `token_status=ok` for all rows.
- `parse_status=partial_tags` for all rows.
- Suspicious rows: 0.
- Average verbalization length: about 10.55 words.
- Duplicate text rows remain high: 925 total, mostly reference/source context repetition; category breakdown is candidate 77, reference 550, source 298.
- Token positions are healthier than the first negative long NLA run: 888 candidate rows, 300 source rows, 564 reference rows.
- Activation summary stats are still not present in the persisted artifact despite the intended support, so this needs a code/artifact audit before the next NLA-heavy run.

Interpretation:

- This fixed-NLA long run is not a clean positive GEPA optimization result because the optimized prompt is exactly the seed prompt.
- It is positive relative to the old long PPL-only optimized run on paper-primary Topical-Chat correlations, but negative on MAE/agreement.
- It is also slightly positive relative to the matched current-code PPL-only long control on all recorded metrics, but this comparison remains weak because the prompt is byte-identical and only two final-test examples change.
- It is much healthier than the first old NLA long run, which had Pearson/Spearman drops and a large MAE increase caused by weak/repetitive first-token feedback.
- The remaining ambiguity is now different: fixed-NLA did not produce a better selected prompt, so the small metric gain is not enough for a strong thesis claim that NLA improves GEPA. It should be reported as diagnostic/weak-positive evidence pending an aux-judge or compressed-NLA variant.
- Action taken: `11913587` completed and was compared against `11913284`; generated `nla_fixed_long_vs_current_ppl_long_20260612.md` and updated the aggregate evidence report.

## Future Step Roadmap

Step 1: finish and analyze smoke diagnostics.

- Completed artifact recovery for `11912914`, `11912915`, `11912916`, `11913111`, `11913112`, and `11913113`; use these artifacts to verify real-NLA precompute and runner behavior on SummEval, QAGS-CNN, and QAGS-XSUM.
- Recovered and audited 1-GPU Topical-Chat jobs `11913130` and `11913131`; they are not a valid matched comparison because split sizes differ, final-test examples do not overlap, proposer is not Qwen 35B, and `11913131` contains dry-run fallback feedback.
- Completed the matched Topical-Chat PPL-only vs fixed-NLA smoke comparison using `11913161` vs replacement `11913262`.
- Generated diagnostic report `gepa-experiments/results/diagnostics/nla_vs_ppl_fixed_smoke_20260610.md`.
- Decision gate result: fixed-NLA smoke is positive on the 12-example final-test slice and has valid NLA coverage/useful rows, so proceed to one longer fixed-NLA run while treating the smoke as suggestive rather than conclusive.
- Completed matched single-GPU dataset control diagnostics for SummEval consistency, QAGS-CNN consistency, and QAGS-XSUM consistency. SummEval is directionally useful and negative for NLA versus PPL-only; QAGS smokes are `n=2` and should be treated only as plumbing checks.

Step 2: launch and analyze one longer fixed-NLA Topical-Chat run.

- Dataset/dimension: Topical-Chat engagingness.
- Compare against the closest PPL-only control with same seed, split, proposer, and GEPA budget.
- Use fixed candidate-prioritized NLA token selection.
- Keep Qwen 35B as proposer.
- Do not enable auxiliary judge yet.
- Goal: verify whether fixed-NLA alone can recover or improve over the previous PPL-only result.
- Use new config `gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b.env` so the new fixed-NLA long run does not mix artifacts with the older negative long NLA run.
- Keep `SLURM_TIME=16:00:00` as a safety limit; the target is a long 8-12 hour style run, not necessarily a full 16 hours.
- Submitted this long fixed-NLA run as `11913284`.
- Completion status: `11913284` completed successfully on `faretra`, final artifacts were recovered locally, and `nla_fixed_long_vs_ppl_long_20260611.md` was generated.
- Startup status: NLA activation extraction completed for 300 manifest rows and 1752 real precomputed NLA feedback rows were written. This covers the GEPA train+validation manifest and is not affected by the incomplete single-GPU smoke issue.
- Final runtime: about 8h28m39s.
- Result: baseline and optimized metrics are identical because GEPA selected the unchanged seed prompt. The fixed-NLA run beats the old long PPL-only optimized run on Pearson/Spearman but loses on MAE/agreement.
- Matched follow-up control `11913587` completed with NLA disabled. Fixed-NLA is slightly better than this current-code control on all recorded metrics, but both runs kept the same seed prompt and only two final-test predictions changed. Treat this as weak-positive diagnostic evidence, not a clean GEPA+NLA optimization win.

Step 3: add Qwen 35B auxiliary LLM-as-a-judge feedback.

- Run a matched Topical-Chat engagingness ablation:
  - `ppl`
  - `ppl_nla`
  - `ppl_nla_auxjudge`
- Keep the same seed, data split, GEPA budget, proposer model, and base 7B judge.
- Only the third variant enables Qwen 35B LLM-as-a-judge feedback.
- Goal: isolate whether auxiliary judge feedback adds useful semantic feedback beyond perplexity and NLA.
- Implementation update: the auxiliary judge now receives the already-computed NLA text as extra weak feedback for the same example. Its prompt explicitly asks it to convert this into a general rubric-level lesson and not to copy token-level strings.
- Smoke config: `gepa-experiments/config/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke.env`.
- Submitted job: `11913424`, dependency `afterok:11913415`, 2 x RTX 3090, flexible node selection, `deeplearn2` excluded. It was cancelled after `11913415` failed, because the dependency became `DependencyNeverSatisfied` and because the aux-judge/NLA-context ablation needs to be resubmitted with an explicit, scientifically unambiguous config.
- Current submitted aux-judge smoke: `11913731`, submitted 2026-06-12 with the same smoke config.
- Scientific setting for `11913731`: Topical-Chat engagingness smoke, same 4/2/2 context split and seed as the matched Qwen35B smoke controls, PPL enabled, fixed real NLA enabled, Qwen35B llama.cpp used both as proposer and auxiliary judge.
- Aux-judge context for `11913731`: the auxiliary judge receives NLA text through `extra_feedback=nla_feedback_text`, so this run answers "can Qwen35B compress/interpret raw NLA into useful rubric feedback?", not merely "does an independent second judge help?"
- Cluster status after submission: pending with `Reason=ReqNodeNotAvail,_UnavailableNodes:faretra,moro43`. `moro232` is idle but has only one RTX 3090, while this job needs two RTX 3090 GPUs on the same node. `faretra` has 4/4 GPUs allocated by other users, so the job must wait for `faretra`.
- Interpretation rule: this smoke does not replace the matched long control. It only tests whether a semantic compression layer can make NLA usable by GEPA.
- If the future aux-judge smoke improves over both `ppl_smoke_q35` and `fixed_nla_smoke_q35`, the next action is a matched longer `ppl_nla_auxjudge` Topical-Chat run with the same 40/10/10 context split as `11913284` and the completed current-code PPL control.
- If it fails, do not launch another raw NLA long run. Instead inspect its auxiliary-judge records to decide whether the judge ignored NLA, copied noisy token strings, or produced rubric feedback that GEPA still overfit.

Step 4: scale to the paper-aligned dimension matrix.

- The full thesis matrix is no longer limited to one representative dimension
  per dataset family. It must cover all paper-aligned G-Eval dimensions listed
  in `Paper-Aligned Targets`.
- The exhaustive matrix is documented in
  `gepa-experiments/status/full_matrix_execution_plan_20260613.md`.
- For each dataset/dimension target, preserve all prompt artifacts, NLA
  artifacts, diagnostic reports, final predictions, paper-aligned metrics, and
  runtime/stage-timing artifacts.
- Prioritize jobs according to the matrix document rather than launching
  duplicates of already completed smoke/long runs.

Step 4b: add a separate multi-dimension joint-prompt pipeline.

- This is an additive experimental family, not a replacement for Step 4.
- It should optimize one prompt that scores all dimensions for one dataset in a
  single response.
- It must use separate entrypoints/configs/output directories and explicit
  `joint_prompt` result labels.
- Reuse common loaders, metrics, feedback providers, Slurm/Docker helpers, and
  artifact writers where that avoids duplication.
- Do not change the current single-dimension runner semantics while adding this
  path.

Step 5: if NLA remains negative or unstable after the fixed selector.

- Keep all creative NLA strategy experiments separate from the current main pipeline.
- Do not change the default NLA selector, runner behavior, or production configs for these experiments.
- Separate experimental scripts/configs are now available for candidate-content token strategies:
  - `gepa-experiments/scripts/experimental_nla_token_strategy_analysis.py`
  - `gepa-experiments/scripts/experimental_build_nla_precomputed.py`
  - `gepa-experiments/slurm/run_experimental_nla_strategy_job.sh`
  - `gepa-experiments/slurm/submit_experimental_nla_strategy.sh`
- Initial CPU-only token strategy analysis is saved in:
  - `gepa-experiments/results/diagnostics/experimental_nla_token_strategies_20260608/token_strategy_report.md`
  - `gepa-experiments/results/diagnostics/experimental_nla_token_strategies_20260608/token_strategy_summary.csv`
- The two best isolated strategies from the CPU analysis are:
  - `candidate_content_6`: candidate-only, weak token rows 0.00%, duplicate row pct 35.58%.
  - `candidate_content_10`: candidate-only, weak token rows 0.00%, duplicate row pct 34.89%.
- `candidate_source_content_8` remains a secondary cross-check, but it is not first priority because it has higher duplicate row pct 51.76%.
- Candidate-content NLA was tested as separate smoke experiments before considering any merge into the main pipeline. Both `11912947` (`candidate_content_6`) and `11912948` (`candidate_content_10`) completed and were negative, so candidate-only selection is not a current merge candidate.
- `11912947` completed successfully and is negative on final metrics, despite cleaner token-selection statistics.
- Hybrid deduplicated NLA is implemented as a separate experimental strategy:
  - `gepa-experiments/scripts/experimental_build_nla_precomputed.py` supports `hybrid_context_dedup_6` and `hybrid_context_dedup_8`.
  - `gepa-experiments/scripts/experimental_nla_token_strategy_analysis.py` reports both hybrid strategies.
  - `gepa-experiments/config/experimental_nla_hybrid_context_dedup_6_topical_chat_smoke.env` is ready for a future isolated smoke.
  - `NlaFeedbackProvider` can consume shared context rows with `example_id="__group__:<group_id>"`.
- Increase `NLA_MAX_TOKENS_PER_EXAMPLE` to 8 or 10 only in separate experimental configs.
- Lower proposer temperature to 0.2 or 0.0 only in matched experimental controls.
- Test summarized/compressed NLA feedback before giving it to the proposer, but keep it isolated from the current runner path unless it clearly wins.
- Test whether different NLA layers/checkpoints are available and compatible with Qwen2.5-7B.
- Merge an experimental NLA strategy into the main pipeline only after it improves the task under a fair 1-to-1 comparison with enough evidence to remove reasonable doubt.
- Only after these ablations decide whether NLA should be reported as negative/diagnostic rather than performance-improving.

Experimental NLA strategy rules:

- Every experimental strategy must keep the same dataset, dimension, seed, split, base 7B judge, proposer 35B, GEPA budget, and final metric computation as its matched control.
- Each experiment must have a paired PPL-only or current fixed-NLA control.
- Each experiment must save a diagnostic report with `diagnose_nla_run.py`.
- Each experiment must report both final metrics and feedback-health metrics: token positions, duplicate verbalizations, parse status, token status, and prediction-level error movement.
- A strategy is not allowed to replace the current pipeline just because it improves one smoke run; it needs at least one longer matched Topical-Chat run and one cross-dataset smoke before merge.

## Current Cluster Queue

The old pinned jobs `11912818`, `11912819`, and `11912820` were cancelled because they had `ReqNodeList=moro232`.

Replacement chain submitted without a node pin, keeping `ExcNodeList=deeplearn2`:

- `11912914`: SummEval consistency smoke, PPL + real NLA
- `11912915`: QAGS-CNN consistency smoke, PPL + real NLA, dependency `afterany:11912914`
- `11912916`: QAGS-XSUM consistency smoke, PPL + real NLA, dependency `afterany:11912915`

Latest status:

- `11912914`, `11912915`, and `11912916` were initially missing from `faretra` checks because they wrote artifacts on `moro232`'s local filesystem. Their artifacts were recovered together with `11913111`, `11913112`, and `11913113`.
- `sacct` is currently unusable because SlurmDB returns `Connection refused`; use node-local artifacts plus `/var/log/slurm/jobcomp.log` when old jobs age out of `scontrol`.
- The recovered 2026-06-08 and 2026-06-09 dataset smoke runs are scientifically usable only as small runner/NLA sanity checks, not as final paper-level comparisons.
- Re-submission on 2026-06-10 01:02 CEST to exploit the idle single-GPU `moro232` node without occupying `faretra`:
  - `11913111`: SummEval consistency smoke, PPL + real NLA, pinned to `moro232`.
  - `11913112`: QAGS-CNN consistency smoke, PPL + real NLA, pinned to `moro232`, dependency `afterany:11913111`.
  - `11913113`: QAGS-XSUM consistency smoke, PPL + real NLA, pinned to `moro232`, dependency `afterany:11913112`.
- This pinning is intentional and limited to 1-GPU dataset smoke jobs. It should not be used for Qwen35B proposer jobs, which need 2 GPUs on the same node.
- Latest check: 2026-06-10 10:38 CEST. Jobs `11913111`, `11913112`, and `11913113` are no longer visible in `squeue` or `scontrol` because `MinJobAge=300 sec` and SlurmDB accounting is down. The text job completion log shows they did run on `moro232` and completed with exit code `0:0`:
  - `11913111`: start 2026-06-10 01:01:56, end 01:27:28.
  - `11913112`: start 2026-06-10 01:27:28, end 01:35:07.
  - `11913113`: start 2026-06-10 01:35:07, end 01:41:47.
- Final diagnosis: the jobs wrote their stdout and result artifacts to the local filesystem visible on `moro232`, not to the filesystem visible from `faretra`. Direct SSH from the local machine to `moro232` works and confirmed the artifacts.
- Recovered and synchronized artifacts from `moro232` to local workspace and then to `faretra`, preserving the expected experiment directories.
- Extra diagnostic jobs `11913124` and `11913129` were cancelled because direct SSH recovery made them unnecessary.
- Important operational consequence: when jobs run on a node whose home/workdir is not shared with `faretra`, result checks must query that node or explicitly sync artifacts back to `faretra`.
- Recovered metrics:
  - `11913111` SummEval consistency smoke: baseline Pearson 0.659360, Spearman 0.615213, MAE 1.375; optimized Pearson 0.618512, Spearman 0.696582, MAE 1.125.
  - `11913112` QAGS-CNN consistency smoke: baseline and optimized both Pearson 1.0, Spearman 1.0, MAE 1.055556 on `n=2`.
  - `11913113` QAGS-XSUM consistency smoke: baseline and optimized both Pearson 0.0, Spearman 0.0, MAE 1.333333 on `n=2`.

Additional Topical-Chat diagnostic chain submitted after the dataset smoke chain:

- `11912917`: failed Topical-Chat engagingness smoke, PPL-only control, llama.cpp proposer
- `11913161`: replacement Topical-Chat engagingness smoke, PPL-only control, llama.cpp proposer, fixed ports
- `11912918`: Topical-Chat engagingness smoke, PPL + fixed NLA, llama.cpp proposer
- `11913262`: replacement Topical-Chat engagingness smoke, PPL + fixed NLA, llama.cpp proposer, parser/useful-row fix

These are intended to validate whether the NLA token-selection fix improves verbalization quality before launching another long NLA run.

Latest status:

- `11912917`: failed on `faretra` because llama.cpp tried to bind occupied port `127.0.0.1:8080`.
- `11913161`: replacement PPL-only control completed on `faretra`; artifacts were recovered to the local workspace from `gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke`.
- `11912918`: started on `faretra` at 2026-06-10 16:01 CEST and failed after 13m16s with exit code `1:0`.
- `11913262`: replacement PPL+fixed-NLA smoke completed on `faretra` with exit code `0:0`.
- Latest check: 2026-06-10 16:49 CEST.
- Reason: `11913161` and the downstream Qwen35B proposer jobs need 2 x RTX 3090 on the same node. `moro232` has only 1 x RTX 3090, so `faretra` remains the only eligible node.
- Failure root cause: the NLA precompute succeeded and wrote 210 rows with `token_status=ok`, but all rows were marked `parse_status=missing_tags` because the NLA verbalizer emitted text ending in `</explanation>` without the opening `<explanation>` tag. The runner's useful-row guard was too strict and rejected all 210 non-empty verbalizations as unusable.
- Fix: `parse_explanation` now treats closing-tag-only output as `partial_tags` and strips the tag; the NLA useful-row validator now accepts non-empty `missing_tags` rows with ok/unknown token status.
- Validation: local targeted tests and the full `gepa-experiments/tests/test_data_and_metrics.py` suite pass; the recovered `11912918` precomputed file now reports `covered_examples=36`, `rows=210`, `useful_rows=210`.
- Replacement job `11913262` used the same Topical-Chat engagingness PPL+fixed-NLA smoke setting with `ExcNodeList=deeplearn2`, 2 x RTX 3090, 64G memory, and 4h time limit.
- `11913262` artifacts are visible under `gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke`.
- Downstream experimental job `11912947` was rewired from the failed `11912918` to `afterok:11913262`, ran on `faretra`, and completed successfully at 2026-06-10 16:54:41 CEST after 16m08s.
- `11912947` final artifacts are visible under `gepa-experiments/results/experimental_nla_candidate_content_6_topical_chat_smoke`.
- Diagnostic report for `11912947` vs PPL-only control is saved at `gepa-experiments/results/diagnostics/nla_candidate_content_6_vs_ppl_smoke_20260610.md`.
- `11912948` completed successfully and artifacts were recovered under `gepa-experiments/results/experimental_nla_candidate_content_10_topical_chat_smoke`.
- Diagnostic report for `11912948` vs PPL-only control is saved at `gepa-experiments/results/diagnostics/nla_candidate_content_10_vs_ppl_smoke_20260611.md`.
- `11912948` result: optimized Pearson 0.402090, Spearman 0.371727, Kendall tau 0.310087, agreement 0.625000, MAE 0.750000 on the 12-example smoke slice.
- Against the matched PPL-only smoke, candidate-content-10 changed 1 example for the better, 2 for the worse, and 9 unchanged. It is less bad than candidate-content-6, but still negative on all paper-primary correlations and worse on MAE.
- Do not submit duplicate matched Qwen35B proposer smoke jobs; the matched smoke comparison is complete. Do submit the longer fixed-NLA run and other scientifically distinct queued work if it answers a planned question and can age in the queue.

`11913161` PPL-only smoke result:

- Dataset/dimension: Topical-Chat engagingness.
- Split: 24 GEPA train examples, 12 GEPA validation examples, 12 final-test examples.
- Feedback: perplexity only, no NLA, no auxiliary judge.
- Proposer: Qwen 35B via llama.cpp on `127.0.0.1:18144`.
- Baseline final-test metrics: Pearson 0.603136, Spearman 0.590879, Kendall tau 0.531588, MAE 0.555556, agreement 0.722222.
- Optimized final-test metrics: Pearson 0.536400, Spearman 0.527410, Kendall tau 0.459933, MAE 0.722222, agreement 0.638889.
- Interpretation: this small smoke control validates the fixed-port and artifact path, but the optimized prompt overfit or degraded final-test behavior on the 12-example test slice. It remains useful as the matched control for `11912918`, not as evidence that GEPA improves this setting.

Additional 1-GPU queue-aging jobs submitted on 2026-06-10:

- `11913130`: Topical-Chat engagingness PPL-only single-GPU smoke, pinned to `moro232`, 4 hour limit.
- `11913131`: Topical-Chat engagingness PPL + real-NLA single-GPU smoke, pinned to `moro232`, dependency `afterany:11913130`, 4 hour limit.
- These jobs completed on `moro232` and artifacts were recovered locally.
- Final audit: they are useful only as operational evidence that the single-GPU path can run. They are not a valid matched NLA comparison because split sizes differ, final-test examples do not overlap, proposer is Qwen2.5-7B rather than Qwen35B, and `11913131` contains dry-run fallback feedback caused by incomplete NLA precompute.
- Prevention action: the single-GPU NLA config was fixed and precomputed NLA no longer silently falls back to dry-run rows.

Failure and recovery action on 2026-06-10:

- `11912917` started on `faretra` at 11:32 CEST and failed after 3 seconds with exit code `125`.
- Cause: llama.cpp sidecar attempted to bind `127.0.0.1:8080`, which was already in use on `faretra`.
- Immediate action: held `11912918`, `11912947`, and `11912948` so the treatment and experimental NLA jobs cannot run without a valid PPL-only control.
- Config fix: changed `geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke.env` from judge/proposer ports `8000/8080` to `18143/18144` and added explicit `SLURM_TIME=04:00:00`.
- Relaunched PPL-only control as `11913161`, then set `11912918` to depend on it with `afterok`.
- Dependency policy changed for the remaining chain: use `afterok` instead of `afterany` so failed controls/treatments do not trigger downstream comparison jobs.
- Prevention fix: `gepa-experiments/slurm/run_docker.sh` now checks the llama.cpp proposer host port before launching the sidecar. If the configured port is busy and `PROPOSER_API_BASE` was not explicitly set, it automatically selects the next free port and passes that endpoint to the main GEPA container.
- Additional config hygiene: old 2-GPU llama.cpp config `geval_gepa_engaging_qwen25_2h_ppl_llamacpp35b_proposer.env` no longer uses risky default ports `8000/8080`; it now uses `18153/18154`.

Additional isolated experimental strategy jobs to queue after `11912918`:

- Topical-Chat engagingness smoke, PPL + experimental `candidate_content_6` NLA, llama.cpp proposer.
- Topical-Chat engagingness smoke, PPL + experimental `candidate_content_10` NLA, llama.cpp proposer.

These jobs must remain outside the main pipeline. They answer only whether alternate NLA token selection makes the feedback condition healthier and whether that translates to better GEPA behavior on a matched smoke setting.

Submission status:

- Local implementation and validation completed.
- Scripts/configs/status and token-strategy diagnostic report/CSV are synced to `faretra`.
- `11912947`: Topical-Chat engagingness smoke, PPL + experimental `candidate_content_6` NLA, llama.cpp proposer, completed successfully; final metrics are worse than fixed-NLA and PPL-only.
- `11912948`: Topical-Chat engagingness smoke, PPL + experimental `candidate_content_10` NLA, llama.cpp proposer, completed successfully; final metrics are also worse than fixed-NLA and PPL-only.
- `11913284`: Topical-Chat engagingness long run, PPL + fixed NLA, llama.cpp proposer, completed successfully on `faretra`; final artifacts recovered locally.
- `11913415`: Topical-Chat engagingness long current-code PPL-only control, llama.cpp proposer, failed before vLLM readiness because the allocated GPU was already heavily occupied by other users' processes.
- `11913482`: replacement Topical-Chat engagingness long current-code PPL-only control, same scientific config, failed at llama.cpp startup because the proposer GPU had only about 7.4 GiB free.
- `11913557`: second replacement attempt, same scientific config, failed at llama.cpp startup for the same reason.
- `11913587`: replacement Topical-Chat engagingness long current-code PPL-only control, same scientific config, completed on `faretra`; final artifacts recovered locally and diagnostic report generated.
- `11913424`: cancelled after `11913415` failed; the control path is now repaired via completed `11913587`.
- `11913731`: aux-judge smoke submitted as the replacement for `11913424`. Pending for two RTX 3090 GPUs on `faretra`; Telegram monitor started and watches `gepa-experiments/results/slurm/slurm-11913731-geval-gepa-topical_chat-engagingness-ppl-nla-auxjudge-llamacpp35b-smoke.out`.
- `11912947` and `11912948` are intentionally serial and outside the main pipeline.
- `11913284` is not part of that experimental serial chain. It is the current main-pipeline long fixed-NLA run and is independent of the candidate-only smoke jobs.
- Slurm priority check after `11913587`: no active user jobs remain visible in `squeue` on `faretra` or `moro232`; `sacct` is still unavailable because SlurmDB refuses connections.
- Single-GPU Topical-Chat smoke work `11913130` and `11913131` is complete and audited. It does not replace the Qwen35B proposer chain and should not be cited as NLA evidence.
- `11913388`: Topical-Chat engagingness single-GPU matched smoke, PPL + real NLA, pinned to `moro232`, submitted after `moro232` became free. It uses the same split and budget as PPL-only `11913130`: 4 train groups, 2 validation groups, 2 final-test groups, seed 42, `MAX_FULL_EVALS=2`, `NUM_THREADS=2`.
- Purpose of `11913388`: technical audit of the corrected precomputed-NLA path after removing the partial-precompute failure mode from `11913131`. It remains secondary evidence because the proposer is Qwen2.5-7B rather than Qwen35B.
- Pre-submit checks for `11913388`: `moro232` GPU was free, port `18213` was free, dataset cache existed, Docker image `geval_gepa:latest` existed, config had no `NLA_PRECOMPUTE_LIMIT`, `NLA_MIN_COVERAGE=0.95` was explicit, and the precomputed-NLA no-dry-run-fallback code was present on `moro232`.
- Startup status for `11913388`: preflight passed on `moro232`, split sizes were `gepa_train=24`, `gepa_validation=12`, `final_test=12`, and a 36-row NLA manifest was written before NLA checkpoint loading.
- Operational note for `11913388`: `moro232` now has `~/.telegram_credentials` copied from `faretra` with mode `600`. A node-local Telegram monitor is active on `moro232` with PID `4108546` and watches the node-local Slurm stdout file. The monitor originally started from `faretra` may still miss node-local log alerts, so use the `moro232` monitor as the authoritative one for this job.
- Completion status for `11913388`: completed on `moro232`, artifacts recovered locally, diagnostic report generated. Final interpretation is audit-positive but not thesis-level: NLA prevented the PPL-only optimized degradation, but the final optimized prompt stayed identical to the seed prompt.
- Submitted and completed after SSH to `moro232` became available:
  - `11913404`: SummEval consistency PPL-only single-GPU control.
  - `11913405`: QAGS-CNN consistency PPL-only single-GPU control.
  - `11913406`: QAGS-XSUM consistency PPL-only single-GPU control.
- These are matched technical controls for the already recovered real-NLA smoke jobs and should be interpreted as secondary 7B-proposer ablations.
- The configs and latest GEPA code path were synced to `moro232` before submission.
- A first submission attempt accidentally expanded the config loop variable locally and submitted three wrong default Topical-Chat jobs: `11913401`, `11913402`, `11913403`. They were cancelled immediately; `11913401` briefly entered `COMPLETING`, then disappeared, and stale Telegram monitors for `11913401-03` were killed.
- Correct PPL-only dataset controls were then submitted on `moro232` and completed:
  - `11913404`: SummEval consistency PPL-only single-GPU control, runtime 666.171 seconds, final artifacts complete.
  - `11913405`: QAGS-CNN consistency PPL-only single-GPU control, runtime 73.002 seconds, final artifacts complete.
  - `11913406`: QAGS-XSUM consistency PPL-only single-GPU control, runtime 52.894 seconds, final artifacts complete.
- Startup and final artifact checks show the intended configs were used: dataset/dimension matched the plan, perplexity feedback was enabled, NLA feedback was disabled, and metrics/runtime/prompt/trajectory artifacts were written.

## Cluster Scheduling Rule

Queue scientifically useful pending work even when the target node is currently occupied, so jobs accrue Slurm priority aging.

Use flexible node selection by default. Pin to `moro232` only when there is a concrete reason, such as:

- the job needs exactly one RTX 3090 and should not compete with the 2-GPU `faretra` Qwen35B proposer chain
- the job is recovering or checking node-local artifacts written on `moro232`
- the experiment is explicitly a 1-GPU smoke/control

Keep:

- `SLURM_EXCLUDE=deeplearn2`
- two GPUs for jobs using llama.cpp proposer sidecar
- serial dependencies for heavy jobs unless explicitly testing independent short smoke runs

Submit-script guard added:

- `gepa-experiments/slurm/submit_gepa_engaging.sh` and `gepa-experiments/slurm/submit_experimental_nla_strategy.sh` now print the scheduling mode before submission.
- `SLURM_NODE=auto` is treated as no pin, so flexible jobs can use any eligible node.
- If a job is pinned, the submit script checks the target node GPU capacity via `scontrol show node`.
- The submit fails immediately if the requested GPU count is larger than the pinned node capacity, preventing mistakes such as a 2-GPU Qwen35B proposer job pinned to single-GPU `moro232`.
- The submit also fails if `PROPOSER_BACKEND=llamacpp` is configured with fewer than 2 GPUs.
- This does not make a 2-GPU job runnable on `moro232`; it only prevents invalid submissions and makes the scheduler constraint explicit at submit time.

## Pre-Submit Checklist

Before submitting any new job, complete this checklist. Do not rely only on the submit script guards; they catch scheduling mistakes, not scientific or artifact mistakes.

Scientific purpose:

- Confirm the job answers a distinct question already in this plan, or update the plan before submitting it.
- Confirm the matched control exists, is queued, or is submitted in the same action.
- Confirm the job is not a duplicate of a queued, running, or completed artifact unless the previous job failed or is explicitly being rerun.
- Confirm whether the job is main-pipeline evidence, secondary smoke evidence, or isolated experimental NLA evidence.
- Confirm the final comparison metrics are paper-aligned for that dataset/dimension.
- Confirm GEPA optimization uses validation examples, not final test examples.
- Confirm final test is used only after optimization.

Config file:

- Confirm `CONFIG_FILE` exists locally and on the target cluster path.
- Confirm `DATASET`, `DIMENSION`, `LABEL`, `SEED`, `TRAIN_CONTEXTS`, `VAL_CONTEXTS`, and `TEST_CONTEXTS` are intentional.
- Confirm split sizes are large enough for the purpose: smoke, diagnostic, long run, or thesis-level comparison.
- Confirm `OUTPUT_DIR` is unique or intentionally reuses a directory with timestamped artifacts.
- Confirm `SLURM_TIME` is set; smoke jobs must not appear as `UNLIMITED`.
- Confirm `SLURM_MEM` is set and consistent with vLLM, NLA, and llama.cpp memory needs.
- Confirm `JOB_SLUG` is unique enough that Slurm stdout files are readable and not ambiguous.
- Run shell syntax validation on changed Slurm scripts before submission.

Scheduling:

- Inspect current queue with `squeue` and current node state with `scontrol show node`.
- Confirm requested GPU count is compatible with the selected backend:
  - llama.cpp proposer sidecar requires 2 GPUs on one node.
  - single-GPU smoke jobs must not pretend to be matched Qwen35B proposer comparisons.
- Confirm node pinning is intentional:
  - use flexible scheduling by default.
  - pin to `moro232` only for 1-GPU smoke/control jobs or node-local artifact work.
  - do not pin 2-GPU Qwen35B proposer jobs to `moro232`.
- Keep `SLURM_EXCLUDE=deeplearn2` unless there is an explicit reason to change it.
- Confirm dependencies use `afterok` for scientific chains where downstream jobs require a valid upstream result.
- Use `afterany` only for cleanup/recovery jobs or cases where downstream execution is valid after failure.
- Confirm held jobs are intentionally held; release only after dependencies and configs are corrected.
- Queue useful pending work even when nodes are occupied, but do not queue duplicate jobs that answer the same question.

Ports and networking:

- Confirm vLLM `SERVER_PORT` is not an old risky default unless intentionally isolated.
- Confirm llama.cpp `PROPOSER_PORT` is not `8080` unless there is a specific reason.
- Confirm `run_docker.sh` port auto-selection is synced on the cluster before submitting llama.cpp proposer jobs.
- If `PROPOSER_API_BASE` is explicitly set, confirm it matches `PROPOSER_PORT`; otherwise the auto-port fallback cannot safely rewrite it.
- Confirm the expected proposer endpoint is printed in the Slurm log at startup.

Model/cache/container readiness:

- Confirm judge model cache exists under `/llms/hub/...` on the execution node.
- Confirm NLA checkpoint cache exists when NLA is enabled.
- Confirm llama.cpp GGUF exists under `/llms/llamacpp-cache` for Qwen35B proposer jobs.
- Confirm the Docker image expected by `IMAGE_NAME` exists on the node or can be pulled/built intentionally.
- Confirm vLLM/Flash Attention compatibility has not changed for the active Docker image.
- Confirm no stale container with the same `CONTAINER_NAME` or `SIDECAR_NAME` can block startup; names include the Slurm job id by default.

Data and split integrity:

- Confirm data source exists on the execution node.
- Confirm preflight checks pass for dataset, split counts, context disjointness, response ids, model cache, and NLA cache.
- For Topical-Chat, confirm the run uses the intended USR data source and dimension.
- For SummEval/QAGS, confirm the run uses the intended paper-aligned dataset/dimension and not a legacy Topical-Chat label path.

NLA/perplexity feedback:

- If `NLA_FEEDBACK=1`, confirm whether this is real NLA or dry-run NLA.
- Dry-run NLA is allowed only for plumbing tests, not scientific comparison.
- If real NLA is enabled, confirm `NLA_BACKEND=precomputed`, `NLA_PRECOMPUTED_AUTO=1`, token budget, layer, dtypes, and checkpoint.
- Confirm `NLA_PRECOMPUTE_LIMIT` is intentional; do not accidentally cap scientific runs below train/validation coverage.
- Confirm `NLA_MIN_COVERAGE` is strict enough for the run purpose.
- Confirm perplexity feedback is computed on the base Qwen2.5-7B judge model, not on the 35B proposer.

Artifacts and monitoring:

- Confirm Slurm stdout path is under `gepa-experiments/results/slurm`.
- Confirm Telegram monitor credentials exist if monitor is enabled.
- Confirm monitor log and pid paths will be unique for the job id.
- If the job can run on `moro232`, plan artifact retrieval from `moro232` because results may be node-local.
- After submission, record the job id, dependency chain, config path, output dir, node pin/exclude, and expected comparison in this file.
- After completion, sync artifacts from the execution node to local workspace and to `faretra` if needed.
- After completion, verify required artifacts: metrics, run config, baseline and optimized predictions, seed and optimized prompts, prompt trajectory, GEPA viz, runtime manifest, Slurm log, vLLM log, llama.cpp log when applicable, and NLA verbalizations when applicable.

Post-submit sanity:

- Immediately inspect `scontrol show job` for the submitted job.
- Confirm `TimeLimit`, `ReqNodeList`, `ExcNodeList`, `TRES`, `Dependency`, `StdOut`, and `WorkDir`.
- If any field is wrong, fix it with `scontrol update` before the job starts, or cancel and resubmit.
- If the job starts and fails before model loading, inspect Slurm stdout first; common startup failures include occupied ports, missing image, missing model cache, wrong node filesystem, or bad dependency/config path.
- If a control job fails, hold or rewire downstream treatment jobs before they start.

## SSH / IPS Mitigation

The UniBo network can quarantine a client when it detects an IPS-like pattern. In this project the risky pattern is many short-lived SSH/SCP/rsync connections from Codex, especially parallel checks and aborted retries. The mitigation is to keep command throughput high while reducing new TCP/SSH handshakes.

Current local SSH configuration for `faretra` and `moro232` uses:

- `ControlMaster auto`
- `ControlPersist 30m`
- `ControlPath ~/.ssh/cm-%C`
- public-key-only batch authentication
- short connect timeout and one connection attempt

Operational rule:

- Prefer one persistent master connection to `faretra`.
- Use `faretra` as the main Slurm entrypoint.
- Direct SSH to `moro232` is appropriate for node-local artifact inspection and sync, because `moro232` result files may not be visible from `faretra`.
- Do not run parallel local SSH status checks.
- Use `gepa-experiments/slurm/cluster_status_snapshot.sh` for status: it collects queue, node state, job state, log tails, artifacts, and monitor logs in one remote command.
- When a job seems to disappear without artifacts, first check the execution node filesystem directly or sync from that node. Use `gepa-experiments/slurm/submit_slurm_stdout_smoke.sh` only if direct node checks do not explain the missing artifacts.

## 2026-06-17 Follow-Up Jobs Submitted

Status snapshot recorded at 2026-06-17 13:57 CEST.

Directly runnable follow-up jobs for the advisor request have been submitted. The local manifest is `gepa-experiments/status/submitted_followup_jobs_20260617.tsv`.

Submitted groups:

- A: NLA token-strategy wiring probe, job `11920199`. Pending because compatible multi-GPU nodes are currently unavailable.
- B: NLA token-position sweep jobs `11920242` to `11920253`. Pending on dependency after the A probe.
- C: soft-prompt/SIPIT controls and robustness jobs `11920236` to `11920241`. Job `11920236` is running on `moro232`; the others are pending for resources/priority.
- D: auxiliary-judge and matched no-aux controls `11920201`, `11920202`, `11920233`, `11920234`, `11920235`. The first smoke jobs are pending because compatible multi-GPU nodes are currently unavailable; downstream jobs are dependency-gated.

Runtime sanity already checked for `11920236`:

- Slurm state is `RUNNING` on `moro232`.
- Docker container `geval_gepa_11920236` is alive.
- Qwen2.5-7B loaded successfully.
- `gepa-experiments/results/soft_prompt_sipit_init_control/nearest_tokens.jsonl` has been produced with 16 rows.
- GPU utilization is active, so the SIPIT recovery is running rather than stuck at startup.

Update at 2026-06-17 16:10 CEST:

- `11920236`, `11920237`, `11920238`, `11920239`, and `11920240` completed successfully on `moro232`.
- `11920241` did not start while `moro232` was idle because it requested `48G` host memory; `moro232` has about `32G`, so Slurm considered only larger unavailable nodes.
- The memory request for the precision16 SIPIT job was reduced to `28G`, matching the other SIPIT/soft-prompt jobs.
- After becoming eligible, `11920241` started on `moro232` but failed with `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half`.
- Root cause: the precision16 model runs in fp16, while the continuous soft-prompt prefix was passed to SIPIT target-state computation in fp32.
- Fix: `sipit_soft_prompt_recover.py` now casts the continuous prefix to the model input-embedding dtype before calling SIPIT target-state computation.
- Replacement retry `11925716` was submitted and started on `moro232`; it passed the previous crash point and is running.

Monitoring caveat:

- Telegram monitors are active from `faretra` for all submitted job ids and have sent start/state-change messages.
- For jobs that execute on `moro232`, Slurm stdout and artifacts can be node-local. The monitor running from `faretra` can still track Slurm state, but log-alert inspection may require checking `moro232` directly.

## Acceptance Criteria

NLA can be considered thesis-ready only if at least one of these is true:

- It improves paper-aligned metrics under a fair 1-to-1 comparison.
- Or, if it does not improve, we have a reproducible diagnostic report explaining the bottleneck and next implementation step.

Before launching another long fixed-NLA run, the fixed-NLA smoke comparison must show that the feedback condition itself is healthier:

- NLA emitted artifacts preserve `token_status=ok`.
- Verbalizations are not dominated by first-token source/reference activations.
- Duplicate verbalization rows are substantially lower than the first long NLA run.
- `partial_tags` or missing tags are confirmed not to truncate semantic content; after the parser fix, closing-tag-only outputs are normalized and accepted only when they contain non-empty text.
- The PPL-only smoke and fixed-NLA smoke are compared with `diagnose_nla_run.py`.
- Current status: this gate passed on smoke job `11913262`, with the caveat that duplicate source/reference verbalizations remain high and must be monitored in the long run.

Before launching `ppl_nla_auxjudge`, the fixed-NLA run should have a valid diagnostic report. The auxiliary judge experiment should answer a separate question: whether Qwen 35B feedback helps the proposer use NLA/perplexity signals better, not whether the base judge model itself changes.

The full experiment matrix is considered scientifically usable only when every dataset/dimension target has:

- baseline/seed evaluation
- optimized evaluation
- paper-aligned metrics
- config artifact
- prompt artifacts
- runtime artifact
- prompt trajectory artifact

## Thesis Writing Readiness

Material already usable for thesis writing:

- Background/method section: GEPA pipeline, G-Eval judge setup, Qwen2.5-7B base judge, Qwen35B proposer, PPL feedback, NLA activation verbalization feedback, and optional auxiliary judge design.
- Reproducibility section: Slurm/Docker/vLLM/llama.cpp setup, artifact requirements, split semantics, pre-submit checklist, SSH/IPS mitigation, and node-local artifact recovery caveat.
- Topical-Chat engagingness results section:
  - old PPL long showed GEPA+PPL can improve over its seed/initial run;
  - old NLA long was negative and diagnosed as a weak/noisy NLA condition;
  - fixed-NLA smoke was positive;
  - fixed-NLA long was weak-positive against the current-code PPL control but did not produce a better prompt.
- Negative/diagnostic NLA section: candidate-only NLA, SummEval single-GPU smoke, QAGS plumbing smokes, duplicate/completion-like NLA verbalizations, and the conclusion that raw NLA text is not yet reliably metric-aligned.
- Code/provenance section: all key artifact paths and diagnostic scripts are present locally, including `nla_evidence_deep_dive_20260612.md` and `nla_fixed_long_vs_current_ppl_long_20260612.md`.

Still missing before writing the main experimental claim:

- Decide with the advisor whether the thesis claim should be framed as "NLA improves GEPA" or as "raw NLA is diagnostically informative but needs semantic compression/auxiliary judging to help GEPA reliably".
- Run or explicitly postpone the auxiliary-judge smoke with Qwen35B, because it is the planned next test for turning raw NLA into rubric-conditioned feedback.
- If claiming improvement, replicate the weak-positive fixed-NLA long result or obtain a stronger aux-judge/compressed-NLA result where GEPA selects a genuinely different and better prompt.
- For paper-aligned full-matrix tables, expand beyond the current smoke coverage to all planned dimensions or clearly label the current tables as pilot/smoke evidence.
- Add final paper-comparison context: exact G-Eval paper Table 2 numbers, dataset sizes/splits used in our experiments, and which comparisons are data-aligned but not model-identical.
- Decide whether to include QAGS-CNN and QAGS-XSUM smoke numbers; current `n=2` results are plumbing-only and should not be presented as scientific performance claims.
- Collect final prompt examples for the thesis appendix: seed prompt, old PPL optimized prompt, old NLA optimized prompt, fixed-NLA final prompt, and at least one failed candidate-only prompt.
- Fill a clean results table with confidence caveats: n, split, proposer, feedback variant, Pearson/Spearman/Kendall, agreement/MAE diagnostics, prompt changed yes/no, and artifact path.

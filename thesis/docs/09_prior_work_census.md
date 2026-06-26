# Prior Work Census

Status date: 2026-06-26

This document is the inventory that prevents the thesis from becoming a
GEPA-only writeup. The local repository and the cluster contain several earlier
research tracks that must be represented in the thesis narrative, even when
some of them end as negative or diagnostic evidence.

## High-Level Story

The thesis should be framed around semantic fidelity in latent-to-text methods.
The original advisor framing was broader than prompt optimization: the goal was
to study whether inversion or verbalization methods preserve the meaning of
inputs that are logically difficult, negated, counterfactual, or contrary to
commonsense, instead of reconstructing a more plausible but semantically
different text.

GEPA/G-Eval is therefore not the whole thesis by itself. The current proposed
component is Latent-GEPA: use NLA-style activation verbalizations and other
internal signals as feedback for GEPA while optimizing LLM-as-a-judge prompts.
The earlier inversion work remains relevant because it motivates why activation
verbalizations might expose semantic failures that ordinary surface metrics do
not capture.

## Research Questions To Preserve

The original inversion-oriented questions should stay visible:

1. Do state-of-the-art inversion methods reconstruct standard in-distribution
   text well?
2. Do standard reconstruction metrics capture logical semantic fidelity?
3. Is negation preserved or systematically weakened/removed?
4. Are counterfactual or commonsense-violating inputs reconstructed faithfully,
   or normalized toward a plausible commonsense version?
5. Does this bias differ across inversion families?

Latent-GEPA adds a second layer:

1. Can GEPA improve G-Eval-style judge prompts on paper-aligned datasets and
   metrics?
2. Does perplexity feedback from the base model help the proposer?
3. Can NLA verbalizations from the base model improve GEPA, or do raw
   verbalizations need transformation before they become useful feedback?
4. Does an auxiliary 35B judge help compress NLA into rubric-level proposer
   feedback?

## Local Inventory

| Path | Role | Current evidence | Thesis use |
|---|---|---|---|
| `README.md` | Top-level project framing | Already states the semantic-fidelity thesis direction and the active components. | Use as the bridge between old inversion work and current GEPA work. |
| `embedding-inversion-demo/` | Jina-style embedding inversion reproduction and diagnostics. | Contains training, inference, evaluation, provenance, failure-mode, and experiment decision docs. | Main evidence for the early embedding-inversion phase and for why reproduction/metric fidelity is non-trivial. |
| `thesis-datasets/` | Canonical dataset pipeline for semantic/logical stress tests. | Builds 2080 rows across standard text, negation, commonsense/counterfactual pairs; validation currently passes. | Central dataset contribution for inversion and NLA experiments. |
| `spit/SIPIT/` | SIPIT reproduction and extensions. | Contains Table 5 reproduction protocol, collision check, logical dataset export, and random-prefix diagnostics. | Main hidden-state inversion method and baseline for exact prompt recovery. |
| `natural_language_autoencoders/` | Local NLA codebase and checkpoint integration. | Contains AV/AR repo, Qwen2.5-7B layer-20 checkpoint references, inference code, training docs, patches. | Related work plus implementation basis for activation verbalization. |
| `nla-experiments/` | NLA SummEval extraction pipeline. | Extracts Qwen2.5-7B layer-20 residual activations and verbalizes them with Qwen NLA AV. | Independent validation of NLA plumbing before GEPA integration. |
| `nla-artifacts/` | Curated NLA artifact repo. | Stores SummEval manifests, sample activations, verbalizations, reports, and Slurm logs. | Artifact evidence for NLA extraction validation. |
| `gepa-experiments/` | Current GEPA/G-Eval pipeline. | Contains multi-dataset runner, PPL, NLA, aux judge, Qwen35B proposer, matrix plan, diagnostics, results. | Main current Latent-GEPA implementation. |
| `gepa-experiments/soft_prompting/` | Soft-prompt tuning and SIPIT readout on the G-EVAL-style judge task. | Contains random/text initialization support, nearest-token diagnostics, and SIPIT-style recovery scripts. | Creates learned continuous prompt targets to test what SIPIT can invert and whether the inversion has semantic value. |
| `prompt-waywardness/` | Related prompt inversion / continuous prompt discretization work. | Upstream code and paper PDF. | Related work and conceptual support for continuous-to-discrete prompt mismatch. |
| `towards_interpretable_softprompts/` | Related soft-prompt interpretability material. | Upstream notebook/code material. | Related work or appendix context if useful. |
| `tesi_t_simoneMazzacano/` | Previous student thesis reference. | Local-only reference folder, intentionally not modified. | Style/structure reference, not content source. |

## Cluster Inventory

Remote check at this census date:

- Remote workspace: `faretra:~/tesi`.
- Direct node-local artifact checks may still be needed on `moro232`, because
  single-GPU soft-prompt/SIPIT jobs write under `moro232:~/tesi` before being
  synchronized.
- Visible queued thesis jobs on `faretra` are follow-up jobs, not completed
  results:
  - `11920199`: NLA token-strategy wiring probe, pending multi-GPU resources.
  - `11920242` to `11920253`: dependent NLA token-position sweep jobs.
  - `11920201`, `11920202`, `11920233`, `11920234`, and `11920235`:
    aux-judge and matched no-aux follow-up jobs, dependency-gated or pending
    resources.
- GEPA status files exist remotely:
  - `gepa-experiments/status/current_plan_status.md`
  - `gepa-experiments/status/full_matrix_execution_plan_20260613.md`
  - `gepa-experiments/status/relatore_call_brief_20260611.md`
- GEPA result directories exist remotely for the historical Topical-Chat PPL
  runs, fixed/raw NLA runs, dataset smokes, candidate-only NLA smokes, and
  soft-prompt/SIPIT artifacts.
- SIPIT evidence is now reflected in the thesis result chapter through the
  recovered GPT-2 exact-recovery tables, logical20 recovery, and soft-prompt
  readout/control summaries. Older remote snapshots may still contain partial
  or duplicated copies and should not override the curated local artifacts.

Remote check on `moro232` succeeded for the soft-prompt/SIPIT result
directories. Lightweight artifacts have been synchronized locally for the
completed random-init soft-prompt runs, SIPIT readouts, and SIPIT controls.

## Embedding Inversion Work

Path: `embedding-inversion-demo/`

This study reproduces and audits an embedding inversion method based on
conditional masked diffusion. The important thesis content is not only whether
the paper was reproduced, but also which failure modes were found while trying
to reproduce it.

Relevant files:

- `embedding-inversion-demo/README.md`
- `embedding-inversion-demo/RESULTS.md`
- `embedding-inversion-demo/FAILURE_MODES.md`
- `embedding-inversion-demo/EXPERIMENTS.md`
- `embedding-inversion-demo/PROVENANCE_LEDGER.md`
- `embedding-inversion-demo/tools/probe_embedding_provenance.py`
- `embedding-inversion-demo/tools/probe_gradient_health.py`
- `embedding-inversion-demo/tools/probe_embedding_information.py`
- `embedding-inversion-demo/tools/probe_aux_bow_training.py`
- `embedding-inversion-demo/tools/prepare_encoder_subset.py`

Evidence already documented:

- A long Jina-v3 run before the paper-equation correction was not paper-level:
  job `11108134` reached about 0.127 full-mask token accuracy at 97.5K steps.
- A corrected v3 10K probe also remained far below paper-level behavior.
- A tiny full-mask overfit positive control did pass: job `11108213` reached
  1.000 train accuracy with the paper-equation v2 architecture and Eq. 4-style
  loss. This proves that the implementation can learn in a small controlled
  setting.
- Full-data probes such as `11108216` and `11108226` did not recover strong
  full-mask behavior; sequential decoding did not rescue the checkpoint.
- Embedding ablations showed weak functional dependence on the embedding in
  failed full-data settings.
- Gradient-health probes found severe gradient clipping, but not enough
  evidence to call it the unique cause.
- Provenance checks showed the cached Jina-v3 embeddings match the public repo
  no-task/no-adapter path exactly. Jina task-adapter mode remains a provenance
  grey area, not evidence of a local accidental data bug.
- Qwen3-Embedding probes learned much more easily than Jina-v3 on tiny
  full-mask overfit, suggesting encoder/tokenizer choice materially affects
  inversion difficulty.

Thesis interpretation:

- This evidence is diagnostic/negative with respect to exact Jina paper
  reproduction.
- It is still valuable because it motivates the thesis emphasis on faithful
  semantic reconstruction, careful metric choice, and artifact-level
  reproducibility rather than treating high-level claims as automatically
  reproducible.

Open gaps:

- Decide whether this appears as a full results subsection or as an initial
  failed-reproduction/diagnostic appendix.
- If it stays central, write a compact table of job ids, architecture/loss
  variants, and final diagnostic conclusions.

## Canonical Logical Dataset

Path: `thesis-datasets/`

This is the dataset bridge between the original inversion topic and later NLA
experiments.

Current canonical dataset:

- total rows: 2080
- total pairs/groups: 1060
- block A, standard controlled sentences: 40 rows
- block B, negation: 720 rows
- block C, commonsense/counterfactual: 1320 rows
- split counts: train 962, validation 290, test 828
- validation status: PASS

Sources:

- manual controlled standard sentences
- `jinaai/negation-dataset`
- `HiTZ/This-is-not-a-dataset`
- SemEval 2020 Task 4 ComVE
- controlled synthetic commonsense violations

Relevant files:

- `thesis-datasets/scripts/build_canonical.py`
- `thesis-datasets/scripts/validate_corpus.py`
- `thesis-datasets/scripts/export_for_sipit.py`
- `thesis-datasets/scripts/export_for_nla.py`
- `thesis-datasets/reports/build_report.md`
- `thesis-datasets/reports/validation_report.md`
- `thesis-datasets/reports/sipit_export_report.md`
- `thesis-datasets/reports/nla_export_report.md`

SIPIT export:

- CPU-side smoke export: 140 rows, up to 50 per block, 32-token placeholder
  ids.
- Real SIPIT experiments tokenize the same canonical rows with the target model
  tokenizer.

NLA export:

- NLA manifest: 140 rows.
- Target model: Qwen/Qwen2.5-7B-Instruct.
- Target layer: 20.
- Keeps canonical ids, pair ids, paired text, labels, and phenomenon metadata.

Thesis interpretation:

- This dataset is a concrete thesis contribution. It operationalizes the
  advisor's semantic-fidelity question into standard, negation, and
  counterfactual/commonsense blocks.
- The split and pair metadata must be explained in Chapter 4 because it controls
  which claims are legitimate.

Open gaps:

- Decide whether Block A should remain controlled manual text or be expanded
  with dataset-based standard text.
- Decide which metrics will score semantic flips in inversion/verbalization
  outputs, beyond ordinary lexical overlap.

## SIPIT Work

Path: `spit/SIPIT/`

SIPIT is the main hidden-state inversion method in the earlier thesis plan. It
targets exact prompt recovery from decoder-only hidden states, unlike NLA AV,
which verbalizes an activation rather than necessarily reconstructing the exact
input.

Relevant files:

- `spit/SIPIT/notes/REPRODUCTION.md`
- `spit/SIPIT/notes/REPRODUCTION_RUN_2026-05-20.md`
- `spit/SIPIT/notes/INTERIM_METRICS_2026-05-22.md`
- `spit/SIPIT/notes/METRICS_SNAPSHOT_2026-05-22_1110.md`
- `spit/SIPIT/notes/MISTRAL_CANCELLED_METRICS_2026-05-22_1127.md`
- `spit/SIPIT/scripts/reproduce/build_logical_dataset.py`
- `spit/SIPIT/scripts/reproduce/run_logical_sipit.sh`
- `spit/SIPIT/scripts/random_prefix/README.md`
- `spit/SIPIT/scripts/random_prefix/run_random_prefix_sipit.py`
- `spit/SIPIT/tests/test_random_prefix_diagnostics.py`

Paper reproduction targets:

- GPT-2 Table 5 exact-match reproduction.
- BruteForce and HardPrompts baselines.
- FP4 vocabulary scaling for Mistral/Llama where model access allows.
- Collision checks where hidden states must not collide under
  `torch.allclose` and min L2 distance should remain above `1e-6`.

Completed / documented evidence:

- GPT-2 collision check completed on 100 prompts:
  - first layer min L2 6.1603, 0 collisions
  - middle layer min L2 27.1380, 0 collisions
  - last layer min L2 110.2129, 0 collisions
- Interim Table 5 evidence on `faretra`:
  - SIPIT completed 47 observed prompts and found every token before vocabulary
    exhaustion on all 47.
  - Mean token steps about 2216, median 25, mean vocabulary explored about
    4.41%.
  - BruteForce scanned almost the full vocabulary on completed prompts and had
    exhausted-token failures.
  - HardPrompts and Mistral FP4 were cancelled before final exact-match reports.
- The paper did not publish exact Table 5 prompts, so the reproduction builder
  uses a deterministic public-source recipe with metadata.
- Runtime is diagnostic only because the paper used an A100-SXM 64GB, while
  the available cluster nodes use RTX-class GPUs.

Logical dataset preparation:

- The SIPIT logical run uses the canonical thesis dataset rather than an
  ad-hoc prompt list.
- Blocks selected: B and C.
- Phenomena: negation and commonsense/counterfactual violation.
- Model tokenizer: `openai-community/gpt2`.
- Clean 20-token dataset path:
  `spit/SIPIT/data/reproduce/logical20_gpt2_clean/`.
- Dataset size: 40 prompts, balanced as:
  - `B:negative`: 10
  - `B:positive`: 10
  - `C:commonsense_corrected`: 10
  - `C:counterfactual`: 10
- The clean variant uses only `input_text`; it does not concatenate
  `paired_text` into the prompt.

Random-prefix SIPIT extension:

- The extension tests whether SIPIT can recover a prompt when the input begins
  with random continuous embeddings.
- Standard SIPIT recovery is the easier and cleaner reference setting: the
  hidden states come from ordinary vocabulary tokens and the target is that same
  discrete prompt.
- `full-sequence` asks SIPIT to recover both random continuous prefix positions
  and real prompt tokens as vocabulary tokens.
- `known-prefix-control` gives SIPIT the exact continuous prefix as fixed
  context and recovers only the real prompt.
- This separation is essential: a failure in `full-sequence` may reflect
  discretizing an off-vocabulary continuous prefix, not a failure to recover the
  prompt once the real prefix context is known.
- The diagnostic records nearest-token ranks, L2/cosine/dot ranks, top-k
  prefix recovery indicators, and verification status.

Thesis interpretation:

- SIPIT should be presented as the strongest exact hidden-state inversion
  baseline.
- The logical dataset and random-prefix extension connect SIPIT to the thesis
  question about semantic stressors and continuous activation/prompt mismatch.
- The known-prefix vs full-sequence distinction must not be collapsed in the
  thesis; they answer different questions.

Open gaps:

- The curated thesis results now include GPT-2 exact-recovery evidence,
  logical20 recovery, and soft-prompt readout/control summaries. Older partial
  remote copies should be treated as provenance history, not as the source of
  the final tables.

## NLA Extraction Work

Paths: `natural_language_autoencoders/`, `nla-experiments/`, `nla-artifacts/`

NLA AV is not the same task as SIPIT. SIPIT attempts exact prompt recovery from
hidden states. NLA AV maps an activation vector to a natural-language
description. The thesis should make this distinction explicit.

Relevant files:

- `natural_language_autoencoders/README.md`
- `natural_language_autoencoders/nla_inference.py`
- `natural_language_autoencoders/docs/inference.md`
- `nla-experiments/README.md`
- `nla-experiments/summeval/prepare_summeval.py`
- `nla-experiments/summeval/extract_qwen_activations.py`
- `nla-experiments/summeval/verbalize_nla.py`
- `nla-experiments/summeval/summarize_verbalizations.py`
- `nla-artifacts/summeval/RUNS.md`

Checkpoint situation:

- The compatible base model for current thesis experiments is
  Qwen/Qwen2.5-7B-Instruct.
- The public AV checkpoint used locally is
  `kitft/nla-qwen2.5-7b-L20-av`.
- Layer target is 20 of 28, residual stream, hidden size 3584.
- Other public NLA checkpoint families exist locally in docs for Gemma-3-12B,
  Gemma-3-27B, and Llama-3.3-70B, but the current GEPA base model is Qwen 7B.

Standalone runs:

- 2026-05-27 GPU extraction smoke:
  - job `11911035`
  - host `faretra`
  - output: two activation rows for Qwen2.5-7B-Instruct, layer 20
  - token positions: `prompt_final` and `generated_score`
- 2026-05-27 sample12 extraction and first8 verbalization:
  - jobs `11911217`, `11911223`
  - 24 activation rows across four SummEval dimensions and two token positions
  - first 8 verbalizations readable with injection check ok
- 2026-05-28 full streamed verbalization:
  - job `11911233`
  - 24 verbalization rows
  - parse status `partial_tags` for all rows
  - injection check ok for all rows

Thesis interpretation:

- This evidence proves that Qwen2.5-7B activations can be extracted and passed
  through the Qwen NLA AV checkpoint before the more complex GEPA integration.
- It also motivates the later problem seen in GEPA: raw NLA verbalizations are
  not automatically rubric-aligned feedback.

Open gaps:

- Decide whether to include raw examples of NLA verbalizations in the main text
  or only in an appendix.
- If the thesis claims anything about semantic preservation of NLA on the
  logical dataset, a dedicated analysis must be run or clearly marked as future
  work.

## GEPA / G-Eval Work

Path: `gepa-experiments/`

This is the current main engineering component. It evaluates whether GEPA can
improve G-Eval-style prompts, and whether perplexity/NLA/auxiliary-judge
signals improve the proposer.

Relevant files:

- `gepa-experiments/README.md`
- `gepa-experiments/status/current_plan_status.md`
- `gepa-experiments/status/full_matrix_execution_plan_20260613.md`
- `gepa-experiments/status/relatore_call_brief_20260611.md`
- `gepa-experiments/status/relatore_results_index_20260614.md`
- `gepa-experiments/geval_gepa/runner.py`
- `gepa-experiments/geval_gepa/tasks.py`
- `gepa-experiments/geval_gepa/metrics.py`
- `gepa-experiments/geval_gepa/prompts.py`
- `gepa-experiments/geval_gepa/proposers.py`
- `gepa-experiments/geval_gepa/perplexity.py`
- `gepa-experiments/geval_gepa/nla_precompute.py`
- `gepa-experiments/geval_gepa/nla_feedback.py`
- `gepa-experiments/geval_gepa/aux_judge.py`
- `gepa-experiments/scripts/analyze_nla_evidence.py`
- `gepa-experiments/scripts/diagnose_nla_run.py`

Reference notebook/zip influence:

- `gepa-experiments/GEPA_tutorial (1).ipynb` contains the earlier custom
  proposer pattern from another student.
- The key lesson was that default GEPA proposals could overfit to specific
  training examples and produce prompts with literal names/entities/phrases.
- The local implementation generalizes that idea for G-Eval judges:
  - it summarizes inputs instead of passing raw examples;
  - it summarizes judge output instead of copying full generated text;
  - it strips human means, predicted scores, normalized agreement, metric
    names, example ids, generated outputs, PPL fields, and other optimizer
    artifacts from proposed prompts;
  - it rejects prompts that mention the wrong dimension or lose the required
    score contract.

Current model roles:

- Base judge / trained task model: Qwen/Qwen2.5-7B-Instruct.
- Perplexity model: the same Qwen2.5-7B-Instruct.
- NLA activation source: the same Qwen2.5-7B-Instruct, layer 20.
- NLA checkpoint: Qwen2.5-7B layer-20 AV.
- Proposer: Qwen35B through llama.cpp.
- Optional auxiliary judge: Qwen35B through llama.cpp, used only to produce
  feedback for the proposer. It does not replace the base judge or final
  metrics.

Main result status:

- First PPL long run was positive on Topical-Chat engagingness.
- First raw-NLA long run was negative and is treated as a diagnostic condition.
- Fixed-NLA smoke was positive on a 12-example final test.
- Fixed-NLA long was only weakly positive against the current-code PPL control:
  final metrics were slightly better, but both runs kept the seed prompt
  byte-identical.
- Candidate-only NLA reduced duplicate feedback but worsened metrics, so the
  duplicate-only hypothesis is not sufficient.
- The first auxiliary-judge smoke exposed a reasoning-only / empty-feedback
  issue and is not valid evidence of auxiliary-judge benefit.
- The next planned critical condition remains auxiliary-judge compression of NLA
  into rubric-level feedback before it reaches the proposer, but only after a
  valid non-empty-feedback smoke.

Full matrix plan:

- Paper-aligned datasets/dimensions:
  - SummEval: fluency, coherence, consistency, relevance.
  - Topical-Chat: naturalness, coherence, engagingness, groundedness.
  - QAGS-CNN: consistency.
  - QAGS-XSUM: consistency.
- Variants:
  - `base_gepa`
  - `ppl`
  - `ppl_nla`
  - `ppl_nla_auxjudge`
- Joint-prompt multi-dimension path is planned as an independent pipeline and
  must not be mixed with paper-aligned single-dimension results.

Thesis interpretation:

- GEPA is the component where NLA is used not merely for interpretation, but as a
  candidate feedback source for an optimizer.
- Current evidence does not yet prove that raw NLA improves the task. It does
  support a more nuanced thesis: raw activation verbalizations require
  selection, compression, or rubric grounding before they reliably help GEPA.

Open gaps:

- Valid auxiliary-judge smoke/long results are still pending.
- Multi-dimension joint-prompt runner is planned but not yet part of the
  existing reproducible single-dimension path.
- The full matrix is too expensive to complete blindly before the deadline; the
  thesis must distinguish thesis-critical long jobs from exploratory matrix
  jobs.

## Soft-Prompt / SIPIT Readout Work

Path: `gepa-experiments/soft_prompting/` and
`gepa-experiments/results/soft_prompt_*`

This diagnostic follows the advisor suggestion based on the soft-prompting
notebook: train only continuous virtual prompt tokens on the same
G-EVAL-style task, then inspect what SIPIT and nearest-token projection recover
from those vectors. The main purpose is not to introduce soft prompting as a
new performance method, but to test whether the inversion of trained soft
tokens exposes semantic or rubric-level information useful for the thesis.

Relevant files:

- `gepa-experiments/soft_prompting/train_soft_judge.py`
- `gepa-experiments/soft_prompting/sipit_soft_prompt_recover.py`
- `gepa-experiments/slurm/submit_soft_prompt.sh`
- `gepa-experiments/slurm/run_soft_prompt_sipit_job.sh`
- `gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048_random_init/metrics.json`
- `gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048_random_init_vtokens8/metrics.json`
- `gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048_random_init_vtokens32/metrics.json`
- `gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048_random_init_seed43/metrics.json`
- `gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048_random_init_seed44/metrics.json`
- `gepa-experiments/results/soft_prompt_*_sipit/sipit_recovery.json`
- `gepa-experiments/results/soft_prompt_sipit_*_control/sipit_recovery.json`

Completed task-learning sanity evidence:

- Main setting: Topical-Chat engagingness, Qwen/Qwen2.5-7B-Instruct frozen,
  240 train / 60 validation / 60 final-test rows.
- Main clean random-init run: 16 virtual tokens, max sequence length 2048,
  seed 42.
  - validation Pearson: 0.5371 -> 0.5684
  - validation Spearman: 0.5354 -> 0.5732
  - final-test Pearson: 0.7568 -> 0.8125
  - final-test Spearman: 0.7482 -> 0.8098
- Length sweep:
  - 8 tokens: positive but weaker on validation and final test.
  - 32 tokens: validation improves but final-test Pearson/Spearman worsen.
  - 16 tokens with 1024 context: positive but weaker than 2048 and tokenizes
    only 234/240 training rows.
- Seed sweep:
  - seed 43 worsens validation but improves final test.
  - seed 44 worsens both validation and final test.

Completed SIPIT/readout evidence:

- Random-init learned soft prompts are far from the discrete token embedding
  manifold: nearest mean L2 is about 60 and nearest mean cosine is about 0.07
  across the 8/16/32-token random-init readouts.
- SIPIT-style bounded recovery does not verify learned soft prompts exactly:
  `all_positions_verified=false` for all random-init soft-prompt readouts.
- Random hard-token control verifies exactly with `all_positions_verified=true`,
  showing that the bounded recovery path can succeed when the target is a
  sequence of actual vocabulary tokens.
- Initialization-prompt and random-continuous controls separate two different
  failure modes. The initialization-prompt target is still made of real token
  embeddings, so its nearest-token target is the seed instruction; its failed
  full verification diagnoses bounded recovery exhaustion after the first
  positions. The random-continuous target is genuinely off-manifold and tests
  the expected failure mode for arbitrary continuous vectors.
- Therefore the soft-prompt controls should not be described as ordinary SIPIT
  reproductions. They calibrate what the readout means when the target is a
  learned continuous prompt rather than a natural discrete prompt.

Thesis interpretation:

- The clean random-init result shows that the trained soft tokens are
  task-relevant enough to be interesting inversion targets, but robustness is
  mixed across seeds and token counts.
- The readout result is mainly negative/diagnostic for interpretability:
  SIPIT/nearest-token inversion of learned useful continuous tokens does not
  automatically produce faithful natural-language or semantically useful prompt
  content.
- This diagnostic supports the broader thesis theme that task usefulness,
  nearest-token readability, and exact latent-to-text recovery are distinct
  notions.

Open gaps:

- Decide whether to run additional seeds or a longer controlled sweep before
  making a stronger soft-prompt performance claim.
- Decide whether higher-budget SIPIT recovery is worth running after the
  precision16 check, or whether current controls already make the off-manifold
  conclusion strong enough.

## Related Repositories

`prompt-waywardness/` and `towards_interpretable_softprompts/` are not current
active thesis experiments, but they should be scanned for related-work value:

- `prompt-waywardness/` is directly relevant to the danger of interpreting or
  discretizing continuous prompts as if they were ordinary natural-language
  prompts.
- `towards_interpretable_softprompts/` is relevant to the broader literature on
  making soft/continuous prompts interpretable.

These should not be presented as completed thesis experiments unless actual
local results are produced. They can support Chapter 2.

## Required Thesis Placement

Chapter 1 should explain:

- text embeddings, hidden states, residual streams, logits, and soft prompts;
- embedding/activation inversion;
- why lexical overlap can miss logical semantic errors;
- SIPIT exact hidden-state inversion;
- NLA AV as activation verbalization;
- LLM-as-a-judge and Latent-GEPA as the later application component.

Chapter 2 should cover:

- embedding inversion papers including Jina-style conditional masked diffusion;
- SIPIT and its baselines;
- NLA;
- prompt waywardness / interpretable soft prompts;
- G-Eval and LLM-as-a-judge;
- GEPA and prompt optimization.

Chapter 3 should describe the method in this order:

- Semantic-Fidelity Corpus;
- semantic-fidelity readout diagnostics;
- G-EVAL-style judge task;
- Latent-GEPA optimization with PPL, NLA, and auxiliary feedback;
- soft-prompt tuning on the same judge task and SIPIT-style readout of learned
  virtual tokens.

Chapter 4 should include:

- exact dataset counts and splits;
- model roles and checkpoint compatibility;
- cluster/Docker/vLLM/llama.cpp/NLA environment details;
- metrics for both inversion fidelity and G-Eval agreement;
- artifact and versioning policy.

Chapter 5 should include:

- Jina/embedding inversion diagnostic results;
- SIPIT reproduction/interim evidence and collision checks;
- logical/random-prefix SIPIT findings if final outputs exist;
- NLA extraction smoke evidence;
- GEPA PPL, raw-NLA, fixed-NLA, candidate-only, and aux-judge results;
- soft-prompt random-init results and SIPIT readout/control diagnostics;
- runtime and failure-mode analysis.

## Ambiguities To Resolve With The Advisor

1. Should the thesis main title/framing remain broad
   "semantic fidelity of activation/embedding inversion", with GEPA as an NLA
   application, or should GEPA become the dominant thesis contribution?
2. Should the Jina embedding-inversion reproduction be a main experimental
   chapter, or a preliminary diagnostic appendix?
3. For SIPIT, are log-derived interim Table 5 metrics sufficient to discuss if
   final CSV/JSON outputs are unavailable, or must those remote artifacts be
   recovered/rerun?
4. Should the logical SIPIT dataset be evaluated with final exact-match metrics,
   or only used to motivate the NLA/semantic-fidelity dataset design?
5. Should NLA semantic-fidelity evaluation on the canonical logical dataset be
   run directly, separately from GEPA, before thesis writing?
6. How much of the GEPA full matrix is thesis-critical versus optional
   robustness evidence?

## Immediate Follow-Up Actions

1. Keep the curated SIPIT exact-recovery, logical20, and soft-prompt readout
   results aligned between Chapter 5 and the artifact index.
2. Keep the embedding-inversion diagnostic table synchronized with
   `embedding-inversion-demo/RESULTS.md`.
3. Keep the Semantic-Fidelity Corpus table synchronized with the validated
   dataset reports.
4. Maintain the method diagram that links:
   latent representation -> inversion/verbalization -> semantic-fidelity
   evaluation -> Latent-GEPA feedback, plus the soft-prompt readout diagnostic.
5. Keep GEPA result docs as detailed appendices, but make Chapter 5 read as one
   coherent thesis progression instead of a disconnected set of cluster runs.

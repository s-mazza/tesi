# Chapter 3 Method Plan

This document defines the intended structure of Chapter 3. It should explain
what was built and how the pieces connect, without turning into an environment
or result chapter.

Chapter 3 should be written as a coherent method, not as a list of scripts. The
reader should understand the full pipeline before seeing exact hyperparameters
and numerical results.

## Goal

Describe the thesis method for testing semantic fidelity in latent-to-text
methods and for defining `Latent-GEPA`: the GEPA-style prompt-optimization
pipeline that uses internal model signals, especially perplexity and NLA
verbalizations, as proposer feedback.

Target length: enough to make the implementation reproducible at the conceptual
level. Exact command lines, machine details, and long tables belong to Chapter
4 or appendices.

## Scope Boundary

Include:

- the Semantic-Fidelity Corpus design;
- a concise readout-diagnostic framing for embedding inversion, hidden-state
  recovery, activation verbalization, and soft-prompt readout;
- G-EVAL-style judge task definition;
- the Latent-GEPA optimization loop;
- perplexity feedback;
- NLA feedback generation for Latent-GEPA;
- auxiliary Qwen35B judge feedback;
- soft-prompt training on the same judge task as a way to create learned
  continuous targets for SIPIT-style readout;
- a short bridge to the reproducibility requirements that Chapter 4 defines in
  detail.

Exclude from Chapter 3:

- exact cluster setup;
- Docker image details;
- vLLM and llama.cpp deployment commands;
- Slurm scheduling;
- full hyperparameter tables;
- final metric values;
- detailed failure analysis of individual runs.

Those topics belong to Chapter 4 for setup and Chapter 5 for results. Chapter 3
should define the method independently of one specific run.

## Proposed Section Structure

### 3.1 Method Overview

Purpose: present the thesis as a coherent method centered on two objectives:
semantic-fidelity diagnostics and Latent-GEPA prompt optimization.

Diagnostic objective:

- define controlled semantic-fidelity inputs;
- apply latent-to-text readouts introduced in related work;
- inspect whether the readout preserves fragile meaning.

Latent-GEPA objective:

- test whether internal signals from the base model can improve G-EVAL-style
  prompt optimization.

- base judge predicts scores;
- GEPA optimizes the judge prompt;
- perplexity, NLA, and auxiliary-judge feedback are added as ablations.
- soft-prompt tuning trains only continuous virtual prompt tokens on the same
  judge task. The main purpose is then to pass those learned vectors through
  nearest-token and SIPIT-style readouts, checking whether the inversion
  contains semantic or rubric-level information useful for the thesis.

The section should explain that Latent-GEPA is the main proposed pipeline, while
diagnostic readouts provide the semantic and interpretability context.

### 3.2 Semantic-Fidelity Corpus

Purpose: define the dataset used to stress latent-to-text methods.

Each canonical row should include stable ids, source metadata, split,
phenomenon, label, input text, optional paired text, and pair id.

Dataset blocks:

- Block A: controlled standard sentences;
- Block B: negation pairs;
- Block C: commonsense/counterfactual pairs.

Expected discussion:

- why standard sentences are needed as controls;
- why negation and counterfactual examples are semantically hard;
- how paired rows make semantic flips visible;
- how stable ids allow SIPIT, NLA, and later analyses to be joined.

Value-adding example:

- Best format: a small table with one row per block.
- Structure: `block`, `phenomenon`, `input text`, `paired text`, `expected
  semantic issue`.
- Reason: this is the clearest way to show why lexical similarity can be high
  while logical meaning changes. Avoid a long dataset dump in the method
  chapter.

### 3.3 Semantic-Fidelity Readout Diagnostics

Purpose: connect the corpus to latent-to-text analyses without re-explaining
SIPIT, NLA, or embedding inversion as new thesis methods.

Expected discussion:

- embedding inversion reconstructs text from sentence embeddings;
- hidden-state recovery reconstructs discrete token sequences from decoder
  states;
- activation verbalization describes selected residual-stream activations;
- soft-prompt readout asks whether trained virtual tokens have readable
  projections.

Value-adding example:

- Best format: a compact diagnostic table.
- Structure: `diagnostic view`, `readout object`, `question answered`.
- Reason: this keeps Chapter 3 conceptual and moves setup/results details to
  later chapters.

### 3.4 G-EVAL-Style Judge Task

Purpose: define the task optimized by GEPA.

Each example contains:

- an input context;
- a candidate output;
- optional reference, fact, or source document depending on dataset;
- one or more human scores.

The base judge receives the example and a dimension-specific prompt, then
produces a rationale and a discrete score. A metric function compares this
prediction to the human target.

Expected discussion:

- the task is LLM-as-a-judge scoring, not free-form generation;
- the final target is agreement with human judgments;
- train/validation examples are used during GEPA optimization;
- final-test examples are used only after prompt selection.

Value-adding example:

- Best format: a minimal input/output schema block.
- Structure: one JSON-like example showing `context`, `candidate`, `dimension`,
  `human_score`, `judge_score`, and `parsed_score_status`.
- Reason: this is more useful than a prose-only description because later
  prediction artifacts are JSONL.

### 3.5 Latent-GEPA Optimization Loop

Purpose: explain the prompt optimizer used in the Latent-GEPA component and name the
proposed pipeline as Latent-GEPA when latent-derived feedback is part of the
proposer context.

Latent-GEPA workflow:

1. start from a seed judge prompt;
2. sample minibatches of validation examples;
3. evaluate candidate prompts and compute agreement with human scores;
4. collect metric feedback and optional PPL/NLA/auxiliary feedback;
5. ask the proposer model to mutate or merge selected prompts;
6. update the candidate pool and Pareto-style frontier;
7. evaluate the selected prompt on the held-out final-test split.

Model roles:

- base judge: Qwen2.5-7B-Instruct;
- proposer: Qwen35B through llama.cpp;
- final metrics: computed from base-judge predictions versus human scores.

The proposer does not replace the base judge. It only proposes prompt edits.

Value-adding example:

- Best format: pseudocode shown in the same `tcolorbox`-style figure
  environment used for prompts, following the supervisor's formatting request.
- Structure: `seed_prompt`, `trainset`, `valset`, feedback providers, proposer,
  prompt selection, final-test evaluation.
- Reason: GEPA is an algorithmic loop. Pseudocode will be clearer than a
  diagram alone and avoids binding the method to a specific Python file.

### 3.6 Feedback Variants

Purpose: define the experimental method variants as ablations.

Variants:

- `base_gepa`: metric feedback only;
- `ppl`: metric feedback plus response-only perplexity;
- `ppl_nla`: metric feedback plus perplexity plus NLA verbalizations;
- `ppl_nla_auxjudge`: metric feedback plus perplexity plus NLA, compressed or
  interpreted by the auxiliary Qwen35B judge.

Expected discussion:

- each variant adds one feedback source or transformation;
- comparisons must be made against matched controls;
- these are method variants, not final conclusions.

Value-adding example:

- Best format: an ablation table.
- Structure: rows are variants, columns are `metric feedback`, `perplexity`,
  `NLA`, `auxiliary judge`, `scientific question`.
- Reason: this table will be reused by Chapter 4 and makes the experimental
  matrix easier to read.

### 3.7 Perplexity Feedback

Purpose: explain how response-only perplexity is used.

Perplexity is computed on the candidate response under the base
Qwen2.5-7B judge model.

Expected discussion:

- lower perplexity means the model assigned higher probability to the response;
- the signal describes model surprise/confidence;
- perplexity is included in proposer feedback;
- perplexity is not a final evaluation metric and does not replace human-score
  agreement.

Value-adding example:

- Best format: a one-line formula plus a short numeric toy row.
- Structure: define average negative log-likelihood and `ppl = exp(avg_nll)`;
  then show `tokens`, `avg_nll`, `ppl`, and feedback phrase.
- Reason: this is enough to explain the quantity without turning the chapter
  into a probability tutorial.

### 3.8 NLA Feedback For Latent-GEPA

Purpose: define how activation verbalizations become proposer feedback.

In Latent-GEPA, NLA verbalizes activation vectors for selected prompt/example
tokens under the base Qwen2.5-7B judge model.

Token sources:

- source/context text;
- reference/fact text;
- candidate output text.

Expected discussion:

- selected token positions are converted to activation vectors;
- the AV turns those vectors into text;
- verbalizations are attached to GEPA feedback;
- token selection is part of the method because weak tokens can produce weak
  or repetitive feedback;
- current evidence shows candidate-only selection reduces duplicates but does
  not automatically improve GEPA.

This Latent-GEPA use of NLA is different from direct semantic-fidelity evaluation of
NLA. Here, verbalizations are weak proposer feedback, not final labels.

Value-adding example:

- Best format in the main method chapter: show the seed judge prompt as a
  plain-text block copied from the saved artifact inside a `tcolorbox` figure,
  then explain its method role in prose.
- The concrete PPL/NLA reflection example should be shown as a Python/JSON-like
  record with the same fields used by the code (`example`, `prediction`,
  `score`, `feedback`) plus the serialized `reflection_data` view sent to the
  proposer.
- Reason: in this part, the exact format matters. A table is too abstract and
  hides the fact that GEPA receives multiline textual feedback through Python
  objects rather than a clean conceptual schema.

### 3.9 Auxiliary Judge Feedback

Purpose: explain the Qwen35B auxiliary judge as a feedback compressor.

The auxiliary judge receives:

- the example;
- the human target;
- the base judge prediction;
- the agreement/error signal;
- the base judge rationale;
- optional NLA feedback.

It returns a short prompt-level lesson for the proposer. It does not return the
final score used in paper metrics.

Validity rule:

- auxiliary feedback must be non-empty;
- empty output is an error and invalidates the run;
- the auxiliary judge does not replace the base Qwen2.5-7B judge.

Value-adding example:

- Best format: a three-column table.
- Structure: `raw evidence`, `auxiliary judge lesson`, `prompt-level use`.
- Reason: this makes clear that the auxiliary judge compresses feedback rather
  than becoming the evaluated judge.

### 3.10 Soft-Prompt Training And SIPIT Readout

Purpose: explain the advisor-requested soft-prompt diagnostic as an interpretability
probe for learned continuous prompts, not as a replacement for GEPA prompt
optimization.

Suggested subsections:

- `Purpose`: contrast GEPA-readable prompt search with frozen-model
  soft-prompt tuning, and separate task utility, discrete proximity, and
  semantic readout.
- `Readout Pipeline`: describe random initialization, nearest-token projection,
  cosine/L2 diagnostics, and SIPIT-style bounded recovery.
- `Controls`: explain why hard-token, init-prompt, text-init, and random
  continuous controls are required.
- `Interpretation Rule`: state that a useful continuous prompt is not
  automatically interpretable.

In this diagnostic the base judge model remains Qwen2.5-7B-Instruct, but its
weights are frozen. The only trained parameters are PEFT prompt-tuning virtual
tokens placed before the judge input. The task is the same Topical-Chat
engagingness scoring task used in Latent-GEPA, so any learned soft prompt
is tied to the same rubric-like behavior that GEPA tries to improve.

The main question is what SIPIT-style inversion recovers from those trained
soft tokens, and whether the recovered text or nearest-token projection contains
semantic information that is useful for the thesis. Task-performance metrics are
therefore a sanity check that the soft tokens learned something nontrivial; the
interpretability readout is the main object of analysis.

Expected discussion:

- GEPA searches over readable prompt text; soft-prompt tuning optimizes
  continuous vectors that become targets for inversion/readout.
- The main experiments use random initialization. Text initialization is kept
  only as an explicit control because it biases nearest-token interpretation
  toward the seed instruction.
- After training, nearest-token projection maps each virtual token vector to
  the closest vocabulary embedding by L2 distance. Cosine similarity is saved
  as a qualitative geometric diagnostic.
- SIPIT-style bounded recovery is then run on the learned continuous vectors to
  test whether they can be represented as discrete text tokens.
- Control conditions are required: random hard vocabulary tokens, the initial
  prompt-token embeddings, and random continuous vectors.

The control modes used in this diagnostic should be explained explicitly because
they are easy to confuse with standard SIPIT:

- `soft_prompt` loads the trained PEFT virtual-token embeddings and asks what
  discrete tokens, if any, can reproduce their hidden-state trajectory.
- `random_hard_tokens` samples real vocabulary ids, uses their exact embedding
  vectors as the target, and acts as the positive control. If this fails, the
  readout pipeline is broken even for discrete token targets.
- `init_prompt` tokenizes the seed instruction text and uses the exact
  vocabulary embeddings of those tokens. This is not a random/off-manifold
  target; it checks how a text-initialized control behaves under the same
  bounded recovery budget. Nearest-token projection should recover the seed
  tokens even if iterative SIPIT verification stops after a prefix.
- `random_continuous` samples Gaussian vectors and norm-matches them to the
  learned soft prompt when possible. It is the negative control for continuous
  off-manifold targets.

Compared with normal SIPIT, the soft-prompt readout is therefore harder and has
a different interpretation: the target vectors may not correspond to any
vocabulary tokens at all. A failed verification can mean that the learned
continuous prompt is off the discrete token manifold, not that ordinary SIPIT
cannot recover natural prompts.

Important interpretation rule:

- A useful soft prompt is not automatically interpretable. If SIPIT verification
  fails and nearest-token distances are large, the recovered text should be
  treated as a diagnostic projection, not as the natural-language prompt that
  the model learned.

Value-adding example:

- Best format: a compact comparison table.
- Structure: `condition`, `trained object`, `initialization`, `readout method`,
  `valid interpretation`.
- Reason: this makes clear why a GEPA prompt, a text-initialized soft prompt,
  a random-initialized soft prompt, and a hard-token control answer different
  questions.

### Reproducibility Bridge To Chapter 4

Chapter 3 should not contain a standalone artifact-policy section. It should
close with a short transition stating that Chapter 4 defines the concrete
artifact requirements: prompts, prompt trajectories, per-example predictions,
split manifests, run configurations, runtime logs, NLA metadata, auxiliary
judge outputs, and SIPIT verification artifacts.

Reason: artifact preservation is essential, but it is an experimental-setup
constraint rather than a method component. Keeping it in Chapter 4 avoids
duplicating implementation details in the method chapter.

## Figures To Add

### Figure 3.1: End-To-End Thesis Method

Required original figure. Current implementation: a TikZ diagram inside the
LaTeX source, so the figure is original and travels with the single Overleaf
project. A draw.io-exported PDF would also be acceptable if the diagram becomes
too dense. The current intended structure is deliberately row-based to avoid
crossing arrows.

Structure:

1. semantic readout diagnostics:
   - canonical semantic-fidelity corpus;
   - embedding inversion, SIPIT-style recovery, NLA verbalization;
   - semantic-fidelity analysis.
2. prompt-optimization component:
   - G-EVAL-style judge examples;
   - Qwen2.5-7B base judge;
   - metric/PPL/NLA/auxiliary feedback;
   - GEPA selection and final-test metrics.
3. soft-prompt interpretability track:
   - same judge task;
   - frozen judge with trained virtual prompt tokens;
   - nearest-token and SIPIT-style readout;
   - continuous-prompt interpretability diagnostic.

Reason: this is the main bridge between the older inversion work and the newer
GEPA/NLA feedback work.

Figure-selection audit:

- Include Figure 3.1 as the main original method figure because Chapter 3 must
  contain at least one original figure.
- Include Figure 3.2 as a companion GEPA-flow figure because Figure 3.1 is
  intentionally high-level and should not carry the full feedback/proposer loop.
- Do not include a separate NLA pipeline figure in Chapter 3. NLA is explained
  as prior work in Chapter 2, configured in Chapter 4, and validated in
  Chapter 5.
- Do not add paper figures in Chapter 3 unless they explain a method detail
  that the original diagrams cannot cover. Paper figures belong primarily in
  related work, where the reader needs to understand prior work.
- Use artifact-derived examples for proposer feedback and prompts because those
  objects are part of the Latent-GEPA method. Detailed SIPIT/NLA examples belong
  in the results chapter unless they are needed to prevent a method ambiguity.

External-figure candidate audit:

- Embedding-inversion diffusion architecture: included in Chapter 2 as the
  related-work figure for the Jina-style conditional masked diffusion line, not
  repeated in Chapter 3 because the method chapter only frames readout
  diagnostics rather than re-explaining the prior architecture.
- SIPIT paper figure or algorithm schematic: not included as an external figure
  in Chapter 3. SIPIT is positioned in related work and appears in Chapter 3 only
  as a SIPIT-style readout tool for soft prompts.
- Prompt-waywardness / soft-prompt interpretability figure: included in Chapter
  2 to motivate why nearest-token interpretations can be misleading. Chapter 3
  uses our own table of soft-prompt/SIPIT readout conditions.
- NLA autoencoder figure from the original NLA work: included in Chapter 2 to
  explain the prior method. Chapter 3 does not include a separate NLA figure;
  it explains only how NLA verbalizations enter Latent-GEPA feedback.
- G-EVAL framework figure: included in Chapter 2 as related work. Chapter 3
  uses a JSON-like schema and the seed prompt because those are closer to the
  concrete judge task implemented in the thesis.
- GEPA overview figure: included in Chapter 2 as related work. Chapter 3 uses
  an original GEPA feedback-flow figure because our method has thesis-specific
  PPL, NLA, and auxiliary-judge feedback paths that are not visible in the
  generic GEPA figure.

Decision: Chapter 3 should contain original method diagrams and artifact-derived
examples. External paper figures are used in Chapter 2 unless a figure explains
something that cannot be expressed cleanly with the thesis-specific diagrams.

### Figure 3.2: GEPA Feedback Flow

Required companion figure after simplifying Figure 3.1. It keeps the global
overview readable while preserving the detailed GEPA feedback path.

Best format: small pipeline diagram or algorithm box, also exported as PDF if
drawn.

Structure:

`example -> base judge -> metric feedback -> optional PPL/NLA -> optional
auxiliary judge -> proposer -> revised prompt -> validation selection`.

Reason: this figure can clarify that the auxiliary judge affects feedback to
the proposer but does not define final metrics.

## Examples To Include

Use examples sparingly. The method chapter should include only examples that
clarify a transformation or prevent a likely misunderstanding.

Recommended examples:

- dataset example table in Section 3.2;
- semantic-readout diagnostic table in Section 3.3;
- JSON-like G-EVAL example schema in Section 3.4;
- Latent-GEPA pseudocode in Section 3.5, formatted as a `tcolorbox` figure;
- feedback-variant ablation table in Section 3.6;
- short PPL formula in Section 3.7;
- Python/JSON-like concrete PPL/NLA proposer-feedback block in the NLA/GEPA
  feedback section, with both important keys and diagnostic values highlighted;
- soft-prompt/SIPIT-style readout comparison table in Section 3.10.

Avoid large log dumps in Chapter 3. Full prompt text is acceptable there only
for the seed prompt, because the prompt itself is the method object. Prompt
changes that are part of the experimental outcome belong in Chapter 5.

## Transition To Chapter 4

Close Chapter 3 by stating that the next chapter fixes the experimental
instantiation of this method: datasets and dimensions, model checkpoints,
cluster environment, hyperparameters, metrics, baselines, and comparison
protocol.

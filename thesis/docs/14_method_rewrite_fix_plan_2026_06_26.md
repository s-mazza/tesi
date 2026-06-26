# Method Rewrite Fix Plan - 2026-06-26

Purpose: track the supervisor comments in `thesis/fixes.txt` and turn them
into a concrete implementation plan. This file is a plan only: do not edit the
LaTeX content until the open questions are answered.

## Source Comments Covered

Primary source:

- `thesis/fixes.txt`

Relevant existing planning files:

- `thesis/docs/00_writing_guidelines.md`
- `thesis/docs/02_thesis_scope.md`
- `thesis/docs/05_method_spec.md`
- `thesis/docs/06_experimental_setup_spec.md`
- `thesis/docs/07_results_inventory.md`
- `thesis/docs/13_reviewer_fix_plan_2026_06_22.md`

## Main Diagnosis

The central issue is not a single missing paragraph. Chapter 3 currently reads
as a list of parallel branches and implementation histories. The supervisor asks
for a clearer method narrative:

- avoid defensive wording such as "this thesis does not do X";
- remove "branch" as a framing device in the final prose;
- avoid inline Python files, local paths, and function/script references;
- keep SIPIT and NLA as related-work methods, not detailed method sections;
- describe GEPA and Latent-GEPA more strongly in Chapter 3;
- explain mini-batch proposal, validation, and Pareto-style selection in GEPA;
- expand the semantic-fidelity dataset construction with examples and stats;
- add the soft-prompt plus SIPIT-readout work if not sufficiently visible;
- remove or merge thesis-internal sections such as "Chapter Summary" and
  "Why GEPA fits this Thesis".

## Proposed Chapter 3 Target Structure

This is the proposed replacement narrative for Chapter 3.

### 3.1 Method Overview

Goal: introduce the method positively and formally.

Content:

- state the two task surfaces:
  - semantic-fidelity analysis of latent-to-text outputs;
  - Latent-GEPA for G-EVAL-style prompt optimization;
- avoid saying what the chapter "does not" report;
- explain that inversion/verbalization outputs are analyzed as evidence, while
  Latent-GEPA uses selected evidence as proposer feedback;
- avoid the word "branch" in final prose.

Action on current draft:

- rewrite lines around `main.tex:910-923`;
- simplify Figure 3.1 caption and surrounding prose to avoid "branches";
- remove "SIPIT, NLA" from the high-level method description except as short
  references to tools already introduced in related work.

### 3.2 Semantic-Fidelity Corpus

Goal: satisfy comments around `main.tex:1194-1195`.

New title proposal:

- `Semantic-Fidelity Corpus`
- alternative if a more explicit name is wanted:
  `Logical Semantic-Fidelity Corpus`

Content to add:

- construction procedure:
  - controlled standard examples;
  - negation/polarity examples;
  - commonsense/counterfactual examples;
  - stable IDs, paired rows, phenomenon labels, and split metadata;
- concrete examples, not only schema;
- stats:
  - total rows;
  - rows per block;
  - split sizes;
  - average/median sentence length if recomputed from current data;
  - optional token-length stats if useful;
- explain how the corpus supports semantic-fidelity checks without describing
  each downstream method in this section.

Data to double-check before editing:

- current docs report 2080 total rows:
  - Block A: 40;
  - Block B: 720;
  - Block C: 1320;
  - train: 962;
  - validation: 290;
  - test: 828.

Before implementation, recompute these from the canonical dataset artifacts
rather than trusting the markdown.

### 3.3 G-EVAL-Style Judge Task

Goal: introduce the actual task before GEPA.

Content:

- define input fields, candidate response, optional reference/fact, evaluation
  dimension, human target, judge output, and parsed score;
- keep this concise and move detailed split/config choices to Chapter 4;
- do not place this after all method variants if it is needed to understand
  Latent-GEPA.

Action on current draft:

- move or rewrite the current `G-EVAL-Style Judge Task` section so it appears
  before GEPA;
- avoid presenting it as a separate "branch".

### 3.4 GEPA Prompt Optimization

Goal: make GEPA the method section's main algorithmic core.

Content to add or strengthen:

- seed prompt;
- train/validation split usage;
- mini-batch proposal:
  - GEPA evaluates prompt candidates on small subsets before full validation;
  - this limits expensive full validation calls;
  - weak candidates can be rejected early;
- textual reflection/proposal:
  - the proposer receives failure summaries and revises the prompt;
- Pareto-style selection:
  - multiple candidates can be useful for different validation examples;
  - GEPA maintains a candidate/frontier view rather than only greedy global
    replacement;
- final-test evaluation:
  - only after prompt selection.

Action on current draft:

- keep GEPA detailed description in Chapter 3;
- make the GEPA pseudocode and prompt boxes comply with the existing
  `tcolorbox` convention;
- consider moving the generic GEPA figure from related work to a shorter
  related-work mention, while keeping the thesis-specific GEPA feedback figure
  in Chapter 3.

### 3.5 Latent-GEPA Feedback Signals

Goal: describe what this thesis adds to GEPA without re-explaining SIPIT or NLA
as full methods.

Content:

- define Latent-GEPA as GEPA plus latent-derived proposer feedback;
- metric feedback as baseline;
- response-only perplexity;
- NLA verbalizations as activation-derived text from the base judge;
- auxiliary judge as optional feedback compressor;
- explain that PPL/NLA/auxiliary feedback affects the proposer context, not the
  final evaluation metric.

Action on current draft:

- merge current `Feedback Variants`, `Perplexity Feedback`, `NLA Feedback for
  GEPA`, and `Auxiliary-Judge Feedback` into a tighter Latent-GEPA method block;
- keep one concrete proposer-feedback example if it helps, but ensure every
  displayed object has a caption and a reader guide;
- remove code/file references from this prose.

### 3.6 Soft-Prompt Readout Diagnostic

Goal: include the advisor-requested soft-prompt plus SIPIT analysis without
turning SIPIT itself into a Chapter 3 method.

Content:

- the base judge is frozen;
- only virtual prompt embeddings are trained;
- task metrics are sanity checks;
- nearest-token and SIPIT-style readouts test whether learned soft tokens have
  interpretable discrete projections;
- distinguish this from standard SIPIT recovery.

Action on current draft:

- retain a concise method section for the soft-prompt diagnostic if approved;
- phrase SIPIT as a readout tool introduced in related work;
- move detailed SIPIT algorithm description and random-prefix reproduction
  material out of Chapter 3 unless it is needed to understand the diagnostic.

### 3.7 Reproducibility Bridge

Goal: close Chapter 3 briefly.

Content:

- one paragraph pointing to Chapter 4 for datasets, splits, models,
  hyperparameters, runtime environment, artifact requirements, and exact run
  settings.

Action on current draft:

- remove standalone `Chapter Summary` sections unless the reference thesis
  strongly requires them;
- avoid long artifact-policy prose in Chapter 3.

## Content To Move Out Of Chapter 3

### SIPIT

Supervisor instruction: SIPIT should be only in related work, because the thesis
does not modify SIPIT itself.

Planned action:

- keep SIPIT explanation and paper positioning in Chapter 2;
- keep SIPIT reproduction/setup details in Chapter 4;
- keep SIPIT results, logical20, and soft-prompt readout outcomes in Chapter 5;
- in Chapter 3, mention SIPIT only as an already-defined readout/recovery tool
  when describing the soft-prompt diagnostic.

### NLA

Supervisor instruction: NLA is a tool. It can be explained in related work and
briefly recalled in method when discussing extra-info feedback for GEPA.

Planned action:

- keep NLA technical explanation in Chapter 2;
- keep model/layer/checkpoint details in Chapter 4;
- keep standalone NLA validation in Chapter 5;
- in Chapter 3, only describe how NLA output enters Latent-GEPA feedback.

### Embedding-Inversion Diagnostic Details

Supervisor did not explicitly forbid embedding inversion in Chapter 3, but the
current text has too much reproduction-boundary and implementation detail.

Planned action:

- keep the semantic-fidelity motivation and corpus construction in Chapter 3;
- move reproduction boundary, Jina cache provenance, task-adapter details, and
  failure diagnostics to Chapter 4/5;
- retain at most one concise example if it clarifies what latent-to-text
  diagnostic means.

## Line-Level Comment Mapping

| Comment location | Issue | Planned action | Draft reply |
|---|---|---|---|
| Teams message | Do not delete Overleaf comments | Do not edit comments directly; provide replies for each comment and use small commits for local changes | "ok, da ora non cancello i commenti, rispondo sotto con risolto/fatto" |
| General method | Narrative too fragmented/defensive | Rewrite Chapter 3 around Semantic-Fidelity Corpus -> Judge Task -> GEPA -> Latent-GEPA feedback -> Soft-prompt diagnostic | "risolto, ho riscritto il metodo con una narrativa unica e meno difensiva" |
| General method | Avoid Python files/functions | Remove inline code/script/file references from thesis prose; keep only displayed code-like artifacts when necessary | "fatto, ho tolto i riferimenti a file/script dal testo" |
| General method | Remove unclear sections | Remove/merge `Chapter Summary`, `Why GEPA fits this Thesis`, and unclear defensive sections | "fatto" |
| Conclusions missing | Chapter 6 not in `main.tex` | Add Chapter 6 later, after method rewrite unless user asks to do it now | "non ancora fatto in questa passata, lo aggiungo dopo il metodo" |
| lines 770 | "gippittierose" section | Remove `Why GEPA Fits This Thesis` or merge one useful sentence into related-work summary | "fatto" |
| lines 910-914 | Intro says what chapter does not do | Rewrite positively: what the chapter defines and proposes | "fatto" |
| lines 921-923 | Defensive / GEPA branch wording | Replace with formal Latent-GEPA objective | "fatto" |
| line 1194 | "Canonical" title unclear | Rename section to `Semantic-Fidelity Corpus` or approved variant | "fatto" |
| line 1195 | Dataset section too thin | Add construction process, examples, stats, and length statistics | "fatto, ho ampliato la descrizione del dataset e aggiunto esempi/statistiche" |
| lines 1256-1260 | Downstream method usage described too early | Remove from dataset section; keep method-specific usage later | "fatto" |
| line 1262 | Avoid "branch" | Rename embedding section or move it out of method | "fatto" |
| line 1361 | Python file reference | Rewrite caption without local file/script names | "fatto" |
| lines 1372-1376 | Unclear reproduction boundary section | Move to setup/results or remove from method | "fatto" |
| lines 1380-1388 | Looks like results/setup | Move Jina cache/provenance details to Chapter 4/5 and rewrite without code paths | "fatto" |
| lines 1398-1400 | Defensive wording | Rewrite or move diagnostic qualification to results | "fatto" |
| line 1428 | SIPIT more related work; use LatentGEPA | Remove standalone SIPIT method section; keep concise references only where needed | "fatto, SIPIT ora è trattato nei related/setup/results e nel metodo resta solo come strumento di readout dove serve" |
| line 1693 | Judge task placement | Move judge task before GEPA and/or setup if only experimental details | "fatto" |
| line 1990 | Captions for everything | Ensure listings/figures/tables have captions and labels | "fatto" |
| lines 3136-3162 | No code/script refs | Rewrite results provenance prose without filenames and low-level implementation references | "fatto" |

## Open Questions Before LaTeX Changes

1. Overleaf comment preservation:
   Should I implement changes locally in `main.tex` and give you small commits
   to copy section-by-section into Overleaf, or should I prepare patch snippets
   plus comment replies without changing `main.tex` until you have handled the
   comments in Overleaf?

   Decision: implement changes locally in `main.tex` and create small commits
   that can be copied into Overleaf section by section.

2. SIPIT in Chapter 3:
   The supervisor says "SIPIT deve andare SOLO in related", but also says to add
   the soft-prompting with SIPIT analysis if missing. Proposed resolution:
   Chapter 3 does not explain SIPIT as a method; it only describes the
   soft-prompt diagnostic and says that the readout uses the SIPIT-style tool
   defined in related work. Is this acceptable?

   Decision: yes. Remove standalone SIPIT method explanation from Chapter 3.
   Keep only the minimum SIPIT-style readout reference needed to describe the
   soft-prompt diagnostic.

3. NLA in Chapter 3:
   Proposed resolution: remove standalone NLA method section from Chapter 3 and
   keep only the Latent-GEPA feedback subsection explaining how NLA
   verbalizations are attached to proposer feedback. Detailed NLA pipeline stays
   in related work/setup/results. Is this the intended interpretation?

   Decision: yes. Remove standalone NLA method explanation from Chapter 3.
   Keep NLA only as one source of feedback in Latent-GEPA.

4. GEPA in related work:
   The supervisor says GEPA can be cited in related work but the real
   description should be in method. Should I remove the related-work GEPA figure
   and keep only a short paragraph under hard prompt optimization, or keep the
   figure in related work and make Chapter 3 more detailed?

   Decision: second option. Keep GEPA in related work as a concise prior method,
   but make Chapter 3 the place where the algorithm and the thesis modification
   are explained properly.

5. Embedding-inversion method content:
   Should Chapter 3 keep a short embedding-inversion diagnostic subsection, or
   should all reproduction-boundary material move to Chapter 4/5 and Chapter 3
   focus only on the semantic-fidelity corpus plus Latent-GEPA?

   Decision: follow the advisor comments as much as possible. Chapter 3 should
   not contain long reproduction-boundary or implementation-detail material. Keep
   only a concise semantic-fidelity diagnostic framing if it supports the method;
   move detailed reproduction/provenance/failure analysis to Chapter 4 or 5.

6. Chapter 6:
   The supervisor notes that conclusions are missing. Should I add a Chapter 6
   skeleton in this same fix batch, or keep this batch focused on Chapter 3 as
   requested?

   Decision: yes, include Chapter 6, but only after the method rewrite so the
   conclusion can use the complete updated thesis narrative. Use the previous
   thesis as structural reference.

7. Dataset statistics:
   The current docs report 2080 canonical rows. I plan to recompute counts and
   length statistics from artifacts before editing. If local artifacts disagree
   with docs, should the LaTeX follow the recomputed artifacts or should I ask
   before changing the numbers?

   Decision: if recomputed artifact values differ from the markdown, update the
   thesis using the recomputed values.

8. Comment replies:
   The user asked that replies to advisor comments be concise but more detailed
   than a bare "fatto" when useful.

   Decision: every reply should state what was changed and why, especially when
   content was moved instead of deleted or when the reasoning cannot be inferred
   from the LaTeX diff alone.

## Implementation Sequence

### Step 1 - Create Comment Tracker And Replies

Files:

- this plan file;
- later, a response file such as
  `thesis/docs/15_reviewer_comment_replies_2026_06_26.md`.

Actions:

- keep one row per comment;
- include exact final reply text;
- mark each as pending/done after implementation.

### Step 2 - Recompute Dataset Stats

Actions:

- locate canonical dataset artifact;
- recompute block counts, split counts, average/median lengths;
- save a small markdown or JSON summary under `thesis/docs` or
  `thesis-datasets/reports` if not already present;
- use these values in the rewritten dataset section.

### Step 3 - Rewrite Chapter 3 Intro And Overview

Actions:

- replace defensive intro with positive method description;
- remove "branch" framing;
- update Figure 3.1 caption/prose if needed.

Commit:

- `Rewrite method overview narrative`

### Step 4 - Expand Semantic-Fidelity Corpus Section

Actions:

- rename section;
- add construction procedure, examples, stats;
- remove downstream method details from this section.

Commit:

- `Expand semantic fidelity corpus method`

### Step 5 - Restructure Method Around Latent-GEPA

Actions:

- move judge task before GEPA;
- rewrite GEPA loop with mini-batch proposal and Pareto-style selection;
- define Latent-GEPA clearly;
- merge feedback variants into a tighter method block.

Commit:

- `Clarify Latent-GEPA method`

### Step 6 - Move Or Reduce SIPIT/NLA/Embedding Details

Actions:

- remove standalone SIPIT method section unless answer says otherwise;
- remove standalone NLA method section unless answer says otherwise;
- move reproduction-boundary/Jina provenance text out of method;
- preserve necessary content in Chapter 4/5 with clearer prose.

Commit:

- `Move diagnostic method details out of Chapter 3`

### Step 7 - Soft-Prompt Diagnostic Section

Actions:

- keep a concise section if approved;
- emphasize the purpose: inspect learned continuous prompt vectors;
- distinguish task sanity metrics from interpretability readout.

Commit:

- `Add soft prompt readout method`

### Step 8 - Remove Code References And Captions Issues

Actions:

- remove inline Python/script/file references in method/results;
- keep artifact paths only in planning docs or footnotes if unavoidable;
- ensure each listing/table/figure has caption and label.

Commit:

- `Remove implementation references from thesis prose`

### Step 9 - Documentation Sync

Actions:

- update `05_method_spec.md`;
- update `06_experimental_setup_spec.md` if content moved there;
- update `07_results_inventory.md` if content moved to results;
- update writing guidelines with any new rule extracted from these comments.

Commit:

- `Sync thesis docs with method rewrite`

### Step 10 - Full Check

Actions:

- compile LaTeX from `thesis/latex`;
- check unresolved refs/citations;
- search for banned/weak patterns:
  - `branch`;
  - local `.py` references in prose;
  - `data_jinav3`, `token_ids_`, `embeddings_` in final prose;
  - `Chapter Summary`;
  - uncaptured `lstlisting`;
- visually inspect page breaks for rewritten sections if possible.

## Commit Policy

Commits should be small enough to copy into Overleaf manually:

- one commit for method overview;
- one commit for dataset section;
- one commit for Latent-GEPA/GEPA loop;
- one commit for moving/removing SIPIT/NLA diagnostic text;
- one commit for soft-prompt diagnostic;
- one commit for results/setup code-reference cleanup;
- one commit for docs/comment replies.

Do not include unrelated experiment files or ignored artifacts.

## Draft Response Messages

These are draft replies to use after implementation.

General Teams reply:

```text
ok, ho dato priorità al metodo. Ho riscritto la narrativa in modo meno
difensivo, tolto la struttura a "branch" dove creava confusione, spostato
SIPIT/NLA fuori dal metodo dettagliato e messo il focus su Latent-GEPA. Ho anche
ampliato la parte del dataset con esempi e statistiche e tolto riferimenti a
file/script dal testo.
```

Short replies:

- comment about "gippittierose": `fatto`
- intro says what chapter does not do: `fatto`
- defensive wording: `fatto`
- canonical title: `fatto`
- dataset too thin: `fatto, ho aggiunto costruzione, esempi e statistiche`
- downstream usage too early: `fatto`
- avoid branch: `fatto`
- python file reference: `fatto`
- unclear section: `fatto`
- result/setup content in method: `fatto`
- SIPIT/NLA placement: `fatto, ho lasciato nel metodo solo il richiamo necessario per Latent-GEPA/readout`
- judge task placement: `fatto`
- captions: `fatto`
- no code/script refs in results: `fatto`

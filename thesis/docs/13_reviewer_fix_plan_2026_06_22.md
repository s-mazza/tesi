# Reviewer Fix Plan - 2026-06-22

Purpose: track the supervisor comments received on the current thesis draft and
turn them into small, reviewable implementation steps. This file is a plan only:
no thesis content changes should be made from it without checking the open
questions at the end.

## Scope

Target files:

- `thesis/latex/main.tex`
- `thesis/docs/04_related_work_map.md`
- `thesis/docs/05_method_spec.md`
- `thesis/docs/06_experimental_setup_spec.md`
- `thesis/docs/11_chapter_1_theoretical_framework.md`
- possibly `thesis/latex/bibliografia.bib` if new citations are added

Commit policy:

- one small commit per coherent group of edits;
- keep unrelated dirty files out of staging;
- compile `thesis/latex/main.tex` after every LaTeX-facing group.

## Fix Checklist

### 1. Cluster hardware table

Supervisor comment: do not specify addresses or unnecessary cluster details;
only list GPU type and count.

Current target:

- `main.tex`, `tab:cluster-hardware`, around the hardware environment section.
- `06_experimental_setup_spec.md`, cluster hardware subsection.

Planned change:

- remove IP/address column from `tab:cluster-hardware`;
- reduce the table to the information needed for experimental interpretation:
  node role/name if useful, GPU count, GPU type, GPU memory;
- remove CPU/RAM if they are not used to justify a result;
- keep the surrounding prose focused on reproducibility constraints: single-GPU
  vs two-GPU jobs, 24 GiB VRAM limit, Slurm allocation.

Expected table shape:

| Resource / node role | GPUs used in thesis |
| --- | --- |
| multi-GPU node | 4 x NVIDIA RTX 3090, 24 GiB |
| single-GPU fallback node | 1 x NVIDIA RTX 3090, 24 GiB |

### 2. Prompt and pseudocode formatting

Supervisor comment: prompts or similar blocks should use a `tcolorbox` figure
environment, like the provided `Knowledge Recall Prompt` example.

Current targets:

- GEPA pseudocode block in `sec:method-gepa-loop`;
- seed judge prompt block in `sec:method-gepa-loop`;
- final optimized prompt in the results chapter;
- JSON/proposer feedback examples may stay as listings if they are code-like data
  structures, but prompt-like text should use the prompt box format.

Planned change:

- define a reusable prompt-box style or use inline `tcolorbox` options;
- convert prompt blocks from `Verbatim` where appropriate to:
  `figure` + `tcolorbox` + `\texttt{...}` or `\begin{Verbatim}` inside
  `tcolorbox` if line wrapping is necessary;
- keep code/data artifacts as `lstlisting` unless they are truly prompt text;
- ensure captions and labels remain clickable.

Decision:

- The user clarified that the GEPA pseudocode must also use the supervisor's
  `tcolorbox`-style presentation. Implement prompt blocks and pseudocode blocks
  with a consistent boxed figure format when they are shown to the reader as
  thesis artifacts.

### 3. Hugging Face model and dataset names

Supervisor comment: when citing models and datasets, do not put the Hugging Face
path in the main prose. Use the normal name plus a citation if a technical paper
exists; put the Hugging Face link in a footnote.

Current targets:

- `jinaai/jina-embeddings-v3` in the embedding-inversion method/results text;
- Qwen model rows in the model-role table and hyperparameter tables;
- NLA checkpoint rows using `kitft/...`;
- Topical-Chat / USR dataset references;
- docs that currently use raw repository/model paths.

Planned change:

- replace prose occurrences like `\texttt{jinaai/jina-embeddings-v3}` with a
  readable model name such as `Jina Embeddings v3`, citation if available, and a
  footnote containing the Hugging Face URL;
- keep full model/checkpoint paths only in implementation tables where exact
  reproducibility requires the identifier, or move them to footnotes;
- for Qwen2.5-7B, use normal model name in prose and add the technical/model
  report citation if already in bibliography or available from DBLP. If the
  only available entry is `CoRR`, use it only as a technical-report citation;
- keep artifact filenames and code identifiers as monospace when they are local
  artifacts, not model names.

### 4. Related-work structure for prompt optimization

Supervisor comment: create a standalone `Prompt Optimization` section. Inside it,
first discuss latent/continuous techniques such as Soft Prompting and Prefix
Tuning, then hard/discrete prompt optimization and GEPA.

Current targets:

- `main.tex`, Chapter 2:
  - current `Prompt Waywardness And Soft-Prompt Interpretability`;
  - current `Prompt Optimization And GEPA`;
- `04_related_work_map.md`;
- possibly Chapter 1 theoretical outline if the concept ordering needs to stay
  aligned.

Planned structure:

```latex
\section{Prompt Optimization}
\subsection{Latent and Continuous Prompt Optimization}
% soft prompting, prefix tuning, prompt tuning, prompt waywardness
\subsection{Hard and Natural-Language Prompt Optimization}
% manual prompt engineering, automatic prompt search, GEPA
\subsection{Why GEPA Fits This Thesis}
% reflective edits, textual feedback, NLA/PPL/aux-judge as proposal aids
```

Planned content changes:

- move soft-prompt interpretability under the new prompt-optimization section;
- explain the boundary between continuous prompt vectors and readable prompt
  strings;
- introduce GEPA as one hard/natural-language prompt optimization method, not as
  the whole category;
- preserve the prompt-waywardness figure if it still helps the continuous branch.

### 5. Expand Natural Language Activations related work

Supervisor comment: the Anthropic technical report contains useful details; go
more specific on how NLA was trained and include stats if helpful.

Current targets:

- `main.tex`, `sec:related-nla`;
- `04_related_work_map.md`;
- possibly `bibliografia.bib` if the citation needs refinement.

Planned additions:

- explain NLA as an autoencoder over activations with two components:
  Activation Verbalizer and Activation Reconstructor;
- describe, at a high level, how the AV is trained to map internal activations to
  natural-language descriptions;
- mention which activation sources/layers/checkpoints are relevant for the
  released models;
- add one or two concrete technical details/statistics from the report, only
  after verifying the exact numbers from the report;
- connect the report back to our method: we use AV outputs as weak textual
  feedback for GEPA, not as ground-truth labels.

### 6. Heading capitalization and connector words

Supervisor comment: do not capitalize connector words such as `And` in section
titles.

Current targets:

- headings such as `Hidden-State Inversion And SIPIT`;
- headings such as `Prompt Optimization And GEPA`;
- headings/tables/captions across all chapters and docs if needed.

Planned change:

- apply title-case cleanup to headings only:
  - `And` -> `and`
  - `As` -> `as`
  - `The` -> `the` where it is not the first word
  - `Of` -> `of`
  - `For` -> `for`
- avoid touching acronym capitalization (`LLM`, `GEPA`, `NLA`, `SIPIT`,
  `G-EVAL`);
- preserve labels so existing references do not break.

### 7. Figure captions from other papers

Supervisor comment: when reproducing a figure, do not mention the original
figure number; citing the paper is enough.

Current targets:

- `fig:related-embedding-diffusion`;
- `fig:related-prompt-waywardness`;
- `fig:related-geval-framework`;
- `fig:related-gepa-overview`;
- any other caption with `Reproduced from Figure~...`.

Planned change:

- rewrite captions from:
  `Reproduced from Figure~1 of Xiao ...`
  to:
  `Adapted/reproduced from Xiao et al. ...`
- verify whether each asset is exactly reproduced or adapted. Use `Reproduced`
  only if unchanged; use `Adapted from` if cropped, redrawn, or reformatted.

### 8. Correlation metrics equations and citations

Supervisor comment: add equations and citations for correlation metrics.

Current targets:

- Chapter 1 `Correlation Metrics`;
- Chapter 4 metric table/metric subsection if equations are more appropriate
  there;
- `bibliografia.bib`.

Planned change:

- in Chapter 1, keep the conceptual introduction but add compact equations for:
  - Pearson correlation;
  - Spearman rank correlation;
  - Kendall rank correlation;
- add citations for the metrics:
  - Pearson correlation coefficient;
  - Spearman rank correlation;
  - Kendall rank correlation / Kendall's tau.
- keep detailed metric boundaries in Chapter 4, so Chapter 1 remains theoretical.

Candidate equations:

```latex
r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}
         {\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
```

```latex
\rho = r(\operatorname{rank}(x), \operatorname{rank}(y))
```

```latex
\tau = \frac{C-D}{\binom{n}{2}}
```

### 9. Agreement rather than absolute truth

Supervisor comment: introduce LLM-human agreement in the correlation discussion
and cite a paper about strong LLM-human alignment, possibly involving
Gemma-3-27B with tricks.

Current targets:

- Chapter 1 `Agreement Rather Than Absolute Truth`;
- Chapter 2 LLM-as-a-judge related work if the cited paper is more related-work
  than theory;
- `bibliografia.bib`.

Planned change:

- clarify that correlation metrics measure agreement with human annotations, not
  access to an objective truth;
- add a short bridge sentence: modern judge papers often evaluate whether model
  judgments can align with human preferences/ratings under controlled prompting
  and calibration;
- identify the exact paper the supervisor has in mind before citing a specific
  Gemma-3-27B claim;
- if not identifiable immediately, cite broader reliable judge-agreement papers
  and leave the specific Gemma paper for a later pass.

### 10. LLM-as-a-judge section intro and cross-domain citations

Supervisor comment: add an intro to LLM-as-a-judge evaluation describing the
technique at high level with citations to papers using it in multiple domains
such as medical and legal.

Current targets:

- Chapter 1 `LLM-As-A-Judge Evaluation`;
- Chapter 2 `LLM-As-A-Judge Evaluation and G-EVAL`;
- `bibliografia.bib`.

Planned change:

- add 1-2 opening paragraphs before rubric-specific details:
  - LLM-as-a-judge uses a language model to approximate human evaluation for
    generated or retrieved text;
  - it is used because human evaluation is expensive and static automatic
    metrics are often insufficient;
  - the method is domain-sensitive and requires careful prompts, parsing, and
    validation against human labels.
- cite G-EVAL plus at least 2-3 broader/domain examples:
  - a general LLM-as-a-judge / evaluation paper;
  - a medical evaluation paper using LLM judges;
  - a legal evaluation paper using LLM judges.
- use DBLP peer-reviewed venue entries when available. Use `CoRR`/arXiv only
  for recent technical reports or papers without a verified conference/journal
  version.

### 11. Explicitly name Latent-GEPA in the thesis outline

Supervisor comment: in the method chapter outline item, explicitly cite
`Latent-GEPA` as the proposed main method.

Current target:

- Introduction outline item for `Chapter 3 - Method`.

Planned change:

- rewrite the method outline sentence to say that the chapter defines
  `Latent-GEPA`, the proposed pipeline that augments GEPA-style prompt
  optimization with latent-derived feedback such as perplexity, NLA
  verbalizations, and optional auxiliary-judge feedback.

### 12. Introduction scope, research framing, and keywords

Supervisor comments:

- remove the claim that the thesis does not aim to show that every method is
  improved by raw NLA and treats negative results as part of the analysis;
- reduce the `This framing leads to...` discussion to two main topics:
  analysis of embedding-inversion methods to identify their potential and
  semantic failure modes; use of inversion / latent-representation analysis to
  optimize GEPA;
- avoid naming specific techniques such as SIPIT and NLA in the introduction
  unless strictly useful. GEPA can be named because it is central to the
  proposed method. If SIPIT or NLA remain in the introduction, they must have a
  citation;
- replace overly specific keywords such as `Semantic Fidelity` and
  `Activation Inversion` with broader keywords such as `Natural Language
  Processing` and `Explainable AI`.

Current targets:

- `main.tex`, abstract/introduction and `\parolechiave`;
- `02_thesis_scope.md`, thesis framing and research-question wording;
- `03_chapter_outline.md`, chapter-level positioning of the two main topics;
- `05_method_spec.md`, method naming and Latent-GEPA framing if the introduction
  points to it;
- possibly `11_chapter_1_theoretical_framework.md` if the theoretical chapter
  still mirrors the old narrower framing.

Planned change:

- remove the raw-NLA negative-result claim from the introduction/scope. If a
  limitation about negative NLA results is useful, move it later to the results
  or conclusion chapter as an empirical observation, not as a front-loaded thesis
  claim;
- rewrite the introductory research framing around two macro topics:
  - semantic-fidelity analysis of embedding/latent inversion methods;
  - `Latent-GEPA`: using latent-representation diagnostics and inversion or
    verbalization signals to improve GEPA-style prompt optimization;
- avoid introducing SIPIT and NLA by name in the introduction unless needed for
  clarity. Prefer broader wording such as `hidden-state inversion methods` and
  `activation verbalization methods`; introduce the concrete techniques in
  Chapter 2 and Chapter 3 with citations;
- update keywords to include:
  - `Natural Language Processing`;
  - `Explainable AI`;
  - likely keep thesis-specific terms such as `Prompt Optimization`,
    `LLM-as-a-Judge Evaluation`, and `Latent Representations` or
    `Semantic Evaluation`, depending on the final keyword count expected by the
    template.

### 13. Documentation consistency pass

This is required because the LaTeX chapters were generated from markdown
planning documents.

Targets:

- `04_related_work_map.md`: update prompt-optimization related-work structure,
  expanded NLA plan, LLM-as-judge citations, heading capitalization.
- `05_method_spec.md`: update prompt-box presentation convention and explicitly
  name Latent-GEPA where method is described.
- `06_experimental_setup_spec.md`: simplify cluster hardware details and adjust
  model naming / HF-link policy.
- `11_chapter_1_theoretical_framework.md`: add correlation-equation plan,
  LLM-human agreement, LLM-as-judge intro expectations, and ensure the
  theoretical chapter does not reintroduce technique-specific framing too early.

## Proposed Commit Sequence

1. `Simplify cluster hardware reporting`
   - hardware table and setup docs only.

2. `Standardize prompt display boxes`
   - prompt/pseudocode formatting policy and selected prompt blocks.

3. `Use publication names for models and datasets`
   - model/dataset naming, footnotes, citations where known.

4. `Restructure prompt optimization related work`
   - Chapter 2 section structure plus related-work markdown.

5. `Expand NLA related work`
   - richer NLA technical report summary and verified details.

6. `Add agreement metric equations`
   - Pearson/Spearman/Kendall formulas and citations.

7. `Broaden LLM-as-judge background`
   - intro, cross-domain citations, agreement framing.

8. `Clean heading capitalization and figure captions`
   - stylistic consistency pass.

9. `Name Latent-GEPA in thesis outline`
   - introduction outline plus method docs.

10. `Refine introduction framing and keywords`
    - remove the raw-NLA negative-result claim, collapse research framing into
      two main topics, avoid unnecessary SIPIT/NLA mentions in the introduction,
      and update the thesis keywords.

## Open Questions Before Implementation

1. Prompt-box scope:
   Resolved. Convert the GEPA pseudocode as well, using the supervisor's
   `tcolorbox`-style presentation.

2. Cluster table scope:
   Resolved. Remove addresses and unnecessary details. Prefer generic resource
   labels unless the older thesis templates show that node labels are useful.
   Check the reference theses before editing the table.

3. Specific Gemma-3-27B agreement paper:
   Resolved. The supervisor confirmed that the intended paper is `Judge's
   Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human
   Agreement` (`arXiv:2510.09738`). Cite this paper for the LLM-human agreement
   discussion and keep the interpretation bounded to judge-agreement evaluation.

4. NLA technical details:
   Resolved. Chapter 2 should include enough Anthropic technical-report detail
   to satisfy the supervisor and make the method understandable to a reader with
   little context. Chapter 4 keeps the concrete checkpoint availability and
   implementation choices.

5. Model naming policy in tables:
   Resolved. Follow the supervisor's preference: prose and table labels use
   normal model/dataset names; exact Hugging Face identifiers go in footnotes or
   reproducibility notes when necessary.

6. Final keyword set:
   Resolved. Use the supervisor's generic keywords and choose the remaining
   keywords during implementation from the actual thesis contribution, without
   asking again unless the template imposes a strict keyword count.

7. Technique names in the introduction:
   Resolved. Follow the supervisor: keep GEPA / Latent-GEPA where needed, avoid
   SIPIT and NLA in the introduction unless strictly useful, and cite any
   technique that remains named there.

Current status:

- Implemented in the current thesis draft:
  - simplified the cluster hardware table to GPU type/count/use only;
  - converted prompt-like text and the GEPA pseudocode to the boxed
    `tcolorbox` presentation requested by the supervisor;
  - moved exact Hugging Face identifiers out of prose/table labels and into
    footnotes or reproducibility notes;
  - reorganized prompt optimization as its own related-work section, with
    latent/continuous prompt optimization before hard/natural-language prompt
    optimization and GEPA;
  - expanded the NLA related-work discussion with AV/AR roles and
    checkpoint/layer constraints;
  - added Pearson, Spearman, and Kendall equations with citations;
  - expanded the LLM-as-a-judge and human-agreement framing;
  - cleaned title capitalization and figure captions;
  - explicitly named `Latent-GEPA` in the thesis outline/method framing;
  - narrowed the introduction framing to the two main thesis axes and updated
    the keyword set.
- Verified after implementation:
  - `latexmk -pdf -g -interaction=nonstopmode main.tex` completes;
  - no unresolved citations or references were reported in `main.log`;
  - the known `Underfull` warnings remain layout warnings, not compilation
    blockers.
- Still to complete:
  - perform a full bibliography audit for all remaining `CoRR`/arXiv entries
    once DBLP is reachable without rate limiting. The current pass fixed the
    verified peer-reviewed replacement for `Sentence Embedding Leaks More
    Information than You Expect`, using the ACL Findings 2023 DBLP entry.
    `Judge's Verdict` remains `CoRR`/arXiv because the supervisor confirmed it
    as the intended paper and no peer-reviewed DBLP venue entry has been
    verified yet;
  - do a final visual pass on the compiled PDF for tables, prompt boxes, and
    page breaks. The automatic compile check is clean, but visual layout still
    matters for the thesis.
- No blocking questions remain before continuing with the next writing pass.
- Confirmed bibliography result: the supervisor's Gemma-3-27B agreement
  reference is `Judge's Verdict: A Comprehensive Analysis of LLM Judge
  Capability Through Human Agreement` (`arXiv:2510.09738`). It reports
  `google/gemma-3-27b-it` among high-correlation judges and argues that
  correlation alone is insufficient. The thesis should cite it as an
  agreement-focused judge paper, not as proof that Gemma is universally aligned.
  It remains a `CoRR`/arXiv citation because no peer-reviewed DBLP venue entry
  has been verified yet, but it is retained because the supervisor confirmed it
  as the intended reference.
- Cross-domain LLM-as-judge citations selected for implementation: MT-Bench /
  Chatbot Arena, G-EVAL, Prometheus, JUDGE-BENCH, Fully Open Meditron, and
  LeMAJ.

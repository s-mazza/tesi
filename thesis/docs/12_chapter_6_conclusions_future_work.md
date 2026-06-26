# Chapter 6 Conclusions And Future Directions Plan

This document defines the intended structure of Chapter 6. The chapter should
close the thesis by summarizing what was learned, what can be claimed, what
remains uncertain, and which future directions naturally follow.

The chapter should be concise, ideally only slightly longer than one page. The
final LaTeX chapter should not use internal section/subsection headings; the
outline sections below are planning notes only. It should not introduce new
experiments, new literature, or new technical details.

## Goal

Summarize the thesis contributions and final evidence position.

The chapter must let a reader answer:

- what the thesis set out to study;
- what was built;
- what evidence was obtained;
- which claims are supported;
- which claims are not yet supported;
- what should be done next.

Target length: about two to three pages, depending on how much of the final
GEPA matrix and auxiliary-judge work is available when writing starts.

## Scope Boundary

Include:

- a compact restatement of the thesis problem;
- a summary of the Semantic-Fidelity Corpus;
- a summary of the embedding-inversion diagnostic evidence;
- a summary of SIPIT evidence;
- a summary of NLA extraction/verbalization;
- a summary of GEPA/G-EVAL results;
- a careful conclusion about NLA feedback;
- future work directions grounded in the actual gaps.

Exclude:

- new background or related work;
- new method details;
- raw metric tables;
- full artifact lists;
- speculative future work that is not connected to the thesis evidence.

## Reference Thesis Style

The reference thesis conclusion is short and practical:

- it briefly recalls the work performed;
- it summarizes the main empirical findings;
- it explains the final practical decision;
- it ends with future directions.

This thesis should keep that style, but with stricter claim control because the
results contain mixed and diagnostic evidence.

## Proposed Section Structure

### 6.1 Summary Of The Thesis

Purpose: restate the full thesis in one coherent paragraph.

Suggested content:

- The thesis studies semantic fidelity in latent-to-text methods.
- The key concern is that a reconstruction or verbalization can look plausible
  while losing negation, polarity, contradiction, or counterfactual content.
- The work connects an inversion/verbalization track with a GEPA/G-EVAL track:
  activation-derived signals are tested as possible feedback for prompt
  optimization.

Avoid:

- presenting GEPA as the whole thesis;
- claiming that all semantic-fidelity questions were fully solved.

### 6.2 Contributions

Purpose: list the concrete contributions that survived the experimental work.

Expected table or bullet list:

| Contribution | Status | Evidence |
|---|---|---|
| Semantic-Fidelity Corpus | Built and validated | Dataset reports |
| Embedding-inversion diagnostics | Completed as diagnostic/negative reproduction evidence | Jina/Qwen probes |
| SIPIT reproduction and extensions | Partially supported | Collision check and interim GPT-2 recovery evidence |
| NLA extraction and verbalization | Operational | Qwen2.5-7B layer-20 extraction and AV verbalization |
| GEPA/G-EVAL pipeline | Operational | Multi-dataset runner and artifacts |
| Perplexity feedback | Positive observed evidence | First PPL long run |
| Raw/fixed NLA feedback analysis | Mixed/diagnostic | Raw-NLA negative run, fixed-NLA weak-positive control |
| Auxiliary-judge compression | Pending or future unless final runs complete | Aux-judge artifacts/status |

This section can be written as prose in the final thesis, but the planning table
keeps the claims disciplined.

### 6.3 Main Findings

Purpose: summarize results without repeating Chapter 5.

Findings to include if still consistent with final artifacts:

- The Semantic-Fidelity Corpus provides a structured way to stress latent-to-text
  methods with standard, negation, and commonsense/counterfactual examples.
- The embedding-inversion diagnostics did not reach a clean paper-level reproduction
  but produced useful failure-mode evidence.
- SIPIT evidence supports the practical injectivity/recovery story in the GPT-2
  setting, but final CSV/JSON outputs must be recovered or the claim must remain
  interim/log-derived.
- NLA extraction/verbalization is technically functional, but this
  alone does not prove semantic faithfulness.
- GEPA with PPL feedback can improve the observed Topical-Chat engagingness
  judge prompt.
- Raw NLA feedback is not reliably helpful in the current GEPA setup.
- Fixed-NLA is technically healthier and weak-positive against a matched PPL
  control, but the unchanged prompt prevents a strong "NLA improved GEPA"
  conclusion.

If auxiliary-judge runs succeed:

- add that semantic compression of NLA feedback appears more promising than raw
  verbalization injection.

If auxiliary-judge runs fail:

- state that transforming activation verbalizations into useful GEPA feedback
  remains open.

### 6.4 Final Claim Position

Purpose: explicitly separate supported and unsupported claims.

Supported claim candidates:

- The semantic-fidelity framing is necessary because standard reconstruction or
  agreement metrics can miss logically important changes.
- The Semantic-Fidelity Corpus and artifact pipeline make these questions more
  testable.
- Perplexity is a useful additional signal for GEPA in the observed setting.
- Raw NLA verbalizations are informative artifacts, but not automatically useful
  prompt-optimization feedback.

Unsupported or not-yet-supported claims:

- NLA robustly improves GEPA.
- NLA extraction/verbalization preserves negation or counterfactual content on the
  Semantic-Fidelity Corpus.
- The thesis exactly reproduces all G-EVAL paper results.
- The thesis exactly reproduces the Jina embedding-inversion paper.
- Smoke-test numbers are final scientific performance evidence.

This section is important because it protects the final conclusion from
overclaiming.

### 6.5 Limitations

Purpose: name limitations that materially affect interpretation.

Limitations to include:

- Some experiment families are diagnostic rather than completed paper
  reproductions.
- SIPIT exact-recovery evidence is available for the recovered GPT-2 setting
  and the thesis logical20 prompts, but runtime remains hardware-dependent.
- NLA semantic-fidelity scoring on the Semantic-Fidelity Corpus has not yet
  been completed.
- GEPA evidence is strongest on Topical-Chat engagingness; full matrix coverage
  is still incomplete unless later jobs finish.
- Model choices differ from the original G-EVAL paper.
- Runtime and cluster constraints affected the number and length of runs.
- Current GEPA artifacts have total runtime, but not always complete per-stage
  timing or peak GPU memory.

Avoid generic limitations such as "more data would be better" unless they are
connected to a specific result.

### 6.6 Future Directions

Purpose: propose concrete next steps that follow from the evidence.

Priority future directions:

1. Run direct NLA semantic-fidelity evaluation on the Semantic-Fidelity Corpus.
2. Complete or recover SIPIT final reports for Table 5, logical dataset, and
   random-prefix experiments.
3. Test NLA transformation strategies before proposer feedback:
   auxiliary-judge compression, rubric-level summarization, token-selection
   ablations, and duplicate-aware context compression.
4. Complete a broader GEPA matrix across Topical-Chat, SummEval, QAGS-CNN, and
   QAGS-XSUM with paper-aligned metrics.
5. Add per-stage timing and GPU-memory profiling for long runs.
6. Explore multi-dimension prompting as an independent pipeline, not as a
   replacement for paper-aligned single-dimension results.

Conditional future work:

- If auxiliary-judge feedback improves GEPA, investigate whether the gain comes
  from NLA compression, stronger critique, or simply better proposer feedback.
- If auxiliary-judge feedback fails, inspect whether NLA verbalizations are too
  noisy, too generic, or misaligned with G-EVAL rubric dimensions.
- If the thesis scope returns to inversion as the main contribution, prioritize
  semantic flip metrics and qualitative analyses over more GEPA runs.

Avoid:

- promising to train new NLA checkpoints unless the thesis actually has the
  resources and motivation for that.
- presenting Qwen35B as a future replacement for the base judge unless the
  experimental question changes.

### 6.7 Closing Paragraph

Purpose: end with the main takeaway in one paragraph.

Suggested message:

- Latent-to-text methods and activation verbalizations should be evaluated not
  only for readability or surface similarity, but for preservation of logically
  important meaning.
- NLA provides a promising interface to internal activations, but raw
  verbalizations are not automatically task-useful feedback.
- The strongest future direction is to transform activation-level evidence into
  rubric-aware feedback and evaluate it under matched, paper-aligned protocols.

## Do Not Forget

- The current LaTeX places acknowledgments in the backmatter as an unnumbered
  chapter, so Chapter 6 remains a standalone conclusion without internal
  sections.
- Thank Prof. Moro first in the acknowledgments, as required by the writing
  guidelines.
- Do not introduce bibliography citations for the first time in the conclusion
  unless absolutely necessary.
- Do not add new result numbers in Chapter 6 that were not already presented in
  Chapter 5.

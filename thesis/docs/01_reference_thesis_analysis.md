# Reference Thesis Analysis

Reference folder: `tesi_t_simoneMazzacano/`.

The reference thesis must remain unchanged. This document records the reusable
structure and style decisions extracted from it. Its structure and style are
the default baseline for the new thesis. Any intentional change to those
reference-derived conventions requires explicit permission first.

## Structural Lessons

- The reference uses a single `main.tex` for the whole thesis body.
- Front matter order:
  1. title page metadata;
  2. abstract;
  3. introduction;
  4. table of contents;
  5. list of figures;
  6. list of tables.
- Main chapters:
  1. Theoretical Framework;
  2. Related Work;
  3. Method;
  4. Experimental Setup;
  5. Results;
  6. Conclusions and Future Directions;
  7. Acknowledgments.
- This chapter sequence matches the advisor's notes and should be reused.

## Style Lessons

- The text introduces the problem from broad context, then narrows to method
  and evaluation.
- The introduction ends with a chapter-by-chapter outline.
- Figures are placed near the text that discusses them and are referenced with
  `Figure~\ref{...}`.
- Tables are used for model variants, hyperparameters, and result summaries.
- The thesis uses numeric citations through `natbib`.

## Template Lessons

- The style files under `tesi_t_simoneMazzacano/style/` can be used as a
  template source when the LaTeX project is created.
- The current reference uses `book` with `12pt`, `a4paper`, `twoside`,
  `openright`.
- Useful packages already present in the reference include `booktabs`,
  `multirow`, `amsmath`, `listings`, `algorithm`, `algpseudocode`, `graphicx`,
  `hyperref`, `pgfplots`, and `natbib`.

## Adaptations For This Thesis

- Keep the same macro-level chapter structure, but rename sections to match
  GEPA, G-EVAL, NLA, LLM-as-a-judge, and prompt optimization.
- Do not copy the previous student's ad hoc content. Replace the topic-specific
  material with this thesis's problem, method, experiments, figures, results,
  and claims.
- Reuse structure, formatting habits, tone, and thesis organization unless
  there is a clear reason to change them and that change has been approved.
- The theory chapter should focus on what is needed to understand the method:
  LLMs as judges, prompt optimization, evaluation metrics, NLA, perplexity,
  and efficient inference.
- The experimental chapter should be more detailed than the reference because
  reproducibility and cluster execution are central to this thesis.

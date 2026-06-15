# Thesis Workspace

This folder is the source area for Simone Mazzacano's thesis work.

The previous student thesis in `tesi_t_simoneMazzacano/` is a read-only
reference. Do not edit it and do not move files from it into this folder unless
there is an explicit reason to copy a template element.

## Current Phase

The thesis now has an initial uploadable LaTeX manuscript in `latex/`, following
the root-level structure of the previous student thesis inside that folder. The
Markdown documents in `docs/` remain the planning source for chapter scope,
experimental decisions, and result mapping.

## LaTeX Manuscript

- `latex/main.tex`: current thesis manuscript, written up to Chapter 1 included.
- `latex/bibliografia.bib`: bibliography file to be filled from DBLP entries where
  available.
- `latex/custom_import.tex`: shared package imports and custom commands.
- `latex/style/`: title-page and formatting files copied from the reference
  thesis template.
- `latex/images/` and `latex/graph/`: manuscript figure folders, currently
  reserved for future thesis figures and plots.

## Documents

- `docs/00_writing_guidelines.md`: advisor writing rules and thesis style
  checklist.
- `docs/01_reference_thesis_analysis.md`: reusable lessons from the previous
  thesis reference.
- `docs/02_thesis_scope.md`: thesis goal, research questions, claims, and
  boundaries.
- `docs/03_chapter_outline.md`: chapter-level thesis outline.
- `docs/04_related_work_map.md`: papers and positioning plan.
- `docs/05_method_spec.md`: dataset, inversion, NLA, GEPA, perplexity, and
  auxiliary judge method.
- `docs/06_experimental_setup_spec.md`: datasets, models, metrics, cluster,
  and hyperparameters.
- `docs/07_results_inventory.md`: current result artifacts and how they map to
  thesis tables.
- `docs/08_overleaf_sync.md`: local Git and Overleaf synchronization workflow.
- `docs/09_prior_work_census.md`: full census of local and cluster work to
  include in the thesis, covering embedding inversion, SIPIT, NLA, and GEPA.
- `docs/10_paper_reading_list.md`: advisor-provided and locally downloaded
  papers, deduplicated and mapped to the thesis related-work groups.
- `docs/11_chapter_1_theoretical_framework.md`: detailed writing plan and
  scope boundary for Chapter 1.

## Local Source Of Truth

This folder is the local source of truth. The Overleaf project is used for
advisor review, while local Git keeps the complete manuscript history. For a
clean Overleaf upload, use only the `latex/` folder.

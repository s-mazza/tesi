# Overleaf Synchronization

Overleaf project:

`https://www.overleaf.com/project/6a18594534f7866ebd435752`

## Goal

Keep local Git as the source of truth while allowing the advisor to review the
thesis on Overleaf.

## Preferred Workflow: Overleaf Git Bridge

Use this if the Overleaf project exposes a Git URL.

Expected flow:

1. Work locally in this repository.
2. Commit changes with descriptive messages.
3. Pull from the Overleaf Git remote before pushing local LaTeX changes.
4. Push the LaTeX project to Overleaf for advisor review.
5. Pull Overleaf edits/comments before continuing locally.

This gives local version history and lets Overleaf remain the review surface.

## Fallback: GitHub Sync

Use this if Overleaf Git Bridge is not available but the project supports
GitHub synchronization.

Expected flow:

1. Local commits go to GitHub.
2. Overleaf imports/syncs from the GitHub branch.
3. Advisor reviews on Overleaf.
4. Local repository remains the canonical history.

## Last Fallback: Manual Upload

Use this only if neither Git Bridge nor GitHub Sync is available.

Expected flow:

1. Upload the `thesis/latex/` folder or a clean zip of it to Overleaf for
   review.
2. Keep the same root-level structure used by the reference thesis:
   `main.tex`, `bibliografia.bib`, `custom_import.tex`, `style/`, `images/`,
   and `graph/`.
3. Pull any advisor edits back manually before continuing.

This is less reliable and should be avoided if Git integration is available.

## Rules

- Do not put experimental result directories directly inside the Overleaf
  manuscript unless they are small tables or figures required for the thesis.
- Do not upload local build artifacts such as `.aux`, `.log`, `.fls`, `.fdb_*`,
  `.synctex.gz`, or generated caches.
- Keep large result artifacts linked through GitHub or summarized in thesis
  tables.
- The uploadable manuscript source is `thesis/latex/`. Inside that folder, the
  structure mirrors the reference thesis: `main.tex`, `bibliografia.bib`,
  `custom_import.tex`, `style/`, `images/`, and `graph/`.
- The `docs/` folder remains local planning material and should not be uploaded
  to Overleaf unless explicitly useful for review.

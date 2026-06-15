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

1. Generate a clean LaTeX folder locally.
2. Upload the folder or zip to Overleaf for review.
3. Pull any advisor edits back manually before continuing.

This is less reliable and should be avoided if Git integration is available.

## Rules

- Do not put experimental result directories directly inside the Overleaf
  manuscript unless they are small tables or figures required for the thesis.
- Do not upload local build artifacts such as `.aux`, `.log`, `.fls`, `.fdb_*`,
  `.synctex.gz`, or generated caches.
- Keep large result artifacts linked through GitHub or summarized in thesis
  tables.
- Before starting LaTeX, decide whether the Overleaf project will mirror only
  `thesis/latex` or the entire `thesis` folder. The recommended target is a
  dedicated `thesis/latex` folder.

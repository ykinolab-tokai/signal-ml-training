# T019 Scaffold LaTeX source layout

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-24
Milestone: M007

## Background
- The user decided that LaTeX will become the new canonical source for the legacy notebook-based text.
- The repository currently has no LaTeX document structure, so conversion work needs a stable destination before chapter-by-chapter migration can start.

## Scope
- In: creating the top-level LaTeX root, chapter directory, markdown mirror directory, figure directories, and a minimal integrated main document.
- Out: migrating every notebook in this task alone.

## Deliverables
- A `latex/` directory that acts as the new book root.
- A minimal `main.tex` and shared preamble/macros.
- Directory conventions for chapter `.tex`, chapter `.md`, extracted figures, and future figure replay scripts.

## Acceptance Criteria
- [x] `latex/` contains a compilable main document skeleton with `\input`-based chapter integration.
- [x] The directory layout distinguishes canonical `.tex`, pandoc-friendly `.md`, and figure assets.
- [x] The structure is concrete enough for one prototype chapter to be generated into it.

## Dependencies / Blockers
- None.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Reuse the established `latex/` layout for the next migration batches instead of creating per-chapter ad hoc structures.

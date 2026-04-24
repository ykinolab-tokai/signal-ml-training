# T024 Convert full notebook corpus into chapter-scoped LaTeX and Markdown

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M007

## Background
- The prototype converter is now stable enough to apply across the whole `text/` corpus.
- The remaining work is bulk generation, chapter ordering, and explicit handling of empty notebooks.

## Scope
- In: converting every `text/*.ipynb` into `latex/chapters/*.tex` and `latex/markdown/*.md`, copying static assets, and generating figure replay scripts where needed.
- Out: hand-redrawing figures in TikZ.

## Deliverables
- Chapter-scoped `.tex` and `.md` outputs for the full notebook corpus.
- Empty `.tex` and `.md` outputs for empty notebooks 13, 14, and 15.
- Generated figure assets and replay scripts for notebooks that contain embedded output figures.

## Acceptance Criteria
- [x] Every notebook under `text/` has a corresponding chapter `.tex` and `.md`.
- [x] Empty notebooks are represented by empty generated chapter files rather than skipped silently.
- [x] Static image references no longer depend on files under `text/`.

## Dependencies / Blockers
- Depends on the repaired converter logic from T022 and T023.

## Next Action
- owner: codex
- due: 2026-04-22
- action: Keep the batch converter as the regeneration entrypoint while later manual chapter cleanup proceeds.

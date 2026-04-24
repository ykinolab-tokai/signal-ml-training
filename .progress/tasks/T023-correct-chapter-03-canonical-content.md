# T023 Correct chapter 03 canonical content after conversion review

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M007

## Background
- Review found that chapter 03's generated canonical outputs still contain stale notebook execution results in the statistics section.
- Even with converter fixes, the current notebook runtime state cannot be trusted as-is, so the new canonical TeX/Markdown must be corrected directly.

## Scope
- In: chapter 03 Markdown and LaTeX outputs, especially the statistics section and migrated static assets.
- Out: re-executing the original notebook in a restored Python environment.

## Deliverables
- Corrected chapter 03 `.tex` and `.md` outputs with fixed statistics outputs and cleaned references.
- Static image assets migrated into the LaTeX workspace so chapter 03 no longer depends on `text/`.

## Acceptance Criteria
- [x] Chapter 03 no longer contains contradictory statistics outputs or local-environment traceback leakage where the content should succeed.
- [x] Chapter 03 references only migrated figure assets under `latex/`.
- [x] Chapter 03 remains buildable from `latex/main.tex`.

## Dependencies / Blockers
- Depends on the converter fixes where possible, but content cleanup may also require direct edits to the canonical outputs.

## Next Action
- owner: codex
- due: 2026-04-22
- action: Carry the corrected chapter 03 forward as the prototype baseline for the remaining-corpus audit.

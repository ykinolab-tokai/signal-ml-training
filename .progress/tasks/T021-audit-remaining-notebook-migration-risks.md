# T021 Audit remaining notebook migration risks

Last updated: 2026-04-22
Status: done
Priority: medium
Owner: codex
Due: 2026-04-25
Milestone: M007

## Background
- The first prototype only proves the path for one representative chapter.
- The remaining notebooks include empty files, external image references, and heavier code/output patterns that need explicit triage before deletion of `.ipynb`.

## Scope
- In: classifying remaining notebooks by migration difficulty, identifying empty notebooks, and documenting unresolved figure/TikZ decisions.
- Out: converting every remaining notebook immediately.

## Deliverables
- A follow-up audit note covering empty notebooks, figure rewrite candidates, and chapters that likely need manual cleanup.
- Updated progress records that separate prototype completion from full migration work.

## Acceptance Criteria
- [x] Empty notebooks are explicitly called out as migration gaps rather than silently skipped.
- [x] Remaining chapters are grouped by migration strategy or risk.
- [x] Open environment/tooling blockers are documented for the next phase.

## Dependencies / Blockers
- None.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Use the audit note as the handoff reference for later TikZ rewrites and chapter polishing.

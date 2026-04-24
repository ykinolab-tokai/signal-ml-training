# T026 Fill empty signal-processing chapters for self-study completeness

Last updated: 2026-04-23
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M008

## Background
- The migrated corpus kept chapters 13, 14, and 15 explicitly empty because the source notebooks were blank.
- As a self-study text, leaving the core signal-processing bridge empty breaks the path from calculus/probability to spectrum analysis and filtering.

## Scope
- In: authoring new canonical content for chapters 13-15 in both `latex/chapters/` and `latex/markdown/`, then rebuilding the integrated LaTeX book.
- Out: TikZ redraws, advanced DSP topics such as z-transform proofs, and chapter-wide visual polishing.

## Deliverables
- New self-contained content for signal basics, spectrum analysis, and LTI systems in the canonical `.tex` sources.
- Matching chapter-scoped Markdown files so the pandoc-friendly parallel representation remains populated.
- A verified integrated `latex/main.pdf` build after the additions.

## Acceptance Criteria
- [x] Chapters 13-15 are no longer empty.
- [x] Each chapter includes definitions, intuition, formulas, and at least one minimal Python example.
- [x] `latexmk` succeeds on the integrated document after the additions.

## Dependencies / Blockers
- Uses the existing LaTeX source tree and integrated build created under M007.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Use these chapters as the new baseline, then identify any other concept gaps that materially block self-study progression.

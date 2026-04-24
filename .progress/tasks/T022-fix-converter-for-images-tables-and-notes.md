# T022 Fix converter for images, tables, notes, and figure replay

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M007

## Background
- Review of the chapter 03 prototype found multiple converter bugs rather than chapter-only mistakes.
- The current converter loses Markdown images, leaves tables and raw HTML notes unconverted in LaTeX, and makes the replay script too permissive to trust for future regeneration.

## Scope
- In: `scripts/ipynb_to_chapter.py`, LaTeX preamble helpers, and generated asset path conventions.
- Out: auditing every remaining notebook.

## Deliverables
- Updated converter logic for static images, internal links, note blocks, and Markdown tables.
- Safer replay-script generation that fails fast and avoids stray output files.
- Regenerated chapter 03 outputs reflecting the converter fixes.

## Acceptance Criteria
- [x] Markdown image syntax no longer degrades into broken LaTeX text.
- [x] Chapter note blocks and tables convert into valid LaTeX structures.
- [x] Replay scripts fail loudly on execution drift and do not emit unmanaged side-effect files.

## Dependencies / Blockers
- None.

## Next Action
- owner: codex
- due: 2026-04-22
- action: Use the repaired converter as the baseline while auditing the remaining notebooks for migration risk.

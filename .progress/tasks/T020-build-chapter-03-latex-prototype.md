# T020 Build chapter 03 LaTeX prototype

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-24
Milestone: M007

## Background
- A representative prototype is needed before migrating all notebooks.
- `text/03_Reviewing_Elementary_Math_with_NumPy_and_Matplotlib.ipynb` contains Markdown, math, code, output, and notebook figure images, so it is the best first test.

## Scope
- In: converting chapter 03 into a chapter-scoped `.tex`, a chapter-scoped `.md`, extracted figure files, and a replay script for later figure regeneration.
- Out: polishing every formatting edge case across the whole notebook corpus.

## Deliverables
- `latex/chapters/` chapter file for chapter 03.
- `latex/markdown/` chapter Markdown for chapter 03.
- Extracted figure image files under `latex/figures/generated/`.
- A future replay script under `latex/figures/scripts/`.

## Acceptance Criteria
- [x] Chapter 03 has both `.tex` and `.md` outputs under the new layout.
- [x] Figure outputs embedded in the notebook are materialized as standalone files.
- [x] The generated LaTeX references figures via file paths rather than notebook output blobs.

## Dependencies / Blockers
- Replay-script execution still needs a Python environment that has NumPy and matplotlib available again, but the extracted prototype figures and compiled PDF are already present.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Use the chapter 03 outputs and converter behavior to define how remaining notebooks should be bucketed into automatic, semi-automatic, and manual migration paths.

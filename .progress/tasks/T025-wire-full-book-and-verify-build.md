# T025 Wire the full LaTeX book and verify integrated build

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M007

## Background
- Full-corpus conversion is only usable if the integrated `latex/main.tex` includes every chapter and still builds.
- Some notebooks contain unsupported assets such as GIFs or unusual Markdown patterns that may require post-conversion cleanup.

## Scope
- In: generating the full `main.tex`, fixing build-breaking chapter output, and verifying `latex/main.pdf`.
- Out: visual polishing beyond what is required for a successful integrated build.

## Deliverables
- Updated `latex/main.tex` that inputs the entire chapter set in order.
- A successful integrated `latexmk` build, or explicit documentation of any residual blockers.

## Acceptance Criteria
- [x] `latex/main.tex` includes the full converted corpus in order.
- [x] `latexmk` succeeds on the integrated document.
- [x] Any non-embedded asset fallbacks are explicit in the generated chapters rather than hidden build failures.

## Dependencies / Blockers
- Depends on T024 finishing the bulk chapter generation.

## Next Action
- owner: codex
- due: 2026-04-22
- action: Treat the current integrated PDF as the baseline while later visual cleanup or TikZ rewrites are scheduled separately.

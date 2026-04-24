# T027 Expand incomplete integral chapter for self-study continuity

Last updated: 2026-04-23
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M008

## Background
- After filling the empty signal-processing chapters, the next most obvious self-study gap was `ch05-basics-of-integral`, which still contained section headings with almost no explanation.
- Leaving calculus incomplete would weaken later chapters that rely on accumulation, area, energy, and expectation concepts.

## Scope
- In: re-authoring chapter 05 in canonical LaTeX and matching Markdown, then verifying the integrated book still builds.
- Out: advanced integration techniques beyond the core seminar prerequisites.

## Deliverables
- A self-contained integral chapter covering indefinite integrals, definite integrals, improper integrals, multiple integrals, and numerical integration.
- Matching Markdown for the chapter.
- Integrated LaTeX build verification after the rewrite.

## Acceptance Criteria
- [x] Chapter 05 no longer stops at empty section headings.
- [x] The rewritten chapter includes definitions, key formulas, and at least one Python example.
- [x] `latexmk` succeeds after the chapter rewrite.

## Dependencies / Blockers
- Uses the existing M007 LaTeX baseline and the broader pedagogical gap review from T026.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Continue future content review by checking for concept coverage gaps rather than line-count parity with the original notebooks.

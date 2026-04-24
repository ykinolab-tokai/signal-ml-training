# T030 Expand matplotlib chapter for graph selection and save-first workflow

Last updated: 2026-04-23
Status: done
Priority: medium
Owner: codex
Due: 2026-04-23
Milestone: M008

## Background
- Chapter 08 had examples, but it still read like a list of plot types rather than a self-study guide that explains when to choose each plot and how to save usable outputs.
- The seaborn section also relied on outdated `distplot` guidance.

## Scope
- In: re-authoring chapter 08 in canonical LaTeX and Markdown around graph selection, labeling, saving, subplot layout, and current seaborn usage.
- Out: advanced style-system topics such as custom rcParams themes or publication-grade figure design.

## Deliverables
- A rewritten chapter 08 with explanation-first coverage of scatter plots, histograms, box plots, line plots, layout control, and seaborn usage.
- Matching Markdown output for the chapter.
- Integrated LaTeX build verification after the rewrite.

## Acceptance Criteria
- [x] Chapter 08 explains when to use each plot type rather than only listing API calls.
- [x] The chapter includes explicit save-first plotting workflow with `savefig`.
- [x] `latexmk` succeeds after the rewrite.

## Dependencies / Blockers
- Builds on the broader self-study gap-filling pass after T026-T029.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Continue reviewing remaining practical-tool chapters and prioritize the ones that still lack worked examples, modern API guidance, or save-and-verify workflow.

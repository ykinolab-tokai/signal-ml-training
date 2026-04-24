# T031 Expand pillow chapter for save-first image preprocessing workflow

Last updated: 2026-04-23
Status: done
Priority: medium
Owner: codex
Due: 2026-04-23
Milestone: M008

## Background
- Chapter 12 still leaned on GUI-style `show()` examples and contained a broken histogram example with `img` undefined.
- As a self-study chapter for image preprocessing, it needed a save-first workflow and a clearer bridge to NumPy-based ML preprocessing.

## Scope
- In: re-authoring chapter 12 in canonical LaTeX and Markdown around image loading, metadata checks, save-first transformations, color conversion, NumPy conversion, and simple preprocessing.
- Out: advanced computer-vision topics better handled by OpenCV or torchvision transforms.

## Deliverables
- A rewritten chapter 12 with explanation-first coverage of practical Pillow workflows.
- Matching Markdown output for the chapter.
- Integrated LaTeX build verification after the rewrite.

## Acceptance Criteria
- [x] Chapter 12 no longer depends on GUI-only `show()`-centric workflow.
- [x] The broken histogram/example flow is replaced with a coherent sequence from loading to saving.
- [x] `latexmk` succeeds after the rewrite.

## Dependencies / Blockers
- Builds on the ongoing self-study gap-filling pass after T030.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Continue reviewing remaining practical chapters and prioritize any that still rely on outdated APIs, display-only workflow, or underexplained preprocessing steps.

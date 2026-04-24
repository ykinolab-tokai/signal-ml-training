# T028 Expand soundfile chapter for audio self-study continuity

Last updated: 2026-04-23
Status: done
Priority: high
Owner: codex
Due: 2026-04-23
Milestone: M008

## Background
- After the larger math and signal-processing gaps were closed, the next thinnest chapter was `ch11-introduction-to-soundfile`.
- The original migrated content was too short to support self-study and even had repeated `\chapter` declarations inside one file.

## Scope
- In: re-authoring chapter 11 in canonical LaTeX and matching Markdown with practical audio I/O guidance and a minimal verification example.
- Out: advanced audio feature extraction such as STFT, mel, or augmentation.

## Deliverables
- A rewritten chapter 11 that explains audio loading, shape conventions, dtype handling, partial reads, writing, and metadata inspection.
- Matching Markdown output for the chapter.
- Integrated LaTeX build verification after the rewrite.

## Acceptance Criteria
- [x] Chapter 11 has a coherent single-chapter structure instead of repeated chapter headings.
- [x] The chapter includes at least one practical Python example that reads, writes, and checks audio data.
- [x] `latexmk` succeeds after the rewrite.

## Dependencies / Blockers
- Uses the post-migration LaTeX baseline and the broader self-study gap review performed after T026 and T027.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Continue checking the shortest remaining chapters and expand the ones that still block a smooth self-study path.

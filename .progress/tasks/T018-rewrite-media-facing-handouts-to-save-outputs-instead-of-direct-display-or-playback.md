# T018 Rewrite media-facing handouts to save outputs instead of direct display or playback

Last updated: 2026-04-22
Status: done
Priority: high
Owner: codex
Due: 2026-04-22
Milestone: M006

## Background
- The user wants image and audio work to default to file output rather than direct display or playback.
- Several handouts still used wording such as `plt.show()` or "表示する", and some deliverables did not require saved output artifacts.

## Scope
- In: handouts that use figures, images, waveforms, spectrograms, or prediction outputs as part of the exercise.
- Out: adding full runnable reference implementations for every session.

## Deliverables
- Updated handouts that instruct students to save figures, audio files, logs, or other output artifacts.
- Example code snippets that create output directories and save representative files.

## Acceptance Criteria
- [x] Direct display or playback is no longer the default expectation in the affected handouts.
- [x] Media-related deliverables explicitly include saved output files.
- [x] Representative snippets show output-file writing for image or audio workflows.

## Dependencies / Blockers
- None.

## Next Action
- owner: codex
- due: 2026-04-22
- action: When future sessions introduce visual or audio checks, require saved output files and predictable output paths.

# T033 Migrate all handouts to the mini-lecture and levelled exercises template

Last updated: 2026-04-24
Status: done
Priority: high
Owner: codex
Due: 2026-04-24
Milestone: M010

## Background
- The latest review found that many handouts were precise enough to follow but still too close to transcription tasks, which weakens the goal of deepening understanding through the exercises themselves.
- The new default template should give students the minimum conceptual bridge they need before coding, then require both execution and interpretation through clearly bounded `基礎レベル` and `発展レベル` exercises.

## Scope
- In: `materials/README.md`, `materials/spring/*.md`, and `materials/autumn/*.md`.
- Out: seminar policy changes in `README.md`, repository layout changes, and deep expansion of `latex/` chapters.

## Deliverables
- All 28 handouts rewritten to the new five-section template.
- Every handout includes a compact `ミニ講義` and an `演習` section split into `基礎レベル` and `発展レベル`.
- Self-check points and fallback references remain available after the rewrite.

## Acceptance Criteria
- [x] All files under `materials/spring/` and `materials/autumn/` use only the new section headings.
- [x] Each handout contains `## ミニ講義` plus `### 基礎レベル` and `### 発展レベル` under `## 演習`.
- [x] Exercises remain concrete enough to start without instructor translation but include at least one reasoning, comparison, or interpretation task.
- [x] `詰まったときに見る資料` still points to the most relevant repo-internal or official references.

## Dependencies / Blockers
- No external blocker. The rewrite must preserve the current curriculum and role expectations described in `README.md`.

## Next Action
- owner: codex
- due: 2026-04-24
- action: Keep future edits aligned with the same five-section template and reject new handouts that reintroduce transcription-only exercises.

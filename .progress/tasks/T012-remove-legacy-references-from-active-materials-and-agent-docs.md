# T012 Remove legacy references from active materials and agent docs

Last updated: 2026-04-11
Status: done
Priority: high
Owner: codex
Due: 2026-04-11
Milestone: M004

## Background
- The review found lingering references to legacy notebook and exercise files in active docs.
- Those references would become stale immediately after cleanup.

## Scope
- In: `materials/`, `templates/`, and `AGENTS.md`.
- Out: progress logs that intentionally preserve task history.

## Deliverables
- Updated active docs with only current internal references or generic supplemental references.

## Acceptance Criteria
- [x] `materials/` no longer referenced legacy notebook or exercise files.
- [x] Templates no longer assumed notebook-based support.
- [x] `AGENTS.md` architecture matched the current repository layout.

## Dependencies / Blockers
- Depended on T011 review findings.

## Next Action
- owner: codex
- due: 2026-04-11
- action: Keep active docs aligned with the actual repository contents.

# T005 Audit text coverage against materials

Last updated: 2026-04-11
Status: done
Priority: high
Owner: codex
Due: 2026-04-11
Milestone: M002

## Background
- Existing materials referenced `text/` heavily, but the current notebooks did not cover all required topics.
- Some signal-processing notebooks were empty, and neural-network coverage in `text/23` was not PyTorch-native.

## Scope
- In: inspect current notebooks, identify direct coverage, identify gaps that materials must absorb.
- Out: rewriting `text/` itself.

## Deliverables
- A concrete mapping of `text/` coverage to session materials.
- A list of gaps to close inside `materials/`.

## Acceptance Criteria
- [x] Current notebook coverage was inspected.
- [x] Empty or non-PyTorch content was identified explicitly.
- [x] Required follow-up edits in `materials/` were determined.

## Dependencies / Blockers
- None.

## Next Action
- owner: codex
- due: 2026-04-11
- action: Re-run this audit whenever `text/` gains new notebooks or major rewrites.

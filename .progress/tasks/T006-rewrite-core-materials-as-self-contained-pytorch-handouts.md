# T006 Rewrite core materials as self-contained PyTorch handouts

Last updated: 2026-04-11
Status: done
Priority: high
Owner: codex
Due: 2026-04-11
Milestone: M002

## Background
- The user required `materials/` to stand alone without relying on `text/` for essential content.
- Core implementation topics needed direct PyTorch examples.

## Scope
- In: add concise PyTorch explanations and minimal runnable snippets to the relevant session handouts.
- Out: large full-length tutorials or rewriting the notebooks under `text/`.

## Deliverables
- Self-contained PyTorch sections in the core autumn implementation sessions.
- Additional minimal implementation sections in spring sessions that lacked concrete code.

## Acceptance Criteria
- [x] Autumn sessions 18-27 contain PyTorch-centered concepts and minimal code.
- [x] Spring sessions needing concrete PyTorch examples were strengthened.
- [x] Materials no longer depend on `text/` for essential PyTorch understanding.

## Dependencies / Blockers
- Depended on T005 coverage audit.

## Next Action
- owner: codex
- due: 2026-04-11
- action: Expand year-specific rotation sessions when their exact topics are fixed.

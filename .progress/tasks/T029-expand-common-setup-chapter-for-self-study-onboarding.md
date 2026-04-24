# T029 Expand common setup chapter for self-study onboarding

Last updated: 2026-04-23
Status: done
Priority: medium
Owner: codex
Due: 2026-04-23
Milestone: M008

## Background
- Chapter 00 had OS-specific installation commands, but the common section still stopped before the practical checks that self-study students usually need after installation.
- That made the onboarding chapter weak at the point where students transition from installation to actual work.

## Scope
- In: extending the common setup section in canonical LaTeX and Markdown with working-directory setup, virtual environments, package installation, first-run checks, Git config, clone flow, and basic troubleshooting.
- Out: institution-specific account administration beyond the existing GitHub onboarding notes.

## Deliverables
- Expanded common setup content in chapter 00 for both `.tex` and `.md`.
- Integrated LaTeX build verification after the additions.

## Acceptance Criteria
- [x] Chapter 00 now includes post-install common workflow rather than stopping at account setup.
- [x] The added content includes at least one concrete Python verification script.
- [x] `latexmk` succeeds after the update.

## Dependencies / Blockers
- Builds on the migrated LaTeX baseline and the continuing gap-filling pass after T028.

## Next Action
- owner: codex
- due: 2026-04-23
- action: Continue reviewing the shortest remaining chapters and prioritize whichever still lacks enough explanation or verification steps for independent study.

# LaTeX Source of Truth

- `main.tex`: integrated root document. Each chapter is included with `\input`.
- `preamble.tex`: shared packages, listing styles, and helper macros.
- `chapters/`: canonical chapter `.tex` files.
- `markdown/`: chapter Markdown kept simple so `pandoc` can consume it later.
- `figures/generated/`: generated figure files referenced from LaTeX and Markdown.
- `figures/scripts/`: Python replay scripts for regenerating selected figures.

Current state:
- `latex/` is the canonical source for the legacy textbook-style content.
- `main.tex` and `chapters/` are edited directly; the old notebook conversion pipeline is no longer part of this repository.
- Replay scripts remain only for chapters whose figures are regenerated from Python.

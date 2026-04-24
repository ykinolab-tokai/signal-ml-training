# Notebook Migration Audit

Date: 2026-04-22
Milestone: M007

## Scope
- Corpus: `text/*.ipynb`
- Output roots: `latex/chapters/`, `latex/markdown/`, `latex/figures/`

## Migration groups
- Empty notebooks: `13_Basics_of_Signal_Processing.ipynb`, `14_Basics_of_Spectrum_Analysis.ipynb`, `15_Basics_of_LTI_Systems.ipynb`
  These now map to intentionally empty `.tex` and `.md` files.
- Markdown-first chapters with local static assets: `01`, `02`, `04`, `06a`, `06b`, `21`, `23`
  The converter copies static assets out of `text/images/` into `latex/figures/source/`.
- Code-heavy chapters with embedded notebook figures: `03`, `08`
  These chapters also emit replay scripts under `latex/figures/scripts/`.
- Code-heavy chapters without embedded figures: `07`, `11`, `12`, `22`
  These convert mechanically after table, code, and output normalization fixes.
- Markdown-first chapters without special assets: `00`, `05`

## Non-blocking residuals
- `23` contains animated GIF assets and similar non-embeddable references that are kept as explicit asset references rather than inlined figures.
- `23` still emits one LaTeX float-size warning for a large figure, but the integrated build succeeds.
- Typewriter rendering of `\textbackslash{}` emits a font substitution warning in `02`, but it does not block the build.

## Tooling status
- The converter now handles empty notebooks, duplicate chapter numbers (`06a`, `06b`), sparse Markdown tables, chapter-local note labels, local fallback for URL-based images, and malformed display-math closing markers.
- `latex/main.pdf` builds successfully from the full converted corpus.
- The `uv` Python environment is available, so replay scripts can be run later with the managed dependencies.

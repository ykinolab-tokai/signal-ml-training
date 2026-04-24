# Agent Guidelines

<!-- Do not restructure or delete sections. Update individual values in-place when they change. -->

## Core Principles

- **Do NOT maintain backward compatibility** unless explicitly requested. Break things boldly.
- **Treat `README.md` as the source of truth** for seminar policy, roles, grading, schedule, and generative-AI rules.

---

## Project Overview

**Project type:** Seminar curriculum repository for image/audio signal processing and deep-learning implementation training  
**Primary language:** Markdown, LaTeX, Python 3  
**Key dependencies:** NumPy, matplotlib, pandas, scikit-learn, Pillow, soundfile, PyTorch, torchvision, torchaudio

---

## Commands

```bash
# Development
git status
git checkout main
git pull origin main

# Testing
# No automated test suite; re-run any edited `.py` script or Markdown example manually.

# Build
cd latex
latexmk -interaction=nonstopmode -halt-on-error main.tex
```

---

## Code Conventions

- Keep repository scope limited to math, signal processing, programming, PyTorch implementation, model understanding, code quality, and reusable implementation assets
- Do not add research design, research evaluation, writing guidance, or presentation guidance here
- When editing `README.md`, preserve the 28-session structure, grade-specific responsibilities, evaluation rules, standard time allocation, and generative-AI policy unless the user explicitly changes them

---

## Architecture

```text
README.md            canonical seminar policy and yearly structure
latex/               canonical LaTeX source for the legacy textbook-style material
materials/           student-facing handouts for spring/autumn sessions
templates/           handout templates for new or revised sessions
.progress/           task and milestone tracking for repository updates
```

---

## Maintenance Notes

<!-- This section is permanent. Do not delete. -->

**Keep this file lean and current:**

1. **Remove placeholder sections** once they are filled
2. **Review regularly** - stale instructions poison the agent's context
3. **Keep this file short** - detailed policy belongs in `README.md`
4. **Update commands immediately** when workflows change
5. **Rewrite Architecture section** when major structural changes occur
6. **Delete anything the agent can infer** from repository contents

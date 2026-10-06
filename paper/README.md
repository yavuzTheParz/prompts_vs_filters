# Paper: Coevolutionary Prompt-Injection Testing (EAAI submission draft)

- `main.tex` — manuscript (`elsarticle`, author–year, Engineering Applications of AI)
- `refs.bib` — proposal references (verbatim from the authors' list) plus five
  additional method/data references marked in the file; verify before submission
- `generated/` — numbers and table bodies, **generated; do not edit by hand**
- `figures/` — result figures (PDF), generated
- `scripts/make_paper_assets.py` — regenerates `generated/` and `figures/` from
  the repository artifacts (`outputs/`, `prompts/`, `tests/fixtures/`)
- `scripts/legacy_quality_constraints.py` — snapshot of the pre-fix quality gates
  from `main`, used only to reproduce the seed-pool audit

```bash
python3 -B paper/scripts/make_paper_assets.py   # from the repository root
cd paper && latexmk -pdf main.tex               # or upload paper/ to Overleaf
```

Every quantity the current campaign has not produced is typeset as
`[not measured]` via the `\nm` macro. Search for `[to be` and `[not measured]`
before submission.

# Paper: Content-Aware Visual Chunking in Screenshot RAG — A Reader–Retriever Tradeoff

LaTeX source for the paper. **Data discipline:** no number in `main.tex` is hand-typed — every figure,
table, and inline number is generated from the committed `results/*.json` by `gen_paper_numbers.py`
(which reuses the `research/analyze_chunking.py` primitives, so the paper and the analysis cannot
disagree).

> **Upstream / naming:** this pipeline builds on **`StarTrail-org/PixelRAG`** (the upstream
> screenshot-RAG system). This repository (`kaushikchaturvedula/PixelRAG`) is a distinct fork that
> shares the name; the content-aware chunker, hierarchical retrieve-then-expand, and end-to-end QA
> harness are additions of this work.

## Build

1. **Generate numbers, tables, and figures** (needs `matplotlib`; stdlib otherwise):
   ```bash
   pip install matplotlib            # if not already installed
   python paper/gen_paper_numbers.py
   ```
   This (re)writes `paper/numbers.tex`, `paper/tables/*.tex`, and `paper/figures/*.{pdf,png}`.

2. **Compile** (standard packages: `natbib`, `booktabs`, `graphicx`, `hyperref`, `authblk`):
   ```bash
   cd paper && latexmk -pdf main.tex          # or: pdflatex; bibtex; pdflatex; pdflatex
   ```
   Or upload the `paper/` folder to Overleaf and compile there.

> **Note:** `main.tex` compiles cleanly with `pdflatex`+`bibtex` (e.g. via Tectonic locally or on
> Overleaf) — 0 overfull/underfull boxes, 0 undefined references or citations. A generated `main.bbl`
> is committed, so the build works even without re-running BibTeX.

## Files
- `main.tex` — the paper (article class, single column).
- `gen_paper_numbers.py` — generator: JSON → `numbers.tex` + `tables/*.tex` + `figures/*`.
- `numbers.tex` — generated `\newcommand` macros (do not edit).
- `tables/*.tex` — generated `booktabs` tabulars (do not edit).
- `figures/*.{pdf,png}` — generated matplotlib figures.
- `refs.bib` — references. All 12 citations are verified against arXiv / official sources.

## Regenerating after data changes
`gen_paper_numbers.py` reads only `results/*.json`; re-run it to refresh every number. The paper stays in
sync with the data by construction.

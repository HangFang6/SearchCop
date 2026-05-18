# Paper Workspace

LaTeX source for the SearchCop paper.

## Layout

```
paper/
├── main.tex                      # entry; compile from here
├── references.bib                # bib database (verify all citations on Day 2-3)
├── sections/
│   ├── 01_introduction.tex       ✅ Day 1 (initial draft)
│   ├── 02_related_work.tex       ⏳ Day 10
│   ├── 03_method.tex             ⏳ Day 11
│   ├── 04_experiments.tex        ⏳ Day 12
│   └── 05_conclusion.tex         ⏳ Day 13
├── figures/                      # PDF/SVG figures for camera-ready
│   └── (architecture.pdf, …)     ⏳ Day 11
└── README.md                     # this file
```

## Build

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex && pdflatex main.tex
```

## Switching to the official CVPR template

1. Download `cvpr.sty` from the CVPR template release.
2. Copy `cvpr.sty` and `ieee_fullname.bst` into this folder.
3. Replace the preamble at the top of `main.tex` per the comments.
4. Replace anonymous author block with the camera-ready author list.

## Macros (defined in main.tex)

- `\method` -> renders **SearchCop** in small-caps consistently
- `\todo{...}` -> red brackets for things to fill from experiments
- `\note{...}` -> blue brackets for internal notes (remove before submission)

## Convention

- One `.tex` file per section; `\input` from `main.tex`.
- Figures live in `figures/`; reference them with `\includegraphics{figures/foo}`.
- All BibTeX keys lowercase and `firstauthor` + `year` + `firstword`
  (e.g. `yao2023react`, `davila2023mevid`).
- Any number that comes from experiments goes inside `\todo{...}` until
  the experiment runs are finalized; this prevents accidentally shipping
  fabricated numbers.

## Status

| Section | Status | Owner | Notes |
|---|---|---|---|
| Abstract           | draft  | -    | numbers stubbed via \todo  |
| 1. Introduction    | draft  | Codebuddy | 5-paragraph CVPR layout, 5 contributions |
| 2. Related Work    | TBD    | Researcher Agent | seed material in `docs/related_work.md` |
| 3. Method          | TBD    | Architect Agent  | seed material in `docs/architecture.md`  |
| 4. Experiments     | TBD    | Experiment Agent | depends on results/summary/all_runs.xlsx |
| 5. Conclusion      | TBD    | Writer Agent | last |

# PySofra

**The missing statistical reporting layer for Python.**

PySofra transforms datasets, statistical model outputs, and summary
statistics into publication-ready tables across HTML, Markdown, DOCX,
LaTeX, and PPTX, with minimal friction. It brings the practical
workflows of R's `tableone`, `gtsummary`, and `flextable` into a
single coherent Pythonic API.

```python
import pandas as pd
import pysofra as ps

df = pd.read_csv("trial.csv")

(
    ps.tbl_one(df, by="arm")
      .add_p()
      .add_smd()
      .add_overall()
      .theme("clinical")
      .to_docx("table1.docx")
)
```

## Why PySofra?

Python already has the statistical computation — pandas, scipy,
statsmodels, lifelines, scikit-learn. What's been missing is the
**reporting ergonomics layer** that turns a fitted model or a stratified
summary into a manuscript-ready, journal-styled artifact. PySofra fills
that gap.

## Highlights

- **One object, every format.** `SofraTable` renders in Jupyter via
  `_repr_html_`, exports to DOCX / LaTeX / PPTX / HTML / Markdown, and
  carries captions, footnotes, and themes through to each backend.
- **Sensible statistical defaults.** Welch / Wilcoxon / ANOVA / Kruskal
  / χ² / Fisher are auto-selected per variable, with per-variable
  overrides (`tests={'age': 'wilcoxon'}`) when you need them.
- **Multi-model regression.** Pass a single fit or a list — statsmodels,
  lifelines, sklearn all work.
- **Multiplicity adjustment.** `.add_q()` adds Benjamini–Hochberg /
  Bonferroni / Holm / Hommel / Šidák / BY q-values.
- **No nonsense.** No metaprogramming, no telemetry, no network calls,
  deterministic by construction.

## Status

PySofra is in **alpha** (`0.1.0a1`). The MVP scope (Table 1, summary,
regression, composition, HTML/MD/DOCX/LaTeX/PPTX export, themes) is
stable; internal modules may change before `1.0`.

See [Quickstart](quickstart.md) to install and run your first table.

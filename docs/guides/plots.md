# Inline plots

PySofra can attach a matplotlib-rendered plot to a `SofraTable` and
serialise it once into **SVG + PNG + PDF**, so each renderer picks the
form it supports:

| Renderer | Plot format used |
|---|---|
| `to_html()` | inline `<svg>` (rendered above or below the table) |
| `to_docx()` | embedded PNG (via `python-docx`'s `add_picture`) |
| `to_pptx()` | embedded PNG (placed on the slide alongside the table) |
| `to_latex()` | embedded PDF (written as a sidecar file referenced by `\includegraphics`) |
| `to_xlsx()` | not embedded (Excel users typically prefer numeric data) |
| `to_image()` | not embedded (the table-as-image path is itself a PNG) |

All three byte streams are post-processed so that two renders of the
same plot are **byte-identical** — matplotlib's wall-clock timestamps,
process-randomised DOM IDs, and PDF creation dates are stripped so
publication artifacts remain reproducible across processes and CI runs.

## Forest plot for regression tables

```python
ps.tbl_regression(fit, exponentiate=True).with_forest_plot()
```

Reads point estimates and CI bounds from the displayed cells, so the
plot is always consistent with the table. For multi-model regression
tables, attach a forest plot per model with `.with_forest_plot()` after
slicing or call `.with_forest_plot()` on the merged table once.

Knobs:

| Parameter | Effect |
|---|---|
| `log_x=True` (default) | Log-scale x-axis (natural for ORs / HRs / IRRs) |
| `null_line=1.0` (default) | Reference vertical line |
| `position='above'` / `'below'` | Plot placement relative to the table |
| `color='#0b3d91'` | Point + segment colour |
| `width_in=6.5`, `height_per_row_in=0.42` | Sizing |

## Kaplan–Meier curves

```python
ps.tbl_survival(df, time='time', event='event', by='arm',
                times=[12, 24]).with_km_plot()
```

The KM curves are fit from the original data (the survival table
preserves a reference) so they match exactly the medians and survival
probabilities shown in the body rows.

Knobs:

| Parameter | Effect |
|---|---|
| `ci=True` (default) | Shaded confidence bands |
| `palette=['#0b3d91', '#c1272d', ...]` | Per-group colours |
| `xlabel='Time'`, `ylabel='Survival probability'` | Axis labels |
| `width_in=6.5`, `height_in=4.0` | Figure size |
| `position='above'` / `'below'` | Plot placement |

## Custom SVGs

For any custom matplotlib output, render to SVG yourself and attach:

```python
import io, matplotlib.pyplot as plt

fig, ax = plt.subplots()
# ... draw whatever ...
buf = io.StringIO()
fig.savefig(buf, format='svg', bbox_inches='tight')
my_svg = buf.getvalue()

table.with_inline_svg(my_svg, position='above')
```

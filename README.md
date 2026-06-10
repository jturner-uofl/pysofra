<div align="center">

# PySofra

### The missing statistical reporting layer for Python

[![CI](https://github.com/jturner-uofl/pysofra/actions/workflows/tests.yml/badge.svg)](https://github.com/jturner-uofl/pysofra/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://jturner-uofl.github.io/pysofra/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1036%20passing-brightgreen.svg)](#status)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#status)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)

</div>

> R has `gtsummary`, `tableone`, and `flextable`. Python doesn't — or didn't.
> PySofra turns datasets, fitted models, and summary statistics into
> **publication-ready tables** across HTML · Markdown · LaTeX · DOCX ·
> PPTX · XLSX · PNG from a single immutable object.

<div align="center">
  <img src="https://raw.githubusercontent.com/jturner-uofl/pysofra/main/assets/readme/table_one.png"
       alt="Baseline characteristics table by treatment arm — clinical theme, p-values, SMDs, Overall column"
       width="820">
  <br>
  <sub><em>Baseline-characteristics Table 1, by treatment arm. JAMA theme. <strong>One call.</strong></em></sub>
</div>

---

## Numerically verified — not just tested

Most reporting packages test that code runs. PySofra tests that **numbers are correct**.

A 54-step audit notebook downloads the 2017–18 US National Health and Nutrition
Examination Survey directly from the CDC, fits models, and asserts **52 numerical
contracts** against independent R reference implementations:

| What | Reference | Tolerance | Observed |
|------|-----------|-----------|----------|
| Weighted mean (5 variables) | R `svymean` | ≤ 10⁻⁹ rel | **< 10⁻¹⁴** |
| Weighted mean — Table 1 cell | `gtsummary` display | ≤ 10⁻⁹ rel | **3.3 × 10⁻¹⁵** |
| Weighted SD — Table 1 cell | `gtsummary` display | ≤ 10⁻⁹ rel | **3.8 × 10⁻¹⁵** |
| Weighted proportion (4 vars) | `gtsummary` display | ≤ 10⁻⁹ rel | **4.6 × 10⁻¹⁵** |
| Survey regression SE (6 coefs) | R `svyglm` | ≤ 1% rel | **< 0.8%** |
| MI pooled β̂ | Rubin (1987) | ≤ 10⁻¹⁰ abs | **< 10⁻¹⁴** |
| KM survival probability | lifelines | ≤ 10⁻¹² abs | **< 10⁻¹⁵** |
| Wilson CI endpoint | Newcombe (1998) | ≤ 10⁻⁹ abs | **< 10⁻¹⁵** |

Weighted means and SDs agree with R at **floating-point machine precision** —
a few multiples of ε ≈ 2.2 × 10⁻¹⁶. That's not approximation; it's the same formula.

Nominal 95% CIs from survey-weighted logistic regression attain **94.2% and 93.8%
empirical coverage** in a 1,000-replicate Monte Carlo study with known truth.

Every contract runs in CI on every push.
[→ Read the full audit](https://jturner-uofl.github.io/pysofra/notebooks/jss_case_study.html) ·
[→ Reproduce it yourself](AUDITOR.md)

---

## Quick start

```bash
pip install "pysofra[all]"
```

```python
import pysofra as ps

# Table 1 — baseline characteristics by treatment arm
tbl = (
    ps.tbl_one(df, by="arm",
               labels={"age": "Age (years)", "bmi": "BMI (kg/m²)"},
               nonnormal=["bmi"])
      .add_p()
      .add_smd()
      .add_overall()
      .theme("clinical")
)

tbl                          # renders in Jupyter / VS Code / Colab
tbl.to_docx("table1.docx")   # publication-quality Word
tbl.to_latex()               # LaTeX fragment
tbl.to_html()                # standalone HTML
```

```python
# Regression table with inline forest plot
import statsmodels.api as sm

fit = sm.Logit(df["event"], sm.add_constant(df[["age", "bmi"]])).fit(disp=False)

(
    ps.tbl_regression(fit, exponentiate=True)
      .with_forest_plot()
      .bold_p()
      .theme("jama")
      .to_docx("table2.docx")
)
```

```python
# Survey-weighted Table 1 (NHANES-style design)
design = ps.SurveyDesign(weights="WTMEC2YR", strata="SDMVSTRA", cluster="SDMVPSU")

(
    ps.tbl_one(df, by="diabetes", design=design)
      .add_p()
      .add_overall()
      .theme("clinical")
      .to_docx("table_nhanes.docx")
)
```

```python
# Multiple-imputation pooling (Rubin's rules)
pooled = ps.pool(fits)   # list of per-imputation model fits

(
    ps.tbl_regression(pooled, exponentiate=True, intercept=False)
      .set_caption(f"Pooled ORs (m = {len(fits)}, Rubin's rules)")
      .to_docx("table_mi.docx")
)
```

---

## Why PySofra

| | R ecosystem | PySofra | Python alternatives |
|---|---|---|---|
| **Table 1 (auto stats)** | `tableone`, `gtsummary` | ✅ | `tableone` (partial) |
| **Regression table** | `gtsummary` | ✅ | — |
| **Survival (KM) summary** | `gtsummary` | ✅ | — |
| **Survey-weighted Table 1** | `gtsummary` + `survey` | ✅ | — |
| **MI pooling built-in** | manual `mice` coordination | ✅ | — |
| **Word + LaTeX + 5 more formats** | separate packages | ✅ | — |
| **Byte-deterministic output** | — | ✅ | — |
| **Safety diagnostics in table** | — | ✅ | — |
| **Machine-precision validation** | — | ✅ | — |

**One immutable object, seven output formats.** Build a `SofraTable` once; render to
HTML, Markdown, LaTeX, DOCX, PPTX, XLSX, or PNG. Output is byte-identical across
processes — a hard requirement for reproducible manuscript artefacts tracked in git.

**Typed-value cells.** Every cell stores both a Python `float` *and* a rendered
string. `bold_p(threshold=0.05)` compares the float directly — it works correctly
whether the display reads `"0.032"`, `"0.03"`, or `"<0.001"`. No fragile string
parsing at threshold-formatted values.

**Auto-dispatched tests.** Welch's *t*, ANOVA, Wilcoxon, Kruskal–Wallis, Fisher,
χ², Taylor-linearised design-adjusted *t*, first-order Rao–Scott χ² — picked per
row by variable kind, overridable per variable.

**Publication-safety diagnostics.** `with_safety_warnings()` appends footnotes
for separation in logistic regression, PH-assumption violations, and sparse
contingency cells — directly into the rendered DOCX, HTML, and LaTeX, not just
a console warning that disappears in batch output.

---

## Outputs

<table>
<tr>
<td width="50%" valign="top" align="center">
  <img src="https://raw.githubusercontent.com/jturner-uofl/pysofra/main/assets/readme/regression_forest.png"
       alt="Adjusted odds ratios with inline forest plot" width="100%">
  <br>
  <sub><em>Adjusted ORs + inline forest plot</em><br><code>tbl_regression(fit).with_forest_plot()</code></sub>
</td>
<td width="50%" valign="top" align="center">
  <img src="https://raw.githubusercontent.com/jturner-uofl/pysofra/main/assets/readme/survival_km.png"
       alt="Kaplan–Meier survival table with embedded KM curve" width="100%">
  <br>
  <sub><em>KM table + inline survival curve</em><br><code>tbl_survival(...).with_km_plot()</code></sub>
</td>
</tr>
</table>

---

## What's in the box

**Builders**

| Builder | What it produces |
|---------|-----------------|
| `ps.tbl_one(df, by=)` | Baseline-characteristics table stratified by group |
| `ps.tbl_summary(df)` | Descriptive summary (no stratification) |
| `ps.tbl_cross(df, row=, col=)` | Two-way cross-tabulation |
| `ps.tbl_regression(fit)` | Regression results — statsmodels, lifelines, sklearn |
| `ps.tbl_uvregression(df, outcome=, predictors=)` | Univariable regression panel |
| `ps.tbl_survival(df, time=, event=, by=)` | Kaplan–Meier summary + optional curves |

**Modifiers** (each returns a new `SofraTable`)

`add_p()` · `add_q()` · `add_smd()` · `add_overall()` · `add_difference()` · `add_ci()` ·
`bold_p()` · `bold_if()` · `highlight_if()` · `theme()` · `set_caption()` · `with_footnotes()` ·
`with_forest_plot()` · `with_km_plot()` · `with_safety_warnings()`

**Composition**

`tbl_merge()` — horizontal merge with spanning headers  
`tbl_stack()` — vertical stack with group-header rows

**Survey & MI**

`SurveyDesign(weights=, strata=, cluster=, fpc=)` — pass to any builder  
`pool(fits)` — Rubin's rules MI pooling; result accepted by `tbl_regression`

**Export**

`.to_html()` · `.to_markdown()` · `.to_latex()` · `.to_docx()` · `.to_pptx()` ·
`.to_xlsx()` · `.to_image()` — all byte-deterministic across processes

**Statistical methods**

Welch *t* · Wilcoxon · ANOVA · Kruskal–Wallis · Fisher · χ² · Rao–Scott ·
Taylor-linearised design *t* · Wilson CI · Newcombe CI · Cohen *d* ·
Hedges *g* · Cramér *V* · η² · ω² · Kaplan–Meier · log-rank ·
Cox PH · Rubin's rules MI · BH · BY · Bonferroni · Holm · Hommel · Šidák

---

## Installation

```bash
pip install pysofra                   # core (numpy, pandas, scipy, statsmodels, python-docx)
pip install "pysofra[survival]"       # + lifelines, matplotlib
pip install "pysofra[plot]"           # + forest plots, table-as-image
pip install "pysofra[pptx]"           # + PowerPoint export
pip install "pysofra[xlsx]"           # + Excel export
pip install "pysofra[polars]"         # + polars DataFrame input
pip install "pysofra[sklearn]"        # + scikit-learn model support
pip install "pysofra[all]"            # everything
```

Requires Python ≥ 3.11.

---

## Documentation

**[→ Full documentation](https://jturner-uofl.github.io/pysofra/)**

- [Quickstart](https://jturner-uofl.github.io/pysofra/quickstart/)
- [Guides](https://jturner-uofl.github.io/pysofra/guides/tbl_one/) — tbl_one, tbl_regression, survey weights, MI pooling, themes, exports
- [API reference](https://jturner-uofl.github.io/pysofra/api/builders/)
- [NHANES validation notebook](https://jturner-uofl.github.io/pysofra/notebooks/jss_case_study.html) — 52 numerical contracts, live results

---

## Status

**Version 0.1.0 — first stable release.**

The public API is pinned by an explicit
[API-stability test](tests/test_api_stability.py) — any unintended rename,
removal, or signature change surfaces as a failed test before merge. The full
deprecation policy is in [Concepts → API stability](docs/concepts/stability.md).

Quality bar at this release:

- **1,036 tests** passing on Python 3.11 and 3.12 (Ubuntu + macOS), 100% line coverage
- **52 numerical contracts** against R `survey`, `gtsummary`, `lifelines`, `scipy`,
  `statsmodels`, and textbook formulas — run in CI on every push
- **Property-based tests** (Hypothesis) enforce universal invariants across 720
  randomised examples per run: *p*-value ∈ [0, 1], ordered CI bounds, percent ∈ [0, 100]
- **Byte-deterministic** renderer output — identical input → identical bytes, across
  processes, required for reproducible git-tracked manuscript artefacts
- **mypy strict** · **ruff clean** · public signatures fully type-annotated

---

## Independently verifying PySofra

Reviewers, sceptics, and downstream users: see [**AUDITOR.md**](AUDITOR.md) for
the single-command recipe that downloads NHANES from the CDC, runs the full
54-step audit notebook, and prints:

```
AUDIT COMPLETE — 52/52 contracts passed | pysofra 0.1.0 | <UTC>
```

The [pre-executed notebook](https://jturner-uofl.github.io/pysofra/notebooks/jss_case_study.html)
is available for read-only review without installing anything.

---

## Citation

If you use PySofra in academic work, cite the software via the
[`CITATION.cff`](CITATION.cff) metadata (GitHub shows a **Cite this repository**
button in the right-hand sidebar). A full-length methods paper is in preparation.

---

## Contributing

Bug reports, feature requests, and pull requests are welcome.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow and quality gates, and
the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).

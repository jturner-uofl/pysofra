# Changelog

All notable changes to PySofra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a5] — 2026-05-25

### Fixed
- **`svyttest` degrees of freedom** now follow the standard survey
  convention `n_PSU − n_strata` (matching Stata `svy: ttest` and R
  `survey::svyttest` with `nest=TRUE`), instead of `N − n_strata`.
- **AFT models (Weibull / LogNormal / LogLogistic) are now labelled
  "TR" (Time Ratio)** instead of "HR".
- **Lifelines regression CIs honour the user-supplied `conf_level`**
  (previously the CIs reflected the model's fit-time `alpha`).
- **SMDs on a weighted Table 1 are now weighted**.
- **`add_ci`, `add_difference`, and `add_global_p` now honour weights**.

### Added
- `conf_level` range validation in `tbl_regression`, `tbl_survival`,
  and `pool`.
- `with_forest_plot()` on a multi-model regression table emits a
  `UserWarning` noting that only the first model is visualised.

## [0.1.0a4] — 2026-05-25

### Added
- Input validation for duplicate names in `variables=` (now raises
  `ValueError` instead of silently accepting duplicates).
- Confidence-level range check in `.add_ci()` and related modifiers
  (must lie in `(0, 1)`).

### Changed
- Renamed several test files for clarity. No public API changes.

## [0.1.0a3] — 2026-05-24

### Changed
- Documentation polish across README, changelog, and inline docstrings.
  No public API or behavioural changes.

## [0.1.0a2] — 2026-05-23

### Fixed
- Theme styling now survives notebook viewers that strip `<style>` blocks
  (e.g. GitHub's notebook viewer). Critical theme properties (font, border,
  padding) are emitted as inline `style` attributes on each table element, so
  `jama` vs `nejm` vs `clinical` vs `minimal` stay visibly distinct everywhere.
- README image and link URLs are now absolute so they render on PyPI.

## [0.1.0a1] — 2026-05-20

### Added

- Initial alpha release.
- Core `SofraTable` object with immutable method chaining.
- `tbl_one()` — baseline characteristic tables (Table 1) with continuous /
  categorical summaries, stratification, missing data summaries, overall
  column, p-values, and standardized mean differences (SMDs).
- `tbl_summary()` — general descriptive summary tables with grouping and
  configurable statistics.
- `tbl_regression()` — regression tables for `statsmodels` linear / logistic
  / Poisson models, with confidence intervals, exponentiation, and p-values.
- `tbl_uvregression()` — univariable regression wrapper: one model per
  predictor stacked into a single table, with optional `adjust_for=`
  covariates.
- `tbl_cross()` — explicit cross-tabulation with selectable cell content.
- `tbl_survival()` — Kaplan–Meier summary table with N total / events /
  censored, median survival, and survival probability at user-supplied time
  points. Requires `lifelines`.
- `tbl_merge()` / `tbl_stack()` — side-by-side and vertical table composition.
- HTML renderer with rich notebook `_repr_html_` output (dark-mode aware,
  responsive, sticky headers).
- Markdown renderer with GFM-compatible output.
- DOCX renderer via `python-docx` (publication-quality Word tables with
  captions, footnotes, merged spanning headers).
- LaTeX renderer (`.to_latex`, `.to_latex_file`) producing `booktabs` tables.
- PowerPoint renderer (`.to_pptx`).
- Excel renderer (`.to_xlsx`) with full style preservation.
- PNG renderer (`.to_image`) via matplotlib.
- Forest plots (`with_forest_plot()`) and Kaplan–Meier curves (`with_km_plot()`)
  embeddable inline across every backend.
- Conditional row formatting: `.bold_if`, `.highlight_if`, `.style_if`.
- Themes: `clinical`, `compact`, `jama`, `nejm`, `minimal`, plus user-defined
  themes via `register_theme`.
- Automatic statistical test selection (Welch's t, Wilcoxon, ANOVA, Kruskal–
  Wallis, Fisher's exact, chi-square, Rao–Scott chi-square) with per-variable
  override hooks.
- Multiplicity adjustment (`.add_q`): BH, BY, Bonferroni, Holm, Hommel, Šidák.
- Survey-weighted Table 1: `tbl_one(weights='col')` and `SurveyDesign` with
  strata, cluster, finite-population correction, replicate weights.
- Effect-size helpers: Cohen's *d*, Hedges' *g*, Cramér's *V*, η², ω², φ.
- Multiple-imputation pooling (`ps.pool`) via Rubin's rules.
- `polars` input support (DataFrames and LazyFrames).
- `lifelines` and `scikit-learn` model integration for `tbl_regression`.
- Snapshot tests for renderer output and property-based invariant testing via
  Hypothesis.

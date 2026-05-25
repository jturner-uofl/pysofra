# Changelog

All notable changes to PySofra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a6] — 2026-05-26

### Fixed
- **`svyttest` now uses full-design Taylor linearisation** of the
  regression coefficient `ȳ_B − ȳ_A` instead of summing per-group
  variances in quadrature. The new formulation accounts for
  cross-group covariance under the survey design. Pinned against
  R `survey::svyttest`: identical t-statistic and df, p-value
  agreement to 7 decimal places on the test fixture. The previous
  per-group formulation could be wildly anti-conservative when
  clusters straddled groups.
- **`svyttest` degrees of freedom** corrected to `n_PSU − n_strata − 1`
  (the design df minus one for the slope parameter). Previously
  off by one.
- **`rao_scott_chisq` normalises weights to `Σw = n` before computing
  the chi-square statistic**, matching R `survey::svychisq`. The
  previous formulation produced statistics that scaled linearly with
  the absolute magnitude of the weights and disagreed with R by
  ~10–15% on typical survey-weighted contingency tables.
- **`tbl_one(..., weights=...)` raises on negative or all-zero
  weights** instead of warning and silently dropping. The earlier
  behaviour could leave `N = -1` or `N = 0` cells in the rendered
  table.
- **`tbl_one(...).add_p()` now emits a UserWarning** when falling
  back to unweighted ANOVA / Kruskal–Wallis for >2-group
  continuous variables under weights (design-adjusted multi-group
  test is not yet implemented).
- **`tbl_one(...).add_global_p()` warns** when the table already
  carries a column added by a prior modifier (`add_difference`,
  `add_significance_stars`); the rebuild path drops such columns
  and the user should call `add_global_p()` first.

## [0.1.0a5] — 2026-05-25

### Fixed
- **`svyttest` degrees of freedom** now follow the standard survey
  convention `n_PSU − n_strata` (matching Stata `svy: ttest` and R
  `survey::svyttest` with `nest=TRUE`), instead of `N − n_strata`. The
  previous formula over-stated df dramatically under clustering and
  produced anti-conservative p-values.
- **AFT models (Weibull / LogNormal / LogLogistic) are now labelled
  "TR" (Time Ratio)** instead of "HR". The two parameters point in
  opposite directions (TR > 1 → longer survival; HR > 1 → shorter
  survival), so the mislabel was potentially misleading.
- **Lifelines regression CIs honour the user-supplied `conf_level`**.
  Previously the CIs reflected the model's fit-time `alpha` regardless
  of `conf_level`, so passing `conf_level=0.90` produced a "90% CI"
  header with 95% CI numbers. The CI is now re-derived from `coef ±
  z·se(coef)` at the requested level.
- **SMDs on a weighted Table 1 are now weighted**. `continuous_smd` and
  `categorical_smd` accept a `weights=` argument; `tbl_one(..., weights=)`
  threads it through automatically. Previously the SMD column was
  always computed on unweighted samples even on a weighted table.
- **`add_ci`, `add_difference`, and `add_global_p` now honour weights**.
  The Welch CI on continuous means, the Newcombe CI on proportion
  differences, and the joint Wald-F test for `add_global_p` all use
  weighted means / variances / proportions (with Kish's effective
  sample size for SEs) when the table was built with `weights=`.

### Added
- `conf_level` range validation in `tbl_regression`, `tbl_survival`, and
  `pool` (raises `ValueError` for values outside `(0, 1)`).
- `with_forest_plot()` on a multi-model regression table now emits a
  `UserWarning` that only the first model is visualised, so the
  presence of additional models is no longer silent.

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
- `tbl_merge()` / `tbl_stack()` — table composition.
- HTML renderer with rich notebook `_repr_html_` output (dark-mode aware,
  responsive, sticky headers).
- Markdown renderer.
- DOCX renderer via `python-docx` (publication-quality Word tables with
  captions, footnotes, merged spanning headers).
- Themes: `clinical`, `compact`, `jama`, `nejm`, `minimal`.
- Automatic statistical test selection with override hooks.
- Snapshot tests for HTML output.

# Changelog

All notable changes to PySofra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a8] — 2026-05-26

### Fixed
- **`SurveyDesign(fpc=...)` now actually applies the finite-population
  correction when no strata are given.** Previously the FPC was
  silently dropped in the unstratified branch of ``design_mean_var``;
  a user who configured ``SurveyDesign(weights=..., fpc=...)`` got the
  same variance as if the FPC had been omitted.
- **Lonely PSU warning.** A stratum (or the whole design) containing
  only one cluster now emits a clear ``UserWarning`` from
  ``design_mean_var``. R's ``survey`` package errors by default in
  this case because the cluster-robust variance is undefined; pysofra
  warns and contributes zero so the rest of the table still renders.
- **Stratified variance centring.** The Taylor-linearised variance for
  the stratified-unclustered case now centres residuals on the
  stratum-specific mean rather than on zero, matching the textbook
  formula and R ``survey::svyrecvar``. The previous form
  systematically inflated variance whenever the influence-function
  mean drifted off zero within a stratum.
- **Multiple-imputation pool uses direct SE** (when available) instead
  of back-deriving from the CI half-width with a z-pivot. Statsmodels
  CIs are t-based; dividing the half-width by ``z`` over-stated SE.
  The fix stores SE on ``ModelSummary`` from extractors that expose it
  (statsmodels ``bse``, lifelines ``se(coef)``) and falls back to the
  z-pivot only for fitters that don't (sklearn).
- **`tbl_regression.add_p()` is genuinely a no-op** on tbl_regression
  tables (previously raised ``RuntimeError`` despite docstring
  promising a no-op).
- **Forest-plot scale auto-detection.** ``with_forest_plot()`` now
  picks ``log_x``/``null_line`` from the table's coefficient column
  header — exponentiated families (OR/HR/TR/IRR/RR) get log-scale at
  null=1; raw-coefficient families (β/Estimate) get linear at null=0.
  Explicit kwargs still override.
- **HTML link scheme allowlist.** ``CellPart(link=...)`` URLs are now
  filtered against an allowlist of safe schemes (http/https/mailto/
  ftp/ftps, plus relative paths and fragments); ``javascript:`` and
  ``data:`` are replaced with ``about:blank``. Closes an XSS vector
  for tables built from untrusted input.
- **`_refit_with_design` raises on invalid weights** instead of
  silently zeroing rows (matches the existing ``tbl_one(weights=)``
  policy).
- **`tbl_regression(design=)` uses `var_weights`** rather than
  ``freq_weights`` when refitting under a survey design. The
  ``freq_weights`` convention scales ``df_resid`` by ``Σw`` (treating
  the weight as an integer count of repeats), inflating effective N
  on non-integer sampling weights. ``var_weights`` keeps
  ``df_resid = n − k``, matching R ``survey::svyglm`` to first order.

### Added
- **`tbl_survival(weights=)`**: lifelines' weighted KM is now exposed.
  Weighted N totals / events / censored are reported as weighted sums;
  the log-rank test is unweighted (lifelines doesn't expose a
  weighted log-rank) and a footnote flags this when weights are
  active.
- **Rebuild-drop warning extended to every spec-changing modifier**
  (was only `add_global_p`). Calling any of ``.add_p``, ``.add_smd``,
  ``.add_overall``, ``.add_n``, ``.add_q``, ``.add_global_p`` after a
  column-adding modifier (``add_difference``, ``add_ci``,
  ``add_significance_stars``) now emits a ``UserWarning`` naming the
  dropped columns and recommending the right call order.

### Changed
- ``_weighted_mean_var`` docstring now explicitly distinguishes
  frequency / reliability weights (where ``Σw − 1`` is appropriate)
  from sampling weights with large ``Σw`` (where ``SurveyDesign``
  should be used instead).
- ``_median_ci`` no longer accepts a redundant ``conf_level`` argument
  (the level is fixed at KMF fit time).

## [0.1.0a7] — 2026-05-26

### Fixed
- **`tbl_survival` validates `time` and `event` content**: negative
  survival times raise `ValueError`; non-`0/1` event codes raise
  `ValueError`. Previously these were passed silently to lifelines,
  which would either clamp negative times to zero or treat any
  nonzero event value as a death — producing a misleading curve
  without complaint.
- **`add_global_p()` on weighted `tbl_one`** now uses
  ``statsmodels.GLM(..., var_weights=w)`` instead of
  ``freq_weights=w``. For non-integer sampling weights ``freq_weights``
  scales ``df_resid`` by ``Σw`` (treating the weight as an integer
  count of repeats), which inflates the effective sample size and
  produces anti-conservative p-values. ``var_weights`` keeps
  ``df_resid = n − k`` — the appropriate SRS-weighted Wald-F
  convention. For full design-based inference (with strata or
  clusters) use ``ps.SurveyDesign`` end-to-end.

### Changed
- **`rao_scott_chisq` docstring** now honestly states a 10–15%
  typical disagreement with R ``survey::svychisq`` on non-trivial
  weighted designs (was: an overoptimistic "~5%"). The first-order
  Kish-DEFF approximation is unchanged; for design-grade chi-square
  inference call R directly.
- **Added published-reference citations** to public statistical
  functions: Welch / Satterthwaite, Wilcoxon (Mann-Whitney 1947),
  Kruskal-Wallis (1952), Fisher (1922), Pearson chi-square (1900),
  Wilson score (1927), Rao-Scott (1981/1984), Kish (1965),
  Benjamini-Hochberg (1995), Benjamini-Yekutieli (2001), Holm
  (1979), Hommel (1988), Šidák (1967), Binder (1983) Taylor
  linearisation.
- **`pool` and `cohen_d` docstrings** now have NumPy-style
  ``Parameters`` / ``Returns`` / ``References`` sections matching
  the other public functions.

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

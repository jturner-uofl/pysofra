# Changelog

All notable changes to PySofra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a16] — 2026-05-29

### Added — maturity contracts
- API stability test now also pins behavioural contracts: builders
  return `SofraTable`; modifiers are copy-on-write; every public
  symbol carries a docstring; a representative build emits zero
  pysofra-originated `DeprecationWarning`. New written policy at
  `docs/concepts/stability.md` (incl. post-1.0 deprecation ladder).
- New `tests/test_cross_backend_consistency.py` pins the
  one-spec → many-renderers numeric-payload preservation contract
  (HTML/LaTeX/Typst/Markdown) and the typed-value-not-string
  modifier contract.
- `tbl_regression()` now attaches an explicit "point estimates only"
  footnote when the source fitter is scikit-learn (new
  `ModelSummary.inference_unavailable` field). New written
  `docs/concepts/limitations.md` consolidates all three documented
  approximations / gaps.

### Notebook
- New Section X (Steps 49–53) — maturity contracts pinned inside
  the notebook itself: API surface manifest, cross-backend
  consistency, typed-value provenance, declarative-vs-pandas
  error-surface comparison, and disciplined limitations footnotes.

## [0.1.0a15] — 2026-05-28

### Fixed — honest CIs for weighted survival
- `tbl_survival(weights=)` warns once (and adds a footnote) when given
  non-integer weights: the KM point estimates are unbiased, but the
  Greenwood-variance CIs are biased (too narrow) under sampling /
  propensity / IPTW weights. Integer frequency weights stay silent
  (Greenwood is exact). PySofra now silences lifelines' raw per-fit
  `StatisticalWarning` and surfaces its own table-level advisory.

### Notebook
- Step 27 (weighted KM) pins the CI-bias warning + footnote as a
  contract; Step 16 (`add_q`) suppresses an incidental Rao–Scott
  re-advisory so the executed notebook stays warning-clean.

## [0.1.0a14] — 2026-05-28

### Fixed — design-based regression SEs (biggest limitation closed)
- `tbl_regression(design=)` now uses a full Taylor-linearisation
  cluster-robust sandwich (`survey_glm_vcov`) and matches R
  `survey::svyglm` on β, SE, and p-value to numerical precision
  (NHANES: 0.0 % SE gap, ≤ 0.04 % p gap). Monte-Carlo CI coverage
  rises from ~85 % (var_weights) to ~95 %.
- New `survey_glm_vcov()` + `SurveyGLMResults` wrapper; t-df =
  (n_PSU − n_strata) − k + 1, matching R svyglm.
- Gaussian-family GLM coefficients now labelled β (linear model).

### Added
- `tests/test_survey_glm_vcov.py` (8 tests). Suite: 1019 passing.

### Notebook
- Step 39 now asserts SE ≤ 1 % / p ≤ 2 % vs R svyglm; Step 47 asserts
  ~95 % CI coverage. Only the first-order Rao–Scott chi-square (Step
  38) remains an open documented approximation.

## [0.1.0a13] — 2026-05-28

### Fixed — rendering hygiene
- Removed CSS `color-mix()` from HTML output (GitHub's .ipynb
  renderer mangled it, leaking raw CSS into cells); now uses
  `rgba()` greys that render everywhere.
- LaTeX now escapes `<` / `>` → `\textless` / `\textgreater`.

### Added
- `tests/test_render_no_css_leak.py` (11 tests) — CI guard against
  renderer-hostile CSS and stray markup in HTML/LaTeX, all themes.
- Step 38 links rendered Table-1 p-values to first-order Rao-Scott
  (1e-9) and documents the 57–69 % gap vs R `svychisq` (race p
  flips 0.62 → 0.023). Step 45 gains an explicit MI≠survey-design
  warning. Step 46 expands to 5 outcome definitions + a fasting-
  subsample-weight (`WTSAF2YR`) audit.

Full pytest suite: 1011 passing.

## [0.1.0a12] — 2026-05-27

### Notebook — Section IX (inferential validity)
- Step 47 — Monte Carlo coverage of `tbl_regression(design=)` 95 % CI:
  ~84–86 % empirical at nominal 95 % (≈10 pp under-coverage,
  documented). Quantifies the inferential consequence of the
  Step-39 SE gap.
- Step 48 — exponentiated-CI asymmetry guard: matches
  `exp(β ± z·SE)` to ≤ 1e-9.

Notebook now has 9 sections, 48 contracts, 107 cells; full pytest
suite 1000 passing.

## [0.1.0a11] — 2026-05-27

### Documentation honesty (external-audit response)
- Rao–Scott docstring corrected: NHANES measurement shows 57 %
  median, 69 % max gap with R `svychisq` (was claimed as 10–15 %).
- `tbl_regression(design=)` SE documented as ~50–100 % different
  from R `svyglm` sandwich estimator (β still matches to machine
  precision).

### Notebook — three new sections, 9 new contracts
- Section VI (Steps 38–40): full inferential parity with R survey
  (β/SE/CI/p, multi-var batteries, quantified Rao-Scott gap).
- Section VII (Steps 41–43): negative-control tests (wrong weights,
  freq_weights, wrong strata).
- Section VIII (Steps 44–46): MI m-sensitivity, CC-vs-MI estimands,
  alternate outcome definitions.
- Scope statement + documented-limitations box added to intro.
- Summary table reorganised: numerical-correctness vs
  structural/interface contracts now clearly distinguished.

## [0.1.0a10] — 2026-05-27

### Added — Capabilities beyond R / gtsummary
- **Snapshot lock** (`snapshot_hash` / `lock_snapshot` / `assert_snapshot`)
  for binding-contract reproducibility — pin a published table to a
  content hash; CI fails on substantive drift.
- **Publication-safety auto-checker** (`check_safety` /
  `with_safety_warnings`) — flags extreme proportions, SD>mean,
  sparse p-values, |SMD|>1, extreme exp(coef), dominant missingness.
- **Quarto export** (`to_quarto(format=, label=, caption=)`) — first-
  class Quarto fenced blocks with cross-references.
- **Typst renderer** (`to_typst` / `to_typst_file`) — first stats
  package in either Python or R with native Typst support.
- **CLI**: `pysofra table data.csv --by arm --out table1.docx`,
  `pysofra check`, `pysofra version`.

### Notebook
- Section V (Steps 33-37) demonstrates each new feature with
  binding asserts. Notebook now covers 37 audited contracts.

## [0.1.0a9] — 2026-05-26

### Fixed
- Numeric `[0.1, 0.2, 0.9, 1.1]` is no longer classified as dichotomous
  (the old `int(x)` truncation collapsed it to `{0, 1}`).
- `tbl_regression` flags logistic / GLM fits with separation
  (|coef| or SE above the non-identification threshold).
- `tbl_one(design=…)` warns when Rao–Scott chi-square falls back on
  the first-order Kish-DEFF approximation under strata or clusters.
- `tbl_regression(CoxPHFitter, data=df)` runs `proportional_hazard_test`
  on the training frame and lists any covariate that violates PH.
- `weighted_continuous_stats`, `_weighted_mean_var_kish`, and the SMD
  weighted-mean helper use `math.fsum` for compensated summation.
- Rebuild-drop warning now catches `add_n`, `add_ci`, and
  `add_significance_stars` columns via a metadata tag (was: only
  `add_difference` / `signif.` via header-text match).
- HTML link allowlist blocks UNC paths (`\\server\share`) and
  control-character-prefixed hrefs.
- DOCX cell / caption / footnote text is stripped of XML 1.0-illegal
  control chars so user-supplied `\x00` doesn't corrupt the file.

### Changed
- `tests/test_scipy_validation.py` → `tests/test_pinned_references.py`;
  `test_matches_r` methods → `test_matches_pinned_reference` (honest
  about the fixture source, which is SciPy, not a live R run).

## [0.1.0a8] — 2026-05-26

### Fixed
- `SurveyDesign(fpc=...)` now applies FPC even when no strata are given.
- Lonely PSU (single cluster in a stratum) emits a `UserWarning`.
- Stratified variance now centres residuals on the stratum mean (was zero).
- `pool()` uses direct SE from `ModelSummary.se` instead of back-deriving
  from the CI half-width with a z-pivot.
- `tbl_regression.add_p()` is genuinely a no-op on tbl_regression tables.
- `with_forest_plot()` auto-detects log/linear scale from the table's
  coefficient column header (OR/HR/TR/IRR → log; β/Estimate → linear).
- HTML link scheme allowlist: `javascript:` / `data:` URLs in
  `CellPart(link=...)` are replaced with `about:blank`.
- `_refit_with_design` raises on invalid weights instead of silently zeroing.
- `tbl_regression(design=)` uses `var_weights` (not `freq_weights`)
  to match R `survey::svyglm` to first order on non-integer weights.

### Added
- `tbl_survival(weights=)`: weighted KM via lifelines.
- Rebuild-drop warning extended to every spec-changing modifier.

### Changed
- `_weighted_mean_var` docstring distinguishes freq/reliability
  weights from large-Σw sampling weights.
- `_median_ci` drops the redundant `conf_level` parameter.

## [0.1.0a7] — 2026-05-26

### Fixed
- `tbl_survival` validates `time` (negative → `ValueError`) and
  `event` (non-`0/1` → `ValueError`); previously passed silently to
  lifelines.
- `add_global_p()` on weighted `tbl_one` uses `var_weights=` instead
  of `freq_weights=` (avoids inflated df_resid on non-integer
  sampling weights).

### Changed
- `rao_scott_chisq` docstring honestly states a 10–15% typical
  disagreement with R `survey::svychisq` (was: optimistic "~5%").
- Added published-reference citations to every public statistical
  function (Welch, Wilson, Rao-Scott, Kish, BH, BY, Holm, Hommel,
  Šidák, Binder, etc.).
- `pool` and `cohen_d` docstrings now have NumPy-style
  Parameters/Returns/References sections.

## [0.1.0a6] — 2026-05-26

### Fixed
- **`svyttest`** now uses full-design Taylor linearisation; matches R
  `survey::svyttest` to 6+ decimal places. Previous per-group variance
  formulation could be anti-conservative under cluster-straddling-group
  designs.
- **`svyttest` df** corrected to `n_PSU − n_strata − 1`.
- **`rao_scott_chisq`** normalises weights to `Σw = n` before
  computing chi-square (matches R `survey::svychisq`).
- **`tbl_one(weights=...)`** raises on negative / all-zero weights
  instead of silently dropping.
- **`tbl_one(...).add_p()`** warns when >2-group continuous under
  weighted Table 1 falls back to unweighted ANOVA.
- **`tbl_one(...).add_global_p()`** warns when a prior modifier's
  column (e.g. `add_difference`) is about to be dropped by the
  rebuild.

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

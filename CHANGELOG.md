# Changelog

All notable changes to PySofra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a14] — 2026-05-28

### Fixed — design-based regression standard errors (closes the
### package's biggest documented limitation)
- **`tbl_regression(design=)` now computes the full Taylor-
  linearisation cluster-robust sandwich covariance**, matching R
  `survey::svyglm` on β, SE, **and p-value** to numerical precision.
  Through 0.1.0a13 the design refit used statsmodels `var_weights`
  SEs that differed from R's sandwich by ~50–100 % on stratified
  clustered designs (documented in 0.1.0a11–a13 as the outstanding
  limitation). It is now closed.
  - New `pysofra.summary.design.survey_glm_vcov(fit, weights, strata=,
    cluster=, fpc=)` implements `V = A⁻¹ B A⁻¹` with PSU-within-
    stratum nesting (the same machinery `design_mean_var` uses for a
    scalar mean, generalised to the GLM score vector). Supports the
    canonical-link Binomial / Poisson / Gaussian families; raises
    `NotImplementedError` for other families (rather than silently
    returning a wrong SE).
  - New `pysofra.models.regression.SurveyGLMResults` results wrapper
    routes the sandwich SE / CI / p through `tbl_regression(design=)`.
    Inference uses a *t* distribution with the R `svyglm` residual df
    `(n_PSU − n_strata) − k + 1`.
  - Verified on NHANES 2017-2018: PySofra vs R `svyglm` agree to
    **0.0 % on SE** and **≤ 0.04 % on p-value** across all six
    coefficients (jss_case_study Step 39).
  - **Monte Carlo CI coverage** for `tbl_regression(design=)` rises
    from ~84–86 % (var_weights) to a correctly-calibrated ~95 %
    (jss_case_study Step 47, now asserted in [92 %, 97 %]).
- **Gaussian-family GLM coefficients are now labelled β** (not
  "Estimate") — a Gaussian/identity GLM is a linear model. Keeps the
  OLS-under-`design=` refit (which routes through a Gaussian GLM)
  labelled consistently with the un-refit OLS fit.

### Added
- `tests/test_survey_glm_vcov.py` (8 tests): structural properties
  (symmetric / PSD / finite), df = n_PSU − n_strata, no-design ≈
  statsmodels HC1, all three families run, unsupported family raises,
  and an end-to-end check that `tbl_regression(design=)` SEs differ
  from the naive model-based SEs.

### Notebook
- Step 39 flips from "documented ~50–100 % SE gap" to **asserting**
  SE ≤ 1 % and p ≤ 2 % vs R `svyglm`.
- Step 47 flips from "documented ~85 % under-coverage" to
  **asserting** ~95 % empirical coverage.
- Top-of-notebook limitations box updated: the svyglm-SE item moves
  to "Resolved in 0.1.0a14"; only the first-order Rao–Scott
  chi-square (Step 38) remains an open documented approximation.

### Tests
- Full pytest suite: 1019 passing (was 1011).

## [0.1.0a13] — 2026-05-28

### Fixed — rendering hygiene (external-reviewer-reported)
- **CSS `color-mix()` removed from rendered HTML.** Borders and faded
  text previously used `color-mix(in srgb, currentColor 25%,
  transparent)` — readable in interactive notebook frontends but the
  GitHub.com .ipynb renderer's HTML sanitiser mangles it mid-attribute,
  leaking raw CSS text into table cells (the "CSS leaking into
  education labels" a reviewer observed). Replaced with fixed neutral
  `rgba(127,127,127,·)` greys that every renderer (including GitHub)
  handles. Trade-off: borders no longer follow the light/dark text
  colour; numbers and structure are unchanged.
- **LaTeX `<` / `>` now escaped** to `\textless{}` / `\textgreater{}`.
  Category labels like `<HS` previously emitted a bare `<`, which
  renders as ¡ under the default OT1 font encoding (correct only under
  T1). Now encoding-robust.

### Added — tests + audit depth
- **`tests/test_render_no_css_leak.py`** (11 tests): regression guard
  that no rendered HTML/LaTeX contains `color-mix()`, no `<style>`
  leaks into `<td>`, no CSS tokens bleed into cell text, LaTeX body
  rows carry no stray markup, and renders are deterministic. Every
  built-in theme is checked.
- **Step 38 now links the rendered Table-1 p-values** to the
  first-order Rao-Scott engine (asserted equal to 1e-9) and to R
  `survey::svychisq` (documented 57–69 % gap). This makes explicit
  that the p-values a user *publishes* from a survey-design Table 1
  are the first-order values — and can flip significance vs R (race:
  PySofra p=0.62 vs R p=0.023). Addresses the reviewer's "Table-1
  p-values, not just regression" concern.
- **Step 45 gains a prominent "NOT A SINGLE INFERENTIAL ANALYSIS"
  warning box** clarifying that the MI (Step 6) and survey-design
  (Step 7) demos are independent feature demonstrations, not two
  routes to one estimand.
- **Step 46 expanded to five outcome definitions** (+ medication use,
  + ADA fasting-glucose FPG≥126) with an explicit **subsample-weight
  audit**: the FPG arm correctly switches to `WTSAF2YR` (fasting
  subsample weight), not `WTMEC2YR`; the audit asserts the two weights
  genuinely differ (mean 129 % on the fasting subsample). NHANES
  `GLU_J` added to the notebook's download set.

### Tests
- Full pytest suite: 1011 passing (was 1000).

## [0.1.0a12] — 2026-05-27

### Notebook — Section IX added (inferential validity)

A second external reviewer raised the framework distinction between
*"the numbers match"* and *"the inference is statistically valid."*
For the `tbl_regression(design=)` path that point lands directly,
since 0.1.0a11 had documented (in Step 39) a ~50–100 % SE gap vs R
`survey::svyglm`'s sandwich estimator. Section IX quantifies the
*inferential consequence* of that gap:

- **Step 47** — Monte Carlo coverage: 500 synthetic stratified-
  clustered datasets, fit through PySofra's design-refit, record
  empirical CI coverage. Result on the representative DGP:
  **~84–86 % empirical coverage at nominal 95 %** (≈10 pp under-
  coverage). Documented honestly; recommendation in the cell:
  *"For publication-grade design-adjusted CIs, fit the model in R
  `svyglm` and use PySofra for the table presentation around the
  R-computed numbers."*
- **Step 48** — CI-asymmetry regression guard: verifies PySofra
  preserves `(exp(β_lo), exp(β_hi))` instead of applying a
  symmetric `OR ± z·SE`. Matches `exp(β ± z·SE)` to ≤ 1e-9; on an
  OR-of-36 fit the upper gap is 3.2× the lower gap, as expected.

### Notebook size
- Now nine sections, **48 audited contracts**, 107 cells.
- 2 new pytest acceptance tests; full suite 1000 passing (was 998).

### Reviewer-scope clarification
The remaining items from the second reviewer's framework — reference
grid construction, EMM weighting semantics, non-estimability, K-R
denominator df, full Monte Carlo EMM coverage — are out of scope
for PySofra (these belong in an estimated-marginal-means package
analogous to R `emmeans`, not in a statistical-reporting layer
analogous to R `gtsummary`). No action taken in PySofra; the
framework is being applied to the parallel `pymmeans` project.

## [0.1.0a11] — 2026-05-27

### Documentation honesty (in response to external audit)
- **Rao–Scott docstring corrected**: previously claimed "10–15 %"
  disagreement with R `survey::svychisq`; direct measurement on
  NHANES 2017-2018 shows **median 57 %, max 69 %** relative-error
  gap on the test statistic (see jss_case_study Step 38). The
  docstring now reports this directly and recommends R for
  publication-grade categorical inference.
- **`tbl_regression(design=)` SE limitation documented**: the
  `var_weights` convention used internally produces standard errors
  that can differ from R `survey::svyglm`'s cluster-robust sandwich
  estimator by ~50–100 % on stratified clustered designs (the β
  estimates remain in machine-precision agreement). Users requiring
  publication-grade design-adjusted CIs/p-values should fit the
  model in R `svyglm` and use PySofra only for table presentation.
  Documented in jss_case_study Step 39.

### Narrative-audit notebook — three new sections (Steps 38-46)
- **Section VI — Full inferential parity with R `survey`**
  - Step 38: Quantified Rao-Scott vs `svychisq` gap per variable.
  - Step 39: Full svyglm β AND SE AND CI AND p comparison;
    documents the var_weights-vs-sandwich gap.
  - Step 40: `svymean` battery (5 vars) + `svyttest` battery
    (3 vars) — all agree to ≤ 1e-9 relative error.
- **Section VII — Negative-control tests**
  - Step 41: Wrong weight column → visibly different estimate.
  - Step 42: `freq_weights` inflates `df_resid` by ~47,000×.
  - Step 43: Wrong strata → 70 % SE gap (wiring responsive).
- **Section VIII — Sensitivity analyses within scope**
  - Step 44: `pool()` SE convergence (m=5 vs m=20 vs m=50) —
    PySofra MI is well-converged even at m=5 (max gap 1.79 %).
  - Step 45: Complete-case vs MI estimates side-by-side, with
    explicit "different estimands" framing.
  - Step 46: Three diabetes-outcome definitions (ADA primary,
    lab-only, self-report-only) compared in weighted prevalence.

### Notebook reframing
- New **scope statement** at the top: explicit in-scope vs out-of-
  scope table. PySofra validates the software; it does not validate
  the epidemiological design.
- New **documented limitations** box: Rao-Scott approximation,
  survey-weighted MI (unsupported), age standardisation (not a
  feature), cluster-robust regression SEs.
- Step 5 / Step 6 / Step 7 prose updated to forward-reference
  Section VI quantifications and resolve the "which estimand?"
  ambiguity Reviewer #2 flagged.
- Summary table reorganised: **numerical-correctness contracts**
  (load-bearing) separated from **structural/interface contracts**
  (regression guards). Both have value; conflating them was a
  fair reviewer critique.

### Tests
- 9 new pytest acceptance tests for Steps 38-46.
- Total acceptance tests: 40 (was 31). Full pytest: 998 passing.

## [0.1.0a10] — 2026-05-27

### Added — Capabilities beyond R / gtsummary
- **Snapshot lock** for binding-contract reproducibility:
  `SofraTable.snapshot_hash()` returns a SHA-256 of the table's
  *logical content* (rendered Markdown + footnotes + spanning
  headers — not its presentational randomness like the per-render
  CSS class). `.lock_snapshot(path)` writes a JSON pin file;
  `.assert_snapshot(path)` raises with a unified diff if the table
  has drifted. The intended workflow is: author runs `lock_snapshot`
  once when the paper is submitted; CI runs `assert_snapshot` on
  every PR to fail loudly if any change to the upstream pipeline
  would alter the published numbers. No equivalent exists in
  gtsummary.
- **Publication-safety auto-checker**: `SofraTable.check_safety()`
  scans a built table for patterns historically associated with
  errata or retractions — 100 %/0 % proportions on n ≥ 30, SD > |Mean|,
  p < 0.001 with cell n < 30, |SMD| > 1.0, exp(coef) outside
  [0.1, 10], variables > 50 % missing — and returns a list of
  `SafetyWarning` objects. `.with_safety_warnings()` attaches them
  as footnotes on the rendered table. No other Python or R
  reporting package does this.
- **Quarto-native export**: `SofraTable.to_quarto(format='html'|'latex',
  label=, caption=)` emits a properly-formatted Quarto fenced
  pass-through block with optional cross-reference label
  (`#tbl-XXX`) and caption, so the table is one `{{< include >}}`
  away from a `.qmd` document.
- **Typst renderer**: `SofraTable.to_typst()` /
  `.to_typst_file(path)` produces a Typst `#table(...)` block with
  per-column alignment, spanning headers, and italic footnotes.
  PySofra is, to the authors' knowledge, the **first stats-reporting
  package in either Python or R** to ship a native Typst backend.
- **Command-line interface**: a `pysofra` console entry point
  exposes `pysofra table data.csv --by arm --vars age,sex --out
  table1.docx` (build a Table 1 in one shot from a tabular file),
  `pysofra check data.csv --by arm` (run the safety checker; exit
  code 2 on any flag — handy for shell pipelines and Makefiles),
  and `pysofra version`.

### Narrative-audit notebook
- Added **Section V (Steps 33-37)** demonstrating each of the five
  additions above with binding assertions. The notebook now covers
  **37 audited seams** in 81 cells, and every assertion fires on
  every commit through the CI workflow.

## [0.1.0a9] — 2026-05-26

### Fixed
- **Float dichotomous misclassification.** A column like
  ``[0.1, 0.2, 0.9, 1.1]`` was previously classified as ``dichotomous``
  because the old detector cast each value through ``int(x)`` before
  membership-testing against ``{0, 1}`` — the truncation collapsed
  ``0.1 → 0`` and ``1.1 → 1``. The dichotomous test now requires exact
  numeric equality to ``0.0`` or ``1.0`` (via ``np.isclose``) so only
  genuine 0/1 indicators trigger the branch.
- **Logit / GLM separation surfaced.** Statsmodels emits a
  ``PerfectSeparationWarning`` at fit time, but by the time the fitter
  reaches ``tbl_regression`` that warning is gone and the rendered
  table shows a finite-but-huge OR with a multi-thousand-unit CI.
  ``tbl_regression`` now inspects ``|coef|`` and ``SE`` thresholds and,
  when either crosses the non-identification boundary, appends a
  ``WARNING: at least one coefficient appears non-identified …``
  footnote pointing the user at Firth logistic / collinearity audit.
- **Rao–Scott design awareness.** ``tbl_one(design=…)`` with a strata
  or cluster column now emits a ``UserWarning`` when the categorical
  chi-square falls back on the first-order Kish-DEFF approximation,
  which doesn't use the strata/cluster covariance. The warning points
  users at R ``survey::svychisq`` for design-grade inference.
- **Cox PH assumption check.** When a fitted ``CoxPHFitter`` is passed
  to ``tbl_regression(..., data=df)``, the table now runs
  ``lifelines.statistics.proportional_hazard_test`` on the training
  frame and appends a footnote listing any covariate whose Schoenfeld
  residual test rejects proportional hazards at *p* < 0.05. Without
  ``data=`` the check is silently skipped (lifelines doesn't stash
  the training X on the fitter).
- **Compensated weighted summation.** ``weighted_continuous_stats``,
  ``_weighted_mean_var_kish``, and the SMD weighted-mean helper now
  use ``math.fsum`` for ``Σ w_i x_i`` and ``Σ w_i (x_i − μ)²``. The
  exactly-rounded accumulator removes order-dependent precision loss
  on long arrays with heterogeneous sampling weights (e.g. NHANES-
  scale weights spanning 4–5 orders of magnitude).
- **Rebuild-drop warning now catches ``add_n`` / ``add_ci`` /
  ``add_significance_stars`` columns**, not just ``add_difference``.
  The detector switched from header-text pattern-matching to a
  metadata-tag inserted by each column-adding modifier so it
  correctly catches headers like ``"N"`` (too generic to match
  safely) and ``""`` (the empty placeholder for significance stars).
- **HTML link allowlist hardened.** ``CellPart(link=…)`` now blocks
  UNC-style paths (``\\server\share``) and ``href`` values that start
  with ASCII control characters, both of which are routes around the
  scheme allowlist on legacy / Windows browsers.
- **DOCX control-character sanitization.** Cell, caption, and footnote
  text are stripped of XML 1.0–illegal control chars (``\x00``–``\x08``,
  ``\x0B``, ``\x0C``, ``\x0E``–``\x1F``, ``\x7F``) before being
  written, so a user-supplied label containing a stray ``\x00`` no
  longer produces a .docx that Word refuses to open.

### Changed
- ``tests/test_scipy_validation.py`` renamed to
  ``tests/test_pinned_references.py``; the ``test_matches_r`` method
  name (which implied a live R cross-check, when the fixtures are
  pinned SciPy reference values) is renamed to
  ``test_matches_pinned_reference``. Fixture provenance is unchanged
  — they remain ``scipy.stats`` / ``statsmodels`` / ``lifelines``
  outputs.

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

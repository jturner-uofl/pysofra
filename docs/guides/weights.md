# Survey-weighted Table 1

Pass `weights=` to `tbl_one()` or `tbl_summary()` to compute weighted
summaries. The weights column should hold non-negative frequency
weights (one weight per row).

```python
ps.tbl_one(df, by='arm', weights='ipw').add_p().theme('clinical')
```

## What gets weighted

| Statistic | Weighted form |
|---|---|
| Mean | $\sum w_i x_i / \sum w_i$ |
| Variance / SD | Frequency-weighted unbiased: $\sum w_i (x_i - \mu)^2 / (\sum w_i - 1)$ |
| Median / IQR | Linear-interpolation weighted quantile |
| n (%) | $\sum w_i \mathbf{1}\{x_i = \text{level}\} / \sum w_i$ |
| Missing (%) | Weighted count of missing values |
| Header N | Sum of weights per group |

## Complex survey designs — `SurveyDesign`

For stratified or clustered designs (and optional finite-population
correction), wrap the design columns in a `SurveyDesign` and pass it
via `design=`:

```python
design = ps.SurveyDesign(
    weights='pweight',
    strata='region',
    cluster='psu',
    fpc='stratum_pop',   # optional FPC column
)

ps.tbl_one(df, by='arm', design=design).add_p()
```

When the design includes strata or clusters, continuous summaries
switch from "mean (SD)" to **"mean (SE)"**, where the SE is computed
via Taylor linearisation across PSUs nested in strata. The footnote
names the variance method explicitly.

Categorical tests use the Rao–Scott corrected chi-square (design
effect computed from the weights). FPC, if provided, shrinks variance
by `(1 − n_h / N_h)` within each stratum.

## What ships today

* **Stratified + clustered designs** — `SurveyDesign(weights=,
  strata=, cluster=)` with Taylor-linearisation variance.
* **Finite population correction (FPC)** —
  `SurveyDesign(..., fpc='colname')`.
* **Replicate weights (JK1 / JKn / bootstrap)** —
  `SurveyDesign(replicate_weights=(...), replicate_type='jk1' | 'jkn' |
  'bootstrap')`. Used in `tbl_one`/`tbl_summary` to compute weighted
  means + replicate-based SEs.
* **Design-adjusted continuous tests** — when `design.strata` or
  `design.cluster` is set, `.add_p()` on a 2-group continuous variable
  routes to PySofra's `svyttest` (an `R::survey::svyttest`
  first-order analogue).
* **Rao–Scott corrected χ²** for categorical tests on weighted data
  (auto-routed when `weights` is set).
* **Cluster-robust regression refit** — `tbl_regression(model,
  design=..., data=...)` honours `design.weights` (WLS or
  `freq_weights=`) and `design.cluster` (cluster-robust SE).
* **Calibration helpers** — `ps.post_stratify(df, weights, margins=)`
  for complete-cross-classification, and `ps.rake(df, weights,
  margins=)` (iterative proportional fitting) for marginal-only
  targets.

## What's still raw

* Multi-stage (≥ 2-stage) cluster designs — currently `cluster=` accepts
  a tuple of two PSU columns but only the first stage contributes to
  variance estimation.
* Wald-F tests for multi-coefficient hypotheses in survey-design
  regression. For now, single-coefficient Wald is the available
  pathway via `.add_p()`.

For these cases, see R's `survey::svyglm` / `survey::regTermTest`.

## Notes

- The weights column is auto-excluded from the variable list — no need
  to pass `variables=[...]` to omit it.
- Equal weights (all `w = 1`) reproduce the unweighted output exactly.
- Zero or `NaN` weights skip the corresponding row.

# Scope & known limitations

A short, honest list of where PySofra makes a deliberate
approximation, exposes a known gap, or simply does not cover a case.
Each item is paired with the renderer-level signal a user sees, the
recommended workaround, and the audit step in the case-study notebook
that quantifies the gap.

This page exists so that PySofra users (and their reviewers) never
encounter a limitation through an unlabelled numeric discrepancy — the
limitation should be visible on the rendered table itself.

## 1. Rao–Scott design-based chi-square is **first-order**

| | |
|---|---|
| Where | `tbl_one(design=...)` p-values for categorical variables |
| What | PySofra uses the first-order Rao–Scott adjustment (Kish DEFF) for design-based chi-square. R `survey::svychisq` uses the second-order (generalised-DEFF / eigen-decomposition) adjustment. |
| Observed | On NHANES 2017-2018 (moderate clustering) the first-order p-values differ from R `svychisq` by 57–69 % on individual variables. Under **high intra-cluster correlation** — designs where the outcome is nearly constant within PSUs — the Kish DEFF approximation is blind to the clustering structure and can be off by an **order of magnitude or more** (empirically: ×10–×20 vs the second-order p-value). The underlying Pearson chi-square statistic matches R exactly in all cases. |
| User signal | Table 1 p-values for categorical variables under `design=` carry a footnote naming the approximation. |
| Workaround | Compute the chi-square statistic via PySofra (matches R exactly), then run `survey::svychisq()` in R for the second-order p-value if a publication requires that exact match. |
| Audit step | jss_case_study Step 38 (quantified gap + Table-1 linkage). |

## 2. Weighted Kaplan–Meier CIs use Greenwood

| | |
|---|---|
| Where | `tbl_survival(weights=...)` median-survival and S(t) CIs |
| What | PySofra delegates to `lifelines.KaplanMeierFitter`, which uses the Greenwood variance. Greenwood is **exact** for integer (frequency) weights, but **biased (too narrow)** for non-integer (sampling, propensity, IPTW) weights. The KM point estimates remain unbiased under any weights. |
| User signal | When `weights=` resolves to non-integer values, `tbl_survival` emits one `UserWarning` and attaches a matching table footnote naming the Greenwood approximation. Integer weights stay silent. |
| Workaround | For design-grade weighted-survival CIs, bootstrap-resample units (or PSUs) and report empirical-percentile CIs. |
| Audit step | jss_case_study Step 27 (pinned CI-bias warning + footnote as a contract). |

## 3. scikit-learn estimators expose point estimates only

| | |
|---|---|
| Where | `tbl_regression(sklearn_fit)` for `LogisticRegression`, `LinearRegression`, etc. |
| What | scikit-learn does not natively expose standard errors, confidence intervals, or p-values on fitted estimators. PySofra renders the point estimates faithfully and leaves the CI / p-value columns blank. |
| User signal | The rendered table carries a footnote naming the source family (e.g., `LogisticRegression (scikit-learn)`) and stating "point estimates only — the source fitter does not expose standard errors, confidence intervals, or p-values". |
| Workaround | Refit the same model with `statsmodels` (`sm.Logit`, `sm.GLM`, `sm.OLS`) when inferential output is required. PySofra then auto-extracts CI + p from the statsmodels result via the same `tbl_regression()` entry point — no other code changes needed. |
| Audit step | jss_case_study Step 53 (no-inference footnote pinned as a contract); unit test `test_sklearn_table_carries_no_inference_footnote` in `tests/test_regressions.py`. |

---

### What is **not** a limitation we intend to fix

Some properties are deliberate, not gaps:

- **`SofraTable` is frozen / immutable.** Modifier methods are
  copy-on-write. This is the foundation of the cross-backend
  consistency proof (`tests/test_cross_backend_consistency.py`) and
  is fixed by the
  [API stability contract](stability.md).
- **PySofra does not fit models on the user's behalf for
  `tbl_regression`.** The function accepts a *fitted* statsmodels /
  lifelines / sklearn result and extracts the right quantities. A
  user fitting their own model preserves all model diagnostics that
  PySofra cannot reproduce (BIC, fit warnings, residual plots).
- **`tbl_survival` re-derives CIs from `lifelines.fit(alpha=)` and
  does not patch lifelines' variance formula.** Doing so would make
  PySofra a fork of lifelines rather than a thin reporting layer
  over it.

Anything else? Open an issue on
[the repository](https://github.com/jturner-uofl/pysofra) tagged
`limitation` and we will either fix it or add it to this page.

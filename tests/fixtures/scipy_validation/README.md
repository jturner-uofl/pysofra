# SciPy validation fixtures

This directory holds reference outputs computed in **Python via SciPy /
statsmodels / lifelines** for canonical datasets. We use them to
cross-validate that PySofra's higher-level wrappers (`continuous_test`,
`categorical_test`, `run_named_test`, `svyttest`) produce the same
numbers as a direct call to the underlying library.

## Why these are useful even though they are not R outputs

SciPy's `scipy.stats` hypothesis tests have themselves been validated
to machine precision against R's `stats` package across releases, and
PySofra delegates to SciPy under the hood. So a PySofra-vs-SciPy
assertion is *also* implicitly a PySofra-vs-R assertion — but the
fixture's `source` field is honest: the reference value came from
SciPy, not from an independent R run.

If you need a direct R cross-validation, the recommended workflow is:

```r
# example_session.R — run alongside the equivalent Python in the tutorial
library(stats); library(survival); library(survey)
# … run the same test in R and confirm output matches PySofra's
```

We do not commit an R-generation script because (a) CI does not have
R installed, and (b) the SciPy→R agreement is already established at
the layer below PySofra. If a future divergence appears we will add
explicit R-output JSONs at that point, with their `source` field
correctly identifying the R version + package versions used.

## How fixtures were generated

Each fixture pairs:

- a **dataset** built in pure Python (deterministic seeding via
  `numpy.random.default_rng(SEED)`) so the input is reproducible
  without any external tooling, and
- a **reference values JSON** that captures the SciPy / statsmodels /
  lifelines output for the same computation.

The reference values are computed inside `pytest` collection (see
`tests/test_pinned_references.py`) — that is the only canonical source
of truth. The JSONs are versioned and checked into the repo for
diff-friendly inspection.

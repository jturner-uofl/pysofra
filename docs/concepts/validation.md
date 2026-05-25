# Validation

PySofra's statistical outputs (test statistics, p-values, SMDs) are
computed via SciPy / statsmodels. The package itself adds:

- variable typing (which test to run on which variable)
- formatting (rounding, p-value display rules)
- multiplicity adjustment via `statsmodels.stats.multitest.multipletests`
- standardised mean differences via the Yang–Dalton (2012) formulation

## What we test

- All statistical computations have unit tests covering correctness
  (e.g. SMD = 0 when groups are identical; t-test reproduces SciPy's
  result; chi-square detects sparse tables).
- HTML / Markdown / LaTeX / DOCX outputs have rendering tests.
- The full `tbl_one` / `tbl_summary` / `tbl_regression` pipeline has
  integration tests across realistic synthetic data.

## Cross-validation against R

PySofra's numeric outputs go through SciPy / statsmodels / lifelines.
Those libraries are individually cross-validated against R's `stats`
package across releases (see the SciPy test suite), so the chain
PySofra→SciPy→R holds to machine precision. The
[`tests/fixtures/scipy_validation/`](https://github.com/jturner-uofl/pysofra/tree/main/tests/fixtures/scipy_validation)
directory pins this with explicit reference JSONs; the next-layer-out
R cross-check is left to dedicated R-vs-SciPy regression suites
upstream.

## Reproducibility

All statistical calculations are deterministic given the same input.
We do not seed any RNG internally, and we explicitly strip every known
source of cross-process non-determinism from renderer output:

* **HTML scope IDs** are derived from a SHA-256 hash of the table's
  content (`render/html.py:_scope_id_for`), so identical tables
  produce identical CSS selectors.
* **matplotlib SVG / PNG / PDF** outputs from `with_forest_plot()` and
  `with_km_plot()` are post-processed to strip wall-clock timestamps,
  process-random hash salts, and randomised DOM IDs
  (`plot/inline.py:_strip_svg_nondeterminism` /
  `_strip_png_nondeterminism` / `_strip_pdf_nondeterminism`).
* **DOCX / PPTX / XLSX** archives — these contain their own
  embedded-content layer, and the embedded PNGs reuse the
  deterministic stream above.

This guarantee is regression-tested in
[`tests/test_renderer_consistency.py`](https://github.com/jturner-uofl/pysofra/blob/main/tests/test_renderer_consistency.py)
(plot-less tables) and
[`tests/test_plot_determinism.py`](https://github.com/jturner-uofl/pysofra/blob/main/tests/test_plot_determinism.py)
(tables with attached forest / KM plots).

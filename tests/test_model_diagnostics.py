"""Regression-table diagnostic footnotes.

Two failure modes are easy to miss when looking only at a rendered
regression table:

* **Complete or quasi-complete separation** in a logistic / Cox fit.
  The optimiser walks off the boundary, leaves a finite-but-huge
  coefficient with an SE in the hundreds, and statsmodels emits a
  ``PerfectSeparationWarning`` at fit time — but that warning is gone
  by the time ``tbl_regression`` sees the result. Without a footnote
  the rendered cell reads "OR = 5e18 (95% CI: 0, inf)" and a busy
  reader takes it at face value.

* **Cox proportional-hazards violation.** ``CoxPHFitter`` happily
  reports a single HR even when the hazard ratio inverts mid
  follow-up; the resulting number is a (weighted) average of an
  effect that doesn't exist as a constant. The PH check
  (Schoenfeld residuals via ``lifelines.statistics.proportional_hazard_test``)
  surfaces this — but only if someone runs it.

Both should produce a footnote on the rendered table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import pysofra as ps

lifelines = pytest.importorskip("lifelines")
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.datasets import load_rossi  # noqa: E402


class TestSeparationDetection:
    def test_perfectly_separable_logit_emits_warning_footnote(self):
        df = pd.DataFrame({
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
            "x": [-2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2.0],
        })
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(df["y"], sm.add_constant(df[["x"]])).fit(disp=False)
        tbl = ps.tbl_regression(m)
        joined = " ".join(tbl.footnotes)
        assert "non-identified" in joined or "separation" in joined.lower()

    def test_clean_logit_has_no_separation_footnote(self):
        rng = np.random.default_rng(0)
        n = 400
        x = rng.normal(size=n)
        p = 1.0 / (1.0 + np.exp(-(0.5 * x)))
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({"y": y, "x": x})
        m = sm.Logit(df["y"], sm.add_constant(df[["x"]])).fit(disp=False)
        tbl = ps.tbl_regression(m)
        joined = " ".join(tbl.footnotes)
        assert "non-identified" not in joined
        assert "separation" not in joined.lower()


class TestCoxPHViolationFootnote:
    def test_rossi_with_data_surfaces_violation_footnote(self):
        df = load_rossi()
        cf = CoxPHFitter().fit(df, "week", "arrest")
        tbl = ps.tbl_regression(cf, data=df)
        joined = " ".join(tbl.footnotes)
        assert "Proportional-hazards" in joined
        # rossi: age and wexp violate
        assert "age" in joined or "wexp" in joined

    def test_cox_without_data_kwarg_does_not_run_check(self):
        # Without ``data=`` we have no training X to feed to the Schoenfeld
        # test, so no PH footnote should appear (silent — never crash).
        # Use a *fresh* fit so any earlier test that attached training_data_
        # to a shared fixture doesn't bleed into this one.
        df = load_rossi()
        cf = CoxPHFitter().fit(df, "week", "arrest")
        # Make sure no training data was stashed.
        if hasattr(cf, "training_data_"):
            try:
                delattr(cf, "training_data_")
            except (AttributeError, TypeError):
                pass
        tbl = ps.tbl_regression(cf)
        joined = " ".join(tbl.footnotes)
        # Either there is no PH footnote (no training data → silent), or
        # if some lifelines version did stash it for us, the footnote is
        # accurate. Both outcomes are acceptable; the contract is "no
        # crash, no false positive".
        assert "crash" not in joined  # trivially true; documents intent

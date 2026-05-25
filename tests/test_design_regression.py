"""Tests for design-aware regression refit (``tbl_regression(design=...)``).

These tests pin the behaviour that PySofra's design refit:

* Re-estimates the coefficients using the design weights (matching
  ``sm.WLS`` for OLS and ``sm.GLM(freq_weights=...)`` for Logit /
  Poisson / GLM), **not** the unweighted point estimates of the
  original fit.
* Uses a cluster-robust variance estimator when ``design.cluster`` is
  set, HC1 otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def trial_with_weights():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "w": rng.uniform(0.5, 2.5, size=n),
        "psu": rng.integers(0, 25, size=n),
    })
    df["y_continuous"] = 2.0 + 1.5 * df["x"] - 0.3 * df["z"] + rng.normal(size=n)
    df["y_binary"] = (rng.uniform(size=n) <
                      1 / (1 + np.exp(-(df["x"] - 0.5 * df["z"])))).astype(int)
    return df


# ----------------------------------------------------------------------
# OLS → WLS
# ----------------------------------------------------------------------
class TestOLSWithDesign:
    def test_design_matches_direct_wls(self, trial_with_weights):
        """The refit estimate must equal ``sm.WLS(weights=)`` exactly."""
        sm = pytest.importorskip("statsmodels.api")
        df = trial_with_weights
        m = sm.OLS(df["y_continuous"], sm.add_constant(df[["x", "z"]])).fit()
        d = ps.SurveyDesign(weights="w", cluster="psu")
        t = ps.tbl_regression(m, intercept=False, design=d, data=df)
        ref = sm.WLS(df["y_continuous"], sm.add_constant(df[["x", "z"]]),
                     weights=df["w"]).fit()
        for name in ("x", "z"):
            row = next(r for r in t.rows if r.cells[0].text == name)
            est = row.cells[1].value
            assert est == pytest.approx(float(ref.params[name]), abs=1e-9)

    def test_design_estimate_differs_from_unweighted(self, trial_with_weights):
        """Sanity: the design refit must NOT silently return the
        unweighted estimate."""
        sm = pytest.importorskip("statsmodels.api")
        df = trial_with_weights
        m = sm.OLS(df["y_continuous"], sm.add_constant(df[["x", "z"]])).fit()
        d = ps.SurveyDesign(weights="w")
        t_design = ps.tbl_regression(m, intercept=False, design=d, data=df)
        t_plain = ps.tbl_regression(m, intercept=False)
        for name in ("x", "z"):
            r_d = next(r for r in t_design.rows if r.cells[0].text == name)
            r_p = next(r for r in t_plain.rows if r.cells[0].text == name)
            # They should differ — the weights matter.
            assert r_d.cells[1].value != r_p.cells[1].value, (
                f"design refit silently produced unweighted estimate for {name}"
            )


# ----------------------------------------------------------------------
# Logit → GLM(Binomial, freq_weights=)
# ----------------------------------------------------------------------
class TestLogitWithDesign:
    def test_design_logit_matches_glm_binomial(self, trial_with_weights):
        sm = pytest.importorskip("statsmodels.api")
        df = trial_with_weights
        m = sm.Logit(df["y_binary"],
                     sm.add_constant(df[["x", "z"]])).fit(disp=False)
        d = ps.SurveyDesign(weights="w", cluster="psu")
        t = ps.tbl_regression(m, intercept=False, design=d, data=df,
                              exponentiate=True)
        # Reference: GLM with freq_weights, same family
        ref = sm.GLM(df["y_binary"],
                     sm.add_constant(df[["x", "z"]]),
                     family=sm.families.Binomial(),
                     freq_weights=df["w"].to_numpy()).fit()
        for name in ("x", "z"):
            row = next(r for r in t.rows if r.cells[0].text == name)
            or_pysofra = row.cells[1].value
            or_ref = float(np.exp(ref.params[name]))
            assert or_pysofra == pytest.approx(or_ref, abs=1e-9)


# ----------------------------------------------------------------------
# Cluster-robust SE when design.cluster is set
# ----------------------------------------------------------------------
class TestClusterRobustSE:
    def test_cluster_se_differs_from_hc1(self, trial_with_weights):
        sm = pytest.importorskip("statsmodels.api")
        df = trial_with_weights
        m = sm.OLS(df["y_continuous"], sm.add_constant(df[["x", "z"]])).fit()
        d_cluster = ps.SurveyDesign(weights="w", cluster="psu")
        d_no_cluster = ps.SurveyDesign(weights="w")
        t_c = ps.tbl_regression(m, intercept=False, design=d_cluster, data=df)
        t_h = ps.tbl_regression(m, intercept=False, design=d_no_cluster, data=df)
        # Same point estimate (same weights), but different CIs.
        for name in ("x", "z"):
            r_c = next(r for r in t_c.rows if r.cells[0].text == name)
            r_h = next(r for r in t_h.rows if r.cells[0].text == name)
            assert r_c.cells[1].value == pytest.approx(r_h.cells[1].value, abs=1e-9)
            # CI widths should differ
            ci_c = r_c.cells[2].value
            ci_h = r_h.cells[2].value
            assert (ci_c[1] - ci_c[0]) != (ci_h[1] - ci_h[0])


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------
class TestDesignErrors:
    def test_raises_without_data_when_weights_set(self, trial_with_weights):
        sm = pytest.importorskip("statsmodels.api")
        df = trial_with_weights
        m = sm.OLS(df["y_continuous"], sm.add_constant(df[["x"]])).fit()
        d = ps.SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="design has weights"):
            ps.tbl_regression(m, intercept=False, design=d)

    def test_raises_when_lengths_disagree(self, trial_with_weights):
        sm = pytest.importorskip("statsmodels.api")
        df = trial_with_weights
        m = sm.OLS(df["y_continuous"], sm.add_constant(df[["x"]])).fit()
        d = ps.SurveyDesign(weights="w")
        wrong = df.iloc[:50]  # half-length
        with pytest.raises(ValueError, match="length"):
            ps.tbl_regression(m, intercept=False, design=d, data=wrong)

"""Design-based (Taylor-linearised) GLM covariance — survey_glm_vcov.

This is the sandwich estimator that gives ``tbl_regression(design=)``
standard errors / CIs / p-values matching R ``survey::svyglm`` to
numerical precision (verified against pinned R output on NHANES in the
case-study notebook Step 39). These unit tests are self-contained —
no NHANES download, no R required — and cover:

* the three supported canonical-link families (Binomial, Poisson,
  Gaussian) all run and return finite, symmetric, PSD covariances;
* the no-strata-no-cluster sandwich agrees with statsmodels' own
  HC1 robust covariance (same estimator family);
* the design degrees of freedom equal ``n_PSU − n_strata``;
* an unsupported family raises ``NotImplementedError`` (so we never
  silently return a wrong SE for, e.g., a Gamma GLM).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

sm = pytest.importorskip("statsmodels.api")

from pysofra.summary.design import survey_glm_vcov  # noqa: E402


@pytest.fixture
def design_data():
    """A small synthetic stratified-clustered survey dataset."""
    rng = np.random.default_rng(0)
    n_strata, n_psu, n_per = 5, 4, 15        # 300 rows, 20 PSUs
    rows = []
    for s in range(n_strata):
        for p in range(n_psu):
            psu_id = s * 10 + p              # globally-unique PSU id
            for _ in range(n_per):
                rows.append((s, psu_id))
    strata = np.array([r[0] for r in rows])
    cluster = np.array([r[1] for r in rows])
    n = len(rows)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, n)
    X = sm.add_constant(np.column_stack([x1, x2]))
    w = rng.uniform(0.5, 3.0, n)
    return X, w, strata, cluster, rng


# ----------------------------------------------------------------------
# Structural properties
# ----------------------------------------------------------------------

class TestStructure:
    def test_binomial_symmetric_psd_finite(self, design_data):
        X, w, strata, cluster, rng = design_data
        y = rng.binomial(1, 0.4, len(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Binomial(),
                         var_weights=w).fit()
            vcov, df = survey_glm_vcov(fit, w, strata=strata,
                                       cluster=cluster)
        assert vcov.shape == (3, 3)
        assert np.allclose(vcov, vcov.T), "vcov not symmetric"
        assert np.all(np.linalg.eigvalsh(vcov) > -1e-9), "vcov not PSD"
        assert np.all(np.isfinite(vcov))

    def test_poisson_runs(self, design_data):
        X, w, strata, cluster, rng = design_data
        y = rng.poisson(2.0, len(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Poisson(),
                         var_weights=w).fit()
            vcov, _ = survey_glm_vcov(fit, w, strata=strata,
                                      cluster=cluster)
        assert np.all(np.isfinite(np.sqrt(np.diag(vcov))))

    def test_gaussian_runs(self, design_data):
        X, w, strata, cluster, rng = design_data
        y = rng.normal(size=len(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Gaussian(),
                         var_weights=w).fit()
            vcov, _ = survey_glm_vcov(fit, w, strata=strata,
                                      cluster=cluster)
        assert np.all(np.isfinite(np.sqrt(np.diag(vcov))))


# ----------------------------------------------------------------------
# Degrees of freedom
# ----------------------------------------------------------------------

class TestDesignDF:
    def test_df_equals_npsu_minus_nstrata(self, design_data):
        X, w, strata, cluster, rng = design_data
        y = rng.binomial(1, 0.4, len(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Binomial(),
                         var_weights=w).fit()
            _, df = survey_glm_vcov(fit, w, strata=strata, cluster=cluster)
        n_psu = len(np.unique(cluster))
        n_strata = len(np.unique(strata))
        assert df == float(n_psu - n_strata)   # 20 - 5 = 15

    def test_df_inf_without_cluster(self, design_data):
        X, w, strata, cluster, rng = design_data
        y = rng.binomial(1, 0.4, len(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Binomial(),
                         var_weights=w).fit()
            _, df = survey_glm_vcov(fit, w)   # no strata, no cluster
        assert df == float("inf")


# ----------------------------------------------------------------------
# Consistency with statsmodels HC1 (no-design case)
# ----------------------------------------------------------------------

class TestHC1Consistency:
    def test_no_design_matches_hc1(self, design_data):
        """With no strata/cluster the linearisation sandwich is an
        HC0-family robust covariance; it should agree with statsmodels'
        HC1 to within the n/(n-1) vs n/(n-k) small-sample factor."""
        X, w, strata, cluster, rng = design_data
        y = rng.binomial(1, 0.4, len(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Binomial(),
                         var_weights=w).fit()
            vcov, _ = survey_glm_vcov(fit, w)
            hc1 = sm.GLM(y, X, family=sm.families.Binomial(),
                         var_weights=w).fit(cov_type="HC1")
        se_sandwich = np.sqrt(np.diag(vcov))
        se_hc1 = np.asarray(hc1.bse)
        rel = np.abs(se_sandwich - se_hc1) / se_hc1
        assert np.all(rel < 0.02), (
            f"no-design sandwich SE differs from HC1 by >2%: {rel}"
        )


# ----------------------------------------------------------------------
# Unsupported family
# ----------------------------------------------------------------------

class TestUnsupportedFamily:
    def test_gamma_raises(self, design_data):
        X, w, strata, cluster, rng = design_data
        y = np.abs(rng.normal(size=len(w))) + 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, X, family=sm.families.Gamma()).fit()
        with pytest.raises(NotImplementedError, match="canonical-link"):
            survey_glm_vcov(fit, w, strata=strata, cluster=cluster)


# ----------------------------------------------------------------------
# End-to-end through tbl_regression(design=)
# ----------------------------------------------------------------------

class TestEndToEnd:
    def test_design_se_differs_from_naive(self, design_data):
        """tbl_regression(design=) must produce different (design-based)
        SEs than the naive model-based SEs — proving the sandwich is
        actually wired into the public API."""
        import pysofra as ps
        X, w, strata, cluster, rng = design_data
        y = rng.binomial(1, 0.4, len(w))
        import pandas as pd
        df = pd.DataFrame({
            "y": y, "x1": X[:, 1], "x2": X[:, 2],
            "w": w, "strata": strata, "psu": cluster,
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(df["y"], sm.add_constant(df[["x1", "x2"]]),
                         family=sm.families.Binomial()).fit()
            naive = ps.tbl_regression(fit, exponentiate=False)
            design = ps.SurveyDesign(weights="w", strata="strata",
                                     cluster="psu")
            robust = ps.tbl_regression(fit, design=design, data=df,
                                       exponentiate=False)
        # Extract the CI half-widths for the x1 row from both tables
        def _ci_width(table, label):
            for r in table.rows:
                if r.cells[0].text.strip() == label:
                    v = r.cells[2].value
                    if isinstance(v, tuple) and len(v) == 2:
                        return abs(v[1] - v[0])
            return None
        w_naive = _ci_width(naive, "x1")
        w_robust = _ci_width(robust, "x1")
        assert w_naive is not None and w_robust is not None
        assert abs(w_naive - w_robust) > 1e-6, (
            "design-based CI identical to naive — sandwich not applied"
        )

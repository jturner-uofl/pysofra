"""Statistical correctness validation against independent references.

Every numeric routine in PySofra is verified against an independent
reference: scipy / lifelines / statsmodels direct calls, hand-computed
textbook formulas, or published reference values. Any number rendered
by ``tbl_one`` / ``tbl_regression`` / ``tbl_survival`` /
``tbl_uvregression`` can be traced back through this file to where it
came from.

Each test names its reference source in the docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

import pysofra as ps

ATOL = 1e-9   # absolute tolerance for exact-match reference checks
RTOL = 1e-9   # relative tolerance for exact-match reference checks


# ======================================================================
# CONTINUOUS HYPOTHESIS TESTS — must match scipy exactly
# ======================================================================
class TestContinuousTests_vs_Scipy:
    """All continuous tests must equal a direct ``scipy.stats`` call.

    Reference: ``scipy.stats.ttest_ind``, ``mannwhitneyu``, ``f_oneway``,
    ``kruskal``.
    """

    def _two_group_data(self, n=80):
        rng = np.random.default_rng(20260521)
        a = rng.normal(0.0, 1.0, n)
        b = rng.normal(0.3, 1.2, n)
        return a, b

    def test_welch_t_matches_scipy(self):
        from pysofra.summary.tests import continuous_test
        a, b = self._two_group_data()
        df = pd.DataFrame({"v": np.r_[a, b],
                           "g": ["A"] * len(a) + ["B"] * len(b)})
        ours = continuous_test(df["v"], df["g"], nonnormal=False)
        ref_stat, ref_p = sp_stats.ttest_ind(a, b, equal_var=False)
        assert ours.p_value == pytest.approx(float(ref_p), abs=ATOL, rel=RTOL)
        assert ours.statistic == pytest.approx(float(ref_stat), abs=ATOL, rel=RTOL)
        assert ours.test == "Welch's t-test"

    def test_wilcoxon_matches_scipy(self):
        from pysofra.summary.tests import continuous_test
        a, b = self._two_group_data()
        df = pd.DataFrame({"v": np.r_[a, b],
                           "g": ["A"] * len(a) + ["B"] * len(b)})
        ours = continuous_test(df["v"], df["g"], nonnormal=True)
        ref_stat, ref_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        assert ours.p_value == pytest.approx(float(ref_p), abs=ATOL, rel=RTOL)
        assert ours.statistic == pytest.approx(float(ref_stat), abs=ATOL, rel=RTOL)

    def test_anova_matches_scipy(self):
        from pysofra.summary.tests import continuous_test
        rng = np.random.default_rng(0)
        a, b, c = (rng.normal(loc, 1.0, 50) for loc in (0.0, 0.4, 1.0))
        df = pd.DataFrame({"v": np.r_[a, b, c],
                           "g": ["A"] * 50 + ["B"] * 50 + ["C"] * 50})
        ours = continuous_test(df["v"], df["g"], nonnormal=False)
        ref_stat, ref_p = sp_stats.f_oneway(a, b, c)
        assert ours.p_value == pytest.approx(float(ref_p), abs=ATOL, rel=RTOL)
        assert ours.test == "One-way ANOVA"

    def test_kruskal_matches_scipy(self):
        from pysofra.summary.tests import continuous_test
        rng = np.random.default_rng(1)
        a, b, c = (rng.exponential(scale, 60) for scale in (1.0, 1.3, 1.8))
        df = pd.DataFrame({"v": np.r_[a, b, c],
                           "g": ["A"] * 60 + ["B"] * 60 + ["C"] * 60})
        ours = continuous_test(df["v"], df["g"], nonnormal=True)
        ref_stat, ref_p = sp_stats.kruskal(a, b, c)
        assert ours.p_value == pytest.approx(float(ref_p), abs=ATOL, rel=RTOL)


# ======================================================================
# CATEGORICAL TESTS — must match scipy
# ======================================================================
class TestCategoricalTests_vs_Scipy:
    """Fisher / chi-square must equal scipy.stats direct calls."""

    def test_fisher_2x2_matches_scipy(self):
        from pysofra.summary.tests import categorical_test
        # 2x2 contingency reproducible with explicit values
        v = pd.Series(["x"] * 10 + ["y"] * 10 + ["x"] * 6 + ["y"] * 14)
        g = pd.Series(["A"] * 20 + ["B"] * 20)
        ours = categorical_test(v, g)
        # Build the expected 2x2 table by hand:
        # A: x=10, y=10; B: x=6, y=14
        observed = np.array([[10, 6], [10, 14]])
        _, ref_p = sp_stats.fisher_exact(observed, alternative="two-sided")
        assert ours.p_value == pytest.approx(float(ref_p), abs=ATOL, rel=RTOL)

    def test_chisq_3x3_matches_scipy(self):
        from pysofra.summary.tests import categorical_test
        rng = np.random.default_rng(2)
        v = rng.choice(["x", "y", "z"], 300)
        g = rng.choice(["A", "B", "C"], 300)
        ours = categorical_test(pd.Series(v), pd.Series(g))
        ctab = pd.crosstab(pd.Series(v), pd.Series(g)).to_numpy()
        _, ref_p, _, _ = sp_stats.chi2_contingency(ctab, correction=False)
        assert ours.p_value == pytest.approx(float(ref_p), abs=ATOL, rel=RTOL)


# ======================================================================
# EFFECT SIZES — verified against textbook formulas
# ======================================================================
class TestEffectSizes_vs_TextbookFormulas:
    """Each effect size verified against its formula on hand-rigged data."""

    def test_cohens_d_known_values(self):
        # Hand-computed: a = [1,2,3,4,5], b = [3,4,5,6,7]
        # mean_a=3, mean_b=5, var_a=var_b=2.5, s_pool=sqrt(2.5)=1.5811...
        # d = (3-5)/1.5811 = -1.2649
        from pysofra.summary.effect_size import cohen_d
        a = np.array([1.0, 2, 3, 4, 5])
        b = np.array([3.0, 4, 5, 6, 7])
        d = cohen_d(a, b)
        expected = (3.0 - 5.0) / math.sqrt(2.5)
        assert d == pytest.approx(expected, rel=1e-12)

    def test_cohens_d_zero_difference(self):
        from pysofra.summary.effect_size import cohen_d
        x = np.array([1.0, 2, 3, 4, 5])
        assert cohen_d(x, x) == 0.0

    def test_hedges_g_bias_correction(self):
        # g = d * J, J = 1 - 3/(4*N - 9) where N = n_a + n_b
        from pysofra.summary.effect_size import cohen_d, hedges_g
        a = np.array([1.0, 2, 3, 4, 5])
        b = np.array([3.0, 4, 5, 6, 7])
        d = cohen_d(a, b)
        J = 1.0 - 3.0 / (4 * (5 + 5) - 9)
        assert hedges_g(a, b) == pytest.approx(d * J, rel=1e-12)

    def test_eta_squared_formula(self):
        # Eta² = SS_between / SS_total
        from pysofra.summary.effect_size import eta_squared
        v = pd.Series([1.0, 2, 3] + [4.0, 5, 6] + [7.0, 8, 9])
        g = pd.Series(["A"] * 3 + ["B"] * 3 + ["C"] * 3)
        # Grand mean = 5
        # SS_between = 3*(2-5)² + 3*(5-5)² + 3*(8-5)² = 27 + 0 + 27 = 54
        # SS_total = sum((x-5)²) = 60
        # eta² = 54/60 = 0.9
        assert eta_squared(v, g) == pytest.approx(0.9, rel=1e-12)

    def test_cramers_v_2x2_equals_phi(self):
        # For 2x2 tables, Cramér's V == phi.
        from pysofra.summary.effect_size import cramers_v, phi_coefficient
        v = pd.Series(["x", "x", "y", "y"] * 10)
        g = pd.Series(["A", "B", "A", "B"] * 10)
        assert cramers_v(v, g) == pytest.approx(phi_coefficient(v, g),
                                                rel=1e-12)


# ======================================================================
# WILSON-SCORE CI — verified against scipy.stats.binomtest
# ======================================================================
class TestWilsonCI_vs_scipy:
    """Wilson-score CI must match the closed-form solution."""

    def test_wilson_known_values(self):
        # Reference: Wilson (1927) for x=5, n=10, alpha=0.05
        # p_hat = 0.5, z=1.96
        # Wilson: (p + z²/(2n) ± z*sqrt(p(1-p)/n + z²/(4n²))) / (1 + z²/n)
        from pysofra.summary.extras import _wilson_ci
        z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
        lo, hi = _wilson_ci(5, 10, z=z)
        # Closed-form:
        # numerator_center = 0.5 + (z²)/(2*10) = 0.5 + 0.19207...
        # numerator_radius = z*sqrt(0.25/10 + z²/400)
        # denom = 1 + z²/10
        center = (0.5 + z * z / 20.0)
        radius = z * math.sqrt(0.25 / 10.0 + z * z / 400.0)
        denom = 1.0 + z * z / 10.0
        expected_lo = (center - radius) / denom
        expected_hi = (center + radius) / denom
        assert lo == pytest.approx(expected_lo, abs=1e-12)
        assert hi == pytest.approx(expected_hi, abs=1e-12)


# ======================================================================
# WEIGHTED STATS — verified against numpy weighted formulas
# ======================================================================
class TestWeightedStats_vs_Numpy:
    def test_weighted_mean_matches_numpy(self):
        from pysofra.summary.weights import weighted_continuous_stats
        v = pd.Series([1.0, 2, 3, 4, 5])
        w = pd.Series([1.0, 2, 1, 4, 2])
        out = weighted_continuous_stats(v, w)
        expected_mean = float(np.average(v, weights=w))
        assert out.mean == pytest.approx(expected_mean, rel=1e-12)

    def test_weighted_sd_unbiased(self):
        # Variance with frequency weights — Bessel-style correction:
        # var = sum w*(x-mean)² / (sum w - 1)
        from pysofra.summary.weights import weighted_continuous_stats
        v = pd.Series([1.0, 2, 3, 4, 5])
        w = pd.Series([1.0, 1, 1, 1, 1])
        # With uniform weight=1, this should equal numpy sample std
        out = weighted_continuous_stats(v, w)
        ref_sd = float(v.std(ddof=1))
        assert out.sd == pytest.approx(ref_sd, rel=1e-6)


# ======================================================================
# SMD — verified against textbook formula
# ======================================================================
class TestSMD_vs_Textbook:
    """Standardized mean difference matches the standard formula."""

    def test_smd_known_values(self):
        # SMD = (mean_1 - mean_2) / sqrt((var_1 + var_2) / 2)
        from pysofra.summary.smd import continuous_smd_pair
        a = np.array([1.0, 2, 3, 4, 5])
        b = np.array([3.0, 4, 5, 6, 7])
        # mean_a=3, mean_b=5, var_a=var_b=2.5
        # pooled = sqrt((2.5 + 2.5) / 2) = sqrt(2.5)
        # SMD magnitude = |3 - 5| / sqrt(2.5) = 1.2649
        expected = abs(3.0 - 5.0) / math.sqrt(2.5)
        assert continuous_smd_pair(a, b) == pytest.approx(expected,
                                                          rel=1e-12)


# ======================================================================
# KAPLAN-MEIER — must match lifelines direct call
# ======================================================================
class TestKM_vs_Lifelines:
    """The median survival reported by tbl_survival must equal the
    median reported by ``KaplanMeierFitter`` directly.
    """

    def test_median_survival_matches_lifelines(self):
        pytest.importorskip("lifelines")
        from lifelines import KaplanMeierFitter
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "time": rng.exponential(20, n),
            "event": rng.integers(0, 2, n),
        })

        kmf = KaplanMeierFitter()
        kmf.fit(df["time"], df["event"])
        ref_med = float(kmf.median_survival_time_)

        t = ps.tbl_survival(df, time="time", event="event")
        # The median row's value cell carries the float
        med_cell = next(
            r.cells[1] for r in t.rows
            if r.cells[0].text.startswith("Median survival")
        )
        assert med_cell.value == pytest.approx(ref_med, rel=1e-9)

    def test_logrank_p_matches_lifelines(self):
        pytest.importorskip("lifelines")
        from lifelines.statistics import multivariate_logrank_test
        rng = np.random.default_rng(0)
        n = 200
        arm = rng.choice(["A", "B"], n)
        df = pd.DataFrame({
            "arm": arm,
            "time": rng.exponential(np.where(arm == "A", 18, 25), n),
            "event": rng.integers(0, 2, n),
        })
        ref = multivariate_logrank_test(df["time"], df["arm"], df["event"])
        ref_p = float(ref.p_value)

        t = ps.tbl_survival(df, time="time", event="event", by="arm")
        # The log-rank p attaches to the median-survival row
        med_row = next(
            r for r in t.rows
            if r.cells[0].text.startswith("Median survival")
        )
        p_cell = next(c for c in med_row.cells if c.kind == "p_value")
        assert p_cell.value == pytest.approx(ref_p, rel=1e-9)


# ======================================================================
# REGRESSION — must match statsmodels direct call
# ======================================================================
class TestRegression_vs_Statsmodels:
    """tbl_regression numbers must match statsmodels' own .params / .pvalues."""

    def test_ols_coefficients_match(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        })
        df["y"] = 1.5 + 2.0 * df["x1"] - 0.5 * df["x2"] + rng.normal(size=n)
        m = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()

        t = ps.tbl_regression(m, intercept=False)
        for name in ("x1", "x2"):
            row = next(r for r in t.rows if r.cells[0].text == name)
            est_cell = row.cells[1]
            ref_est = float(m.params[name])
            assert est_cell.value == pytest.approx(ref_est, rel=1e-9)
            p_cell = next(c for c in row.cells if c.kind == "p_value")
            ref_p = float(m.pvalues[name])
            assert p_cell.value == pytest.approx(ref_p, rel=1e-9)

    def test_logit_or_matches_statsmodels(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = (rng.uniform(size=n) <
                   1.0 / (1.0 + np.exp(-(0.5 + 0.8 * df["x"])))).astype(int)
        m = sm.Logit(df["y"], sm.add_constant(df["x"])).fit(disp=False)

        t = ps.tbl_regression(m, intercept=False, exponentiate=True)
        row = next(r for r in t.rows if r.cells[0].text == "x")
        est_cell = row.cells[1]
        ref_or = float(np.exp(m.params["x"]))
        assert est_cell.value == pytest.approx(ref_or, rel=1e-9)


# ======================================================================
# MI POOLING — verified against hand-computed example
# ======================================================================
class TestPooling_vs_Hand:
    """Rubin's rules: pooled estimate = mean of imputation-specific
    estimates; pooled variance = within + (1 + 1/M) * between."""

    def test_rubin_pooling_hand_example(self):
        # 3 imputations with estimates [1.0, 1.2, 0.8] and within-variances
        # [0.04, 0.04, 0.04]
        # Pooled estimate = 1.0 (mean)
        # Within  W = 0.04 (mean of within-variances)
        # Between B = var of estimates (sample) = ((1-1)² + (1.2-1)² + (0.8-1)²)/2 = 0.04
        # Total T = W + (1 + 1/3) * B = 0.04 + (4/3) * 0.04 = 0.04 + 0.0533 = 0.0933
        # Pooled SE = sqrt(T) = 0.3055...
        # Rubin df = (m-1) * (1 + 1/r)^2 with r = (1+1/m)*B/Ubar = 4/3,
        # so df = 2 * (1 + 3/4)^2 = 6.125.
        import scipy.stats as sp_stats

        from pysofra.models.extract import ModelSummary
        from pysofra.models.pool import pool

        ests = [1.0, 1.2, 0.8]
        ses = [0.2, 0.2, 0.2]  # within-SE → within-var = 0.04
        summaries = []
        for e, se in zip(ests, ses, strict=True):
            # Build a 95% CI symmetric around e using z=1.96
            z = 1.959963984540054
            summaries.append(ModelSummary(
                estimates=pd.Series([e], index=["x"]),
                ci_lo=pd.Series([e - z * se], index=["x"]),
                ci_hi=pd.Series([e + z * se], index=["x"]),
                pvalues=pd.Series([0.5], index=["x"]),
                family="OLS",
                natural_exponentiate=False,
            ))
        pooled = pool(summaries)
        # Pooled estimate
        assert pooled.estimates["x"] == pytest.approx(1.0, abs=1e-12)

        # Mutation-resistant SE check. Recover the SE the pooler used
        # by inverting its t-CI construction. With Ubar = 0.04, B = 0.04,
        # m = 3, the *exact* Rubin total variance is
        #
        #     T = Ubar + (1 + 1/m) * B = 0.04 + (4/3) * 0.04 = 14/150
        #
        # and the Rubin df is (m-1)*(1 + 1/r)^2 with r = (1+1/m)*B/Ubar = 4/3,
        # so df = 2 * (1 + 3/4)^2 = 98/16 = 6.125.
        expected_T = 0.04 + (1.0 + 1.0 / 3.0) * 0.04
        expected_se = math.sqrt(expected_T)
        expected_df = (3 - 1) * (1.0 + 1.0 / (4.0 / 3.0)) ** 2
        t_crit = float(sp_stats.t.ppf(0.975, df=expected_df))
        actual_ci_lo = float(pooled.ci_lo["x"])
        actual_ci_hi = float(pooled.ci_hi["x"])
        actual_half_width = (actual_ci_hi - actual_ci_lo) / 2.0
        actual_se = actual_half_width / t_crit
        # Tight equality: any drift in T, df, or the t-quantile choice
        # surfaces here. Dropping the (1+1/m) factor moves expected_se
        # from sqrt(0.0933) ≈ 0.3055 to sqrt(0.08) ≈ 0.2828, ~7%, well
        # above this tolerance.
        assert actual_se == pytest.approx(expected_se, rel=1e-10)


# ======================================================================
# DESIGN-BASED VARIANCE — hand-computed stratified estimator
# ======================================================================
class TestDesignBasedVariance:
    """Stratified Taylor-linearisation variance against textbook formula."""

    def test_stratified_mean_matches_weighted_average(self):
        from pysofra.summary.design import design_mean_var
        # 2 strata of size 4 each, all weight = 1.
        v = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
        w = pd.Series([1.0] * 8)
        strata = pd.Series(["s1"] * 4 + ["s2"] * 4)
        mean, var, _n = design_mean_var(v, w, strata=strata)
        # With uniform weights, the design mean equals the simple mean
        assert mean == pytest.approx(4.5, rel=1e-12)
        # And the variance must be > 0 (computed from residuals)
        assert var > 0.0

    def test_unweighted_equals_simple_variance(self):
        # With uniform weight = 1, the Taylor-linearised variance estimator
        # reduces algebraically to ``sum((x - mean)²) / (n - 1) · n / (n - 1)``,
        # i.e. the sample variance scaled by ``n / (n - 1)`` rather than
        # divided by ``n``. The exact identity:
        #
        #     var_taylor = (sum((x - mean)²)) · (n / (n - 1)) / (sum w)²
        #                = (sum((x - mean)²) · n) / ((n - 1) · n²)
        #
        # Re-derived directly here so any future drift in the estimator
        # is caught immediately rather than hidden behind slack.
        from pysofra.summary.design import design_mean_var
        rng = np.random.default_rng(0)
        n = 100
        v = pd.Series(rng.normal(size=n))
        w = pd.Series([1.0] * n)
        mean, var, _ = design_mean_var(v, w)
        assert mean == pytest.approx(float(v.mean()), rel=1e-15)
        # Exact formula
        e = v - mean
        expected_var = float((e ** 2).sum()) * (n / (n - 1)) / (w.sum() ** 2)
        assert var == pytest.approx(expected_var, rel=1e-12)


# ======================================================================
# FORMAT — invariants on the formatting helpers
# ======================================================================
class TestFormatInvariants:
    def test_fmt_p_value_journal_rules(self):
        from pysofra.core.format import fmt_p_value
        # Standard journal: p<0.001, p>0.99, else 0.xxx
        assert fmt_p_value(0.0001) == "<0.001"
        assert fmt_p_value(0.995) == ">0.99"
        assert fmt_p_value(0.05) == "0.050"
        assert fmt_p_value(0.123) == "0.123"
        # Boundaries
        assert fmt_p_value(1.5) == "—"   # out of range
        assert fmt_p_value(-0.1) == "—"  # out of range

    def test_fmt_percent_no_unit_drift(self):
        # 0.234 → '23.4' (one decimal by default)
        from pysofra.core.format import fmt_percent
        assert fmt_percent(0.234, digits=1) == "23.4"
        assert fmt_percent(1.0, digits=2) == "100.00"
        assert fmt_percent(0.0, digits=2) == "0.00"


# ======================================================================
# tbl_one — N + percentages must sum correctly
# ======================================================================
class TestTblOneInvariants:
    """Numerical invariants on rendered Table 1 cells."""

    def test_categorical_percentages_sum_to_100(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "race": rng.choice(["A", "B", "C", "D"], 200),
        })
        t = ps.tbl_one(df, variables=["race"], missing="never")
        # Each category cell renders "n (p%)". Sum the percent parts.
        total = 0.0
        for r in t.rows[1:]:  # skip the group-header row
            cell_text = r.cells[1].text
            if "%" not in cell_text:
                continue
            pct_str = cell_text.split("(")[1].split("%")[0]
            total += float(pct_str)
        # Allow 0.5 absolute slack for rounding
        assert abs(total - 100.0) < 0.5

    def test_row_count_matches_variable_count(self):
        # 3 continuous + 1 dichotomous → 4 rows (no missing, no by)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x1": rng.normal(size=80),
            "x2": rng.normal(size=80),
            "x3": rng.normal(size=80),
            "y":  rng.choice([0, 1], size=80),
        })
        t = ps.tbl_one(df, variables=["x1", "x2", "x3", "y"],
                       missing="never")
        # x1, x2, x3, y=1 → 4 body rows
        assert len(t.rows) == 4

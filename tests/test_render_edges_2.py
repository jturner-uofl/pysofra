"""Additional renderer edge-case tests.

These are degenerate-input cases that exercise short-circuit returns in the
stats / effect-size / svy code, plus a few real-world ergonomic paths
(auto-detected predictors, `_safe_exp`, log-rank failures, replicate
weights validation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


# ----------------------------------------------------------------------
# tests.py — degenerate inputs across every dispatch
# ----------------------------------------------------------------------
class TestTestsDegenerate:
    def test_group_arrays_all_nan(self):
        from pysofra.summary.tests import _group_arrays
        v = pd.Series([np.nan, np.nan, np.nan])
        g = pd.Series(["A", "B", "A"])
        assert _group_arrays(v, g) == []

    def test_continuous_test_single_group(self):
        from pysofra.summary.tests import continuous_test
        v = pd.Series([1.0, 2, 3])
        g = pd.Series(["A", "A", "A"])
        r = continuous_test(v, g)
        assert r.p_value is None

    def test_categorical_test_single_level(self):
        from pysofra.summary.tests import categorical_test
        v = pd.Series(["x"] * 5)
        g = pd.Series(["A", "A", "B", "B", "A"])
        r = categorical_test(v, g)
        assert r.p_value is None

    def test_svyttest_empty(self):
        from pysofra.summary.tests import svyttest
        v = pd.Series([np.nan, np.nan])
        g = pd.Series(["A", "B"])
        w = pd.Series([1.0, 2.0])
        r = svyttest(v, g, w)
        assert r.p_value is None

    def test_svyttest_zero_se(self):
        # All within-group values identical → variance 0 → NA
        from pysofra.summary.tests import svyttest
        v = pd.Series([5.0, 5.0, 5.0, 5.0])
        g = pd.Series(["A", "A", "B", "B"])
        w = pd.Series([1.0, 1.0, 1.0, 1.0])
        r = svyttest(v, g, w)
        assert r.p_value is None

    def test_rao_scott_empty(self):
        from pysofra.summary.tests import rao_scott_chisq
        r = rao_scott_chisq(pd.Series([], dtype=object),
                            pd.Series([], dtype=object),
                            pd.Series([], dtype=float))
        assert r.p_value is None

    def test_rao_scott_too_small(self):
        from pysofra.summary.tests import rao_scott_chisq
        r = rao_scott_chisq(pd.Series(["x"] * 5),
                            pd.Series(["A", "A", "B", "B", "A"]),
                            pd.Series([1.0] * 5))
        assert r.p_value is None

    def test_crosstab_empty(self):
        from pysofra.summary.tests import _crosstab
        assert _crosstab(pd.Series([], dtype=object),
                         pd.Series([], dtype=object)) is None

    def test_crosstab_too_small(self):
        from pysofra.summary.tests import _crosstab
        assert _crosstab(pd.Series(["x"] * 5),
                         pd.Series(["A"] * 5)) is None

    def test_chisq_sparse_branch(self):
        from pysofra.summary.tests import _chisq
        # Tiny expected counts to trip the sparse branch.
        obs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        r = _chisq(obs)
        assert "sparse" in r.test

    def test_run_named_test_continuous_insufficient(self):
        from pysofra.summary.tests import run_named_test
        r = run_named_test("welch", pd.Series([1.0, 2]),
                           pd.Series(["A", "A"]), kind="continuous")
        assert r.p_value is None

    def test_run_named_test_categorical_insufficient(self):
        from pysofra.summary.tests import run_named_test
        r = run_named_test("fisher", pd.Series(["x"] * 4),
                           pd.Series(["A", "A", "A", "A"]), kind="categorical")
        assert r.p_value is None


# ----------------------------------------------------------------------
# effect_size.py — degenerate paths
# ----------------------------------------------------------------------
class TestEffectSizeDegenerate:
    def test_eta_zero_variance(self):
        from pysofra.summary.effect_size import eta_squared
        v = pd.Series([1.0, 1, 1, 1])
        g = pd.Series(["A", "A", "B", "B"])
        assert eta_squared(v, g) == 0.0

    def test_omega_empty(self):
        from pysofra.summary.effect_size import omega_squared
        v = pd.Series([np.nan, np.nan])
        g = pd.Series(["A", "B"])
        assert omega_squared(v, g) is None

    def test_omega_single_group(self):
        from pysofra.summary.effect_size import omega_squared
        # k=1 → n - k > 0 but k ≤ 1 → returns None
        assert omega_squared(pd.Series([1.0, 2, 3]),
                             pd.Series(["A", "A", "A"])) is None


# ----------------------------------------------------------------------
# smd.py — degenerate paths
# ----------------------------------------------------------------------
class TestSmdDegenerate:
    def test_continuous_smd_empty(self):
        from pysofra.summary.smd import continuous_smd
        assert continuous_smd(pd.Series([np.nan, np.nan]),
                              pd.Series(["A", "B"])) is None

    def test_continuous_smd_single_group(self):
        from pysofra.summary.smd import continuous_smd
        assert continuous_smd(pd.Series([1.0, 2, 3]),
                              pd.Series(["A", "A", "A"])) is None

    def test_categorical_smd_single_level(self):
        from pysofra.summary.smd import categorical_smd
        # Only one row level → ctab.shape[0] < 2 → returns None
        assert categorical_smd(pd.Series(["x"] * 4),
                               pd.Series(["A", "A", "B", "B"])) is None


# ----------------------------------------------------------------------
# survival.py — sad paths
# ----------------------------------------------------------------------
class TestSurvivalDegeneratePaths:
    def test_group_with_all_nan_falls_through(self):
        pytest.importorskip("lifelines")
        df = pd.DataFrame({
            "arm": ["A"] * 5 + ["B"] * 5,
            "time": [1.0, 2, 3, 4, 5] + [np.nan] * 5,
            "event": [1, 0, 1, 1, 0] + [np.nan] * 5,
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm",
                            show_logrank=False)
        assert any(c.text == "B" for c in t.headers[0].cells)

    def test_survival_at_returns_nan(self):
        from pysofra.models.survival import _survival_at

        class StubKMF:
            def survival_function_at_times(self, t):
                return pd.Series([float("nan")])

        assert _survival_at(StubKMF(), 1.0) is None

    def test_n_at_risk_no_times_before(self):
        from pysofra.models.survival import _n_at_risk

        class StubKMF:
            event_table = pd.DataFrame(
                {"at_risk": [10, 8, 5]}, index=[1.0, 2.0, 3.0],
            )

        # t = 0.5 is before any recorded event → uses iloc[0]
        assert _n_at_risk(StubKMF(), 0.5) == 10


# ----------------------------------------------------------------------
# uvregression.py — auto-detect predictors + safe_exp + edge raises
# ----------------------------------------------------------------------
class TestUvregressionEdges:
    def test_predictors_auto_detected(self):
        # Numeric predictors auto-included; categorical predictors are
        # auto-included AND factor-expanded into one group-header row +
        # one row per non-reference level (see _expand_predictor).
        pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "cat": rng.choice(["a", "b"], size=n),
        })
        t = ps.tbl_uvregression(df, outcome="y")  # predictors=None
        labels = [r.cells[0].text for r in t.rows]
        # 'x' and 'z' are numeric → single rows.
        assert "x" in labels and "z" in labels
        # 'cat' is now factor-expanded → 'cat' group header + one row per level.
        assert "cat" in labels
        assert "a" in labels and "b" in labels

    def test_safe_exp_negative_overflow(self):
        # _safe_exp is a nested helper; exercise it via a Logit fit whose
        # CI lower bound is huge-negative → exp(very negative) = 0.0.
        pytest.importorskip("statsmodels.api")
        import warnings as _w
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"x": rng.normal(size=n)})
        # Perfect-separation-ish
        df["y"] = (df["x"] > 0.5).astype(int)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_uvregression(df, outcome="y", predictors=["x"],
                                    method="Logit", exponentiate=True)
        # We don't assert on values — just that the fit succeeded.
        assert len(t.rows) == 1


# ----------------------------------------------------------------------
# design.py — replicate-weights validation paths
# ----------------------------------------------------------------------
class TestDesignValidation:
    def test_replicate_weights_missing_column(self):
        df = pd.DataFrame({"y": [1.0, 2], "w": [1, 1]})
        d = ps.SurveyDesign(
            weights="w",
            replicate_weights=("nope",),  # not in df
            replicate_type="jk1",
        )
        with pytest.raises(KeyError, match="replicate_weights"):
            d.validate(df)

    def test_replicate_type_invalid(self):
        df = pd.DataFrame({"y": [1.0, 2], "w": [1, 1], "r": [0.5, 0.5]})
        d = ps.SurveyDesign(
            weights="w",
            replicate_weights=("r",),
            replicate_type="bogus",  # invalid
        )
        with pytest.raises(ValueError, match="replicate_type"):
            d.validate(df)

    def test_design_mean_var_zero_weights(self):
        from pysofra.summary.design import design_mean_var
        v = pd.Series([1.0, 2, 3])
        w = pd.Series([0.0, 0.0, 0.0])  # all-zero → degenerate
        m, var, n = design_mean_var(v, w)
        assert np.isnan(m) and np.isnan(var) and n == 0.0

    def test_design_mean_var_single_cluster(self):
        # Only one PSU overall → cluster-robust variance is undefined;
        # pysofra reports 0 and warns (R survey would error).
        from pysofra.summary.design import design_mean_var
        v = pd.Series([1.0, 2, 3, 4])
        w = pd.Series([1.0] * 4)
        cluster = pd.Series(["c1"] * 4)
        with pytest.warns(UserWarning, match=r"only one cluster"):
            m, var, _n = design_mean_var(v, w, cluster=cluster)
        assert var == 0.0


# ----------------------------------------------------------------------
# extras.py — small reachable remnants
# ----------------------------------------------------------------------
class TestExtrasRemnants:
    def test_color_scale_if_short_row_skipped(self):
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable
        from pysofra.summary.extras import color_scale_if

        t = SofraTable(
            rows=(
                Row(cells=(Cell(text="row1"),)),  # only 1 cell — column=2 too high
                Row(cells=(Cell(text="row2"), Cell(text="2.0", value=2.0),
                           Cell(text="3.0", value=3.0))),
                Row(cells=(Cell(text="row3"), Cell(text="4.0", value=4.0),
                           Cell(text="5.0", value=5.0))),
            ),
            headers=(HeaderRow(cells=(
                HeaderCell(text="V"), HeaderCell(text="A"), HeaderCell(text="B"),
            )),),
        )
        t2 = color_scale_if(t, column=2)
        assert t2 is not None
        assert len(t2.rows) == 3  # short row preserved, others coloured


# ----------------------------------------------------------------------
# regression.py + extract.py — small remnants
# ----------------------------------------------------------------------
class TestExtractRemnants:
    def test_lifelines_no_p_column(self):
        # A lifelines-style summary that lacks 'p' but has CI columns.
        from pysofra.models.extract import _extract_lifelines

        class FakeFitter:
            summary = pd.DataFrame({
                "coef": [0.1],
                "coef lower 95%": [0.0],
                "coef upper 95%": [0.2],
                # no 'p' column — should still extract (p=nan)
            }, index=["x"])

        fake = FakeFitter()
        fake.__class__.__module__ = "lifelines.fitters.fake"
        result = _extract_lifelines(fake, conf_level=0.95)
        assert "x" in result.estimates.index

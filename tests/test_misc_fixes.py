"""Targeted regression tests for previously-fixed defects.

Each test pins a specific bug so that any regression is easy to
locate by name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


# ----------------------------------------------------------------------
# Predictor / adjust_for overlap raises cleanly
# ----------------------------------------------------------------------
class TestUvregressionOverlapErrors:
    def test_predictor_also_in_adjust_for_raises(self):
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({
            "y": np.arange(50, dtype=float),
            "x": np.arange(50, dtype=float),
            "z": np.arange(50, dtype=float),
        })
        # Without the guard the column slice returns a DataFrame and
        # _expand_predictor crashes with an obscure AttributeError.
        with pytest.raises(ValueError, match="also appear in adjust_for"):
            ps.tbl_uvregression(df, outcome="y", predictors=["x"],
                                adjust_for=["x"])

    def test_outcome_in_predictors_raises(self):
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"y": [1.0, 2, 3], "x": [4.0, 5, 6]})
        with pytest.raises(ValueError, match="outcome"):
            ps.tbl_uvregression(df, outcome="y", predictors=["y", "x"])

    def test_outcome_in_adjust_for_raises(self):
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"y": [1.0, 2, 3], "x": [4.0, 5, 6]})
        with pytest.raises(ValueError, match="outcome"):
            ps.tbl_uvregression(df, outcome="y", predictors=["x"],
                                adjust_for=["y"])


# ----------------------------------------------------------------------
# Formula-API model works through the design refit
# ----------------------------------------------------------------------
class TestFormulaAPIRoundTrip:
    def test_design_refit_with_smf_ols(self):
        smf = pytest.importorskip("statsmodels.formula.api")
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "x": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], n),
            "w": rng.uniform(0.5, 2, n),
        })
        df["y"] = 2 + df["x"] + rng.normal(size=n)
        m = smf.ols("y ~ x + C(g)", data=df).fit()
        d = ps.SurveyDesign(weights="w")
        t = ps.tbl_regression(m, intercept=False, design=d, data=df)
        # Refit must produce rows for every formula-expanded term.
        labels = [r.cells[0].text for r in t.rows]
        assert "x" in labels
        assert any("g" in lbl for lbl in labels)


# ----------------------------------------------------------------------
# High-cardinality factor scales reasonably
# ----------------------------------------------------------------------
class TestHighCardinalityFactor:
    def test_50_level_factor_completes_quickly(self):
        pytest.importorskip("statsmodels.api")
        import time
        rng = np.random.default_rng(0)
        n = 2000
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "site": rng.choice([f"site_{i:02d}" for i in range(50)], n),
        })
        t0 = time.perf_counter()
        t = ps.tbl_uvregression(df, outcome="y", predictors=["site"])
        elapsed = time.perf_counter() - t0
        # 1 group header + 50 level rows
        assert len(t.rows) == 51
        # Should not take more than a second on a modern laptop.
        assert elapsed < 2.0, f"factor expansion too slow: {elapsed*1000:.0f} ms"


# ----------------------------------------------------------------------
# Newcombe handles imbalanced n
# ----------------------------------------------------------------------
class TestNewcombeImbalanced:
    def test_imbalanced_n_produces_valid_ci(self):
        # n1 = 8, n2 = 500 — extreme imbalance. Newcombe should still
        # produce lo < diff < hi inside [-1, 1].
        df = pd.DataFrame({
            "arm": ["A"] * 8 + ["B"] * 500,
            "y": [1, 1, 1, 0, 0, 0, 0, 0] + [1] * 250 + [0] * 250,
        })
        t = ps.tbl_one(df, by="arm", variables=["y"],
                       types={"y": "dichotomous"},
                       missing="never").add_difference()
        row = next(r for r in t.rows if r.cells[0].text.startswith("y ="))
        diff_cell = next(c for c in row.cells if c.kind == "ci"
                         and isinstance(c.value, tuple))
        d, lo, hi = diff_cell.value
        assert -1.0 <= lo <= d <= hi <= 1.0


# ----------------------------------------------------------------------
# Doc/code consistency — add_global_p docstring NO LONGER says "planned"
# ----------------------------------------------------------------------
class TestAddGlobalPDocstringHonest:
    def test_docstring_does_not_say_planned(self):
        from pysofra.core.table import SofraTable
        doc = SofraTable.add_global_p.__doc__ or ""
        assert "planned" not in doc.lower()
        # And it does describe the actual behaviour.
        assert "tbl_regression" in doc
        assert "NotImplementedError" in doc

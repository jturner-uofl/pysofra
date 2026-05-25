"""Tests for the Rao–Scott corrected chi-square for weighted data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.summary.tests import rao_scott_chisq


class TestRaoScottDirect:
    def test_equal_weights_match_unweighted_chisq(self):
        rng = np.random.default_rng(0)
        n = 400
        v = rng.choice(["a", "b", "c"], n)
        g = rng.choice(["X", "Y"], n)
        w = pd.Series([1.0] * n)
        rs = rao_scott_chisq(pd.Series(v), pd.Series(g), w)
        assert rs.test == "Rao–Scott chi-square"
        # With equal weights, DEFF = 1 → corrected p ≈ raw chi-sq p.
        from pysofra.summary.tests import categorical_test
        raw = categorical_test(pd.Series(v), pd.Series(g))
        assert rs.p_value == pytest.approx(raw.p_value, rel=1e-6, abs=1e-6)

    def test_returns_test_name(self):
        v = pd.Series(["a", "a", "b", "b"] * 10)
        g = pd.Series(["X", "Y"] * 20)
        w = pd.Series([1.0] * 40)
        rs = rao_scott_chisq(v, g, w)
        assert "Rao" in rs.test


class TestRaoScottInTblOne:
    def test_weighted_tbl_one_uses_rao_scott(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "race": rng.choice(["W", "B", "A"], n),
            "w":    rng.uniform(0.5, 2.5, n),
        })
        t = ps.tbl_one(df, by="arm", weights="w").add_p()
        assert any("Rao" in f for f in t.footnotes)

    def test_unweighted_path_still_uses_default(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").add_p()
        # No Rao-Scott footnote
        assert not any("Rao" in f for f in t.footnotes)

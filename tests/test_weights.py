"""Tests for weighted Table 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.summary.weights import (
    weighted_categorical_stats,
    weighted_continuous_stats,
)


class TestWeightedContinuousStats:
    def test_equal_weights_matches_unweighted(self):
        s = pd.Series([1.0, 2, 3, 4, 5])
        w = pd.Series([1.0] * 5)
        st = weighted_continuous_stats(s, w)
        assert st.mean == pytest.approx(3.0)
        assert st.sd == pytest.approx(np.sqrt(2.5))
        assert st.median == pytest.approx(3.0)

    def test_weights_shift_mean(self):
        # Heavy weight on 10 should pull the mean upward.
        s = pd.Series([1.0, 10.0])
        w = pd.Series([1.0, 9.0])
        st = weighted_continuous_stats(s, w)
        assert st.mean == pytest.approx((1 + 9 * 10) / 10)

    def test_handles_missing(self):
        s = pd.Series([1.0, np.nan, 3.0])
        w = pd.Series([1.0, 1.0, 1.0])
        st = weighted_continuous_stats(s, w)
        assert st.n_eff == pytest.approx(2.0)
        assert st.n_missing == pytest.approx(1.0)
        assert st.mean == pytest.approx(2.0)

    def test_zero_weight_excludes_row(self):
        s = pd.Series([1.0, 100.0, 3.0])
        w = pd.Series([1.0, 0.0, 1.0])  # zero weight on outlier
        st = weighted_continuous_stats(s, w)
        assert st.mean == pytest.approx(2.0)


class TestWeightedCategoricalStats:
    def test_basic(self):
        s = pd.Series(["a", "a", "b", "b", "b"])
        w = pd.Series([1.0, 1.0, 2.0, 2.0, 2.0])
        st = weighted_categorical_stats(s, w)
        assert st.counts["a"] == pytest.approx(2.0)
        assert st.counts["b"] == pytest.approx(6.0)


class TestWeightedTblOne:
    def test_weights_kwarg_propagates(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "age": rng.normal(60, 10, n),
            "sex": rng.choice(["F", "M"], n),
            "w":   rng.uniform(0.5, 2.0, n),
        })
        t = ps.tbl_one(df, by="arm", weights="w")
        # weights column excluded from variables list
        labels = [r.cells[0].text for r in t.rows]
        assert all(not label.startswith("w") for label in labels)
        # Header N values are floats (weighted) — should contain a decimal
        # or be different from raw row counts.
        n_a_raw = int((df.arm == "A").sum())
        header_a = t.headers[0].cells[1].text
        n_a_weighted = float(df.loc[df.arm == "A", "w"].sum())
        assert f"{n_a_weighted:,.1f}" in header_a or str(n_a_raw) not in header_a

    def test_unknown_weights_col_raises(self, small_trial):
        with pytest.raises(KeyError):
            ps.tbl_one(small_trial, by="arm", weights="nope")

    def test_equal_weights_match_unweighted(self):
        rng = np.random.default_rng(7)
        n = 100
        df = pd.DataFrame({
            "arm": ["A"] * 50 + ["B"] * 50,
            "age": rng.normal(60, 5, n),
            "w":   [1.0] * n,
        })
        t_w = ps.tbl_one(df, by="arm", weights="w").to_dict()
        t_u = ps.tbl_one(df.drop(columns="w"), by="arm").to_dict()
        # Body row labels and column structure should align
        assert len(t_w["rows"]) == len(t_u["rows"])
        # Continuous summary text should agree to digits
        assert t_w["rows"][0][0] == t_u["rows"][0][0]

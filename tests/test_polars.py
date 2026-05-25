"""Tests for polars DataFrame input support."""

from __future__ import annotations

import pytest

import pysofra as ps
from pysofra.core.frames import to_pandas


class TestFrameAdapter:
    def test_pandas_passthrough(self, small_trial):
        out = to_pandas(small_trial)
        assert out is small_trial  # no copy

    def test_polars_to_pandas(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        out = to_pandas(df)
        assert list(out.columns) == ["x", "y"]
        assert out.shape == (3, 2)

    def test_polars_lazyframe(self):
        pl = pytest.importorskip("polars")
        df = pl.LazyFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        out = to_pandas(df)
        assert list(out.columns) == ["x", "y"]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            to_pandas([{"x": 1}, {"x": 2}])


class TestPolarsTblOne:
    def test_polars_input_works(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({
            "arm": ["A"] * 30 + ["B"] * 30,
            "age": [50 + (i % 20) for i in range(60)],
            "sex": (["F", "M"] * 30),
        })
        t = ps.tbl_one(df, by="arm").add_p()
        # Headers: Characteristic, A, B, p-value
        assert len(t.headers[0].cells) == 4
        assert len(t.rows) >= 2

    def test_polars_matches_pandas(self, small_trial):
        pl = pytest.importorskip("polars")
        pl_df = pl.from_pandas(small_trial)
        t_pl = ps.tbl_one(pl_df, by="arm").add_p().to_dict()
        t_pd = ps.tbl_one(small_trial, by="arm").add_p().to_dict()
        assert t_pl["headers"] == t_pd["headers"]
        assert t_pl["rows"] == t_pd["rows"]

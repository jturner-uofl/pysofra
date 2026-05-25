"""Tests for inline plot embedding (forest + KM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def logit_table():
    sm = pytest.importorskip("statsmodels.api")
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(27, 4, n),
        "sex_M": rng.integers(0, 2, n),
    })
    df["event"] = (df["age"] / 10 + rng.normal(0, 1, n) > 6).astype(int)
    X = sm.add_constant(df[["age", "bmi", "sex_M"]])
    fit = sm.Logit(df["event"], X).fit(disp=False)
    return ps.tbl_regression(fit, exponentiate=True)


@pytest.fixture
def survival_table():
    pytest.importorskip("lifelines")
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "arm":   rng.choice(["A", "B"], n),
        "time":  rng.exponential(24, n),
        "event": rng.integers(0, 2, n),
    })
    return ps.tbl_survival(df, time="time", event="event", by="arm", times=[12, 24])


class TestForestPlot:
    def test_attach_and_render(self, logit_table):
        plotted = logit_table.with_forest_plot()
        assert plotted.inline_svg is not None
        assert "<svg" in plotted.inline_svg

    def test_html_embeds_svg_above(self, logit_table):
        html = logit_table.with_forest_plot(position="above").to_html()
        svg_idx = html.find("<svg")
        table_idx = html.find("<table")
        assert 0 < svg_idx < table_idx

    def test_html_embeds_svg_below(self, logit_table):
        html = logit_table.with_forest_plot(position="below").to_html()
        svg_idx = html.find("<svg")
        table_idx = html.find("<table")
        assert table_idx < svg_idx

    def test_forest_plot_with_no_regression_rows_raises(self):
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable
        from pysofra.plot.forest import forest_plot_svg

        t = SofraTable(
            rows=(Row(cells=(Cell(text="x"),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="x"),)),),
        )
        with pytest.raises(ValueError):
            forest_plot_svg(t)


class TestKMPlot:
    def test_with_km_plot_embeds_svg(self, survival_table):
        plotted = survival_table.with_km_plot()
        assert plotted.inline_svg is not None
        assert "<svg" in plotted.inline_svg

    def test_km_html_includes_curves(self, survival_table):
        html = survival_table.with_km_plot().to_html()
        assert "<svg" in html
        assert "<table" in html

    def test_no_km_plot_without_source_raises(self):
        from pysofra.core.schema import Cell, Row
        from pysofra.core.table import SofraTable

        t = SofraTable(rows=(Row(cells=(Cell(text=""),)),))
        with pytest.raises(ValueError):
            t.with_km_plot()


class TestInlineSvgPlumbing:
    def test_invalid_position_raises(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        with pytest.raises(ValueError):
            t.with_inline_svg("<svg/>", position="middle")

    def test_inline_svg_default_position(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").with_inline_svg("<svg id='x'/>")
        assert t.inline_svg_position == "above"
        assert t.inline_svg == "<svg id='x'/>"

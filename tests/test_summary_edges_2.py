"""Descriptive-summary edge-case tests (companion file).

Targets the leftover uncovered paths in: plot/forest, plot/km,
render/latex, summary/extras, models/survival, summary/weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row, SpanningHeader
from pysofra.core.table import SofraTable


# ======================================================================
# plot/forest.py — forest_plot_svg cleanly produces SVG; _isnan tolerates types
# ======================================================================
class TestForestPlotSvg:
    def test_returns_svg_string(self):
        sm = pytest.importorskip("statsmodels.api")
        pytest.importorskip("matplotlib")
        from pysofra.plot.forest import forest_plot_svg
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)})
        fit = sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit()
        t = ps.tbl_regression(fit)
        svg = forest_plot_svg(t)
        assert svg.startswith("<svg")

    def test_isnan_helper_handles_string(self):
        from pysofra.plot.forest import _isnan
        assert _isnan("not_a_number") is False
        assert _isnan(None) is False
        assert _isnan(float("nan")) is True

    def test_skip_row_with_nan_cells(self):
        # Forest plot should silently skip rows where est/lo/hi is NaN
        pytest.importorskip("statsmodels.api")
        pytest.importorskip("matplotlib")
        from pysofra.plot.forest import forest_plot
        # Build a table with one numeric row and one all-NaN row
        rows = (
            Row(cells=(
                Cell(text="good"),
                Cell(text="1.0", value=1.0, kind="numeric"),
                Cell(text="0.5, 1.5", value=(0.5, 1.5), kind="ci"),
                Cell(text="0.01", value=0.01, kind="p_value"),
            )),
            Row(cells=(
                Cell(text="bad"),
                Cell(text="—", value=float("nan"), kind="numeric"),
                Cell(text="—, —", value=(float("nan"), float("nan")), kind="ci"),
                Cell(text="—", value=float("nan"), kind="p_value"),
            )),
        )
        hdr = (HeaderRow(cells=(
            HeaderCell(text="Variable"),
            HeaderCell(text="β"),
            HeaderCell(text="95% CI"),
            HeaderCell(text="p-value"),
        )),)
        t = SofraTable(rows=rows, headers=hdr)
        plot = forest_plot(t)
        assert plot.svg.startswith("<svg")


# ======================================================================
# plot/km.py — km_curve_svg + risk-table edge cases
# ======================================================================
class TestKmSvg:
    def test_km_curve_svg_returns_string(self):
        pytest.importorskip("lifelines")
        pytest.importorskip("matplotlib")
        from pysofra.plot.km import km_curve_svg
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm":   rng.choice(["A", "B"], 100),
            "time":  rng.exponential(24, 100),
            "event": rng.integers(0, 2, 100),
        })
        svg = km_curve_svg(df, time="time", event="event", by="arm")
        assert svg.startswith("<svg")

    def test_km_overall_only_no_legend(self):
        pytest.importorskip("lifelines")
        pytest.importorskip("matplotlib")
        from pysofra.plot.km import km_curve_svg
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "time":  rng.exponential(24, 50),
            "event": rng.integers(0, 2, 50),
        })
        svg = km_curve_svg(df, time="time", event="event")
        assert "<svg" in svg

    def test_skip_empty_group(self):
        pytest.importorskip("lifelines")
        pytest.importorskip("matplotlib")
        from pysofra.plot.km import km_curve
        # One arm has *all* NaN time → that subset is empty after dropna
        df = pd.DataFrame({
            "arm":   ["A"] * 10 + ["B"] * 10,
            "time":  list(np.arange(1, 11.0)) + [float("nan")] * 10,
            "event": [1] * 10 + [0] * 10,
        })
        plot = km_curve(df, time="time", event="event", by="arm")
        assert plot.svg.startswith("<svg")

    def test_n_at_risk_helper_empty_fallback(self):
        # _n_at_risk's "no time <= t" fallback returns event_table's first
        # row's at_risk.
        from pysofra.plot.km import _n_at_risk
        # Create a minimal stub that the helper can interrogate.
        class StubKMF:
            event_table = pd.DataFrame(
                {"at_risk": [100, 80, 60]},
                index=[10.0, 20.0, 30.0],
            )
        # t=5 → no idx ≤ 5 → returns first at_risk value
        assert _n_at_risk(StubKMF(), 5.0) == 100


# ======================================================================
# render/latex.py — multicolumn-gap padding + alignment cases
# ======================================================================
class TestLatexLayoutEdgeCases:
    def test_spanning_header_gap_padding(self):
        # Spanning header that skips some columns → gap cells emitted
        rows = (Row(cells=(
            Cell(text="X"), Cell(text="1"), Cell(text="2"), Cell(text="3"),
        )),)
        hdr = (HeaderRow(cells=tuple(HeaderCell(text=t)
                                      for t in ["V", "A", "B", "C"])),)
        # Span only covers cols 2..3, leaving col 1 as a gap
        spans = (SpanningHeader(label="GroupBC", start=2, end=3),)
        tex = SofraTable(rows=rows, headers=hdr,
                         spanning_headers=spans).to_latex()
        assert r"\multicolumn{2}{c}{GroupBC}" in tex

    def test_align_marker_right(self):
        # First column always left, second column right
        rows = (Row(cells=(
            Cell(text="X"), Cell(text="1.0", align="right"),
        )),)
        hdr = (HeaderRow(cells=(
            HeaderCell(text="V"), HeaderCell(text="N", align="right"),
        )),)
        tex = SofraTable(rows=rows, headers=hdr).to_latex()
        assert r"\begin{tabular}{lr}" in tex

    def test_align_marker_left(self):
        rows = (Row(cells=(Cell(text="X"), Cell(text="Y", align="left"))),)
        hdr = (HeaderRow(cells=(
            HeaderCell(text="V"), HeaderCell(text="N", align="left"),
        )),)
        tex = SofraTable(rows=rows, headers=hdr).to_latex()
        assert r"\begin{tabular}{ll}" in tex

    def test_italic_cell_run(self):
        rows = (Row(cells=(Cell(text="X", italic=True),)),)
        hdr = (HeaderRow(cells=(HeaderCell(text="V"),)),)
        tex = SofraTable(rows=rows, headers=hdr).to_latex()
        assert r"\textit{X}" in tex

    def test_plot_below_in_to_latex_file(self, tmp_path):
        sm = pytest.importorskip("statsmodels.api")
        pytest.importorskip("matplotlib")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)})
        fit = sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit()
        out = tmp_path / "below.tex"
        ps.tbl_regression(fit).with_forest_plot(position="below").to_latex_file(out)
        assert out.exists()
        assert (tmp_path / "below_plot.pdf").exists()
        # Plot command should come after the tabular block
        src = out.read_text()
        tab_idx = src.find(r"\end{tabular}")
        inc_idx = src.find(r"\includegraphics")
        assert tab_idx < inc_idx

    def test_part_with_link(self, small_trial):
        from pysofra.core.schema import CellPart
        from pysofra.render.latex import _render_part_tex
        out = _render_part_tex(CellPart(text="docs", link="https://example.com"))
        assert "https://example.com" in out
        assert r"\href" in out

    def test_part_with_subscript(self, small_trial):
        from pysofra.core.schema import CellPart
        from pysofra.render.latex import _render_part_tex
        out = _render_part_tex(CellPart(text="2", subscript=True))
        assert r"\textsubscript{2}" in out

    def test_part_with_code(self):
        from pysofra.core.schema import CellPart
        from pysofra.render.latex import _render_part_tex
        out = _render_part_tex(CellPart(text="x", code=True))
        assert r"\texttt" in out

    def test_part_with_bold(self):
        from pysofra.core.schema import CellPart
        from pysofra.render.latex import _render_part_tex
        out = _render_part_tex(CellPart(text="X", bold=True))
        assert r"\textbf{X}" in out

    def test_no_headers_column_spec(self):
        rows = (Row(cells=(Cell(text="X"), Cell(text="Y"))),)
        tex = SofraTable(rows=rows).to_latex()
        # No headers → all "l" columns
        assert r"\begin{tabular}{ll}" in tex


# ======================================================================
# summary/extras.py — uncovered branches
# ======================================================================
class TestExtrasMoreEdges:
    def test_add_difference_with_no_rebuild_raises(self):
        # SofraTable without spec / rebuild
        from pysofra.summary.extras import add_difference
        t = SofraTable(rows=(Row(cells=(Cell(text="X"),)),))
        with pytest.raises(ValueError):
            add_difference(t)

    def test_add_ci_with_no_rebuild_raises(self):
        from pysofra.summary.extras import add_ci
        t = SofraTable(rows=(Row(cells=(Cell(text="X"),)),))
        with pytest.raises(ValueError):
            add_ci(t)

    def test_add_difference_with_categorical_dtype_column(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "sex": pd.Categorical(rng.choice(["F", "M"], n)),
        })
        t = ps.tbl_one(df, by="arm", variables=["sex"]).add_difference()
        sex_row = next(r for r in t.rows if "sex" in r.cells[0].text)
        diff_cell = next(c for c in sex_row.cells if c.kind == "ci")
        assert diff_cell.text != ""

    def test_add_ci_with_overall_column(self):
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "age": rng.normal(60, 10, n),
        })
        t = ps.tbl_one(df, by="arm",
                       variables=["age"]).add_overall().add_ci()
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        # Three group columns now: Overall, A, B — each gets a CI
        cis = sum(1 for c in age_row.cells[1:] if "[" in c.text)
        assert cis >= 1

    def test_wilson_ci_zero_n(self):
        from pysofra.summary.extras import _wilson_ci
        lo, hi = _wilson_ci(0, 0, z=1.96)
        import math
        assert math.isnan(lo) and math.isnan(hi)

    def test_find_variable_for_row_no_match(self):
        from pysofra.summary.extras import _find_variable_for_row
        result = _find_variable_for_row("unknown", ("age", "bmi"),
                                         {"age": "continuous", "bmi": "continuous"})
        assert result is None

    def test_isnan_helper_tolerates_strings(self):
        from pysofra.summary.extras import _isnan
        assert _isnan("not_a_number") is False
        assert _isnan(float("nan")) is True

    def test_data_from_rebuild_no_closure(self):
        from pysofra.summary.extras import _data_from_rebuild
        def fake_rebuild(*args):
            return None
        # No closure cells → returns None
        assert _data_from_rebuild(fake_rebuild) is None


# ======================================================================
# models/survival.py — empty group + log-rank failure + extraction edges
# ======================================================================
class TestSurvivalMoreEdges:
    def test_unknown_by_column(self):
        pytest.importorskip("lifelines")
        df = pd.DataFrame({
            "time":  [1.0, 2, 3],
            "event": [1, 0, 1],
        })
        with pytest.raises(KeyError, match="by column"):
            ps.tbl_survival(df, time="time", event="event", by="nope")

    def test_categorical_by_uses_cat_order(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "arm":   pd.Categorical(rng.choice(["A", "B", "C"], n),
                                     categories=["C", "B", "A"],  # custom order
                                     ordered=True),
            "time":  rng.exponential(24, n),
            "event": rng.integers(0, 2, n),
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm")
        # First non-label header should be 'C' (the first cat)
        first = t.headers[0].cells[1].text
        assert first == "C"

    def test_empty_group_fits_none(self):
        pytest.importorskip("lifelines")
        # Group B has no rows after dropna → fits[B] = None branch
        df = pd.DataFrame({
            "arm":   ["A"] * 10 + ["B"] * 0,
            "time":  np.arange(1.0, 11),
            "event": [1] * 10,
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm")
        # Just confirm it didn't crash and produced some rows
        assert len(t.rows) >= 3

    def test_with_km_plot_invalid_position(self):
        pytest.importorskip("lifelines")
        pytest.importorskip("matplotlib")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm":   rng.choice(["A", "B"], 100),
            "time":  rng.exponential(24, 100),
            "event": rng.integers(0, 2, 100),
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm")
        with pytest.raises(ValueError, match="position"):
            t.with_km_plot(position="upside-down")

    def test_n_at_risk_below_first_event(self):
        # Confirm _n_at_risk fallback when t < first index
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm":   rng.choice(["A", "B"], 50),
            "time":  rng.exponential(24, 50) + 5,
            "event": rng.integers(0, 2, 50),
        })
        # times=[0] → before any event → returns at_risk[0] (initial)
        t = ps.tbl_survival(df, time="time", event="event", by="arm",
                            times=[0])
        labels = [r.cells[0].text for r in t.rows]
        assert any("S(" in lab for lab in labels)


# ======================================================================
# summary/weights.py — all-NaN, length mismatch, categorical levels
# ======================================================================
class TestWeightsEdges:
    def test_length_mismatch_raises(self):
        from pysofra.summary.weights import weighted_continuous_stats
        with pytest.raises(ValueError, match="same length"):
            weighted_continuous_stats(
                pd.Series([1.0, 2, 3]),
                pd.Series([1.0, 2]),  # different length
            )

    def test_all_nan_continuous(self):
        from pysofra.summary.weights import weighted_continuous_stats
        st = weighted_continuous_stats(
            pd.Series([float("nan"), float("nan")]),
            pd.Series([1.0, 1.0]),
        )
        assert st.n_eff == 0
        import math
        assert math.isnan(st.mean)

    def test_categorical_dtype_levels(self):
        from pysofra.summary.weights import weighted_categorical_stats
        s = pd.Series(pd.Categorical(["a", "b", "a"],
                                      categories=["a", "b", "c"]))
        w = pd.Series([1.0, 1, 1])
        st = weighted_categorical_stats(s, w)
        # Should include all 3 categories even though 'c' is absent
        assert "c" in st.levels

    def test_explicit_levels_drop_unseen(self):
        from pysofra.summary.weights import weighted_categorical_stats
        s = pd.Series(["a", "b", "a"])
        w = pd.Series([1.0, 1, 1])
        st = weighted_categorical_stats(s, w, levels=["a", "b"])
        assert list(st.levels) == ["a", "b"]


# ======================================================================
# core/format.py — last formatter branches
# ======================================================================
class TestFormatLast:
    def test_fmt_median_iqr(self):
        from pysofra.core.format import fmt_median_iqr
        assert fmt_median_iqr(2.0, 1.0, 3.0, digits=1) == "2.0 (1.0, 3.0)"

    def test_fmt_ci_with_custom_separator(self):
        from pysofra.core.format import fmt_ci
        assert fmt_ci(1.0, 2.0, sep=" — ") == "1.00 — 2.00"

    def test_fmt_n_pct_zero_pct_digits(self):
        from pysofra.core.format import fmt_n_pct
        assert fmt_n_pct(5, 10, digits=0) == "5 (50%)"


# ======================================================================
# render/html.py — a few remaining branches
# ======================================================================
class TestHtmlLast:
    def test_max_height_only_no_sticky(self, small_trial):
        h = ps.tbl_one(small_trial, by="arm").to_html(max_height="100px")
        assert "max-height:100px" in h
        assert "overflow-y:auto" in h

    def test_part_with_color_renders_span(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").compose(
            0, 0, [ps.CellPart("X", color="#0b3d91")],
        )
        h = t.to_html()
        assert "color:#0b3d91" in h

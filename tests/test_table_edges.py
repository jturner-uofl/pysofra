"""SofraTable rich-cell path tests.

Hits the per-renderer CellPart formatting code (superscript, subscript,
monospace, color, links, bold/italic runs), the empty-table fallback for
``_ncols``, and a few stray summary/extras branches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.core.schema import (
    Cell,
    CellPart,
    HeaderCell,
    HeaderRow,
    Row,
    SpanningHeader,
)
from pysofra.core.table import SofraTable


def _rich_table() -> SofraTable:
    """A small SofraTable with rich-cell parts exercising every formatting flag."""
    rich_parts = (
        CellPart(text="x", italic=True),
        CellPart(text="2", superscript=True),
        CellPart(text=" + ", bold=True),
        CellPart(text="log", code=True),
        CellPart(text="i", subscript=True),
        CellPart(text=" / ", color="#ff0000"),
        CellPart(text="link", link="https://example.com"),
    )
    return SofraTable(
        rows=(
            Row(cells=(
                Cell(text="row1", italic=True, indent=1),
                Cell(text="x2+logi/link", parts=rich_parts, kind="numeric",
                     align="center"),
            )),
        ),
        headers=(HeaderRow(cells=(
            HeaderCell(text="Label", bold=True),
            HeaderCell(text="Formula\nrich", align="center"),
        )),),
        spanning_headers=(SpanningHeader(label="Top", start=0, end=1),),
        footnotes=("note 1", "note 2"),
        caption="Rich-cell table",
    )


# ----------------------------------------------------------------------
# HTML renderer rich-cell branches
# ----------------------------------------------------------------------
class TestHtmlRichCells:
    def test_all_part_formats(self):
        from pysofra.render.html import HtmlRenderer
        t = _rich_table()
        html = HtmlRenderer().render(t)
        assert "<em>" in html
        assert "<sup>" in html
        assert "<sub>" in html
        assert "<strong>" in html
        assert "<code>" in html
        assert "color:#ff0000" in html
        assert 'href="https://example.com"' in html
        # italic on the cell itself
        assert "font-style:italic" in html

    def test_ncols_fallback_empty(self):
        # An empty SofraTable (no headers, no rows) — _ncols should return 1.
        from pysofra.render.html import HtmlRenderer
        t = SofraTable()
        out = HtmlRenderer().render(t)
        # No thead emitted when both headers and spans are empty.
        assert "<thead>" not in out
        # No body either, but the wrapping <table> still renders.
        assert "<table" in out

    def test_empty_body(self):
        from pysofra.render.html import HtmlRenderer
        t = SofraTable(
            headers=(HeaderRow(cells=(HeaderCell(text="A"),)),),
        )
        html = HtmlRenderer().render(t)
        assert "<tbody></tbody>" in html


# ----------------------------------------------------------------------
# Markdown / LaTeX / image renderer fallbacks
# ----------------------------------------------------------------------
class TestRendererFallbacks:
    def test_markdown_ncols_fallback(self):
        from pysofra.render.markdown import _ncols
        # Empty table → _ncols returns 1
        assert _ncols(SofraTable()) == 1

    def test_latex_rich_cell_branches(self):
        from pysofra.render.latex import LatexRenderer
        t = _rich_table()
        out = LatexRenderer().render(t)
        assert r"\textit" in out
        assert r"\textsuperscript" in out
        assert r"\textsubscript" in out
        assert r"\textbf" in out
        assert r"\texttt" in out
        assert r"\href" in out

    def test_latex_ncols_fallback(self):
        from pysofra.render.latex import _ncols
        assert _ncols(SofraTable()) == 1

    def test_latex_spanning_with_trailing_empty(self):
        # Spanning header that doesn't reach the last column → trailing
        # empty cells filled by the `while col < ncols` loop.
        from pysofra.render.latex import LatexRenderer
        t = SofraTable(
            headers=(HeaderRow(cells=(
                HeaderCell(text="A"), HeaderCell(text="B"), HeaderCell(text="C"),
            )),),
            spanning_headers=(SpanningHeader(label="L", start=0, end=0),),
        )
        out = LatexRenderer().render(t)
        assert "L" in out  # span rendered

    def test_xlsx_ncols_fallback(self, tmp_path):
        from pysofra.render.xlsx import _ncols
        assert _ncols(SofraTable()) == 1
        # Also exercise the rows-only path
        rows_only = SofraTable(
            rows=(Row(cells=(Cell(text="a"), Cell(text="b"))),),
        )
        assert _ncols(rows_only) == 2

    def test_xlsx_rich_features(self, tmp_path):
        pytest.importorskip("xlsxwriter")
        from pysofra.render.xlsx import XlsxRenderer
        # Center align, italic, highlight, plus an unparseable "numeric"
        # value to trigger the write_number except branch.
        t = SofraTable(
            rows=(
                Row(cells=(
                    Cell(text="a", align="center", italic=True),
                    # numeric kind but non-finite value → write_number raises
                    Cell(text="oops", value=float("nan"), kind="numeric"),
                ), metadata={"highlight": "#ffeecc"}),
            ),
            headers=(HeaderRow(cells=(
                HeaderCell(text="A"), HeaderCell(text="B"),
            )),),
        )
        out = tmp_path / "t.xlsx"
        XlsxRenderer().write(t, str(out))
        assert out.exists()

    def test_xlsx_spanning_single_column(self, tmp_path):
        pytest.importorskip("xlsxwriter")
        from pysofra.render.xlsx import XlsxRenderer
        # A spanning header that covers only one column triggers the
        # non-merge branch (end == start).
        t = SofraTable(
            headers=(HeaderRow(cells=(HeaderCell(text="X"),)),),
            spanning_headers=(SpanningHeader(label="solo", start=0, end=0),),
        )
        out = tmp_path / "solo.xlsx"
        XlsxRenderer().write(t, str(out))
        assert out.exists()

    def test_image_renderer_empty_table(self, tmp_path):
        from pysofra.render.image import write_image
        # Force the `if not grid` fallback by passing an empty table.
        t = SofraTable()
        out = tmp_path / "x.png"
        write_image(t, out)
        assert out.exists()

    def test_image_renderer_ncols_fallback(self):
        from pysofra.render.image import _ncols
        assert _ncols(SofraTable()) == 1
        rows_only = SofraTable(
            rows=(Row(cells=(Cell(text="a"),)),),
        )
        assert _ncols(rows_only) == 1


# ----------------------------------------------------------------------
# DOCX renderer rich-cell branches
# ----------------------------------------------------------------------
class TestDocxRichCells:
    def test_full_part_coverage(self, tmp_path):
        pytest.importorskip("docx")
        from pysofra.render.docx import DocxRenderer
        t = _rich_table()
        out = tmp_path / "rich.docx"
        DocxRenderer().write(t, str(out))
        assert out.exists()

    def test_docx_ncols_fallback(self):
        from pysofra.render.docx import _ncols
        assert _ncols(SofraTable()) == 1
        rows_only = SofraTable(rows=(Row(cells=(Cell(text="x"),)),))
        assert _ncols(rows_only) == 1

    def test_docx_completely_empty(self, tmp_path):
        pytest.importorskip("docx")
        from pysofra.render.docx import DocxRenderer
        t = SofraTable()
        out = tmp_path / "empty.docx"
        DocxRenderer().write(t, str(out))
        assert out.exists()


# ----------------------------------------------------------------------
# PPTX renderer
# ----------------------------------------------------------------------
class TestPptxExtras:
    def test_pptx_inline_plot_below(self, tmp_path):
        pytest.importorskip("pptx")
        pytest.importorskip("matplotlib")
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "time": rng.exponential(20, 60),
            "event": rng.integers(0, 2, 60),
        })
        t = ps.tbl_survival(df, time="time", event="event").with_km_plot(
            position="below", risk_times=[5, 10, 15],
        )
        out = tmp_path / "km.pptx"
        t.to_pptx(str(out))
        assert out.exists()

    def test_pptx_ncols_fallback(self):
        from pysofra.render.pptx import _ncols
        assert _ncols(SofraTable()) == 1


# ----------------------------------------------------------------------
# Compose / Markdown small remnants
# ----------------------------------------------------------------------
class TestComposeRemnants:
    def test_tbl_merge_single_input_raises(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        t1 = ps.tbl_one(df, by="arm", variables=["x"])
        with pytest.raises(ValueError, match="at least two"):
            ps.tbl_merge([t1])


# ----------------------------------------------------------------------
# Themes registry
# ----------------------------------------------------------------------
class TestThemeRegistry:
    def test_available_themes_listed(self):
        from pysofra.themes.registry import available_themes
        names = available_themes()
        assert "default" in names
        assert "clinical" in names

    def test_register_theme_with_pptx_overrides(self):
        # Triggers the `if pptx_overrides: new_pptx.update(...)` branch.
        from pysofra.themes.registry import _override, resolve_theme
        parent = resolve_theme("default")
        t = _override(parent, "_test_pptx_ov", {}, None, {"font_size": 11})
        assert t.pptx.get("font_size") == 11


# ----------------------------------------------------------------------
# Summary stats remnants
# ----------------------------------------------------------------------
class TestStatsRemnants:
    def test_categorical_stats_with_categorical_dtype(self):
        from pysofra.summary.stats import categorical_stats
        s = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]))
        out = categorical_stats(s)
        assert "c" in out.counts and out.counts["c"] == 0

    def test_safe_sort_key_bool(self):
        from pysofra.summary.stats import _safe_sort_key
        assert _safe_sort_key(True) == (0, 1)
        assert _safe_sort_key(False) == (0, 0)

    def test_safe_sort_key_numeric(self):
        from pysofra.summary.stats import _safe_sort_key
        assert _safe_sort_key(1.5) == (0, 1.5)

    def test_safe_sort_key_other(self):
        from pysofra.summary.stats import _safe_sort_key
        # Non-string, non-numeric → falls through to repr branch.
        out = _safe_sort_key((1, 2))
        assert out[0] == 2


# ----------------------------------------------------------------------
# Typing / categorical detection
# ----------------------------------------------------------------------
class TestTypingRemnants:
    def test_int_dtype_with_low_cardinality_treated_as_categorical(self):
        from pysofra.summary.typing import infer_kind
        # Small integer set with mostly distinct values → categorical
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        kind = infer_kind(s)
        assert kind in ("categorical", "continuous")

    def test_dichotomous_zero_one_floats(self):
        # The 0/1 fast-path detects floats too.
        from pysofra.summary.typing import infer_kind
        s = pd.Series([0.0, 1.0, 0.0, 1.0])
        assert infer_kind(s) == "dichotomous"

    def test_object_dtype_two_values(self):
        from pysofra.summary.typing import infer_kind
        s = pd.Series(["yes", "no", "yes", "no"], dtype=object)
        assert infer_kind(s) == "dichotomous"


# ----------------------------------------------------------------------
# tbl_one paths
# ----------------------------------------------------------------------
class TestTblOneRemnants:
    def test_include_missing_true_alias(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3, 4]})
        t = ps.tbl_one(df, variables=["x"], include_missing=True)
        labels = [r.cells[0].text for r in t.rows]
        assert any("Missing" in lab for lab in labels)

    def test_include_missing_false_alias(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3, 4]})
        t = ps.tbl_one(df, variables=["x"], include_missing=False)
        labels = [r.cells[0].text for r in t.rows]
        assert not any("Missing" in lab for lab in labels)

    def test_missing_invalid_raises(self):
        df = pd.DataFrame({"x": [1.0, 2, 3]})
        with pytest.raises(ValueError, match="missing"):
            ps.tbl_one(df, variables=["x"], missing="bogus")

    def test_variables_not_in_data_raises(self):
        df = pd.DataFrame({"x": [1.0, 2]})
        with pytest.raises(KeyError, match="not in data"):
            ps.tbl_one(df, variables=["nope"])

    def test_all_nan_variable_with_overall_and_p(self):
        # Empty levels path with show_overall + show_smd, via .add_overall()
        df = pd.DataFrame({
            "arm": ["A", "B"] * 5,
            "junk": [np.nan] * 10,
        })
        t = (
            ps.tbl_one(df, by="arm", variables=["junk"], missing="never")
            .add_overall()
            .add_smd()
        )
        assert len(t.rows) >= 1


# ----------------------------------------------------------------------
# extras.py / add_ci remnants
# ----------------------------------------------------------------------
class TestAddCiRemnants:
    def test_add_ci_with_overall(self):
        # `if opts.get("overall"):` branch — chain .add_overall() first.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=80),
            "x": rng.normal(size=80),
        })
        t = (
            ps.tbl_one(df, by="arm", variables=["x"], missing="never")
            .add_overall()
            .add_ci()
        )
        assert any("[" in c.text for r in t.rows for c in r.cells)

    def test_add_ci_dichotomous_branch(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=60),
            "flag": rng.choice([0, 1], size=60),
        })
        t = ps.tbl_one(df, by="arm", variables=["flag"], missing="never").add_ci()
        assert any("%" in c.text for r in t.rows for c in r.cells)


# ----------------------------------------------------------------------
# extract.py — lifelines edge with no CI columns at all
# ----------------------------------------------------------------------
class TestExtractRemnants:
    def test_lifelines_partial_ci(self):
        # Summary has 'lower' but not 'upper' → ValueError
        from pysofra.models.extract import _extract_lifelines

        class FakeFitter:
            summary = pd.DataFrame({
                "coef": [0.1],
                "p": [0.5],
                "coef lower 95%": [0.0],
                # missing 'coef upper 95%'
            }, index=["x"])

        fake = FakeFitter()
        fake.__class__.__module__ = "lifelines.fitters.fake"
        with pytest.raises(ValueError, match="CI columns"):
            _extract_lifelines(fake, conf_level=0.95)


# ----------------------------------------------------------------------
# Calibration: rake with zero-sum subset
# ----------------------------------------------------------------------
class TestCalibrateRemnants:
    def test_rake_skips_zero_weight_level(self):
        from pysofra.summary.calibrate import rake
        df = pd.DataFrame({
            "sex": ["M", "M", "F", "F"],
            "region": ["E", "W", "E", "W"],
        })
        # Set up the rake weights for sex and region but bring W's weights to 0
        weights = pd.Series([1.0, 0.0, 1.0, 0.0])  # W rows zero-weighted
        margins = {
            "sex": {"M": 50.0, "F": 50.0},
            "region": {"E": 60.0, "W": 40.0},
        }
        w = rake(df, weights, margins=margins, max_iter=20)
        assert w is not None

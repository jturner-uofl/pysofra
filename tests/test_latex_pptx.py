"""Tests for the LaTeX and PPTX renderers."""

from __future__ import annotations

import pytest

import pysofra as ps
from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row, SpanningHeader
from pysofra.core.table import SofraTable
from pysofra.render.latex import LatexRenderer


def _example_table() -> SofraTable:
    return SofraTable(
        rows=(
            Row(cells=(Cell(text="Age", align="left"),
                       Cell(text="55.0", kind="numeric", align="right"))),
            Row(cells=(Cell(text="BMI", align="left"),
                       Cell(text="27.0", kind="numeric", align="right"))),
        ),
        headers=(HeaderRow(cells=(HeaderCell(text="Var"),
                                  HeaderCell(text="A\nN = 5"))),),
        caption="Example table",
        footnotes=("Mean (SD) shown.",),
    )


class TestLatex:
    def test_booktabs_skeleton(self):
        out = LatexRenderer().render(_example_table())
        assert r"\begin{table}" in out
        assert r"\toprule" in out
        assert r"\midrule" in out
        assert r"\bottomrule" in out
        assert r"\end{table}" in out

    def test_no_booktabs_uses_hline(self):
        out = LatexRenderer(booktabs=False).render(_example_table())
        assert r"\hline" in out
        assert r"\toprule" not in out

    def test_caption(self):
        out = LatexRenderer().render(_example_table())
        assert r"\caption{Example table}" in out

    def test_footnote_after_tabular(self):
        out = LatexRenderer().render(_example_table())
        # Footnote should follow \end{tabular}, before \end{table}
        i_end_tab = out.find(r"\end{tabular}")
        i_footnote = out.find("Mean (SD) shown.")
        i_end_table = out.find(r"\end{table}")
        assert i_end_tab < i_footnote < i_end_table

    def test_special_char_escaping(self):
        t = SofraTable(
            rows=(Row(cells=(Cell(text="100%"), Cell(text="A & B"))),),
            headers=(HeaderRow(cells=(HeaderCell(text="x"),
                                      HeaderCell(text="y"))),),
        )
        out = LatexRenderer().render(t)
        assert r"100\%" in out
        assert r"A \& B" in out

    def test_multiline_header_uses_shortstack(self):
        out = LatexRenderer().render(_example_table())
        assert r"\shortstack{" in out

    def test_column_spec_first_left(self):
        out = LatexRenderer().render(_example_table())
        # First col left-aligned; second has no explicit align so falls back to "c".
        assert r"\begin{tabular}{lc}" in out

    def test_to_latex_via_table(self, small_trial):
        out = ps.tbl_one(small_trial, by="arm").add_p().to_latex()
        assert r"\begin{table}" in out
        assert r"\bottomrule" in out

    def test_spanning_header(self):
        t = SofraTable(
            rows=_example_table().rows,
            headers=_example_table().headers,
            spanning_headers=(SpanningHeader(label="Numbers", start=0, end=1),),
        )
        out = LatexRenderer().render(t)
        assert r"\multicolumn{2}{c}{Numbers}" in out
        assert r"\cmidrule(lr){1-2}" in out


class TestPptx:
    def test_to_pptx(self, small_trial, tmp_path):
        pytest.importorskip("pptx")
        out = tmp_path / "t.pptx"
        result = (
            ps.tbl_one(small_trial, by="arm")
              .add_p()
              .set_caption("Demo")
              .to_pptx(out)
        )
        assert result.exists()
        assert result.stat().st_size > 5000

    def test_to_pptx_no_caption(self, small_trial, tmp_path):
        pytest.importorskip("pptx")
        out = tmp_path / "t.pptx"
        ps.tbl_one(small_trial, by="arm").add_p().to_pptx(out)
        assert out.exists()

    def test_to_pptx_custom_title(self, small_trial, tmp_path):
        pytest.importorskip("pptx")
        out = tmp_path / "t.pptx"
        ps.tbl_one(small_trial, by="arm").to_pptx(out, slide_title="My slide")
        assert out.exists()

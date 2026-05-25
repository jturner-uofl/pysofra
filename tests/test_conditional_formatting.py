"""Tests for conditional formatting (bold_if / highlight_if / style_if) and
notebook reprs (_repr_html_ / _repr_markdown_ / _repr_latex_)."""

from __future__ import annotations

import pysofra as ps


class TestBoldIf:
    def test_bold_if_specific_row(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm", missing="never")
        out = t.bold_if(lambda r: r.cells[0].text == "age")
        # age row should now have bold cells; bmi row should not
        for r in out.rows:
            if r.cells[0].text == "age":
                assert all(c.bold for c in r.cells if c.text)
            elif r.cells[0].text == "bmi":
                assert not any(c.bold for c in r.cells if c.text)

    def test_bold_p_uses_bold_if(self, small_trial):
        # Sanity — bold_p still works after the refactor.
        # Use threshold 1.5 so even rows with Fisher exact p == 1.0 are bolded.
        t = (
            ps.tbl_one(small_trial, by="arm", missing="never")
              .add_p()
              .bold_p(threshold=1.5)
        )
        for r in t.rows:
            p_cell = next(
                (c for c in r.cells
                 if c.kind == "p_value" and isinstance(c.value, (int, float))
                 and not (isinstance(c.value, float) and c.value != c.value)),  # not NaN
                None,
            )
            if p_cell:
                assert any(c.bold for c in r.cells if c.text)


class TestHighlightIf:
    def test_highlight_metadata(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm", missing="never")
        out = t.highlight_if(lambda r: r.cells[0].text == "age",
                             color="#fff3cd")
        age_row = next(r for r in out.rows if r.cells[0].text == "age")
        assert age_row.metadata.get("highlight") == "#fff3cd"

    def test_highlight_in_html(self, small_trial):
        t = (
            ps.tbl_one(small_trial, by="arm", missing="never")
              .highlight_if(lambda r: r.cells[0].text == "age",
                            color="#ffdada")
        )
        html = t.to_html()
        assert "background:#ffdada" in html


class TestStyleIf:
    def test_combined(self, small_trial):
        t = (
            ps.tbl_one(small_trial, by="arm", missing="never")
              .style_if(lambda r: r.cells[0].text == "age",
                        bold=True, italic=True, color="#fff3cd")
        )
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        assert all(c.bold for c in age_row.cells if c.text)
        assert all(c.italic for c in age_row.cells if c.text)
        assert age_row.metadata.get("highlight") == "#fff3cd"


class TestNotebookReprs:
    def test_repr_html(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        h = t._repr_html_()
        assert "<table" in h
        assert "pysofra-wrap" in h

    def test_repr_markdown(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        md = t._repr_markdown_()
        assert md.startswith("|") or md.startswith("**")

    def test_repr_latex(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        latex = t._repr_latex_()
        assert r"\begin{table}" in latex
        assert r"\bottomrule" in latex


class TestStickyHeader:
    def test_sticky_header_css_present(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        html = t.to_html(sticky_header=True)
        assert "position:sticky" in html

    def test_max_height_creates_scroll_container(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        html = t.to_html(max_height="50vh")
        assert "max-height:50vh" in html
        assert "overflow-y:auto" in html

    def test_no_sticky_by_default(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        html = t.to_html()
        assert "position:sticky" not in html

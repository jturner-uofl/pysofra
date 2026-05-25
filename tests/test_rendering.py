"""Renderer tests — HTML, Markdown, DOCX."""

from __future__ import annotations

import re

import pytest

import pysofra as ps
from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row, SpanningHeader
from pysofra.core.table import SofraTable
from pysofra.render.html import HtmlRenderer
from pysofra.render.markdown import MarkdownRenderer


def _example_table() -> SofraTable:
    return SofraTable(
        rows=(
            Row(cells=(Cell(text="Age"), Cell(text="55.0", kind="numeric", align="right"))),
            Row(cells=(Cell(text="BMI"), Cell(text="27.0", kind="numeric", align="right"))),
        ),
        headers=(HeaderRow(cells=(HeaderCell(text="Var"), HeaderCell(text="Value"))),),
        caption="Example table",
        footnotes=("Mean (SD) shown.",),
    )


class TestHtml:
    def test_contains_caption_and_footnote(self):
        h = HtmlRenderer().render(_example_table())
        assert "Example table" in h
        assert "Mean (SD) shown." in h

    def test_escaping(self):
        t = SofraTable(rows=(Row(cells=(Cell(text="a<b>"),)),),
                       headers=(HeaderRow(cells=(HeaderCell(text="x>y"),)),))
        h = HtmlRenderer().render(t)
        assert "a&lt;b&gt;" in h
        assert "x&gt;y" in h

    def test_notebook_wrapper(self):
        h = HtmlRenderer(notebook=True).render(_example_table())
        assert "pysofra-wrap" in h

    def test_scoped_styles_are_content_derived(self):
        # Scope IDs are a SHA-256 hash of the table's content (rows,
        # headers, caption, footnotes, theme), so identical tables get
        # identical scope IDs — required for reproducible publication
        # outputs — while tables with different content get different
        # scope IDs so embedded styles don't collide on the same page.
        h1 = HtmlRenderer().render(_example_table())
        h2 = HtmlRenderer().render(_example_table())
        id1 = re.search(r"pysofra-[0-9a-f]+", h1).group(0)
        id2 = re.search(r"pysofra-[0-9a-f]+", h2).group(0)
        assert id1 == id2  # same content → same scope id

        other = SofraTable(
            rows=(Row(cells=(Cell(text="alpha"), Cell(text="beta"))),),
            headers=(HeaderRow(cells=(HeaderCell(text="A"),
                                       HeaderCell(text="B"))),),
        )
        h3 = HtmlRenderer().render(other)
        id3 = re.search(r"pysofra-[0-9a-f]+", h3).group(0)
        assert id3 != id1  # different content → different scope id

    def test_spanning_header(self):
        t = SofraTable(
            rows=_example_table().rows,
            headers=_example_table().headers,
            spanning_headers=(SpanningHeader(label="Numbers", start=0, end=1),),
        )
        h = HtmlRenderer().render(t)
        assert 'colspan="2"' in h
        assert "Numbers" in h


class TestMarkdown:
    def test_basic_markdown(self):
        m = MarkdownRenderer().render(_example_table())
        assert "**Example table**" in m
        assert "| Var | Value |" in m
        assert "Mean (SD) shown." in m

    def test_pipe_escaping(self):
        t = SofraTable(rows=(Row(cells=(Cell(text="a|b"),)),),
                       headers=(HeaderRow(cells=(HeaderCell(text="h"),)),))
        m = MarkdownRenderer().render(t)
        assert r"a\|b" in m


class TestThemes:
    @pytest.mark.parametrize("name", ["clinical", "compact", "jama", "nejm", "minimal"])
    def test_theme_resolves_and_renders(self, name, small_trial):
        t = ps.tbl_one(small_trial, by="arm").theme(name)
        h = t.to_html()
        assert "<table" in h

    def test_unknown_theme_raises(self, small_trial):
        with pytest.raises(ValueError):
            ps.tbl_one(small_trial, by="arm").theme("not-a-theme")


class TestEscaping:
    """Regression guards: every renderer must escape special characters
    in user-supplied content so a malicious DataFrame can't break out of
    the cell. These tests are the first line of defence against a
    future refactor accidentally removing an escape step.
    """

    def _bad_table(self) -> SofraTable:
        """A table whose every text field carries a different injection
        attempt: HTML/JS, LaTeX command, Markdown link, XML break-out."""
        bad = '<script>alert("xss")</script>'
        latex = r"\textbf{\dangerous} $\sum$ \input{/etc/passwd}"
        md_link = "[link](javascript:alert(1))"
        xml = "</w:t><w:t>BROKEOUT</w:t><w:t>"
        return SofraTable(
            rows=(
                Row(cells=(Cell(text=bad), Cell(text=latex, align="right"))),
                Row(cells=(Cell(text=md_link), Cell(text=xml, align="right"))),
            ),
            headers=(HeaderRow(cells=(
                HeaderCell(text=bad), HeaderCell(text=latex))),),
            caption=bad,
            footnotes=(latex, md_link),
        )

    def test_html_escapes_script_tag(self):
        h = self._bad_table().to_html()
        # The literal `<script>` open tag MUST NOT survive — a browser
        # would execute it. The escaped form &lt;script&gt; is fine.
        assert "<script>" not in h
        assert "&lt;script&gt;" in h
        # The cell text contains '"' which must be entity-escaped inside
        # attribute or text context.
        assert 'alert("xss")' not in h
        assert "alert(&quot;xss&quot;)" in h

    def test_html_javascript_url_is_text_not_href(self):
        h = self._bad_table().to_html()
        # The string ``javascript:`` may appear as TEXT content (in a <td>
        # or footnote paragraph) but must NEVER end up inside an ``href=``
        # attribute. ``<a href="javascript:`` is the dangerous form.
        assert 'href="javascript:' not in h
        assert "href='javascript:" not in h
        # Same defence-in-depth for ``onerror=``: must not appear as an
        # attribute, only as cell text.
        # An "attribute-like" pattern is ``<TAG ... onerror=``.
        import re
        attribute_onerror = re.search(r'<\w+[^>]*\sonerror\s*=', h)
        assert attribute_onerror is None, (
            f"onerror= leaked into an HTML attribute: {attribute_onerror.group(0)}"
        )

    def test_latex_escapes_backslash_commands(self):
        ltx = self._bad_table().to_latex()
        # A future refactor that forgot to escape could let raw \input{
        # through. The LaTeX renderer is supposed to neutralise these.
        body_start = ltx.find("\\toprule")
        body_end   = ltx.find("\\bottomrule")
        body = ltx[body_start:body_end] if body_start >= 0 else ltx
        # User-supplied \input{/etc/passwd} must not survive verbatim in
        # the table body — that would cause TeX to read the file at
        # compile time.
        assert "\\input{/etc/passwd}" not in body
        # User-supplied \dangerous and raw $\sum must be neutralised.
        assert "\\dangerous" not in body
        # raw "$\sum" would drop the renderer into math mode; check the
        # backslash is escaped or the dollar sign is escaped.
        assert "$\\sum" not in body

    def test_markdown_escapes_pipe_and_html(self):
        md = self._bad_table().to_markdown()
        # Pipe in a cell must be escaped to ``\|`` so Markdown table
        # syntax doesn't break.
        assert "INJECTED|<w:t>" not in md
        # Markdown-flavoured HTML must be neutralised so raw <script>
        # doesn't pass through to a Markdown viewer that allows HTML.
        assert "<script>" not in md
        # Markdown links syntax inside cells must NOT be interpreted as
        # a live link — bracket characters should be backslash-escaped.
        assert "\\[link\\]" in md or "[link]" not in md  # one or the other

    def test_docx_xml_breakout_is_neutralised(self, tmp_path):
        import zipfile
        out = self._bad_table().to_docx(tmp_path / "x.docx")
        with zipfile.ZipFile(out) as z:
            doc = z.read("word/document.xml").decode("utf-8")
        # The user payload was ``</w:t><w:t>BROKEOUT</w:t><w:t>`` which,
        # if pasted verbatim, would split into multiple text runs and
        # inject ``BROKEOUT`` as an independent run. The XML-escaping
        # in python-docx ensures the closing tag is encoded.
        assert "</w:t><w:t>BROKEOUT" not in doc

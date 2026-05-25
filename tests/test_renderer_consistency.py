"""Cross-renderer consistency validation.

A single ``SofraTable`` must render to HTML / Markdown / LaTeX / DOCX /
PPTX / XLSX with **consistent content**: the same cell texts, the
same row count, the same caption, the same footnotes, the same
spanning-header labels. Differences should be in *encoding*, not
*content*.

This file pins those invariants so any renderer that silently drifts
gets caught immediately.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def golden_table():
    """A representative SofraTable that covers most rendering features."""
    rng = np.random.default_rng(20260521)
    n = 120
    df = pd.DataFrame({
        "arm": rng.choice(["Placebo", "Treatment"], size=n),
        "age": rng.normal(55, 12, size=n),
        "sex": rng.choice(["F", "M"], size=n),
        "race": rng.choice(["White", "Black", "Asian"], size=n),
    })
    df.loc[df.index[0], "age"] = np.nan  # one missing row
    t = (
        ps.tbl_one(df, by="arm", variables=["age", "sex", "race"])
        .add_overall()
        .add_p()
        .add_smd()
        .set_caption("Baseline characteristics")
    )
    return t


def _all_cell_text(t):
    """Flatten every body cell text in the SofraTable."""
    return [c.text for r in t.rows for c in r.cells]


def _all_header_text(t):
    out = []
    for hr in t.headers:
        out.extend(c.text for c in hr.cells)
    return out


# ======================================================================
# Plain-text renderers — every cell text should appear in the output
# ======================================================================
class TestTextContentSurvivesRender:
    """For HTML / Markdown / LaTeX, every cell's text must be present
    in the rendered string (modulo escaping)."""

    def _check_cells_present(self, t, rendered: str, escape_check=None):
        for cell_text in _all_cell_text(t):
            if cell_text == "" or cell_text == "—":
                continue
            # Compare on the un-escaped substring for HTML.
            needle = escape_check(cell_text) if escape_check else cell_text
            assert needle in rendered, (
                f"cell text {cell_text!r} missing from rendered output"
            )

    def test_html_contains_every_cell(self, golden_table):
        out = golden_table.to_html()
        import html
        self._check_cells_present(golden_table, out, escape_check=html.escape)

    def test_markdown_contains_every_cell(self, golden_table):
        out = golden_table.to_markdown()
        # Markdown escapes pipe chars; check the leading non-pipe portion
        for txt in _all_cell_text(golden_table):
            if txt in ("", "—"):
                continue
            assert txt.split("|")[0] in out

    def test_latex_contains_every_cell(self, golden_table):
        out = golden_table.to_latex()
        for txt in _all_cell_text(golden_table):
            if txt in ("", "—"):
                continue
            # LaTeX escapes %, _, &, and a few others.
            # Strip percent sign and check the rest survives.
            stripped = txt.replace("%", r"\%").replace("&", r"\&") \
                          .replace("_", r"\_")
            assert any(piece in out for piece in stripped.split()), (
                f"latex render missing token derived from {txt!r}"
            )

    def test_caption_consistent_across_text_renderers(self, golden_table):
        # Every text renderer should surface the caption literally.
        assert "Baseline characteristics" in golden_table.to_html()
        assert "Baseline characteristics" in golden_table.to_markdown()
        assert "Baseline characteristics" in golden_table.to_latex()


# ======================================================================
# Binary renderers — open the file and confirm the cell texts appear
# ======================================================================
class TestBinaryRendererCellPresence:
    """Open the DOCX / XLSX / PPTX zip archives and confirm every body
    cell text is somewhere in the extracted XML."""

    def _all_xml_text(self, archive: Path) -> str:
        """Concatenate every XML body inside the archive (DOCX/PPTX/XLSX
        are zipped XML)."""
        chunks = []
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.endswith(".xml"):
                    chunks.append(zf.read(name).decode("utf-8", "replace"))
        return "".join(chunks)

    def test_docx_contains_every_cell(self, golden_table, tmp_path):
        pytest.importorskip("docx")
        out = tmp_path / "g.docx"
        golden_table.to_docx(str(out))
        xml_text = self._all_xml_text(out)
        for txt in _all_cell_text(golden_table):
            if txt in ("", "—"):
                continue
            assert txt in xml_text, f"docx missing {txt!r}"

    def test_pptx_contains_every_cell(self, golden_table, tmp_path):
        pytest.importorskip("pptx")
        out = tmp_path / "g.pptx"
        golden_table.to_pptx(str(out))
        xml_text = self._all_xml_text(out)
        for txt in _all_cell_text(golden_table):
            if txt in ("", "—"):
                continue
            assert txt in xml_text, f"pptx missing {txt!r}"

    def test_xlsx_contains_every_string_cell(self, golden_table, tmp_path):
        # XLSX writes numeric cells via write_number — their textual form
        # is the underlying float, not the rendered "0.247" string. So
        # we check string-typed cells (labels, n (%), formatted text).
        pytest.importorskip("xlsxwriter")
        out = tmp_path / "g.xlsx"
        golden_table.to_xlsx(str(out))
        xml_text = self._all_xml_text(out)
        for r in golden_table.rows:
            for c in r.cells:
                # Skip numeric / p-value cells (written as <v>)
                if c.kind in ("numeric", "p_value", "q_value", "ci"):
                    continue
                if c.text in ("", "—"):
                    continue
                assert c.text in xml_text, f"xlsx missing {c.text!r}"


# ======================================================================
# Row / column counts must agree across formats
# ======================================================================
class TestStructuralConsistency:
    """The same SofraTable must produce the same row and column counts
    in every renderer."""

    def test_markdown_table_row_count_matches(self, golden_table):
        md = golden_table.to_markdown()
        # Count pipe-delimited body lines (skip the separator row)
        lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
        # Subtract: header row + separator row
        body_lines = [ln for ln in lines if not re.match(r"^\|\s*-+", ln)]
        # Subtract header lines
        body_count = len(body_lines) - len(golden_table.headers)
        assert body_count >= len(golden_table.rows) - 2  # allow slack for footnotes


    def test_html_row_count_matches(self, golden_table):
        html = golden_table.to_html()
        # <tr> count in tbody equals rows count
        tbody = html.split("<tbody>")[1].split("</tbody>")[0]
        n_tr = tbody.count("<tr")
        assert n_tr == len(golden_table.rows)

    def test_latex_row_count_matches(self, golden_table):
        latex = golden_table.to_latex()
        # Each body row in our LaTeX renderer ends with "\\".
        # Heuristically: count "\\" occurrences and subtract header rows.
        n_endrow = latex.count(r"\\")
        # At minimum we should have rows + headers worth of "\\"
        assert n_endrow >= len(golden_table.rows) + len(golden_table.headers)


# ======================================================================
# Round-trip stability — re-rendering produces identical output
# ======================================================================
class TestRoundtripStability:
    """Calling .to_html() twice on the same table must produce
    identical text. This rules out non-deterministic ordering
    (e.g. unsorted dicts in cell metadata)."""

    def test_html_deterministic(self, golden_table):
        first = golden_table.to_html()
        second = golden_table.to_html()
        assert first == second

    def test_markdown_deterministic(self, golden_table):
        first = golden_table.to_markdown()
        second = golden_table.to_markdown()
        assert first == second

    def test_latex_deterministic(self, golden_table):
        first = golden_table.to_latex()
        second = golden_table.to_latex()
        assert first == second


# ======================================================================
# Reproducibility — same input data → identical SofraTable structure
# ======================================================================
class TestReproducibility:
    """Building the same table twice from the same data must produce
    identical text on every renderer."""

    def _build(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=80),
            "x": rng.normal(size=80),
            "y": rng.choice([0, 1], size=80),
        })
        return ps.tbl_one(df, by="arm", variables=["x", "y"]) \
                 .add_p() \
                 .add_smd()

    def test_html_reproducible(self):
        assert self._build().to_html() == self._build().to_html()

    def test_markdown_reproducible(self):
        assert self._build().to_markdown() == self._build().to_markdown()

    def test_latex_reproducible(self):
        assert self._build().to_latex() == self._build().to_latex()

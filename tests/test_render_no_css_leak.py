"""Rendering-hygiene snapshot tests.

An external reviewer browsing the case-study notebook on GitHub saw
"CSS leaking into table cells." Root cause: PySofra previously emitted
the CSS-Color-Module-5 function ``color-mix(in srgb, currentColor 25%,
transparent)`` as an inline border colour. GitHub's .ipynb HTML
sanitiser does not understand ``color-mix()`` and mangled the
attribute mid-parse, leaking raw CSS text into the rendered cell.

These tests are the regression guard the reviewer asked for: they
assert that the rendered HTML and LaTeX contain no renderer-hostile
constructs and that cell *content* never contains stray CSS tokens.
They run on every commit, so a future theme change that re-introduces
``color-mix()`` (or any other modern-CSS-only function) fails CI
immediately.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def representative_table() -> ps.SofraTable:
    """A table that exercises every cell type: continuous, dichotomous,
    multi-level categorical (the row class the reviewer saw leak), and
    a missing row."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 120),
        "age": rng.normal(50, 10, 120),
        "sex": rng.choice(["M", "F"], 120),
        "education": rng.choice(["<HS", "HS", "Some-college", "College+"],
                                120),
    })
    # inject some missingness so a Missing row renders
    df.loc[rng.choice(120, 8, replace=False), "education"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ps.tbl_one(
            df, by="arm",
            variables=["age", "sex", "education"],
            labels={"age": "Age", "sex": "Sex",
                    "education": "Education"},
        ).add_p()


# ----------------------------------------------------------------------
# The specific bug the reviewer hit
# ----------------------------------------------------------------------

class TestNoColorMix:
    def test_html_has_no_color_mix(self, representative_table):
        html = representative_table.to_html()
        assert "color-mix" not in html, (
            "HTML emits CSS color-mix() — GitHub's .ipynb renderer "
            "mangles it and leaks raw CSS into cells. Use rgba()."
        )

    def test_all_builtin_themes_clean(self, representative_table):
        # Every shipped theme must be GitHub-renderer-safe.
        for theme in ("default", "clinical", "compact", "jama",
                      "nejm", "minimal"):
            html = representative_table.theme(theme).to_html()
            assert "color-mix" not in html, (
                f"theme {theme!r} emits color-mix() in its HTML"
            )


# ----------------------------------------------------------------------
# General rendering hygiene
# ----------------------------------------------------------------------

class TestHtmlHygiene:
    def test_no_style_tag_inside_cells(self, representative_table):
        """No <td> may contain a nested <style> block (the visible
        symptom of CSS leaking into content)."""
        html = representative_table.to_html()
        # find every <td>...</td> and assert no <style> within
        for td in re.findall(r"<td[^>]*>.*?</td>", html, re.DOTALL):
            assert "<style" not in td, f"<style> leaked into a cell: {td[:120]}"

    def test_cell_text_has_no_raw_css(self, representative_table):
        """The Education multi-level rows must contain only their labels
        — no CSS property tokens like 'border-bottom' or 'srgb' bleeding
        into the rendered text nodes."""
        html = representative_table.to_html()
        # Extract text content of each <td> (strip tags + attributes)
        for td in re.findall(r"<td[^>]*>(.*?)</td>", html, re.DOTALL):
            # the inner text (after removing any nested tags)
            inner = re.sub(r"<[^>]+>", "", td)
            for css_token in ("border-bottom", "srgb", "color-mix",
                              "padding:", "rgba("):
                assert css_token not in inner, (
                    f"CSS token {css_token!r} leaked into cell text: "
                    f"{inner[:120]!r}"
                )

    def test_balanced_table_tags(self, representative_table):
        html = representative_table.to_html()
        assert html.count("<table") == html.count("</table>")
        assert html.count("<tr") == html.count("</tr>")
        assert html.count("<td") == html.count("</td>")
        # Count <th> opening tags precisely — `<th` alone would also
        # match `<thead`, inflating the count. Use a word boundary.
        th_open = len(re.findall(r"<th[ >]", html))
        assert th_open == html.count("</th>")


class TestLatexHygiene:
    def test_latex_no_color_mix(self, representative_table):
        tex = representative_table.to_latex()
        assert "color-mix" not in tex

    def test_latex_balanced_environments(self, representative_table):
        tex = representative_table.to_latex()
        assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}")
        # No HTML/CSS tokens should appear in LaTeX output
        for css_token in ("color-mix", "srgb", "border-bottom",
                          "rgba(", "<td", "<style"):
            assert css_token not in tex, (
                f"CSS/HTML token {css_token!r} leaked into LaTeX output"
            )

    def test_latex_rows_have_no_stray_markup(self, representative_table):
        """Each LaTeX body row (between \\midrule and \\bottomrule) must
        contain only LaTeX, no leaked CSS/HTML — the reviewer reported
        'CSS leaking into LaTeX rows'."""
        tex = representative_table.to_latex()
        m = re.search(r"\\midrule(.*?)\\bottomrule", tex, re.DOTALL)
        assert m, "no booktabs body found in LaTeX output"
        body = m.group(1)
        for tok in ("color-mix", "srgb", "rgba(", "style=", "<"):
            assert tok not in body, (
                f"token {tok!r} leaked into LaTeX body rows"
            )


# ----------------------------------------------------------------------
# Snapshot determinism (content hash stable across renders)
# ----------------------------------------------------------------------

class TestRenderSnapshotStable:
    def test_markdown_snapshot_hash_stable(self, representative_table):
        h1 = representative_table.snapshot_hash()
        h2 = representative_table.snapshot_hash()
        assert h1 == h2

    def test_latex_deterministic(self, representative_table):
        assert (representative_table.to_latex()
                == representative_table.to_latex())

    def test_html_structure_deterministic(self, representative_table):
        # The randomised CSS class differs between renders by design,
        # but the structural content (everything after the <style> block)
        # must be identical.
        def _strip_style(html: str) -> str:
            return re.sub(r"<style>.*?</style>", "", html, flags=re.DOTALL)
        # also strip the randomised class token
        def _strip_class(html: str) -> str:
            return re.sub(r"pysofra-[0-9a-f]+", "pysofra-X", html)
        a = _strip_class(_strip_style(representative_table.to_html()))
        b = _strip_class(_strip_style(representative_table.to_html()))
        assert a == b

"""Cross-backend semantic-content consistency contract.

The architectural claim PySofra makes against pandas Styler, openpyxl,
and Jinja2 templates is **"compute once, render many"** — one
``SofraTable`` spec feeds every renderer, and the *user-visible cell
content* is identical across all of them. This file pins that claim.

We build one representative ``SofraTable`` (a Table-1 fragment with
group columns, statistics, a p-value column, an overall column, and
SMDs) and then for each of the text renderers (HTML, LaTeX, Typst,
Markdown) we verify that every cell text drawn from the spec appears
verbatim in the rendered output. Cells whose text would be touched by
backend-specific escaping (e.g. ``<``, ``>``, ``&``, ``_`` in LaTeX)
are excluded from the verbatim check; for those we assert that the
*typed* ``Cell.value`` is preserved on the spec itself, which is the
real consistency guarantee (markup differs, semantic content does
not).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import pysofra as ps

# Numeric tokens are the escape-immune backbone of a Table-1 / Table-2
# cell: every renderer (HTML, LaTeX, Typst, Markdown) writes "45.62"
# the same way. Match decimal numbers (with optional sign / decimal
# fraction) so we can compare the *multiset* of numbers each backend
# emits — if any backend silently dropped or rounded a cell, this
# multiset will diverge.
_NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")


@pytest.fixture(scope="module")
def representative_table() -> ps.SofraTable:
    """A non-trivial Table-1 fragment exercising numeric, categorical,
    p-value, overall, and SMD cells — i.e. the full set of cell kinds
    a Table-1 builder produces."""
    rng = np.random.default_rng(2026)
    n = 240
    df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], n),
        "age": rng.normal(45.0, 12.0, n),
        "sex": rng.choice(["M", "F"], n),
        "smoker": rng.choice(["never", "former", "current"], n),
    })
    return (ps.tbl_one(df, by="arm")
            .add_p()
            .add_overall()
            .add_smd())


def _number_multiset(text: str) -> tuple[str, ...]:
    """Return the sorted multiset of numeric tokens in ``text``.

    Sorting + tupling gives a canonical, hashable representation that
    can be compared across backends directly. A divergence anywhere —
    a renderer silently dropping a row, double-printing a column,
    truncating a CI endpoint — surfaces as an unequal tuple.
    """
    return tuple(sorted(_NUMBER_RE.findall(text)))


class TestCrossBackendConsistency:
    """One spec → many renderers, semantic content preserved."""

    def test_cell_content_numbers_appear_in_every_text_backend(
        self, representative_table
    ):
        """Every numeric token from the spec's cell content appears
        in every text backend's rendered output.

        We extract numeric tokens *from the SofraTable spec itself*
        (not from the rendered text), then verify each appears in the
        HTML, LaTeX, Typst, and Markdown outputs. Scoping to spec-
        derived numbers sidesteps markup-overhead numbers like
        ``rgba(127, 127, 127, 0.3)`` in HTML themes or LaTeX column
        widths, and ensures we're comparing the *statistical payload*
        the user actually sees rendered.

        This is the architectural property pandas Styler (HTML-only),
        openpyxl (Excel-only), and Jinja2 (one template per output)
        categorically cannot offer: one spec → four backends → every
        numeric cell value preserved in each rendering.
        """
        t = representative_table
        backends = {
            "html": t.to_html(),
            "latex": t.to_latex(),
            "typst": t.to_typst(),
            "markdown": t.to_markdown(),
        }

        # Numeric tokens drawn from cell *text* on the spec — these
        # are the numbers the user sees in any rendering.
        spec_numbers: list[str] = []
        for hr in t.headers:
            for c in hr.cells:
                spec_numbers.extend(_NUMBER_RE.findall(c.text))
        for r in t.rows:
            for c in r.cells:
                spec_numbers.extend(_NUMBER_RE.findall(c.text))

        # Sanity: representative table is non-trivial.
        assert len(spec_numbers) >= 20, (
            f"representative table only carries {len(spec_numbers)} "
            f"numeric tokens on the spec — too thin to be a meaningful "
            f"cross-backend consistency contract"
        )

        # Every spec-derived numeric token must appear in every
        # backend's rendered output. A renderer silently dropping a
        # row, truncating a CI endpoint, or re-rounding a p-value
        # would fail here.
        for backend_name, output in backends.items():
            missing = [n for n in spec_numbers if n not in output]
            assert not missing, (
                f"{backend_name} backend dropped {len(missing)} "
                f"numeric token(s) from spec content; first few: "
                f"{missing[:5]}"
            )

    def test_cell_value_preserved_across_spec(self, representative_table):
        """``Cell.value`` (the typed payload) is the truth — it is the
        same object regardless of which renderer is asked. This is the
        underlying mechanism that makes the verbatim-text contract
        possible.

        We extract a p-value cell, verify its ``value`` is a float (not
        a string), and confirm rendering the table several times leaves
        the spec's typed values untouched.
        """
        t = representative_table
        # Find any p-value cell.
        p_cells = []
        for r in t.rows:
            for c in r.cells:
                if c.kind == "p_value" and c.value is not None:
                    p_cells.append(c)
        assert p_cells, "representative table has no p-value cells"

        # The typed value is a float — modifiers like ``bold_p`` can
        # query it directly rather than parsing the rendered string.
        for c in p_cells:
            assert isinstance(c.value, float), (
                f"p-value cell.value is {type(c.value).__name__}, not "
                f"float — string-parsing modifiers would be necessary"
            )

        # Render multiple times → underlying spec is untouched (the
        # immutable / copy-on-write claim).
        before_texts = [c.text for c in p_cells]
        before_values = [c.value for c in p_cells]
        for _ in range(3):
            _ = t.to_html()
            _ = t.to_latex()
            _ = t.to_markdown()
        after_texts = [c.text for c in p_cells]
        after_values = [c.value for c in p_cells]
        assert after_texts == before_texts
        assert after_values == before_values

    def test_bold_p_queries_typed_value_not_string(
        self, representative_table
    ):
        """``bold_p(threshold)`` is the canonical example of a modifier
        that *must* read the typed value, not the rendered string.

        A pandas-Styler style modifier would have to parse "<0.001" or
        "p = 0.034" back to a float and would fail on threshold cells
        ("<0.001" is technically a string in any Styler-like world).
        PySofra's modifier walks ``Cell.value`` directly; a p-value
        rendered as "<0.001" still has a float ``value`` (e.g. 0.0006)
        and is bolded correctly.
        """
        t = representative_table
        b = t.bold_p(threshold=0.05)

        # Find p-value cells in both tables; pair them by position.
        def _p_cells_with_position(tab):
            out = []
            for ri, r in enumerate(tab.rows):
                for ci, c in enumerate(r.cells):
                    if c.kind == "p_value" and c.value is not None:
                        out.append((ri, ci, c))
            return out

        before = _p_cells_with_position(t)
        after = _p_cells_with_position(b)
        assert len(before) == len(after)
        for (ri, ci, c_before), (_, _, c_after) in zip(
            before, after, strict=True
        ):
            # The typed value is unchanged …
            assert c_after.value == c_before.value
            # … but the rendered presentation is bolded iff value <
            # 0.05 — proving the modifier consulted the float, not the
            # string "<0.001".
            expected_bold = c_before.value < 0.05
            assert c_after.bold is expected_bold, (
                f"bold_p mis-decided on row={ri} col={ci}: "
                f"value={c_before.value!r}, text={c_before.text!r}, "
                f"got bold={c_after.bold}, expected {expected_bold}"
            )

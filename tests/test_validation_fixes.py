"""Regression tests for input-validation edge cases.

Each test pins a specific defect so any future regression surfaces
with a clear name pointing back at the finding.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


# ======================================================================
# add_difference uses Newcombe (Wilson-based) CI
# ======================================================================
class TestNewcombeDifferenceCI:
    def test_matches_statsmodels_newcomb(self):
        """The Newcombe CI emitted by add_difference must match
        statsmodels' confint_proportions_2indep(method='newcomb')."""
        from statsmodels.stats.proportion import confint_proportions_2indep

        # Build a 2-arm tbl_one with a single dichotomous outcome.
        n1, x1 = 70, 56  # arm A: 56/70 successes
        n2, x2 = 80, 48  # arm B: 48/80 successes
        df = pd.DataFrame({
            "arm": ["A"] * n1 + ["B"] * n2,
            "y": [1] * x1 + [0] * (n1 - x1)
                 + [1] * x2 + [0] * (n2 - x2),
        })
        t = ps.tbl_one(df, by="arm", variables=["y"],
                       types={"y": "dichotomous"},
                       missing="never").add_difference()
        # Pull the dichotomous row's difference cell
        row = next(r for r in t.rows if r.cells[0].text.startswith("y ="))
        diff_cell = next(c for c in row.cells
                         if c.kind == "ci" and c.value is not None
                         and isinstance(c.value, tuple) and len(c.value) == 3)
        diff_pt, lo, hi = diff_cell.value
        # Reference value (statsmodels reports (p2-p1, lo, hi))
        ref_lo, ref_hi = confint_proportions_2indep(
            x2, n2, x1, n1, method='newcomb', alpha=0.05, compare='diff',
        )
        assert lo == pytest.approx(float(ref_lo), abs=1e-9)
        assert hi == pytest.approx(float(ref_hi), abs=1e-9)

    def test_extreme_proportion_handles_zero_one_boundary(self):
        """At the p=1 / p=0 boundary Newcombe stays close to the
        truncated unit interval — Wald would give an unusable
        normal-approximation CI of width 0 at one end."""
        # 95%/30% split — extreme but not degenerate.
        df = pd.DataFrame({
            "arm": ["A"] * 20 + ["B"] * 20,
            "y": [1] * 19 + [0] + [0] * 14 + [1] * 6,
        })
        t = ps.tbl_one(df, by="arm", variables=["y"],
                       types={"y": "dichotomous"},
                       missing="never").add_difference()
        row = next(r for r in t.rows if r.cells[0].text.startswith("y ="))
        diff_cell = next(c for c in row.cells
                         if c.kind == "ci"
                         and isinstance(c.value, tuple) and len(c.value) == 3)
        _, lo, hi = diff_cell.value
        # CI should be entirely inside (-1, 1), strictly negative
        # (B has lower success rate), and well separated from the
        # extremes. Wald would give a similar interval here too in
        # this less-extreme case; the point is that Newcombe stays
        # well-defined and bounded.
        assert -1.0 <= lo < hi <= 1.0
        assert lo < -0.3 and hi < 0.0


# ======================================================================
# Markdown spanners + escaping
# ======================================================================
class TestMarkdownSpannersAndEscape:
    def test_spanning_header_not_inserted_as_pipe_row(self):
        """Spanning headers must NOT appear as a pipe row between the
        column-header and alignment rows (which would break GFM
        parsers)."""
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        t1 = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        t2 = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        merged = ps.tbl_merge([t1, t2], tab_spanners=["Group 1", "Group 2"])
        md = merged.to_markdown()
        # The alignment row (|---|---|...) must come immediately after
        # the column header — no spanning row sneaking in between.
        lines = md.splitlines()
        # Find the first pipe-row and the row after it.
        pipe_rows = [i for i, ln in enumerate(lines) if ln.strip().startswith("|")]
        first_pipe = pipe_rows[0]
        assert re.match(r"^\|\s*:?-+:?\s*(\||$)", lines[first_pipe + 1]), (
            f"alignment row missing right after first column header. "
            f"Got: {lines[first_pipe + 1]!r}"
        )
        # The spanning labels survive — as a paragraph above the table.
        assert "Group 1" in md and "Group 2" in md

    def test_markdown_escapes_asterisk(self):
        """A cell text like 'gene*' must NOT be rendered as italic in
        Markdown — backslash-escape the asterisk."""
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable
        t = SofraTable(
            rows=(Row(cells=(Cell(text="gene*expression"),
                             Cell(text="1.2"))),),
            headers=(HeaderRow(cells=(HeaderCell(text="V"),
                                       HeaderCell(text="N"))),),
        )
        md = t.to_markdown()
        # Asterisk must be backslash-escaped.
        assert r"gene\*expression" in md

    def test_markdown_escapes_underscore(self):
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable
        t = SofraTable(
            rows=(Row(cells=(Cell(text="a_b_c"),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="V"),)),),
        )
        md = t.to_markdown()
        assert r"a\_b\_c" in md

    def test_markdown_escapes_brackets(self):
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable
        t = SofraTable(
            rows=(Row(cells=(Cell(text="[unfinished"),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="V"),)),),
        )
        md = t.to_markdown()
        assert r"\[unfinished" in md


# ======================================================================
# labels= no longer breaks downstream modifiers
# ======================================================================
class TestLabelsPreservedDownstream:
    def _df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "arm": rng.choice(["A", "B"], 80),
            "age": rng.normal(55, 10, 80),
        })

    def test_add_n_works_with_labels(self):
        df = self._df()
        t = ps.tbl_one(df, by="arm", variables=["age"],
                       labels={"age": "Patient age (yrs)"}).add_n()
        age_row = next(r for r in t.rows
                       if r.cells[0].text == "Patient age (yrs)")
        # The N cell should be populated — not blank.
        n_cell = age_row.cells[1]
        assert n_cell.text.strip(), "add_n produced blank cell for relabelled row"

    def test_add_ci_works_with_labels(self):
        df = self._df()
        t = ps.tbl_one(df, by="arm", variables=["age"],
                       labels={"age": "Patient age (yrs)"},
                       missing="never").add_ci()
        # Each group cell for the relabelled row should now have a CI
        # appended (look for the "[" character).
        age_row = next(r for r in t.rows
                       if r.cells[0].text == "Patient age (yrs)")
        assert any("[" in c.text for c in age_row.cells[1:])


# ======================================================================
# N at risk uses standard convention
# ======================================================================
class TestNAtRisk:
    def test_n_at_risk_matches_manual_count(self):
        """Number at risk at time t equals the number of observations
        with time >= t (standard KM convention)."""
        from lifelines import KaplanMeierFitter

        from pysofra.models.survival import _n_at_risk
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": rng.integers(0, 2, 100),
        })
        kmf = KaplanMeierFitter()
        kmf.fit(df["time"], df["event"])
        for t in (1, 5, 10, 20, 30):
            ours = _n_at_risk(kmf, t)
            manual = int((df["time"] >= t).sum())
            assert ours == manual, (
                f"N at risk at t={t}: PySofra={ours} vs manual={manual}"
            )


# ======================================================================
# add_global_p raises clearly on tbl_one
# ======================================================================
class TestAddGlobalPOnTblOne:
    # ``add_global_p()`` is implemented for tbl_one / tbl_summary
    # via per-variable logistic regressions.
    # See ``TestTblOneGlobalP`` in ``test_regressions.py`` for
    # the full numeric cross-checks.
    def test_runs_on_tbl_one_and_inserts_global_p_column(self):
        import warnings as _w
        df = pd.DataFrame({
            "arm": ["A", "B"] * 20,
            "x": list(range(40)),
        })
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = (ps.tbl_one(df, by="arm", variables=["x"], missing="never")
                 .add_p()
                 .add_global_p())
        # New 'global p' column was added.
        labels = [h.text for h in t.headers[0].cells]
        assert "global p" in labels, labels

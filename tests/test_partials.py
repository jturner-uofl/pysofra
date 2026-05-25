"""Tests for partial-coverage items.

Covers tbl_uvregression, add_difference, add_global_p, add_ci,
with_pvalue_fmt / with_estimate_fmt, inline_text, modify_spanning_header,
to_image, and risk-times under KM curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def synth():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], n),
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(27, 4, n),
        "sex": rng.choice(["F", "M"], n),
        "race": rng.choice(["W", "B", "Other"], n),
    })
    df["event"] = (df["age"] / 10 + rng.normal(0, 1, n) > 6).astype(int)
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df["time"] = rng.exponential(24, n)
    return df


# ----------------------------------------------------------------------
# tbl_uvregression
# ----------------------------------------------------------------------
class TestTblUvregression:
    def test_ols_basic(self, synth):
        sm = pytest.importorskip("statsmodels.api")
        del sm
        t = ps.tbl_uvregression(synth, outcome="age",
                                predictors=["bmi"], method="OLS")
        assert len(t.rows) == 1
        assert t.rows[0].cells[0].text == "bmi"

    def test_logit_exponentiate(self, synth):
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            synth, outcome="event",
            predictors=["age", "bmi", "sex_M"],
            method="Logit", exponentiate=True,
        )
        # OR label
        assert t.headers[0].cells[2].text == "OR"
        # 3 rows
        assert len(t.rows) == 3

    def test_failed_predictors_footnoted(self, synth):
        pytest.importorskip("statsmodels.api")
        bad = synth.copy()
        bad["bad"] = np.nan  # no data
        t = ps.tbl_uvregression(bad, outcome="age",
                                predictors=["bmi", "bad"])
        # bad predictor should be footnoted
        assert any("bad" in f for f in t.footnotes)


# ----------------------------------------------------------------------
# add_difference
# ----------------------------------------------------------------------
class TestAddDifference:
    def test_continuous_and_dichotomous(self, synth):
        t = ps.tbl_one(synth, by="arm",
                       variables=["age", "sex"]).add_difference()
        # Difference column inserted before any p-value/SMD column.
        labels = [c.text for c in t.headers[0].cells]
        assert any("Diff" in lab for lab in labels)
        # age row has a "X (Y, Z)"-shaped diff cell
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        diff_cell = next(c for c in age_row.cells
                         if c.kind == "ci" and "(" in c.text)
        assert "," in diff_cell.text

    def test_three_groups_raises(self, synth):
        synth3 = synth.copy()
        synth3["arm"] = np.random.default_rng(0).choice(["A","B","C"], len(synth3))
        with pytest.raises(ValueError, match="2 groups"):
            ps.tbl_one(synth3, by="arm",
                       variables=["age"]).add_difference()


# ----------------------------------------------------------------------
# add_global_p
# ----------------------------------------------------------------------
class TestAddGlobalP:
    def test_runs_on_tbl_one_with_per_variable_regressions(self, synth):
        # ``add_global_p()`` is implemented for tbl_one via per-variable
        # logistic regressions. Each row's "global p"
        # cell is the joint Wald p-value across that variable's
        # coefficients in ``Logit(by == ref ~ variable)``.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = (ps.tbl_one(synth, by="arm",
                            variables=["age", "race"])
                 .add_p()
                 .add_global_p())
        labels = [h.text for h in t.headers[0].cells]
        assert "global p" in labels


# ----------------------------------------------------------------------
# add_ci
# ----------------------------------------------------------------------
class TestAddCi:
    def test_continuous_gets_bracketed_ci(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"]).add_ci()
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        # group cells should now contain "[...]"
        assert any("[" in c.text and "]" in c.text for c in age_row.cells[1:])

    def test_dichotomous_gets_wilson_pct(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["sex"]).add_ci()
        sex_row = next(r for r in t.rows if "sex = " in r.cells[0].text)
        assert any("[" in c.text and "%]" in c.text for c in sex_row.cells[1:])

    def test_footnote_added(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"]).add_ci(conf_level=0.95)
        assert any("95%" in f for f in t.footnotes)


# ----------------------------------------------------------------------
# Formatter hooks
# ----------------------------------------------------------------------
class TestFormatterHooks:
    def test_with_pvalue_fmt_applied(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"]).add_p()
        t2 = t.with_pvalue_fmt(lambda p: f"p={p:.4f}")
        # Find a p-value cell with a numeric value
        cell = next(
            c for r in t2.rows for c in r.cells
            if c.kind == "p_value" and isinstance(c.value, (int, float))
        )
        assert cell.text.startswith("p=")

    def test_with_estimate_fmt_applied(self, synth):
        pytest.importorskip("statsmodels.api")
        import statsmodels.api as sm
        X = sm.add_constant(synth[["age"]])
        fit = sm.OLS(synth["bmi"], X).fit()
        t = ps.tbl_regression(fit).with_estimate_fmt(lambda x: f"β={x:.4f}")
        cell = next(
            c for r in t.rows for c in r.cells
            if c.kind == "numeric" and isinstance(c.value, (int, float))
        )
        assert cell.text.startswith("β=")


# ----------------------------------------------------------------------
# inline_text
# ----------------------------------------------------------------------
class TestInlineText:
    def test_by_label(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"]).add_p()
        val = t.inline_text(row="age", column="p-value")
        assert val  # non-empty string

    def test_by_index(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"])
        assert t.inline_text(row=0, column=0) == "age"

    def test_unknown_row_raises(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"])
        with pytest.raises(KeyError):
            t.inline_text(row="not_there", column=0)


# ----------------------------------------------------------------------
# modify_spanning_header
# ----------------------------------------------------------------------
class TestModifySpanningHeader:
    def test_adds_span(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"]).add_p()
        t = t.modify_spanning_header("Treatment groups", start=1, end=2)
        assert any(s.label == "Treatment groups" for s in t.spanning_headers)

    def test_invalid_range(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"])
        with pytest.raises(ValueError):
            t.modify_spanning_header("Bad", start=5, end=10)

    def test_overlapping_replaces(self, synth):
        t = ps.tbl_one(synth, by="arm", variables=["age"]).add_p()
        t = t.modify_spanning_header("A", start=1, end=2)
        t = t.modify_spanning_header("B", start=1, end=2)
        # The first should be replaced.
        labels = [s.label for s in t.spanning_headers]
        assert "A" not in labels and "B" in labels


# ----------------------------------------------------------------------
# to_image
# ----------------------------------------------------------------------
class TestToImage:
    def test_writes_png(self, synth, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "table.png"
        t = ps.tbl_one(synth, by="arm", variables=["age", "sex"]).add_p()
        t.to_image(out)
        assert out.exists()
        assert out.stat().st_size > 1000
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------------
# KM curve with risk_times
# ----------------------------------------------------------------------
class TestKmRiskTable:
    def test_risk_times_renders(self, synth):
        pytest.importorskip("lifelines")
        pytest.importorskip("matplotlib")
        t = (
            ps.tbl_survival(synth, time="time", event="event", by="arm",
                            times=[12, 24])
              .with_km_plot(risk_times=[0, 12, 24])
        )
        assert t.inline_plot is not None
        # The risk-table heading should be in the SVG
        assert "Number at risk" in t.inline_svg

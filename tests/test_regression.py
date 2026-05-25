"""Tests for tbl_regression()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def logit_fit():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(7)
    n = 250
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(27, 4, n),
    })
    df["event"] = (df["age"] / 10 + rng.normal(0, 1, n) > 6).astype(int)
    X = sm.add_constant(df[["age", "bmi"]])
    return sm.Logit(df["event"], X).fit(disp=False)


@pytest.fixture
def ols_fit():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    df["y"] = 2 * df["x1"] - 0.5 * df["x2"] + rng.normal(0, 1, n)
    X = sm.add_constant(df[["x1", "x2"]])
    return sm.OLS(df["y"], X).fit()


class TestTblRegression:
    def test_logit_or_label(self, logit_fit):
        t = ps.tbl_regression(logit_fit, exponentiate=True)
        assert t.headers[0].cells[1].text == "OR"

    def test_logit_beta_label(self, logit_fit):
        t = ps.tbl_regression(logit_fit, exponentiate=False)
        # Logit non-exponentiated → "Estimate"
        assert t.headers[0].cells[1].text in ("Estimate", "β", "exp(β)")

    def test_ols_beta_label(self, ols_fit):
        t = ps.tbl_regression(ols_fit, exponentiate=False)
        assert t.headers[0].cells[1].text == "β"

    def test_no_intercept_by_default(self, ols_fit):
        t = ps.tbl_regression(ols_fit)
        labels = [r.cells[0].text for r in t.rows]
        assert "const" not in labels
        assert "Intercept" not in labels

    def test_intercept_included(self, ols_fit):
        t = ps.tbl_regression(ols_fit, intercept=True)
        labels = [r.cells[0].text for r in t.rows]
        assert "const" in labels or "Intercept" in labels

    def test_bold_p(self, logit_fit):
        t = ps.tbl_regression(logit_fit, exponentiate=True).bold_p(threshold=0.05)
        # At least one row's p-value cell should be bold (age effect is strong)
        bolded = sum(
            1 for r in t.rows for c in r.cells if c.kind == "p_value" and c.bold
        )
        assert bolded >= 1

    def test_labels(self, ols_fit):
        t = ps.tbl_regression(ols_fit, labels={"x1": "Treatment"})
        labels = [r.cells[0].text for r in t.rows]
        assert "Treatment" in labels

    def test_html_renders(self, logit_fit):
        h = ps.tbl_regression(logit_fit, exponentiate=True).to_html()
        assert "<table" in h
        assert "OR" in h

    def test_docx_writes(self, logit_fit, tmp_path):
        out = tmp_path / "reg.docx"
        ps.tbl_regression(logit_fit, exponentiate=True).to_docx(out)
        assert out.exists()

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError):
            ps.tbl_regression(object())

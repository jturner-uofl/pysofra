"""Tests for multi-model regression and lifelines + sklearn extractors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.models.extract import extract


@pytest.fixture
def synth_df():
    rng = np.random.default_rng(123)
    n = 250
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(27, 4, n),
        "tx": rng.integers(0, 2, n),
    })
    linpred = -3 + 0.05 * df["age"] + 0.1 * df["bmi"] - 0.5 * df["tx"]
    df["event"] = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-linpred))).astype(int)
    df["time"] = rng.exponential(scale=10, size=n)
    return df


@pytest.fixture
def logit_fits(synth_df):
    sm = pytest.importorskip("statsmodels.api")
    df = synth_df
    Xu = sm.add_constant(df[["age"]])
    Xa = sm.add_constant(df[["age", "bmi", "tx"]])
    fit_u = sm.Logit(df["event"], Xu).fit(disp=False)
    fit_a = sm.Logit(df["event"], Xa).fit(disp=False)
    return fit_u, fit_a


class TestMultiModel:
    def test_two_logit_models(self, logit_fits):
        fu, fa = logit_fits
        t = ps.tbl_regression([fu, fa], exponentiate=True,
                              model_labels=["Unadjusted", "Adjusted"])
        # Spanning headers should appear
        labels = [s.label for s in t.spanning_headers]
        assert "Unadjusted" in labels and "Adjusted" in labels
        # Header has Variable + (OR, CI, p) x 2 = 7 cells
        assert len(t.headers[0].cells) == 7

    def test_default_model_labels(self, logit_fits):
        fu, fa = logit_fits
        t = ps.tbl_regression([fu, fa])
        labels = [s.label for s in t.spanning_headers]
        assert "Model 1" in labels and "Model 2" in labels

    def test_wrong_model_labels_length_raises(self, logit_fits):
        fu, fa = logit_fits
        with pytest.raises(ValueError):
            ps.tbl_regression([fu, fa], model_labels=["only_one"])

    def test_single_model_still_works(self, logit_fits):
        fu, _ = logit_fits
        t = ps.tbl_regression(fu)
        # No spanning headers for single model
        assert t.spanning_headers == ()

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            ps.tbl_regression([])

    def test_union_of_coefs(self, logit_fits):
        fu, fa = logit_fits
        t = ps.tbl_regression([fu, fa], exponentiate=True)
        labels = [r.cells[0].text for r in t.rows]
        # Union: age (in both), bmi + tx (only in adjusted)
        assert "age" in labels
        assert "bmi" in labels
        assert "tx" in labels
        # Unadjusted's row for bmi/tx is "—"
        bmi_row = next(r for r in t.rows if r.cells[0].text == "bmi")
        # First model's estimate cell should be "—"
        assert bmi_row.cells[1].text == "—"


class TestLifelinesExtract:
    def test_cox_extract(self, synth_df):
        lifelines = pytest.importorskip("lifelines")
        df = synth_df
        cph = lifelines.CoxPHFitter()
        cph.fit(df[["age", "bmi", "tx", "time", "event"]],
                duration_col="time", event_col="event")
        summary = extract(cph)
        assert "age" in summary.estimates.index
        assert summary.natural_exponentiate is True
        assert "CoxPHFitter" in summary.family

    def test_cox_in_tbl_regression(self, synth_df):
        lifelines = pytest.importorskip("lifelines")
        df = synth_df
        cph = lifelines.CoxPHFitter()
        cph.fit(df[["age", "bmi", "tx", "time", "event"]],
                duration_col="time", event_col="event")
        t = ps.tbl_regression(cph)
        # Cox naturally exponentiates → "HR" header
        assert t.headers[0].cells[1].text == "HR"
        assert len(t.rows) == 3  # age, bmi, tx


class TestSklearnExtract:
    def test_logistic_regression(self, synth_df):
        sklearn = pytest.importorskip("sklearn.linear_model")
        df = synth_df
        clf = sklearn.LogisticRegression(max_iter=1000)
        clf.fit(df[["age", "bmi", "tx"]], df["event"])
        summary = extract(clf)
        assert len(summary.estimates) == 3
        # CIs and p-values are NaN for sklearn
        assert summary.ci_lo.isna().all()
        assert summary.pvalues.isna().all()

    def test_linear_regression(self, synth_df):
        sklearn = pytest.importorskip("sklearn.linear_model")
        df = synth_df
        reg = sklearn.LinearRegression()
        reg.fit(df[["age", "bmi"]], df["tx"])
        t = ps.tbl_regression(reg)
        labels = [r.cells[0].text for r in t.rows]
        assert "age" in labels and "bmi" in labels

"""Descriptive-summary edge-case tests.

Hits the new code paths added by the blockers/highs pass so the suite
keeps 100% line coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


# ----------------------------------------------------------------------
# regression.py — design refit dispatch for GLM / Logit / Poisson
# ----------------------------------------------------------------------
class TestDesignRefitFamilies:
    @pytest.fixture
    def trial(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.5, size=n),
            "psu": rng.integers(0, 25, size=n),
        })
        df["y_bin"] = (rng.uniform(size=n) <
                       1 / (1 + np.exp(-df["x"]))).astype(int)
        df["y_count"] = rng.poisson(lam=np.exp(0.5 * df["x"]), size=n)
        return df

    def test_glm_design_refit(self, trial):
        sm = pytest.importorskip("statsmodels.api")
        m = sm.GLM(trial["y_bin"], sm.add_constant(trial[["x"]]),
                   family=sm.families.Binomial()).fit()
        d = ps.SurveyDesign(weights="w", cluster="psu")
        t = ps.tbl_regression(m, intercept=False, design=d, data=trial,
                              exponentiate=True)
        assert any(r.cells[0].text == "x" for r in t.rows)

    def test_poisson_design_refit(self, trial):
        sm = pytest.importorskip("statsmodels.api")
        m = sm.Poisson(trial["y_count"], sm.add_constant(trial[["x"]])).fit(disp=False)
        d = ps.SurveyDesign(weights="w")
        t = ps.tbl_regression(m, intercept=False, design=d, data=trial,
                              exponentiate=True)
        assert any(r.cells[0].text == "x" for r in t.rows)

    def test_logit_design_refit(self, trial):
        sm = pytest.importorskip("statsmodels.api")
        m = sm.Logit(trial["y_bin"], sm.add_constant(trial[["x"]])).fit(disp=False)
        d = ps.SurveyDesign(weights="w")
        t = ps.tbl_regression(m, intercept=False, design=d, data=trial,
                              exponentiate=True)
        # Refit through GLM(Binomial) → returns ORs
        row = next(r for r in t.rows if r.cells[0].text == "x")
        assert row.cells[1].value is not None

    def test_design_without_data_raises_for_cluster(self):
        sm = pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [1.0, 2, 3, 4, 5]})
        m = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
        d = ps.SurveyDesign(weights="x", cluster="x")
        # data=None with cluster set → ValueError before reaching cluster check.
        # Weights are checked first, so it raises with the weights message.
        with pytest.raises(ValueError, match="design has"):
            ps.tbl_regression(m, intercept=False, design=d)

    def test_design_cluster_length_mismatch(self):
        sm = pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0),
                           "psu": [0] * 10})
        m = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
        d = ps.SurveyDesign(weights="x", cluster="psu")
        wrong = df.iloc[:5]
        with pytest.raises(ValueError, match="length"):
            ps.tbl_regression(m, intercept=False, design=d, data=wrong)


    def test_design_no_weights_no_cluster_uses_hc1(self, trial):
        # A SurveyDesign that has neither weights nor cluster → the
        # weight-free + HC1 path (line 458).
        sm = pytest.importorskip("statsmodels.api")
        df = trial
        m = sm.OLS(df["y_count"].astype(float),
                   sm.add_constant(df[["x"]])).fit()
        d = ps.SurveyDesign(weights=None, cluster=None,
                            fpc="psu")  # only FPC → not honored by refit yet
        t = ps.tbl_regression(m, intercept=False, design=d, data=df)
        assert any(r.cells[0].text == "x" for r in t.rows)


# ----------------------------------------------------------------------
# uvregression.py — categorical-only edge paths
# ----------------------------------------------------------------------
class TestUvregressionCategoricalEdges:
    def test_single_level_categorical_predictor(self):
        # A single-level "categorical" predictor has no contrast to test.
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({
            "y": [1.0, 2, 3, 4, 5, 6],
            "cat": ["a"] * 6,
        })
        t = ps.tbl_uvregression(df, outcome="y", predictors=["cat"])
        # No body row produced for the single-level predictor;
        # it's flagged as failed.
        assert any("failed" in fn for fn in t.footnotes)

    def test_pandas_categorical_with_unused_category(self):
        # Predictor has a Categorical dtype with a level present in
        # `categories` but not in the data — _expand_predictor's
        # "continue" branch.
        pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({
            "x": pd.Categorical(rng.choice(["a", "b"], n),
                                categories=["a", "b", "c_unused"]),
            "y": rng.normal(size=n),
        })
        t = ps.tbl_uvregression(df, outcome="y", predictors=["x"])
        labels = [r.cells[0].text for r in t.rows]
        assert "a" in labels and "b" in labels
        # c_unused is in categories but unused → no row
        assert "c_unused" not in labels


# ----------------------------------------------------------------------
# km.py — _n_at_risk Exception branch (defensive)
# ----------------------------------------------------------------------
class TestKMNAtRiskException:
    def test_n_at_risk_returns_zero_on_error(self):
        from pysofra.plot.km import _n_at_risk

        class Broken:
            @property
            def event_table(self):
                raise RuntimeError("synthetic")

        assert _n_at_risk(Broken(), 1.0) == 0

    def test_n_at_risk_happy_path(self):
        # The plot module duplicates _n_at_risk — exercise its happy
        # path directly so the line is covered without re-fitting a
        # KM in every plot test.
        from pysofra.plot.km import _n_at_risk

        class Stub:
            event_table = pd.DataFrame(
                {"at_risk": [10, 8, 5]},
                index=[1.0, 5.0, 10.0],
            )

        assert _n_at_risk(Stub(), 4.0) == 8
        assert _n_at_risk(Stub(), 100.0) == 0


# ----------------------------------------------------------------------
# extras.py — labels=…  display-relabelled-row resolver branches
# ----------------------------------------------------------------------
class TestLabelsResolver:
    def test_labels_match_relabelled_dichotomous(self):
        # The labels-resolver path: a row labelled "Patient sex = M"
        # is matched back to source variable "sex".
        from pysofra.summary.extras import _find_variable_for_row
        assert _find_variable_for_row(
            "Patient sex = M",
            variables=("sex",),
            kinds={"sex": "dichotomous"},
            labels={"sex": "Patient sex"},
        ) == "sex"

    def test_labels_match_exact_relabelled(self):
        from pysofra.summary.extras import _find_variable_for_row
        assert _find_variable_for_row(
            "Patient age (yrs)",
            variables=("age",),
            kinds={"age": "continuous"},
            labels={"age": "Patient age (yrs)"},
        ) == "age"

    def test_labels_resolver_returns_none_when_blank(self):
        # When labels dict has an empty value, that entry is skipped.
        from pysofra.summary.extras import _find_variable_for_row
        assert _find_variable_for_row(
            "unrelated",
            variables=("age",),
            kinds={"age": "continuous"},
            labels={"age": ""},
        ) is None


# ----------------------------------------------------------------------
# render/markdown.py — single-column spanning header
# ----------------------------------------------------------------------
class TestMarkdownSpanningSingleColumn:
    def test_single_column_span_renders_col_n(self):
        # A spanning header that covers exactly one column produces
        # the "col 1" form rather than "cols 1–1".
        from pysofra.core.schema import HeaderCell, HeaderRow, SpanningHeader
        from pysofra.core.table import SofraTable
        t = SofraTable(
            headers=(HeaderRow(cells=(HeaderCell(text="A"),
                                       HeaderCell(text="B"))),),
            spanning_headers=(SpanningHeader(label="Solo", start=0, end=0),),
        )
        md = t.to_markdown()
        assert "col 1" in md
        assert "cols" not in md.split("\n")[0]  # not the range form

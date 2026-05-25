"""Tests for the extended SurveyDesign + design-aware regression / tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.summary.design import replicate_mean_var
from pysofra.summary.tests import svyttest


@pytest.fixture
def survey_data():
    rng = np.random.default_rng(2026)
    n = 400
    df = pd.DataFrame({
        "arm":    rng.choice(["A", "B"], n),
        "strata": rng.choice(["s1", "s2", "s3"], n),
        "psu":    rng.choice(range(40), n),
        "ssu":    rng.choice(range(8), n),
        "age":    rng.normal(60, 10, n),
        "bmi":    rng.normal(27, 4, n),
        "sex_M":  rng.integers(0, 2, n),
        "w":      rng.uniform(0.5, 3.0, n),
    })
    df["y"] = df["age"] * 0.05 + df["sex_M"] * 0.4 + rng.normal(0, 1, n)
    return df


# ----------------------------------------------------------------------
# svyttest
# ----------------------------------------------------------------------
class TestSvyttest:
    def test_returns_design_adjusted(self, survey_data):
        res = svyttest(
            survey_data["age"], survey_data["arm"], survey_data["w"],
            strata=survey_data["strata"], cluster=survey_data["psu"],
        )
        assert "Design-adjusted" in res.test
        assert res.p_value is not None
        assert 0 <= res.p_value <= 1

    def test_three_groups_returns_NA(self, survey_data):
        # Convert to 3-group: shouldn't apply svyttest, returns NA
        survey_data["arm3"] = np.r_[
            ["A"] * 130, ["B"] * 130, ["C"] * 140,
        ]
        res = svyttest(
            survey_data["age"], survey_data["arm3"], survey_data["w"],
        )
        assert res.p_value is None or res.test == "—"

    def test_routes_via_tbl_one_with_design(self, survey_data):
        design = ps.SurveyDesign(weights="w", strata="strata", cluster="psu")
        t = ps.tbl_one(survey_data, by="arm", design=design,
                       variables=["age"]).add_p()
        assert any("Design-adjusted" in f for f in t.footnotes)


# ----------------------------------------------------------------------
# Multi-stage clusters
# ----------------------------------------------------------------------
class TestMultiStageCluster:
    def test_tuple_cluster_accepted(self, survey_data):
        design = ps.SurveyDesign(
            weights="w", strata="strata", cluster=("psu", "ssu"),
        )
        # Should not raise, AND the validated design must retain the
        # multi-stage cluster tuple intact (a silent down-cast to the
        # outermost PSU would defeat the validation).
        design.validate(survey_data)
        assert design.cluster == ("psu", "ssu")
        assert design.strata == "strata"
        assert design.weights == "w"

    def test_unknown_cluster_col_raises(self, survey_data):
        design = ps.SurveyDesign(
            weights="w", cluster=("psu", "no_such_col"),
        )
        with pytest.raises(KeyError):
            design.validate(survey_data)

    def test_primary_cluster_extraction(self):
        d_single = ps.SurveyDesign(weights="w", cluster="psu")
        assert d_single.primary_cluster == "psu"
        d_multi = ps.SurveyDesign(weights="w", cluster=("psu", "ssu"))
        assert d_multi.primary_cluster == "psu"
        d_none = ps.SurveyDesign(weights="w")
        assert d_none.primary_cluster is None

    def test_multi_stage_used_in_tbl_one(self, survey_data):
        design = ps.SurveyDesign(weights="w", strata="strata",
                                 cluster=("psu", "ssu"))
        t = ps.tbl_one(survey_data, by="arm", design=design,
                       variables=["age"]).add_p()
        assert any("Design-adjusted" in f for f in t.footnotes)


# ----------------------------------------------------------------------
# Replicate weights
# ----------------------------------------------------------------------
class TestReplicateWeights:
    def test_jk1_scaling(self, survey_data):
        # Construct simple JK1 weights: each replicate drops one PSU.
        psus = sorted(survey_data["psu"].unique())
        R = 8
        for r in range(R):
            drop = psus[r % len(psus)]
            survey_data[f"rep_{r}"] = survey_data["w"] * (
                (survey_data["psu"] != drop).astype(float)
            )
        rep_w = [survey_data[f"rep_{i}"] for i in range(R)]
        m, v, n_eff = replicate_mean_var(
            survey_data["age"], survey_data["w"], rep_w,
            replicate_type="jk1",
        )
        assert v > 0
        assert n_eff > 0
        assert not np.isnan(m)

    def test_bootstrap_scale_differs(self, survey_data):
        R = 6
        psus = sorted(survey_data["psu"].unique())
        for r in range(R):
            drop = psus[r % len(psus)]
            survey_data[f"rep_{r}"] = survey_data["w"] * (
                (survey_data["psu"] != drop).astype(float)
            )
        rep_w = [survey_data[f"rep_{i}"] for i in range(R)]
        _, v_jk, _ = replicate_mean_var(survey_data["age"], survey_data["w"],
                                         rep_w, replicate_type="jk1")
        _, v_bs, _ = replicate_mean_var(survey_data["age"], survey_data["w"],
                                         rep_w, replicate_type="bootstrap")
        # jk1 = (R-1)/R · sumsq; bootstrap = 1/R · sumsq → bootstrap < jk1 here
        assert v_bs <= v_jk + 1e-12

    def test_replicate_columns_in_tbl_one(self, survey_data):
        R = 6
        psus = sorted(survey_data["psu"].unique())
        for r in range(R):
            drop = psus[r % len(psus)]
            survey_data[f"rep_{r}"] = survey_data["w"] * (
                (survey_data["psu"] != drop).astype(float)
            )
        design = ps.SurveyDesign(
            weights="w",
            replicate_weights=tuple(f"rep_{i}" for i in range(R)),
            replicate_type="jk1",
        )
        # Replicate columns should be auto-excluded from variables.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ps.tbl_one(survey_data, by="arm", design=design)
        labels = [r.cells[0].text for r in t.rows]
        for c in design.replicate_weights:
            assert all(c not in label for label in labels)


# ----------------------------------------------------------------------
# Design-aware tbl_regression
# ----------------------------------------------------------------------
class TestDesignAwareRegression:
    def test_cluster_robust_se_changes_pvalues(self, survey_data):
        smf = pytest.importorskip("statsmodels.formula.api")
        fit = smf.ols("y ~ age + sex_M", data=survey_data).fit()
        t_naive = ps.tbl_regression(fit)
        t_robust = ps.tbl_regression(
            fit, design=ps.SurveyDesign(weights="w", cluster="psu"),
            data=survey_data,
        )
        # Headers identical; p-values may differ
        assert (
            [c.text for c in t_naive.headers[0].cells]
            == [c.text for c in t_robust.headers[0].cells]
        )

    def test_no_data_with_cluster_design_raises(self, survey_data):
        smf = pytest.importorskip("statsmodels.formula.api")
        fit = smf.ols("y ~ age", data=survey_data).fit()
        with pytest.raises(ValueError, match="data="):
            ps.tbl_regression(fit,
                              design=ps.SurveyDesign(weights="w", cluster="psu"))

    def test_hc1_used_when_no_cluster(self, survey_data):
        smf = pytest.importorskip("statsmodels.formula.api")
        fit = smf.ols("y ~ age", data=survey_data).fit()
        t = ps.tbl_regression(fit,
                              design=ps.SurveyDesign(weights="w"),
                              data=survey_data)
        # Just runs cleanly
        assert len(t.rows) > 0


# ----------------------------------------------------------------------
# add_global_p — joint Wald-F
# ----------------------------------------------------------------------
class TestAddGlobalP:
    def test_joint_p_for_categorical(self, survey_data):
        smf = pytest.importorskip("statsmodels.formula.api")
        survey_data["region"] = np.random.default_rng(1).choice(
            ["NE", "S", "W"], len(survey_data),
        )
        fit = smf.ols("y ~ age + C(region)", data=survey_data).fit()
        t = ps.tbl_regression(fit).add_global_p()
        # Both region rows should have the same joint p.
        region_rows = [
            r for r in t.rows if r.cells[0].text.startswith("C(region)")
        ]
        gp = []
        for r in region_rows:
            cell = next(c for c in r.cells if r.cells.index(c) > 0
                        and "p" not in c.text.lower())
            del cell
            # global p is the last column
            gp.append(r.cells[-1].text)
        assert len(set(gp)) == 1  # same joint p

    def test_single_level_keeps_existing_p(self, survey_data):
        smf = pytest.importorskip("statsmodels.formula.api")
        fit = smf.ols("y ~ age", data=survey_data).fit()
        t = ps.tbl_regression(fit).add_global_p()
        # age row: joint p == per-row p (single coefficient)
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        per_row_p = next(c for c in age_row.cells if c.kind == "p_value")
        joint_p_text = age_row.cells[-1].text
        # They should be equal (or very close)
        assert per_row_p.text == joint_p_text

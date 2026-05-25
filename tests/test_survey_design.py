"""Tests for SurveyDesign with strata + cluster + FPC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.summary.design import design_mean_var


@pytest.fixture
def survey_df():
    rng = np.random.default_rng(2026)
    n = 400
    return pd.DataFrame({
        "arm":    rng.choice(["A", "B"], n),
        "strata": rng.choice(["s1", "s2", "s3"], n),
        "psu":    rng.choice(range(50), n),
        "fpc":    [10000] * n,
        "age":    rng.normal(60, 10, n),
        "sex":    rng.choice(["F", "M"], n),
        "w":      rng.uniform(0.5, 3.0, n),
    })


class TestSurveyDesignDataclass:
    def test_construction(self):
        d = ps.SurveyDesign(weights="w", strata="s", cluster="psu", fpc="fpc")
        assert d.weights == "w"
        assert d.strata == "s"
        assert d.cluster == "psu"
        assert d.fpc == "fpc"

    def test_validate_missing_column_raises(self, survey_df):
        d = ps.SurveyDesign(weights="not_there")
        with pytest.raises(KeyError):
            d.validate(survey_df)

    def test_excludes_design_columns_from_variables(self, survey_df):
        design = ps.SurveyDesign(weights="w", strata="strata",
                                 cluster="psu", fpc="fpc")
        t = ps.tbl_one(survey_df, by="arm", design=design)
        labels = [r.cells[0].text for r in t.rows]
        for col in ("w", "strata", "psu", "fpc"):
            assert all(col not in lab for lab in labels), \
                f"Column {col!r} leaked into table labels"


class TestDesignMeanVar:
    def test_unstratified_unclustered_matches_unbiased_var(self):
        v = pd.Series([1.0, 2, 3, 4, 5])
        w = pd.Series([1.0] * 5)
        mean, var, n = design_mean_var(v, w)
        # With equal weights, design var of mean = sample-var / n
        expected_var = float(np.var(v, ddof=1)) / len(v)
        assert mean == pytest.approx(3.0)
        # Allow modest tolerance for the design-based linearisation formula.
        assert var == pytest.approx(expected_var, rel=0.05)

    def test_stratified(self, survey_df):
        mean, var, n = design_mean_var(
            survey_df["age"], survey_df["w"],
            strata=survey_df["strata"],
        )
        assert var >= 0
        assert not np.isnan(mean)

    def test_cluster(self, survey_df):
        mean, var, n = design_mean_var(
            survey_df["age"], survey_df["w"],
            cluster=survey_df["psu"],
        )
        assert var >= 0
        assert not np.isnan(mean)

    def test_stratified_plus_cluster(self, survey_df):
        mean, var, n = design_mean_var(
            survey_df["age"], survey_df["w"],
            strata=survey_df["strata"],
            cluster=survey_df["psu"],
        )
        assert var >= 0

    def test_fpc_reduces_variance(self, survey_df):
        mean, v0, _ = design_mean_var(
            survey_df["age"], survey_df["w"],
            strata=survey_df["strata"],
        )
        mean_fpc, v1, _ = design_mean_var(
            survey_df["age"], survey_df["w"],
            strata=survey_df["strata"],
            fpc=survey_df["fpc"],
        )
        # FPC always shrinks variance — strictly, when n_h < N_h.
        assert v1 < v0
        # And not by a microscopic amount: the smallest sane FPC
        # factor on this fixture is (1 - n_h/N_h) for n_h=5,
        # N_h=10000 -> 0.9995, so v0 - v1 should be a measurable
        # multiple of the unscaled variance, not just rounding noise.
        assert (v0 - v1) > 1e-9 * v0

    def test_fpc_applies_exact_multiplier(self):
        """Mutation-resistant regression: removing ``* (1 - f_h)`` from
        ``design.py`` must change this number, not merely sit within
        a tolerance band of the no-FPC variance.
        """
        v = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        w = pd.Series([1.0] * 8)
        strata = pd.Series(["s1"] * 4 + ["s2"] * 4)
        # fpc column reports the stratum population size N_h. With
        # n_h = 4 per stratum, the FPC factor is 1 - 4/8 = 0.5 in
        # each stratum, so the total weighted-mean variance must be
        # exactly half of the un-FPC'd value.
        fpc = pd.Series([8] * 8)
        _, v0, _ = design_mean_var(v, w, strata=strata)
        _, vf, _ = design_mean_var(v, w, strata=strata, fpc=fpc)
        assert vf == pytest.approx(0.5 * v0, rel=1e-12)

    def test_fpc_full_population_zeros_variance(self):
        """At n_h == N_h the FPC factor is 0, so the variance must be 0."""
        v = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = pd.Series([1.0] * 4)
        strata = pd.Series(["s1"] * 4)
        fpc = pd.Series([4] * 4)  # n_h / N_h = 4/4 = 1
        _, vf, _ = design_mean_var(v, w, strata=strata, fpc=fpc)
        assert vf == pytest.approx(0.0, abs=1e-12)


class TestTblOneWithDesign:
    def test_design_with_strata_shows_SE_footnote(self, survey_df):
        design = ps.SurveyDesign(weights="w", strata="strata")
        t = ps.tbl_one(survey_df, by="arm", design=design)
        assert any("design-based" in f for f in t.footnotes)
        assert any("Mean (SE)" in f for f in t.footnotes)

    def test_weights_only_design_still_uses_SD_footnote(self, survey_df):
        design = ps.SurveyDesign(weights="w")  # no strata / cluster
        t = ps.tbl_one(survey_df, by="arm", design=design)
        assert any("Mean (SD)" in f for f in t.footnotes)

    def test_design_routes_through_rao_scott(self, survey_df):
        design = ps.SurveyDesign(weights="w", strata="strata")
        t = ps.tbl_one(survey_df, by="arm", design=design,
                       variables=["age", "sex"]).add_p()
        assert any("Rao" in f for f in t.footnotes)

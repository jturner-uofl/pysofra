"""Tests for R-ecosystem feature parity: tbl_cross, effect sizes,
add_significance_stars / add_n / add_stat_label / color_scale_if,
MixedLM/GEE extractor, and multiple-imputation pooling (Rubin's rules)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def small_df():
    rng = np.random.default_rng(2026)
    n = 200
    return pd.DataFrame({
        "arm":  rng.choice(["A", "B"], n),
        "sex":  rng.choice(["F", "M"], n),
        "race": rng.choice(["W", "B", "Other"], n),
        "age":  rng.normal(60, 10, n),
    })


# ----------------------------------------------------------------------
# tbl_cross
# ----------------------------------------------------------------------
class TestTblCross:
    def test_default_n_col_pct(self, small_df):
        t = ps.tbl_cross(small_df, row="sex", column="arm")
        # 2 row levels + 1 margin row = 3 rows
        assert len(t.rows) == 3
        # Column header includes the column variable as spanning header
        assert any(s.label == "arm" for s in t.spanning_headers)
        # Body cells contain "n (xx.x%)"
        first = t.rows[0]
        assert "%" in first.cells[1].text

    def test_row_pct(self, small_df):
        t = ps.tbl_cross(small_df, row="sex", column="arm",
                         cell="row_pct", margins=False)
        first = t.rows[0]
        # cell text ends with %
        assert "%" in first.cells[1].text

    def test_n_only(self, small_df):
        t = ps.tbl_cross(small_df, row="sex", column="arm",
                         cell="n", margins=False)
        first = t.rows[0]
        # cell text is a digit string
        assert first.cells[1].text.replace(",", "").isdigit()

    def test_no_margins(self, small_df):
        t_no = ps.tbl_cross(small_df, row="sex", column="arm", margins=False)
        t_yes = ps.tbl_cross(small_df, row="sex", column="arm", margins=True)
        # margins=True adds a "Total" row and a "Total" column
        assert len(t_yes.rows) == len(t_no.rows) + 1
        assert len(t_yes.headers[0].cells) == len(t_no.headers[0].cells) + 1

    def test_unknown_cell_style_raises(self, small_df):
        with pytest.raises(ValueError, match="cell"):
            ps.tbl_cross(small_df, row="sex", column="arm", cell="bad")

    def test_unknown_row_column_raises(self, small_df):
        with pytest.raises(KeyError):
            ps.tbl_cross(small_df, row="nope", column="arm")
        with pytest.raises(KeyError):
            ps.tbl_cross(small_df, row="arm", column="nope")

    def test_labels_remap_levels(self, small_df):
        t = ps.tbl_cross(small_df, row="sex", column="arm",
                         labels={"F": "Female", "M": "Male"})
        first_row = t.rows[0]
        assert first_row.cells[0].text in ("Female", "Male")

    def test_all_nan_emits_placeholder(self):
        df = pd.DataFrame({"r": [None, None], "c": [None, None]})
        t = ps.tbl_cross(df, row="r", column="c")
        assert any("No non-missing" in fn for fn in t.footnotes)

    def test_categorical_preserves_order(self):
        df = pd.DataFrame({
            "r": pd.Categorical(["mid", "low", "high"] * 3,
                                 categories=["low", "mid", "high"],
                                 ordered=True),
            "c": ["A", "B"] * 4 + ["A"],
        })
        t = ps.tbl_cross(df, row="r", column="c", margins=False)
        labels = [r.cells[0].text for r in t.rows]
        assert labels == ["low", "mid", "high"]


# ----------------------------------------------------------------------
# Effect-size helpers
# ----------------------------------------------------------------------
class TestEffectSizes:
    def test_cohen_d_zero_when_identical(self):
        v = np.array([1.0, 2, 3, 4, 5])
        assert ps.cohen_d(v, v.copy()) == pytest.approx(0.0)

    def test_cohen_d_signed(self):
        a = np.array([1.0, 2, 3, 4, 5])
        b = np.array([3.0, 4, 5, 6, 7])
        d = ps.cohen_d(a, b)
        assert d is not None and d < 0

    def test_hedges_g_smaller_than_cohen_d_in_small_samples(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 6)
        b = rng.normal(1, 1, 6)
        d = ps.cohen_d(a, b)
        g = ps.hedges_g(a, b)
        # Small-sample correction shrinks magnitude
        assert abs(g) < abs(d)

    def test_eta_squared_range(self):
        rng = np.random.default_rng(0)
        v = pd.Series(rng.normal(0, 1, 90))
        g = pd.Series(["A", "B", "C"] * 30)
        es = ps.eta_squared(v, g)
        assert 0 <= es <= 1

    def test_omega_less_than_eta(self):
        rng = np.random.default_rng(7)
        v = pd.Series(rng.normal(0, 1, 60))
        g = pd.Series(["A", "B", "C"] * 20)
        eta = ps.eta_squared(v, g)
        omega = ps.omega_squared(v, g)
        # ω² is always less than η² (less biased downward)
        assert omega <= eta + 1e-9

    def test_cramers_v_range(self, small_df):
        v = ps.cramers_v(small_df["sex"], small_df["arm"])
        assert 0 <= v <= 1

    def test_phi_only_for_2x2(self, small_df):
        # 'race' has 3 levels → φ returns None
        assert ps.phi_coefficient(small_df["race"], small_df["arm"]) is None
        # 'sex' is 2-level → φ valid
        assert ps.phi_coefficient(small_df["sex"], small_df["arm"]) is not None

    def test_auto_effect_size_continuous_2grp(self, small_df):
        name, val = ps.auto_effect_size(small_df["age"], small_df["arm"])
        assert name == "Cohen's d"
        assert val is not None

    def test_auto_effect_size_continuous_3grp(self, small_df):
        name, val = ps.auto_effect_size(small_df["age"], small_df["race"])
        assert name == "η²"
        assert val is not None

    def test_auto_effect_size_categorical(self, small_df):
        name, val = ps.auto_effect_size(small_df["race"], small_df["arm"])
        assert name in ("Cramér's V", "φ")
        assert val is not None

    def test_degenerate_inputs_return_none(self):
        assert ps.cohen_d([1.0], [2.0]) is None
        assert ps.eta_squared(pd.Series([], dtype=float),
                               pd.Series([], dtype=str)) is None
        assert ps.cramers_v(pd.Series([]), pd.Series([])) is None


# ----------------------------------------------------------------------
# add_significance_stars / add_n / add_stat_label / color_scale_if
# ----------------------------------------------------------------------
class TestExtraModifiers:
    def test_significance_stars(self, small_df):
        # Inject a definite significant variable
        df = small_df.copy()
        df["highly_sig"] = df["age"] * (df["arm"] == "A").astype(int) * 2
        t = ps.tbl_one(df, by="arm",
                       variables=["highly_sig"]).add_p().add_significance_stars()
        # Star column is the last column
        star_text = t.rows[0].cells[-1].text
        assert star_text in {"*", "**", "***", ""}

    def test_significance_stars_no_p_value(self, small_df):
        # Without add_p, every row should get empty stars
        t = ps.tbl_one(small_df, by="arm",
                       variables=["age"]).add_significance_stars()
        assert all(r.cells[-1].text == "" for r in t.rows)

    def test_add_n_inserts_n_column(self, small_df):
        t = ps.tbl_one(small_df, by="arm", variables=["age", "sex"]).add_n()
        # Second column is the new "N" column
        assert t.headers[0].cells[1].text == "N"
        # age row's N should equal data size (no missing)
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        assert int(age_row.cells[1].text.replace(",", "")) == len(small_df)

    def test_add_stat_label(self, small_df):
        t = ps.tbl_one(small_df, by="arm",
                       variables=["age", "sex"]).add_stat_label()
        assert t.headers[0].cells[1].text == "Statistic"
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        assert age_row.cells[1].text == "Mean (SD)"

    def test_add_stat_label_nonnormal(self, small_df):
        t = ps.tbl_one(small_df, by="arm", variables=["age"],
                       nonnormal=["age"]).add_stat_label()
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        assert age_row.cells[1].text == "Median (Q1, Q3)"

    def test_color_scale_if_applies_html_style(self, small_df):
        t = ps.tbl_one(small_df, by="arm",
                       variables=["age"]).add_p().add_smd()
        # SMD column is now col index 4 (Char, A, B, p, SMD)
        t_col = t.color_scale_if(column=4)
        html = t_col.to_html()
        # At least one background:#... should be present
        assert "background:#" in html

    def test_color_scale_if_no_numerics(self, small_df):
        # apply to the first (label) column → no numerics → table unchanged
        t = ps.tbl_one(small_df, by="arm", variables=["age"])
        t2 = t.color_scale_if(column=0)
        assert t2.rows == t.rows

    def test_chain_order_rebuild_first_then_postprocess(self, small_df):
        # Correct chain order: rebuild-based modifiers first
        t = (
            ps.tbl_one(small_df, by="arm", variables=["age", "sex"])
              .add_p()
              .add_n()
              .add_stat_label()
              .add_significance_stars()
        )
        headers = [c.text for c in t.headers[0].cells]
        assert "N" in headers
        assert "Statistic" in headers
        assert "p-value" in headers


# ----------------------------------------------------------------------
# MixedLM extractor
# ----------------------------------------------------------------------
class TestMixedLMExtractor:
    def test_mixedlm_extract(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n_subj = 30
        n_obs = 4
        df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_subj), n_obs),
            "time":    np.tile(np.arange(n_obs), n_subj),
            "y":       rng.normal(0, 1, n_subj * n_obs),
        })
        df["y"] += 0.5 * df["time"] + rng.normal(0, 0.3, n_subj * n_obs)

        md = sm.MixedLM.from_formula("y ~ time", df, groups=df["subject"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = md.fit()
        t = ps.tbl_regression(fit)
        # Should at least include the time coefficient
        labels = [r.cells[0].text for r in t.rows]
        assert "time" in labels
        # Family label mentions MixedLM
        assert "MixedLM" in t.metadata.get("family", "")


# ----------------------------------------------------------------------
# GEE extractor
# ----------------------------------------------------------------------
class TestGEEExtractor:
    def test_gee_extract(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n_subj = 30
        n_obs = 5
        df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_subj), n_obs),
            "x":       rng.normal(0, 1, n_subj * n_obs),
        })
        df["y"] = 0.5 * df["x"] + rng.normal(0, 1, n_subj * n_obs)
        fam = sm.families.Gaussian()
        gee = sm.GEE.from_formula("y ~ x", "subject", df,
                                   family=fam,
                                   cov_struct=sm.cov_struct.Exchangeable())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = gee.fit()
        t = ps.tbl_regression(fit)
        assert "GEE" in t.metadata.get("family", "")


# ----------------------------------------------------------------------
# Multiple imputation pooling (Rubin's rules)
# ----------------------------------------------------------------------
class TestMiPooling:
    def test_pool_two_fits(self):
        sm = pytest.importorskip("statsmodels.api")
        # Build 5 imputed datasets with slightly different y noise
        fits = []
        for seed in range(5):
            rng_i = np.random.default_rng(seed)
            n = 100
            df = pd.DataFrame({
                "x":   rng_i.normal(0, 1, n),
                "z":   rng_i.normal(0, 1, n),
            })
            df["y"] = 0.5 * df["x"] - 0.2 * df["z"] + rng_i.normal(0, 1, n)
            X = sm.add_constant(df[["x", "z"]])
            fits.append(sm.OLS(df["y"], X).fit())

        pooled = ps.pool(fits)
        # Pooled point estimates should be close to the per-fit means
        for coef in ("x", "z"):
            per_fit = [float(f.params[coef]) for f in fits]
            assert pooled.estimates[coef] == pytest.approx(
                np.mean(per_fit), abs=1e-9,
            )
        # Pooled family label
        assert "Rubin" in pooled.family

    def test_pool_into_tbl_regression(self):
        sm = pytest.importorskip("statsmodels.api")
        fits = []
        for seed in range(4):
            rng_i = np.random.default_rng(seed)
            n = 80
            df = pd.DataFrame({"x": rng_i.normal(size=n)})
            df["y"] = 0.5 * df["x"] + rng_i.normal(size=n)
            fits.append(sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit())
        pooled = ps.pool(fits)
        t = ps.tbl_regression(pooled)
        assert any(r.cells[0].text == "x" for r in t.rows)

    def test_pool_requires_at_least_two(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = rng.normal(size=n)
        fit = sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit()
        with pytest.raises(ValueError, match="at least two"):
            ps.pool([fit])

    def test_pool_coef_name_mismatch(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        df["y"] = rng.normal(size=n)
        f1 = sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit()
        f2 = sm.OLS(df["y"], sm.add_constant(df[["x", "z"]])).fit()
        with pytest.raises(ValueError, match="same coefficient"):
            ps.pool([f1, f2])

"""Edge-case tests for modifier chains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
from pysofra.core.table import SofraTable


# ----------------------------------------------------------------------
# extras.py — "needs source data" error raises
# ----------------------------------------------------------------------
class TestExtrasErrorRaises:
    def _plain_table(self) -> SofraTable:
        return SofraTable(
            rows=(Row(cells=(Cell(text="x"),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="V"),)),),
        )

    def test_add_difference_without_spec_raises(self):
        t = self._plain_table()
        with pytest.raises(ValueError, match="tbl_one"):
            t.add_difference()

    def test_add_ci_without_spec_raises(self):
        t = self._plain_table()
        with pytest.raises(ValueError, match="source data"):
            t.add_ci()


# ----------------------------------------------------------------------
# tbl_one helpers — direct calls
# ----------------------------------------------------------------------
class TestTblOneHelpers:
    def test_fmt_level_bool(self):
        from pysofra.summary.tbl_one import _fmt_level
        assert _fmt_level(True) == "Yes"
        assert _fmt_level(False) == "No"

    def test_fmt_weighted_n_pct_zero_total(self):
        from pysofra.summary.tbl_one import _fmt_weighted_n_pct
        # total <= 0 → "n (—)"
        out = _fmt_weighted_n_pct(2.0, 0.0, pct_digits=1)
        assert "—" in out

    def test_fmt_weighted_n_pct_fractional(self):
        from pysofra.summary.tbl_one import _fmt_weighted_n_pct
        out = _fmt_weighted_n_pct(2.5, 10.0, pct_digits=1)
        assert "2.5" in out  # fractional n_str path

    def test_sort_key_bool(self):
        from pysofra.summary.tbl_one import _sort_key
        assert _sort_key(True) == (0, 1)

    def test_sort_key_numeric(self):
        from pysofra.summary.tbl_one import _sort_key
        assert _sort_key(2.5) == (0, 2.5)


# ----------------------------------------------------------------------
# tbl_one continuous "—" return branches
# ----------------------------------------------------------------------
class TestTblOneContinuousEmpty:
    def test_empty_continuous_var_renders_dash(self):
        df = pd.DataFrame({
            "arm": ["A", "B"] * 5,
            "x": [np.nan] * 10,
        })
        t = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        # Every cell except the label should be "—"
        body_cells = [c.text for r in t.rows for c in r.cells[1:]]
        assert any(c == "—" for c in body_cells)

    def test_continuous_nonnormal_with_data(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=40),
            "x": rng.exponential(size=40),
        })
        t = ps.tbl_one(df, by="arm", variables=["x"], nonnormal=["x"],
                       missing="never")
        # Median (Q1, Q3) format — parentheses appear in each group cell
        assert any("(" in c.text and "," in c.text
                   for r in t.rows for c in r.cells[1:])


# ----------------------------------------------------------------------
# tbl_one — multi-group continuous with weights (no svy strata)
# ----------------------------------------------------------------------
class TestTblOneMultiGroupContinuous:
    def test_three_arm_continuous_svy_fallback(self):
        # When design is provided but groups > 2, svyttest is not applicable
        # → falls back to plain continuous_test
        rng = np.random.default_rng(0)
        n = 90
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B", "C"], size=n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 1.5, size=n),
            "strata": rng.choice(["s1", "s2"], size=n),
        })
        d = ps.SurveyDesign(weights="w", strata="strata")
        t = ps.tbl_one(df, by="arm", variables=["x"], design=d,
                       missing="never")
        assert len(t.rows) >= 1


# ----------------------------------------------------------------------
# tbl_one with weights + missing row
# ----------------------------------------------------------------------
class TestTblOneWeightedMissingRow:
    def test_weighted_missing_row(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=20),
            "x": [1.0, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 2,
            "w": rng.uniform(0.5, 1.5, size=20),
        })
        t = ps.tbl_one(df, by="arm", variables=["x"], weights="w",
                       missing="always")
        labels = [r.cells[0].text for r in t.rows]
        assert any("Missing" in lab for lab in labels)


# ----------------------------------------------------------------------
# tbl_cross helpers
# ----------------------------------------------------------------------
class TestTblCrossHelpers:
    def test_sort_key_bool(self):
        from pysofra.summary.tbl_cross import _sort_key
        assert _sort_key(True) == (0, 1)

    def test_sort_key_numeric(self):
        from pysofra.summary.tbl_cross import _sort_key
        assert _sort_key(1.5) == (0, 1.5)

    def test_sort_key_other(self):
        from pysofra.summary.tbl_cross import _sort_key
        out = _sort_key((1, 2))
        assert out[0] == 2

    def test_pct_nan(self):
        from pysofra.summary.tbl_cross import _pct
        # NaN → "—"
        out = _pct(float("nan"), digits=1)
        assert out == "—"

    def test_unknown_cell_style_raises(self):
        df = pd.DataFrame({"a": ["x", "y"] * 5, "b": ["A", "B"] * 5})
        with pytest.raises(ValueError, match="cell must be"):
            ps.tbl_cross(df, row="a", column="b", cell="bogus_style")

    def test_categorical_column_dtype(self):
        # Triggers the ``df[column].cat.categories`` branch.
        df = pd.DataFrame({
            "a": ["x", "y", "x", "y"],
            "b": pd.Categorical(["A", "B", "A", "B"], categories=["A", "B"]),
        })
        t = ps.tbl_cross(df, row="a", column="b")
        assert len(t.rows) >= 2


# ----------------------------------------------------------------------
# weights — helpers
# ----------------------------------------------------------------------
class TestWeightsHelpers:
    def test_safe_sort_key_bool(self):
        from pysofra.summary.weights import _safe_sort_key
        assert _safe_sort_key(True) == (0, 1)

    def test_safe_sort_key_numeric(self):
        from pysofra.summary.weights import _safe_sort_key
        assert _safe_sort_key(1.5) == (0, 1.5)

    def test_safe_sort_key_other(self):
        from pysofra.summary.weights import _safe_sort_key
        assert _safe_sort_key((1, 2))[0] == 2

    def test_weighted_quantile_empty(self):
        from pysofra.summary.weights import _weighted_quantile
        # Empty inputs → returns NaN
        out = _weighted_quantile(np.array([]), np.array([]), 0.5)
        assert np.isnan(out)


# ----------------------------------------------------------------------
# typing.py — int-overflow / TypeError fallbacks
# ----------------------------------------------------------------------
class TestTypingFallbacks:
    def test_int_coerce_failure_falls_through(self):
        # Float series with NaN — int(nan) raises ValueError, exercising
        # the (TypeError, ValueError) except branch in the 0/1 fast-path.
        from pysofra.summary.typing import infer_kind
        s = pd.Series([0.0, 1.0, 2.5, 3.0, np.nan])
        kind = infer_kind(s)
        assert kind in ("continuous", "categorical", "dichotomous")

    def test_object_int_dtype(self):
        # Object dtype that holds Python ints → np.issubdtype path may
        # take the TypeError branch on some dtypes.
        from pysofra.summary.typing import infer_kind
        s = pd.Series([1, 2, 3, 4, 5, 6], dtype=object)
        out = infer_kind(s)
        assert out in ("continuous", "categorical", "dichotomous")


# ----------------------------------------------------------------------
# smd.py — singular covariance & negative numerical value
# ----------------------------------------------------------------------
class TestSmdSingular:
    def test_categorical_smd_pair_singular(self):
        # Both groups have identical zero-only proportions → covariance
        # matrix is all-zero / singular. The np.linalg.pinv branch isn't
        # the LinAlgError path (pinv always succeeds), but val < 0 from
        # numerical noise can trip the clamp.
        from pysofra.summary.smd import categorical_smd_pair
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        out = categorical_smd_pair(p1, p2)
        assert out is not None
        assert out >= 0  # the clamp prevents negative numerical noise


# ----------------------------------------------------------------------
# effect_size.py — degenerate paths
# ----------------------------------------------------------------------
class TestEffectSizeDegenerate2:
    def test_cramers_v_all_zero_total(self):
        from pysofra.summary.effect_size import cramers_v
        # An empty series → returns None
        out = cramers_v(pd.Series([], dtype=object),
                        pd.Series([], dtype=object))
        assert out is None

    def test_cramers_v_min_dim_zero(self):
        from pysofra.summary.effect_size import cramers_v
        # All same row + same group → cramer's V returns None before computation
        out = cramers_v(pd.Series(["x"] * 5),
                        pd.Series(["A"] * 5))
        assert out is None

    def test_phi_empty(self):
        from pysofra.summary.effect_size import phi_coefficient
        out = phi_coefficient(pd.Series([], dtype=object),
                              pd.Series([], dtype=object))
        assert out is None


# ----------------------------------------------------------------------
# uvregression — failed predictor / "no rows" cases
# ----------------------------------------------------------------------
class TestUvregressionFailures:
    def test_predictor_fit_failure_is_caught(self):
        # A predictor that causes fit() to raise (singular X). The
        # exception MUST be caught internally so the table can still
        # be built from the survivors; nothing escapes to the caller.
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({
            "y": [1.0, 2, 3, 4, 5],
            "constant_col": [1.0] * 5,  # zero variance — perfect collinear with intercept
            "good": [1.0, 2, 3, 4, 5],
        })
        t = ps.tbl_uvregression(df, outcome="y",
                                predictors=["good", "constant_col"])
        # 'good' fits; 'constant_col' may or may not fit (depends on
        # statsmodels' tolerance), but no exception escapes.
        assert len(t.rows) >= 1

    def test_no_predictors_succeed_returns_failed_only(self):
        # Every predictor fails — falls through to no-rows footnote
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({
            "y": [1.0] * 10,
            "bad": [np.nan] * 10,
        })
        t = ps.tbl_uvregression(df, outcome="y", predictors=["bad"])
        # Should produce a footnote listing the failure
        assert any("failed" in fn or "bad" in fn for fn in t.footnotes)


# ----------------------------------------------------------------------
# extract.py — GEE / unsupported family-label paths
# ----------------------------------------------------------------------
class TestExtractFamilyLabel:
    def test_gee_family_label(self):
        # Build a fake statsmodels GEE-ish result with cov_struct
        from pysofra.models.extract import _statsmodels_family_label

        class CovStruct:
            pass

        class FakeInnerGEE:
            cov_struct = CovStruct()

        class FakeWrapper:
            model = FakeInnerGEE()

        results = FakeWrapper()
        # Inner model class must contain "GEE" in name
        FakeInnerGEE.__name__ = "GEE"
        out = _statsmodels_family_label(results)
        assert "GEE" in out

    def test_gee_family_label_no_cov_struct(self):
        # GEE inner without cov_struct attribute
        from pysofra.models.extract import _statsmodels_family_label

        class FakeInnerGEE:
            pass

        FakeInnerGEE.__name__ = "GEE"

        class FakeWrapper:
            model = FakeInnerGEE()

        out = _statsmodels_family_label(FakeWrapper())
        assert out.endswith("(GEE)")

    def test_label_bare_class_no_family_no_inner(self):
        from pysofra.models.extract import _statsmodels_family_label

        class Bare:
            pass

        out = _statsmodels_family_label(Bare())
        assert out == "Bare"


# ----------------------------------------------------------------------
# survival — fixed-time row with kmf is None
# ----------------------------------------------------------------------
class TestSurvivalNonePath:
    def test_fixed_time_with_empty_group(self):
        pytest.importorskip("lifelines")
        # Group "B" has all NaN — its kmf is None → "—" cells emitted.
        df = pd.DataFrame({
            "arm": ["A"] * 10 + ["B"] * 5,
            "time": list(range(1, 11)) + [np.nan] * 5,
            "event": [1, 0] * 5 + [np.nan] * 5,
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm",
                            times=[5.0, 10.0], show_logrank=False)
        # The 'B' column at time 5 / 10 should be "—"
        body = [c.text for r in t.rows for c in r.cells]
        assert "—" in body


# ----------------------------------------------------------------------
# core.table — error raises
# ----------------------------------------------------------------------
class TestTableErrorRaises:
    def test_with_forest_plot_bad_position(self):
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        # Build a regression-style table so forest_plot doesn't fail upstream.
        t = SofraTable(
            rows=(Row(cells=(
                Cell(text="x"),
                Cell(text="1.0", value=1.0, kind="numeric"),
                Cell(text="0.5, 1.5", value=(0.5, 1.5), kind="ci"),
                Cell(text="0.05", value=0.05, kind="p_value"),
            )),),
            headers=(HeaderRow(cells=(
                HeaderCell(text="V"), HeaderCell(text="β"),
                HeaderCell(text="CI"), HeaderCell(text="p"),
            )),),
        )
        with pytest.raises(ValueError, match="above"):
            t.with_forest_plot(position="middle")


# ----------------------------------------------------------------------
# Renderers — small final remnants
# ----------------------------------------------------------------------
class TestRendererSmallRemnants:
    def test_html_ncols_rows_only(self):
        from pysofra.render.html import _ncols
        t = SofraTable(rows=(Row(cells=(Cell(text="x"), Cell(text="y"))),))
        assert _ncols(t) == 2

    def test_html_parts_empty_falls_back_to_escape(self):
        from pysofra.render.html import _render_parts
        # When c.parts is None → returns the escaped text
        c = Cell(text="plain & text")
        out = _render_parts(c)
        assert "&amp;" in out

    def test_pptx_ncols_rows_only(self):
        from pysofra.render.pptx import _ncols
        t = SofraTable(rows=(Row(cells=(Cell(text="x"),)),))
        assert _ncols(t) == 1

    def test_image_renderer_with_caption(self, tmp_path):
        from pysofra.render.image import write_image
        t = SofraTable(
            headers=(HeaderRow(cells=(HeaderCell(text="A"),)),),
            rows=(Row(cells=(Cell(text="x"),)),),
            caption="My table",
        )
        out = tmp_path / "cap.png"
        write_image(t, out)
        assert out.exists()

    def test_latex_header_no_bold(self):
        from pysofra.render.latex import LatexRenderer
        t = SofraTable(
            headers=(HeaderRow(cells=(HeaderCell(text="A", bold=False),)),),
            rows=(Row(cells=(Cell(text="x"),)),),
        )
        out = LatexRenderer().render(t)
        assert "A" in out and r"\textbf{A}" not in out

    def test_docx_existing_border_replaced(self, tmp_path):
        # Two header rows so the existing-border replacement path is hit.
        pytest.importorskip("docx")
        from pysofra.render.docx import DocxRenderer
        t = SofraTable(
            headers=(
                HeaderRow(cells=(HeaderCell(text="A"), HeaderCell(text="B"))),
                HeaderRow(cells=(HeaderCell(text="C"), HeaderCell(text="D"))),
            ),
            rows=(Row(cells=(Cell(text="1"), Cell(text="2"))),),
        )
        out = tmp_path / "borders.docx"
        DocxRenderer().write(t, str(out))
        assert out.exists()


# ----------------------------------------------------------------------
# design.py — final remnants
# ----------------------------------------------------------------------
class TestDesignFinal:
    def test_design_mean_var_stratum_single_psu(self):
        # Stratified with single cluster per stratum → contrib=0 branch
        from pysofra.summary.design import design_mean_var
        v = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = pd.Series([1.0] * 4)
        strata = pd.Series(["s1", "s1", "s2", "s2"])
        cluster = pd.Series(["c1", "c1", "c2", "c2"])
        m, var, _n = design_mean_var(v, w, strata=strata, cluster=cluster)
        assert not np.isnan(m)

    def test_replicate_mean_var_zero_weights(self):
        # base weights all zero → returns nan
        from pysofra.summary.design import replicate_mean_var
        v = pd.Series([1.0, 2, 3])
        bw = pd.Series([0.0, 0.0, 0.0])
        rep = [pd.Series([0.0, 0.0, 0.0])]
        m, var, n = replicate_mean_var(v, bw, rep, replicate_type="jk1")
        assert np.isnan(m)

    def test_replicate_mean_var_no_replicates(self):
        # Empty replicate list → returns (theta_hat, 0.0, total_w)
        from pysofra.summary.design import replicate_mean_var
        v = pd.Series([1.0, 2, 3])
        bw = pd.Series([1.0, 1.0, 1.0])
        m, var, _n = replicate_mean_var(v, bw, [], replicate_type="jk1")
        assert var == 0.0

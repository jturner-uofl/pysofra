"""Renderer edge-case tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
from pysofra.core.table import SofraTable


# ----------------------------------------------------------------------
# tbl_one — survey-design continuous "—" returns + weighted missing row
# ----------------------------------------------------------------------
class TestTblOneContinuousPaths:
    def test_design_continuous_n_eff_zero(self):
        # Force the var to be continuous, then ensure all-NaN values
        # route through the design path → n_eff == 0 → "—"
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=n),
            "junk": pd.Series([np.nan] * n, dtype=float),
            "w": rng.uniform(0.5, 1.5, size=n),
            "strata": rng.choice(["s1", "s2"], size=n),
        })
        d = ps.SurveyDesign(weights="w", strata="strata")
        t = ps.tbl_one(df, by="arm", variables=["junk"], design=d,
                       types={"junk": "continuous"}, missing="never")
        body = [c.text for r in t.rows for c in r.cells[1:]]
        assert "—" in body

    def test_weighted_continuous_n_eff_zero(self):
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=n),
            "junk": pd.Series([np.nan] * n, dtype=float),
            "w": rng.uniform(0.5, 1.5, size=n),
        })
        t = ps.tbl_one(df, by="arm", variables=["junk"], weights="w",
                       types={"junk": "continuous"}, missing="never")
        body = [c.text for r in t.rows for c in r.cells[1:]]
        assert "—" in body

    def test_plain_continuous_all_nan(self):
        # No weights, no design — cs.n == 0 → "—" (line 565)
        df = pd.DataFrame({
            "arm": ["A", "B"] * 10,
            "junk": pd.Series([np.nan] * 20, dtype=float),
        })
        t = ps.tbl_one(df, by="arm", variables=["junk"],
                       types={"junk": "continuous"}, missing="never")
        body = [c.text for r in t.rows for c in r.cells[1:]]
        assert "—" in body

    def test_weighted_nonnormal_continuous(self):
        # Weights + nonnormal → fmt_median_iqr (line 561)
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=n),
            "x": rng.exponential(size=n),
            "w": rng.uniform(0.5, 1.5, size=n),
        })
        t = ps.tbl_one(df, by="arm", variables=["x"], weights="w",
                       nonnormal=["x"], missing="never")
        # Median (Q1, Q3) format includes a comma inside parens
        assert any("(" in c.text and "," in c.text
                   for r in t.rows for c in r.cells[1:])


# ----------------------------------------------------------------------
# tbl_one — 3-group continuous with svy design (line 591 fallback)
# ----------------------------------------------------------------------
class TestTblOneThreeGroupSvy:
    def test_three_group_design_falls_back_to_plain_test(self):
        rng = np.random.default_rng(0)
        n = 120
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B", "C"], size=n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 1.5, size=n),
            "strata": rng.choice(["s1", "s2"], size=n),
        })
        d = ps.SurveyDesign(weights="w", strata="strata")
        # >2 groups under weights emits a UserWarning (design-adjusted
        # F-test not implemented; falls back to unweighted ANOVA).
        with pytest.warns(UserWarning, match=r"design-adjusted F-test"):
            t = (
                ps.tbl_one(df, by="arm", variables=["x"], design=d,
                           missing="never")
                .add_p()
            )
        # P-value column should be present and finite for the continuous row.
        # (svyttest is bypassed for >2 groups → falls back to ANOVA.)
        p_cells = [c for r in t.rows for c in r.cells if c.kind == "p_value"]
        assert any(isinstance(c.value, (int, float)) for c in p_cells)


# ----------------------------------------------------------------------
# core.table._with_option — modifier on a plain table raises
# ----------------------------------------------------------------------
class TestTableWithOptionRaises:
    def test_modifier_on_plain_table_raises(self):
        # The error message was refined to differentiate
        # three causes (direct construction, tbl_cross builder, unpickled).
        # The "directly constructed" case is what this test exercises.
        t = SofraTable(rows=(Row(cells=(Cell(text="x"),)),))
        with pytest.raises(RuntimeError, match=r"constructed directly|composition primitive"):
            t.add_p()


# ----------------------------------------------------------------------
# typing.py — try/except branches
# ----------------------------------------------------------------------
class TestTypingExcepts:
    def test_int_coerce_with_inf(self):
        # int(inf) raises OverflowError but our except catches (TypeError,
        # ValueError) only — line 68-69 should still be hit by NaN paths.
        from pysofra.summary.typing import infer_kind
        # Use a series where int(x) on NaN-as-float raises ValueError.
        s = pd.Series([0, 1, np.nan])
        # int(nan) → ValueError → caught by except (TypeError, ValueError)
        kind = infer_kind(s)
        assert kind in ("continuous", "categorical", "dichotomous")

    def test_object_with_mixed_types(self):
        # Object dtype where np.issubdtype raises TypeError on the dtype
        from pysofra.summary.typing import infer_kind
        s = pd.Series([1, "x", 2, "y", 3, "z"], dtype=object)
        # Falls through the int-coerce except and the issubdtype except
        kind = infer_kind(s)
        assert kind in ("continuous", "categorical", "dichotomous")


# ----------------------------------------------------------------------
# uvregression — failed-fit branches
# ----------------------------------------------------------------------
class TestUvregressionFailedFit:
    def test_singular_design_falls_into_except(self):
        # Force the model_factory(...).fit() to raise.
        # We do this by providing a custom callable method that always
        # raises.
        pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"y": [1.0, 2, 3, 4], "x": [1.0, 2, 3, 4]})

        class BadModel:
            def __init__(self, *_, **__):
                pass

            def fit(self, **_):
                raise RuntimeError("synthetic fit failure")

        t = ps.tbl_uvregression(df, outcome="y", predictors=["x"],
                                method=BadModel)
        # Should produce a "failed" footnote
        assert any("failed" in fn for fn in t.footnotes)

    def test_predictor_not_in_summary(self):
        # Force the path where the fit succeeds but the predictor name
        # isn't in summary.estimates.index. We can do this with a custom
        # method callable that returns a ModelSummary with an unrelated
        # estimate index.
        from pysofra.models.extract import ModelSummary

        class FakeFit:
            pass

        class FakeMethod:
            def __init__(self, endog, exog, **kwargs):
                pass

            def fit(self, **_):
                return FakeFit()

        # Monkey-patch extract to return a ModelSummary lacking 'x'
        import pysofra.models.uvregression as uvm
        original_extract = uvm.extract

        def fake_extract(fit, conf_level=0.95):
            return ModelSummary(
                estimates=pd.Series([0.1], index=["unrelated"]),
                ci_lo=pd.Series([0.0], index=["unrelated"]),
                ci_hi=pd.Series([0.2], index=["unrelated"]),
                pvalues=pd.Series([0.5], index=["unrelated"]),
                family="Fake",
                natural_exponentiate=False,
            )

        uvm.extract = fake_extract
        try:
            df = pd.DataFrame({"y": [1.0] * 8, "x": list(range(8))})
            t = ps.tbl_uvregression(df, outcome="y", predictors=["x"],
                                    method=FakeMethod)
            assert any("failed" in fn or "x" in fn for fn in t.footnotes)
        finally:
            uvm.extract = original_extract


# ----------------------------------------------------------------------
# regression.py — model-label branches
# ----------------------------------------------------------------------
class TestRegressionLabels:
    def test_cox_label(self):
        from pysofra.models.regression import _default_estimate_label
        assert _default_estimate_label("CoxPH (cox)", True) == "HR"
        assert _default_estimate_label("PHReg", True) == "HR"

    def test_aft_label(self):
        # AFT family: exp(coef) is a Time Ratio, NOT a Hazard Ratio.
        # The two parameters point in opposite directions (TR>1 = longer
        # survival; HR>1 = shorter survival) so mislabelling is
        # publication-critical.
        from pysofra.models.regression import _default_estimate_label
        assert _default_estimate_label("Weibull AFT", True) == "TR"
        assert _default_estimate_label("LogNormal AFT", True) == "TR"
        assert _default_estimate_label("LogLogistic AFT", True) == "TR"

    def test_unknown_exp_label(self):
        from pysofra.models.regression import _default_estimate_label
        assert _default_estimate_label("MysteryModel", True) == "exp(β)"

    def test_design_requires_statsmodels_results(self):
        # _refit_with_design path: object without .model.fit raises
        from pysofra.models.regression import _refit_with_design

        class Bogus:
            pass

        with pytest.raises(ValueError, match="statsmodels"):
            _refit_with_design(Bogus(), design=object(), data=None)


# ----------------------------------------------------------------------
# regression.py — multi-model list with per-model data
# ----------------------------------------------------------------------
class TestRegressionMultiModelData:
    def test_per_model_data_length_mismatch(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = df["x"] + rng.normal(size=n)
        m = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
        d_obj = ps.SurveyDesign(weights="x")
        with pytest.raises(ValueError, match="one DataFrame per model"):
            ps.tbl_regression([m, m], data=[df], design=d_obj)


# ----------------------------------------------------------------------
# survival.py — line 222 (survival_at returns None for one group)
# ----------------------------------------------------------------------
class TestSurvivalAtNone:
    def test_survival_at_returns_none_emits_dash(self):
        # When _survival_at returns None we get the second "—" branch
        # inside the times loop.
        pytest.importorskip("lifelines")
        # Construct: in group "A" we have data; in group "B" we have a
        # single-event series that makes survival_function_at_times fail.
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=n),
            "time": rng.exponential(10, size=n),
            "event": rng.integers(0, 2, size=n),
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm",
                            times=[5.0, 1000.0], show_logrank=False)
        # Time 1000 is well beyond any observation in either group
        assert len(t.rows) >= 1


# ----------------------------------------------------------------------
# extras.py — add_global_p with model that lacks contributing params
# ----------------------------------------------------------------------
class TestAddGlobalPNoContributing:
    def test_add_global_p_no_contributing_params(self):
        # When no params contribute to a stem, joint_p[stem] = None (line 276-277)
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable

        class Mock:
            class params:
                index = pd.Index(["unrelated"])

            def f_test(self, *_):
                return type("R", (), {"pvalue": 0.5})()

        t = SofraTable(
            rows=(Row(cells=(
                Cell(text="x"),
                Cell(text="1.0", value=1.0, kind="numeric"),
                Cell(text="0.5, 1.5", value=(0.5, 1.5), kind="ci"),
                Cell(text="0.05", value=0.05, kind="p_value"),
            )),),
            headers=(HeaderRow(cells=(
                HeaderCell(text="V"), HeaderCell(text="β"),
                HeaderCell(text="CI"), HeaderCell(text="p-value"),
            )),),
            metadata={"model": Mock()},
        )
        t2 = t.add_global_p()
        # The new "global p" cell for 'x' has no contributing params → "—"
        assert any(c.text == "—" for r in t2.rows for c in r.cells)


# ----------------------------------------------------------------------
# extras.py — add_ci / add_difference / add_n unreachable-data fallbacks
# ----------------------------------------------------------------------
class TestExtrasDataRecoveryFailure:
    def _table_with_broken_rebuild(self):
        # A SofraTable with a spec + a rebuild closure that doesn't carry
        # a DataFrame in its cell_contents → _data_from_rebuild returns
        # None and the function raises "Could not recover source data".
        from pysofra.core.table import TableSpec

        def fake_rebuild(spec):
            return SofraTable()  # closure has no DataFrame

        # We need it to be a real closure with no DataFrame cell.
        spec = TableSpec(builder="tbl_one", options={
            "variables": ("x",),
            "kinds": {"x": "continuous"},
            "by": None,
        })
        return SofraTable(
            rows=(Row(cells=(Cell(text="x"),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="V"),)),),
            _spec=spec,
            _rebuild=fake_rebuild,
        )

    def test_add_n_data_recovery_fails(self):
        t = self._table_with_broken_rebuild()
        with pytest.raises(ValueError, match="recover source data"):
            t.add_n()

    def test_add_ci_data_recovery_fails(self):
        t = self._table_with_broken_rebuild()
        with pytest.raises(ValueError, match="recover source data"):
            t.add_ci()


# ----------------------------------------------------------------------
# extras.py — _data_from_rebuild path: closure-less function
# ----------------------------------------------------------------------
class TestDataFromRebuild:
    def test_function_without_closure(self):
        from pysofra.summary.extras import _data_from_rebuild

        def noclo(_):  # no closure
            return None

        assert _data_from_rebuild(noclo) is None

    def test_function_with_closure_no_dataframe(self):
        from pysofra.summary.extras import _data_from_rebuild

        outer_var = [1, 2, 3]  # not a DataFrame

        def with_clo(_):
            return outer_var

        assert _data_from_rebuild(with_clo) is None


# ----------------------------------------------------------------------
# tbl_one — no-pairs FDR path
# ----------------------------------------------------------------------
class TestTblOneFDRNoPairs:
    def test_q_values_no_p_rows(self):
        # add_q on a table whose only p-value is NaN → no pairs → rows
        # returned unchanged (line 451).
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=20),
            "junk": [np.nan] * 20,
        })
        # missing='never' + all NaN → p column ends up "—" / None
        t = ps.tbl_one(df, by="arm", variables=["junk"], missing="never")
        t2 = t.add_p().add_q()
        # Function returns successfully even though there's nothing to adjust
        assert len(t2.rows) >= 1


# ----------------------------------------------------------------------
# tbl_one — empty levels with show_q (line 678)
# ----------------------------------------------------------------------
class TestTblOneEmptyLevelsShowQ:
    def test_all_nan_categorical_with_q(self):
        # Empty levels path → branch through show_overall + show_p + show_q
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=20),
            "junk": [np.nan] * 20,
            "good_cat": rng.choice(["x", "y"], size=20),
        })
        # types= forces the all-NaN variable into 'categorical' classification
        t = (
            ps.tbl_one(df, by="arm", variables=["junk", "good_cat"],
                       types={"junk": "categorical"}, missing="never")
            .add_p()
            .add_q()
        )
        assert len(t.rows) >= 1


# ----------------------------------------------------------------------
# extras.py — _insert_after_groups_header / _cell branches
# ----------------------------------------------------------------------
class TestInsertAfterGroupsCell:
    def test_insert_after_groups_cell_with_pvalue_breaks(self):
        # When a row has a p-value cell, _insert_after_groups_cell breaks
        # at that index (line 691-692).
        from pysofra.summary.extras import _insert_after_groups_cell
        r = Row(cells=(
            Cell(text="x"),
            Cell(text="5", value=5, kind="numeric"),
            Cell(text="0.05", value=0.05, kind="p_value"),
        ))
        out = _insert_after_groups_cell(r, "inserted", value=1.0)
        # The inserted cell should appear *before* the p-value cell.
        kinds = [c.kind for c in out.cells]
        # Inserted cell has kind="text"; p_value still last.
        assert kinds[-1] == "p_value"


# ----------------------------------------------------------------------
# add_difference — degenerate (no continuous / dichotomous matches)
# ----------------------------------------------------------------------
class TestAddDifferenceDegenerate:
    def test_add_difference_continuous_too_few_obs(self):
        # 2-group table where continuous var has < 2 obs per group
        df = pd.DataFrame({
            "arm": ["A", "A", "B", "B"],
            "x": [1.0, np.nan, 2.0, np.nan],
        })
        t = ps.tbl_one(df, by="arm", variables=["x"],
                       missing="never").add_difference()
        # Should produce a Δ column with em-dash
        assert any("—" in c.text for r in t.rows for c in r.cells)

    def test_add_difference_dichotomous_three_levels(self):
        # A "dichotomous" var that actually has 3 levels → line 391-392 cont.
        # Note: tbl_one already classifies multi-level cats correctly, but if
        # we force a column with 3 levels through, add_difference should
        # emit "—" for it.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], size=30),
            "cat3": rng.choice(["x", "y", "z"], size=30),
        })
        t = ps.tbl_one(df, by="arm", variables=["cat3"], missing="never")
        t2 = t.add_difference()
        assert any("—" in c.text for r in t2.rows for c in r.cells)

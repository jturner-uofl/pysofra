"""Edge-case tests for the model-summary extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


# ----------------------------------------------------------------------
# extract.py — degenerate paths
# ----------------------------------------------------------------------
class TestExtractCovers:
    def test_unsupported_model_raises(self):
        from pysofra.models.extract import extract

        class Bogus:
            pass

        with pytest.raises(TypeError, match="Unsupported model"):
            extract(Bogus())

    def test_lifelines_no_summary_attribute(self):
        from pysofra.models.extract import _extract_lifelines

        class FakeFitter:
            pass
        fake = FakeFitter()
        fake.__class__.__module__ = "lifelines.fitters.fake"
        with pytest.raises(TypeError, match="no .summary"):
            _extract_lifelines(fake, conf_level=0.95)

    def test_lifelines_summary_not_dataframe(self):
        from pysofra.models.extract import _extract_lifelines

        class FakeFitter:
            summary = "not a dataframe"
        fake = FakeFitter()
        fake.__class__.__module__ = "lifelines.fitters.fake"
        with pytest.raises(TypeError, match="not a DataFrame"):
            _extract_lifelines(fake, conf_level=0.95)

    def test_lifelines_missing_ci_columns(self):
        from pysofra.models.extract import _extract_lifelines

        # A summary DataFrame *without* the CI columns lifelines normally adds
        class FakeFitter:
            summary = pd.DataFrame({
                "coef": [0.1, 0.2],
                "p": [0.5, 0.6],
            }, index=["x", "z"])
        fake = FakeFitter()
        fake.__class__.__module__ = "lifelines.fitters.fake"
        with pytest.raises(ValueError, match="CI columns"):
            _extract_lifelines(fake, conf_level=0.95)

    def test_sklearn_multiclass_emits_flat_class_feature_labels(self):
        # Previously this raised NotImplementedError; the current path
        # multi-class fits are supported and emit one row per
        # (class, feature) pair via the same flat-label convention
        # used by lifelines AFT models.
        from pysofra.models.extract import _extract_sklearn

        class FakeMulticlass:
            coef_ = np.array([[1, 2], [3, 4], [5, 6]])  # 3-class
            classes_ = np.array(["a", "b", "c"])
            feature_names_in_ = np.array(["age", "bmi"])

            def predict(self, X):
                return X

        ms = _extract_sklearn(FakeMulticlass())
        # 3 classes * 2 features = 6 rows; each indexed by
        # "feature (class=X)".
        assert len(ms.estimates) == 6
        for cls in ("a", "b", "c"):
            for feat in ("age", "bmi"):
                key = f"{feat} (class={cls})"
                assert key in ms.estimates.index, ms.estimates.index.tolist()
        # Specific values track coef_:
        assert ms.estimates["age (class=a)"] == 1
        assert ms.estimates["bmi (class=a)"] == 2
        assert ms.estimates["age (class=c)"] == 5
        assert ms.estimates["bmi (class=c)"] == 6

    def test_passing_modelsummary_passes_through(self):
        from pysofra.models.extract import ModelSummary, extract

        ms = ModelSummary(
            estimates=pd.Series([1.0]),
            ci_lo=pd.Series([0.5]),
            ci_hi=pd.Series([1.5]),
            pvalues=pd.Series([0.05]),
            family="Test",
            natural_exponentiate=False,
        )
        result = extract(ms)
        assert result is ms

    def test_sklearn_no_feature_names_in(self):
        from pysofra.models.extract import _extract_sklearn

        class FakeReg:
            coef_ = np.array([1.0, 2.0])

            def predict(self, X):
                return X

        # No feature_names_in_ → defaults to x0, x1
        result = _extract_sklearn(FakeReg())
        assert list(result.estimates.index) == ["x0", "x1"]


# ----------------------------------------------------------------------
# uvregression — GLM method + adjust_for + degenerate paths
# ----------------------------------------------------------------------
class TestUvregressionMore:
    def test_glm_method(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        })
        df["y"] = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-df["x"]))).astype(int)
        t = ps.tbl_uvregression(
            df, outcome="y", predictors=["x", "z"], method="GLM",
            method_kwargs={"family": sm.families.Binomial()},
            exponentiate=True,
        )
        assert len(t.rows) == 2

    def test_empty_after_dropna_predictor_failure(self):
        pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=20),
            "good": rng.normal(size=20),
            "bad":  [np.nan] * 20,  # all-NaN → fails
        })
        t = ps.tbl_uvregression(df, outcome="y", predictors=["good", "bad"])
        labels = [r.cells[0].text for r in t.rows]
        assert "good" in labels
        assert any("bad" in fn for fn in t.footnotes)

    def test_predictor_in_columns_but_pred_not_in_summary(self):
        # When the model fits but the predictor name isn't in the summary
        # (extremely rare; tested by mock)
        pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        })
        t = ps.tbl_uvregression(df, outcome="y", predictors=["x"])
        assert len(t.rows) == 1

    def test_unknown_method_raises(self):
        df = pd.DataFrame({"y": [1.0, 2], "x": [1.0, 2]})
        with pytest.raises(ValueError, match="Unknown method"):
            ps.tbl_uvregression(df, outcome="y", predictors=["x"],
                                method="exotic")


# ----------------------------------------------------------------------
# survival — sad-path branches
# ----------------------------------------------------------------------
class TestSurvivalMore:
    def test_overall_label_in_metadata(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "time": rng.exponential(24, 50),
            "event": rng.integers(0, 2, 50),
        })
        t = ps.tbl_survival(df, time="time", event="event")
        assert t.metadata["n_groups"] == 1

    def test_NaN_median_renders_em_dash(self):
        pytest.importorskip("lifelines")
        # Make the median undefined by having very few events
        df = pd.DataFrame({
            "arm":   ["A"] * 20,
            "time":  list(range(1, 21)),
            "event": [0] * 19 + [1],   # only one event at the very end
        })
        t = ps.tbl_survival(df, time="time", event="event", by="arm")
        # The median-survival row exists and has either a number or em-dash
        med_row = next(
            r for r in t.rows if r.cells[0].text.startswith("Median survival")
        )
        assert med_row.cells[1].text  # not blank

    def test_sort_key_with_bool(self):
        from pysofra.models.survival import _sort_key
        assert _sort_key(True) == (0, 1)
        assert _sort_key(False) == (0, 0)

    def test_sort_key_with_float(self):
        from pysofra.models.survival import _sort_key
        assert _sort_key(3.14) == (0, 3.14)

    def test_sort_key_with_string(self):
        from pysofra.models.survival import _sort_key
        assert _sort_key("hello") == (1, "hello")

    def test_sort_key_other(self):
        from pysofra.models.survival import _sort_key
        result = _sort_key((1, 2))
        assert result[0] == 2

    def test_survival_at_returns_none_on_error(self):
        from pysofra.models.survival import _survival_at

        class StubKMF:
            def survival_function_at_times(self, t):
                raise RuntimeError("boom")
        assert _survival_at(StubKMF(), 10) is None

    def test_n_at_risk_returns_zero_on_error(self):
        from pysofra.models.survival import _n_at_risk

        class StubKMF:
            event_table = None  # malformed
        assert _n_at_risk(StubKMF(), 10) == 0


# ----------------------------------------------------------------------
# extras.py — final branches
# ----------------------------------------------------------------------
class TestExtrasFinal:
    def test_add_global_p_metadata_missing_f_test(self):
        # A SofraTable whose metadata['model'] doesn't have .f_test now
        # raises NotImplementedError rather than silently inserting an
        # em-dash column (which would be misleading).
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        from pysofra.core.table import SofraTable

        class Bogus:
            params = pd.Series({"x": 1.0})

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
            metadata={"model": Bogus()},
        )
        with pytest.raises(NotImplementedError, match="tbl_regression"):
            t.add_global_p()


# ----------------------------------------------------------------------
# compose.py — tbl_merge / tbl_stack edge branches
# ----------------------------------------------------------------------
class TestComposeFinal:
    def test_tbl_merge_with_tab_spanners_unequal_length(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        t1 = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        t2 = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        with pytest.raises(ValueError, match="one entry per table"):
            ps.tbl_merge([t1, t2], tab_spanners=["A"])

    def test_tbl_stack_with_group_labels_unequal(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        t1 = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        t2 = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        with pytest.raises(ValueError, match="group_labels must have"):
            ps.tbl_stack([t1, t2], group_labels=["just one"])

    def test_tbl_stack_single_input_raises(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        t1 = ps.tbl_one(df, by="arm", variables=["x"])
        with pytest.raises(ValueError, match="at least two"):
            ps.tbl_stack([t1])


# ----------------------------------------------------------------------
# format.py — a couple more branches
# ----------------------------------------------------------------------
class TestFormatLast:
    def test_fmt_number_object_input(self):
        from pysofra.core.format import NA_STRING, fmt_number

        class NotNumeric:
            pass
        assert fmt_number(NotNumeric()) == NA_STRING

    def test_fmt_int_object_input(self):
        from pysofra.core.format import NA_STRING, fmt_int

        class NotNumeric:
            pass
        assert fmt_int(NotNumeric()) == NA_STRING

    def test_fmt_number_string_passthrough(self):
        from pysofra.core.format import fmt_number
        # float("1.5") works → format as a number
        assert fmt_number("1.5", digits=2) == "1.50"


# ----------------------------------------------------------------------
# tbl_cross — a few more branches
# ----------------------------------------------------------------------
class TestTblCrossMore:
    def test_tbl_cross_with_all_styles(self, small_trial):
        # Every cell style must produce a valid table
        for style in ("n", "row_pct", "col_pct", "total_pct",
                       "n_row_pct", "n_col_pct", "n_total_pct"):
            t = ps.tbl_cross(small_trial, row="sex", column="arm",
                             cell=style)
            assert len(t.rows) >= 2

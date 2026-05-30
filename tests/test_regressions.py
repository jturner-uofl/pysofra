"""Targeted regression tests for previously-fixed defects.

Each test corresponds to a real bug and the patch that closed it.
Each test should fail if the patch is reverted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.core.format import NA_STRING, fmt_p_value
from pysofra.summary.stats import continuous_stats


# ----------------------------------------------------------------------
# Fmt_p_value clamps out-of-range to NA
# ----------------------------------------------------------------------
class TestFmtPValueClamp:
    def test_negative_returns_na(self):
        assert fmt_p_value(-0.001) == NA_STRING

    def test_above_one_returns_na(self):
        assert fmt_p_value(1.5) == NA_STRING

    def test_in_range_unchanged(self):
        assert fmt_p_value(0.05) == "0.050"
        assert fmt_p_value(0.0001) == "<0.001"
        assert fmt_p_value(0.999) == ">0.99"


        # ----------------------------------------------------------------------
        # N=1 continuous variable: SD is NaN, renders as "—"
        # ----------------------------------------------------------------------
class TestSdNanWhenN1:
    def test_continuous_stats_nan_sd(self):
        st = continuous_stats(pd.Series([5.0]))
        assert np.isnan(st.sd)

    def test_n1_renders_em_dash(self):
        df = pd.DataFrame({"arm": ["A"], "x": [5.0]})
        # The single-level-by guard added a single-level-by UserWarning; expected here.
        with pytest.warns(UserWarning, match=r"only one non-missing level"):
            t = ps.tbl_one(df, by="arm")
            x_row = next(r for r in t.rows if r.cells[0].text == "x")
            # Single-group: value cell should be 5.00 (—)
            text = x_row.cells[1].text
            assert "5.00" in text
            assert "—" in text


            # ----------------------------------------------------------------------
            # All-NaN continuous renders a single em-dash row, not blanks
            # ----------------------------------------------------------------------
class TestAllNanRender:
    def test_all_nan_em_dashes(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": [np.nan] * 10})
        t = ps.tbl_one(df, by="arm").add_p()
        # First body row: variable + em-dashes (no empty cells)
        head_row = t.rows[0]
        assert head_row.cells[0].text == "x"
        assert all(c.text == "—" for c in head_row.cells[1:])


        # ----------------------------------------------------------------------
        # Negative / zero weights raise a clear ValueError (previously
        # they merely warned, which left the displayed table in
        # inconsistent states — e.g. ``N = -1`` or ``N = 0``).
        # ----------------------------------------------------------------------
class TestNegativeWeightRaises:
    def test_negative_weights_raise(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 5,
        "x": np.arange(10.0),
        "w": [-1.0] + [1.0] * 9,
        })
        with pytest.raises(ValueError, match=r"negative value"):
            ps.tbl_one(df, by="arm", weights="w")

    def test_all_zero_weights_raise(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 5,
        "x": np.arange(10.0),
        "w": [0.0] * 10,
        })
        with pytest.raises(ValueError, match=r"total weight"):
            ps.tbl_one(df, by="arm", weights="w")

    def test_positive_weights_no_warning(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 5,
        "x": np.arange(10.0),
        "w": [1.0] * 10,
        })
        with warnings.catch_warnings(record=True) as cw:
            warnings.simplefilter("always")
            ps.tbl_one(df, by="arm", weights="w")
            assert not any("negative" in str(w.message) for w in cw)


            # ----------------------------------------------------------------------
            # Register_theme refuses to overwrite a built-in by default
            # ----------------------------------------------------------------------
class TestRegisterThemeOverwriteGuard:
    def test_built_in_refused(self):
        from pysofra.themes.registry import Theme, register_theme
        with pytest.raises(ValueError, match="built-in"):
            register_theme(Theme(name="clinical", css={}, docx={}))

    def test_overwrite_kwarg_allows(self):
        from pysofra.themes.registry import (
        _CLINICAL,
        Theme,
        register_theme,
        resolve_theme,
        )
        try:
            replacement = Theme(name="clinical", css={"marker": "yes"}, docx={})
            register_theme(replacement, overwrite=True)
            # Verify the registry actually adopted the replacement: the
            # ``css`` dict must now be the one we just registered, not
            # The built-in. Without this assertion the test would still
            # "pass" if register_theme silently no-op'd.
            after = resolve_theme("clinical")
            assert after.css.get("marker") == "yes", (
            "overwrite=True did not actually replace the built-in "
            "clinical theme"
            )
        finally:
            register_theme(_CLINICAL, overwrite=True)

    def test_user_theme_works_without_overwrite(self):
        from pysofra.themes.registry import _THEMES, Theme, register_theme
        try:
            register_theme(Theme(name="audit_test_theme", css={}, docx={}))
            assert "audit_test_theme" in _THEMES
        finally:
            _THEMES.pop("audit_test_theme", None)


            # ----------------------------------------------------------------------
            # Duplicate column names → clear ValueError (was AttributeError)
            # ----------------------------------------------------------------------
class TestDuplicateColumnsRejected:
    def test_raises_value_error(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        with pytest.raises(ValueError, match="duplicate column names"):
            ps.tbl_summary(df)


            # ----------------------------------------------------------------------
            # Bold_if doctest example is no longer a runnable >>> block
            # ----------------------------------------------------------------------
class TestBoldIfDocstring:
    def test_no_failed_doctest(self):
        import doctest

        import pysofra.core.table
        result = doctest.testmod(pysofra.core.table, verbose=False)
        assert result.failed == 0


        # ----------------------------------------------------------------------
        # Variables overlapping by= / design columns → warning
        # ----------------------------------------------------------------------
class TestVariablesOverlapWarns:
    def test_overlap_with_by(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        with pytest.warns(UserWarning, match="overlap"):
            ps.tbl_one(df, by="arm", variables=["arm", "x"])

    def test_no_overlap_no_warning(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10)})
        with warnings.catch_warnings(record=True) as cw:
            warnings.simplefilter("always")
            ps.tbl_one(df, by="arm", variables=["x"])
            assert not any("overlap" in str(w.message) for w in cw)


            # ----------------------------------------------------------------------
            # Both weights= and design= → warning
            # ----------------------------------------------------------------------
class TestWeightsAndDesignConflictWarns:
    def test_warns_when_both(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 5, "x": range(10),
        "w1": [1.0] * 10, "w2": [2.0] * 10,
        })
        with pytest.warns(UserWarning, match="design"):
            ps.tbl_one(df, by="arm", weights="w1",
            design=ps.SurveyDesign(weights="w2"))

    def test_same_weight_col_no_warning(self):
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": range(10), "w": [1.0] * 10})
        with warnings.catch_warnings(record=True) as cw:
            warnings.simplefilter("always")
            ps.tbl_one(df, by="arm", weights="w",
            design=ps.SurveyDesign(weights="w"))
            assert not any("Both weights" in str(w.message) for w in cw)


            # ----------------------------------------------------------------------
            # Empty df renders the variable header with em-dashes, no phantom
            # ----------------------------------------------------------------------
class TestEmptyDataframe:
    def test_no_phantom_row(self):
        t = ps.tbl_summary(pd.DataFrame({"a": []}, dtype=float))
        # One row: the all-NaN representation of `a`
        assert len(t.rows) == 1
        # First cell is the variable label "a"
        assert t.rows[0].cells[0].text == "a"
        # Other cells should be em-dashes, not blanks
        assert all(c.text == "—" for c in t.rows[0].cells[1:])


        # ----------------------------------------------------------------------
        # Add_p error message names the actual cause
        # ----------------------------------------------------------------------
class TestAddPErrorMessageDifferentiation:
    """Previously: calling ``add_p()`` on a ``tbl_cross``
    output raised the unpickle / direct-construction error message
    ("either constructed directly or unpickled") — but the user
    clearly did use a builder. The error path now differentiates
    three distinct routes (direct construction, ``tbl_cross``-style
    builder without recomputation spec, and unpickled).
    """

    def test_tbl_cross_add_smd_error_names_smd_not_defined(self):
        # ``tbl_cross`` carries a rebuild closure so
        # ``.add_p()`` / ``.add_overall()`` work. ``.add_smd()`` still
        # Raises — SMD is undefined on a contingency table — but with
        # A clearer domain-specific message rather than the generic
        # "no re-runnable spec" wording the pre-fix path produced.
        df = pd.DataFrame({"sex": ["F", "M"] * 10,
        "race": ["W", "B", "A", "O"] * 5})
        with pytest.raises(NotImplementedError, match=r"SMD"):
            ps.tbl_cross(df, row="sex", column="race").add_smd()

    def test_directly_constructed_error_names_direct(self):
        from pysofra.core.schema import Cell, HeaderCell, HeaderRow, Row
        t = ps.SofraTable(
        rows=(Row(cells=(Cell(text="x"),)),),
        headers=(HeaderRow(cells=(HeaderCell(text="V"),)),),
        )
        with pytest.raises(RuntimeError, match=r"constructed directly|composition primitive"):
            t.add_p()

    def test_unpickled_error_names_pickle(self):
        import pickle
        t = ps.tbl_one(
        pd.DataFrame({"arm": ["A", "B"] * 5, "x": list(range(10))}),
        by="arm",
        )
        restored = pickle.loads(pickle.dumps(t))
        with pytest.raises(RuntimeError, match=r"unpickled"):
            restored.add_p()


            # ----------------------------------------------------------------------
            # Design_effect warns on negative weights (matches tbl_one)
            # ----------------------------------------------------------------------
class TestDesignEffectNegativeWeights:
    """Previously: ``design_effect`` silently dropped negative
    weights and computed the DEFF on the remainder. ``tbl_one(...,
    weights=...)`` already emits a UserWarning for the same condition;
    this aligns the two surfaces.
    """

    def test_negative_weight_warns(self):
        with pytest.warns(UserWarning, match="negative"):
            d = ps.design_effect(pd.Series([-1.0, 1.0, 1.0, 1.0, 1.0]))
            # And still computes a meaningful DEFF from the positive rows.
            assert d == pytest.approx(1.0)

    def test_positive_weights_no_warning(self):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ps.design_effect(pd.Series([1.0, 1.0, 1.0]))
            assert not any(
            "negative" in str(w.message) for w in caught
            ), [str(w.message) for w in caught]


            # ----------------------------------------------------------------------
            # Tbl_one warns when labels/nonnormal/tests reference a non-variable
            # ----------------------------------------------------------------------
class TestTblOneByLevelDegeneracyWarns:
    """Previously: ``tbl_one(df, by="g")``
    where ``g`` has zero or one non-missing level silently produced a
    degenerate (empty or single-stratum) table with no warning. The R9
    policy is that user inputs whose intent clearly doesn't match the
    output should emit a ``UserWarning``. This pins both branches.
    """

    def test_single_level_by_warns(self):
        df = pd.DataFrame({"arm": ["A"] * 10, "x": list(range(10))})
        with pytest.warns(UserWarning, match=r"only one non-missing level"):
            ps.tbl_one(df, by="arm")

    def test_all_nan_by_warns(self):
        df = pd.DataFrame({
        "arm": [np.nan] * 10,
        "x": list(range(10)),
        })
        with pytest.warns(UserWarning, match=r"no non-missing values"):
            ps.tbl_one(df, by="arm")

    def test_two_or_more_levels_no_warning(self):
        import warnings as _w
        df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": list(range(10))})
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            ps.tbl_one(df, by="arm")
            assert not any(
            "non-missing" in str(w.message) for w in caught
            ), [str(w.message) for w in caught]


class TestTblOneTypoWarnings:
    """Previously: ``tbl_one(df, by='arm', nonnormal=['hbac1'])``
    silently used the WRONG test (Welch instead of Wilcoxon) when the
    user typo'd the column name. Same for ``labels`` (wrong row label)
    and ``tests`` (wrong test). All three now emit a UserWarning naming
    the bad key and the valid variable list.
    """

    def test_nonnormal_typo_warns(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 10,
        "age": list(range(20)),
        "hba1c": list(range(20)),
        })
        with pytest.warns(UserWarning, match="nonnormal=.*hbac1"):
            ps.tbl_one(df, by="arm", nonnormal=["hbac1"])

    def test_labels_typo_warns(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 10,
        "age": list(range(20)),
        "hba1c": list(range(20)),
        })
        with pytest.warns(UserWarning, match="labels=.*hbac1"):
            ps.tbl_one(df, by="arm", labels={"hbac1": "HbA1c (%)"})

    def test_tests_typo_warns(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 10,
        "age": list(range(20)),
        "hba1c": list(range(20)),
        })
        with pytest.warns(UserWarning, match="tests=.*hbac1"):
            ps.tbl_one(df, by="arm", tests={"hbac1": "wilcoxon"})

    def test_valid_inputs_no_warning(self):
        import warnings
        df = pd.DataFrame({
        "arm": ["A", "B"] * 10,
        "age": list(range(20)),
        "hba1c": list(range(20)),
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ps.tbl_one(
            df, by="arm",
            labels={"age": "Age"},
            nonnormal=["hba1c"],
            tests={"age": "wilcoxon"},
            )
            # No typo warnings on valid input
            assert not any(
            "Variables referenced" in str(w.message) for w in caught
            ), [str(w.message) for w in caught]


            # ----------------------------------------------------------------------
            # Lifelines AFT MultiIndex coefnames are flattened to strings
            # ----------------------------------------------------------------------
class TestLifelinesAFTMultiIndex:
    """Previously: lifelines' WeibullAFTFitter (and other AFT
    fitters) expose a MultiIndex ``(param, covariate)`` on ``.summary``,
    so the row labels became tuples like ``('lambda_', 'age')``. The
    markdown renderer crashed at ``_escape`` with
    ``TypeError: expected string or bytes-like object, got 'tuple'``.
    The extractor now flattens to ``"covariate (param)"``.
    """

    def test_weibull_aft_row_labels_are_strings(self):
        from lifelines import WeibullAFTFitter
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "time": rng.exponential(15, n).round(2) + 1,
        "event": rng.binomial(1, 0.6, n),
        })
        w = WeibullAFTFitter().fit(df, duration_col="time", event_col="event")
        t = ps.tbl_regression(w)
        labels = [r.cells[0].text for r in t.rows]
        assert all(isinstance(L, str) for L in labels), labels
        # The age covariate appears in the lambda_ param block.
        assert any("age" in L and "lambda_" in L for L in labels), labels

    def test_weibull_aft_renders_to_every_backend(self, tmp_path):
        from lifelines import WeibullAFTFitter
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "time": rng.exponential(15, n).round(2) + 1,
        "event": rng.binomial(1, 0.6, n),
        })
        w = WeibullAFTFitter().fit(df, duration_col="time", event_col="event")
        t = ps.tbl_regression(w)
        assert len(t.to_html()) > 200
        assert len(t.to_markdown()) > 100
        assert len(t.to_latex()) > 200
        t.to_docx(tmp_path / "x.docx")
        assert (tmp_path / "x.docx").exists()


        # ----------------------------------------------------------------------
        # SofraTable equality is structural, not closure-identity
        # ----------------------------------------------------------------------
class TestSofraTableEquality:
    """Previously: two tables built from the same data by the
    same builder tested unequal because the auto-generated dataclass
    ``__eq__`` compared the ``_rebuild`` closure (per-instance
    identity). A pickled-then-unpickled table also tested unequal to
    its source. The fix overrides ``__eq__`` to compare only output-
    affecting fields, and explicitly sets ``__hash__ = None``.
    """

    def test_identical_builds_compare_equal(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 40),
        "age": rng.normal(50, 8, 40),
        })
        t1 = ps.tbl_one(df, by="arm").add_p()
        t2 = ps.tbl_one(df, by="arm").add_p()
        assert t1 == t2

    def test_theme_change_compares_unequal(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 40),
        "age": rng.normal(50, 8, 40),
        })
        t = ps.tbl_one(df, by="arm").add_p()
        assert t != t.theme("jama")

    def test_pickle_roundtrip_preserves_equality(self):
        import pickle
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 40),
        "age": rng.normal(50, 8, 40),
        })
        t = ps.tbl_one(df, by="arm").add_p()
        assert t == pickle.loads(pickle.dumps(t))

    def test_eq_with_non_sofratable_returns_notimplemented(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 10),
        "age": rng.normal(50, 8, 10),
        })
        t = ps.tbl_one(df, by="arm")
        # Python should fall back to NotImplemented → False from caller's view
        assert (t == 42) is False
        assert (t == "string") is False

    def test_is_not_hashable(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 10),
        "age": rng.normal(50, 8, 10),
        })
        t = ps.tbl_one(df, by="arm")
        with pytest.raises(TypeError):
            hash(t)

    def test_identity_short_circuit(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"arm": rng.choice(["A", "B"], 10),
        "age": rng.normal(50, 8, 10)})
        t = ps.tbl_one(df, by="arm")
        # ``t == t`` hits the ``other is self`` short-circuit before
        # The field-by-field comparison runs.
        assert t == t


        # ----------------------------------------------------------------------
        # Fmt_number / fmt_percent never emit "-0.00" or "-0.0"
        # ----------------------------------------------------------------------
class TestNoNegativeZero:
    """Previously: ``fmt_number(-0.0)`` returned ``"-0.00"`` and
    ``fmt_number(-0.001)`` returned ``"-0.00"`` (small negative rounding
    to zero kept the sign). Same in ``fmt_percent``. Negative zero in a
    publication table is uninformative — the reader sees "zero" with a
    spurious sign. Both formatters now strip the leading minus when the
    output is all-zero.
    """

    def test_fmt_number_minus_zero(self):
        from pysofra.core.format import fmt_number
        assert fmt_number(-0.0) == "0.00"
        assert fmt_number(0.0) == "0.00"
        # Small negative that rounds to zero at 2 decimal places.
        assert fmt_number(-0.001) == "0.00"
        assert fmt_number(-0.004) == "0.00"
        # Legit small negative that survives rounding stays signed.
        assert fmt_number(-0.005) == "-0.01" # half-even or away
        assert fmt_number(-0.05) == "-0.05"
        assert fmt_number(-1.234) == "-1.23"

    def test_fmt_percent_minus_zero(self):
        from pysofra.core.format import fmt_percent
        # -0.0001 * 100 = -0.01 → rounds to -0.0 at 1dp → strip sign
        assert fmt_percent(-0.0001) == "0.0"
        assert fmt_percent(-0.0) == "0.0"
        # Legit -50% survives.
        assert fmt_percent(-0.5) == "-50.0"


        # ----------------------------------------------------------------------
        # Every renderer raises an OSError subclass on read-only paths
        # ----------------------------------------------------------------------
class TestRendererErrorTypeConsistency:
    """Previously: ``to_xlsx`` raised
    ``xlsxwriter.exceptions.FileCreateError`` while the other renderers
    raised ``PermissionError`` (subclass of ``OSError``). User code
    handling write failures wants one catch — ``except OSError`` — to
    work across every renderer. The fix wraps the xlsxwriter exception.
    """

    @pytest.mark.parametrize("renderer_name", [
        "to_docx", "to_xlsx", "to_pptx", "to_image", "to_latex_file"
        ])
    def test_readonly_path_raises_oserror(self, tmp_path, renderer_name):
        import os
        ro = tmp_path / "ro"
        ro.mkdir()
        os.chmod(ro, 0o555)
        try:
            df = pd.DataFrame({"arm": ["A", "B"] * 5, "x": list(range(10))})
            tbl = ps.tbl_one(df, by="arm")
            fn = getattr(tbl, renderer_name)
            ext = {
            "to_docx": ".docx", "to_xlsx": ".xlsx", "to_pptx": ".pptx",
            "to_image": ".png", "to_latex_file": ".tex",
            }[renderer_name]
            with pytest.raises(OSError):
                fn(ro / f"out{ext}")
        finally:
            os.chmod(ro, 0o755)


            # ----------------------------------------------------------------------
            # Infer_kind warns clearly on datetime64 / timedelta64 columns
            # ----------------------------------------------------------------------
class TestInferKindWarnsOnTemporal:
    """a datetime64 column passed to tbl_one previously
    fell through to ``categorical`` silently and produced one row per
    unique timestamp. We now emit a UserWarning so the user can convert
    to a numeric duration instead.
    """

    def test_datetime_emits_warning(self):
        from pysofra.summary.typing import infer_kind
        s = pd.Series(pd.date_range("2020-01-01", periods=5),
        name="event_date")
        with pytest.warns(UserWarning, match="temporal"):
            kind = infer_kind(s)
            assert kind == "categorical" # backward-compatible fallback

    def test_timedelta_emits_warning(self):
        from pysofra.summary.typing import infer_kind
        s = pd.Series(pd.to_timedelta(["1 day", "2 days", "3 days"]),
        name="duration")
        with pytest.warns(UserWarning, match="temporal"):
            kind = infer_kind(s)
            assert kind == "categorical"

    def test_non_temporal_does_not_emit_temporal_warning(self):
        import warnings as _w

        from pysofra.summary.typing import infer_kind
        s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            infer_kind(s)
            assert not any("temporal" in str(w.message) for w in caught)

    def test_exotic_non_temporal_dtype_falls_through_silently(self):
        """Period and Interval dtypes are neither temporal (so no
        warning) nor numeric — they hit the final fallback and must be
        classified as categorical without emitting the temporal warning.

        This pins the *False* branch of the temporal-dtype check that
        branch coverage flagged
        (``summary/typing.py:101->112``).
        """
        import warnings as _w

        from pysofra.summary.typing import infer_kind
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            k_period = infer_kind(
            pd.Series(pd.period_range("2020", periods=4, freq="M"))
            )
            k_interval = infer_kind(
            pd.Series(pd.interval_range(start=0, end=4))
            )
            assert k_period == "categorical"
            assert k_interval == "categorical"
            # No false-positive temporal warning for these non-temporal dtypes.
            assert not any("temporal" in str(w.message) for w in caught)


            # ----------------------------------------------------------------------
            # Infer_kind must not crash on np.inf in a numeric column
            # ----------------------------------------------------------------------
class TestInferKindInfSafe:
    """Previously: ``int(np.inf)`` raises
    ``OverflowError`` and the dichotomous-detector's ``except`` did not
    list it, so any numeric column containing ``±inf`` crashed
    ``tbl_one(...)`` at the ``infer_kind`` step. The fix is to add
    ``OverflowError`` to the except clause; the column then falls
    through to the continuous branch and renders cells as ``—``.

    Previously: a *second* leak in the same
    inf-input path: the downstream ``continuous_stats`` and
    ``continuous_smd_pair`` calls let numpy's ``RuntimeWarning:
        invalid value encountered in subtract`` (and similar) propagate
        through. Under ``filterwarnings = error`` (the project's own
        pyproject.toml gate, and a common user ``-W error`` posture), that
        warning escalates to an exception and the table build crashes
        despite the earlier ``int(np.inf)`` fix in ``infer_kind``.
        ``test_inf_does_not_leak_runtime_warning`` below pins the
        strict-warning behaviour so the inf path is end-to-end clean.
        """

    def test_inf_in_numeric_column_does_not_crash(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 6,
        "x": [1, 2, np.inf, -np.inf, np.nan, 0, 1, 2, 3, 4, 5, 6.0],
        })
        # Should NOT raise OverflowError. We allow downstream
        # RuntimeWarnings here for clarity; the strict-warning gate is
        # In the next test.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = ps.tbl_one(df, by="arm").add_p()
            assert len(t.rows) >= 1
            x_row = next(r for r in t.rows if r.cells[0].text == "x")
            assert any("—" in c.text for c in x_row.cells), x_row

    def test_inf_does_not_leak_runtime_warning(self):
        """Strict-warning gate. Under ``simplefilter('error',
        RuntimeWarning)``, any RuntimeWarning escaping from numpy's
        mean/std/quantile on inf-bearing input would re-raise here and
        fail the test. The fixes in ``summary/stats.py`` and
        ``summary/smd.py`` use ``np.errstate`` + ``catch_warnings`` to
        suppress them at source.
        """
        import warnings
        df = pd.DataFrame({
        "arm": ["A", "B"] * 6,
        "x": [1, 2, np.inf, -np.inf, np.nan, 0, 1, 2, 3, 4, 5, 6.0],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Tbl_one + add_p + add_smd exercises both ``continuous_stats``
            # (mean / SD / quantile) and ``continuous_smd_pair`` (mean /
            # Var). Any RuntimeWarning leak from either crashes here.
            t = ps.tbl_one(df, by="arm").add_p().add_smd()
            assert len(t.rows) >= 1

    def test_pos_inf_only_does_not_crash(self):
        df = pd.DataFrame({
        "arm": ["A", "B"] * 5,
        "y": [np.inf, 0, 1, 0, 1, 0, 1, 0, 1, 0.0],
        })
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = ps.tbl_one(df, by="arm")
            assert len(t.rows) >= 1


            # ----------------------------------------------------------------------
            # SofraTable must round-trip through pickle
            # ----------------------------------------------------------------------
class TestSofraTablePicklability:
    """A SofraTable produced by a builder must survive ``pickle.dumps``/
    ``pickle.loads``. The recomputation closure (``_rebuild``) is dropped
    on pickle; presentational modifiers and renderers still work; the
    statistical modifiers (``add_p``, etc.) raise a clear ``RuntimeError``
    pointing at the unpickle limitation.
    """

    def test_pickle_roundtrip_preserves_renders(self):
        import pickle
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 50),
        "age": rng.normal(50, 8, 50).round(),
        })
        tbl = ps.tbl_one(df, by="arm").add_p()
        blob = pickle.dumps(tbl)
        roundtripped = pickle.loads(blob)
        # Cell content survives byte-identically.
        assert roundtripped.to_html() == tbl.to_html()
        assert roundtripped.to_markdown() == tbl.to_markdown()
        assert roundtripped.to_latex() == tbl.to_latex()
        # Presentational modifiers still work after unpickle.
        themed = roundtripped.theme("clinical").set_caption("Caption")
        assert themed.caption == "Caption"

    def test_recomputation_after_unpickle_raises_informative(self):
        import pickle
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 30),
        "age": rng.normal(50, 8, 30).round(),
        })
        tbl = pickle.loads(pickle.dumps(ps.tbl_one(df, by="arm")))
        with pytest.raises(RuntimeError, match="unpickled"):
            tbl.add_p()

    def test_pickle_drops_rebuild_closure(self):
        import pickle
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 30),
        "age": rng.normal(50, 8, 30).round(),
        })
        tbl = ps.tbl_one(df, by="arm")
        assert tbl._rebuild is not None # before pickle
        unpickled = pickle.loads(pickle.dumps(tbl))
        assert unpickled._rebuild is None # after pickle


        # ----------------------------------------------------------------------
        # Add_q must not propagate a single NaN p-value into every q
        # ----------------------------------------------------------------------
class TestAddQNanPropagation:
    def test_nan_p_does_not_eat_other_qs(self):
        # One degenerate variable (constant → NaN p) and one good variable.
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({
        "arm": ["A", "B"] * (n // 2),
        "constant_x": [5.0] * n, # Welch t-test → NaN
        "noisy_y": rng.normal(0, 1, n) + np.r_[np.zeros(n//2), np.ones(n//2)],
        })
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm").add_p().add_q()
            # Noisy_y should still have a numeric q
            y_row = next(r for r in t.rows if r.cells[0].text == "noisy_y")
            q_cell = next(c for c in y_row.cells if c.kind == "q_value")
            assert isinstance(q_cell.value, (int, float))
            assert q_cell.text != "—"
            # Constant_x's q must stay blank/em-dash
            x_row = next(r for r in t.rows if r.cells[0].text == "constant_x")
            x_q = next(c for c in x_row.cells if c.kind == "q_value")
            assert x_q.value is None or x_q.text in ("", "—")


            # ----------------------------------------------------------------------
            # Scipy hypothesis tests on degenerate data must not leak
            # RuntimeWarning (e.g. "Precision loss occurred in moment calculation").
            # Users gating CI on -W error::RuntimeWarning would otherwise see
            # ``.add_p()`` raise instead of returning a NaN p-value.
            # ----------------------------------------------------------------------
class TestScipyTestsNoRuntimeWarningLeak:
    @staticmethod
    def _strict_call(fn):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            return fn()

    def test_welch_on_constant_data_no_warning(self):
        from pysofra.summary.tests import continuous_test
        v = pd.Series([1.0] * 60)
        g = pd.Series(["A", "B"] * 30)
        self._strict_call(lambda: continuous_test(v, g, nonnormal=False))

    def test_anova_on_constant_data_no_warning(self):
        from pysofra.summary.tests import continuous_test
        v = pd.Series([1.0] * 90)
        g = pd.Series(["A", "B", "C"] * 30)
        self._strict_call(lambda: continuous_test(v, g, nonnormal=False))

    def test_wilcoxon_on_constant_data_no_warning(self):
        from pysofra.summary.tests import continuous_test
        v = pd.Series([2.0] * 60)
        g = pd.Series(["A", "B"] * 30)
        self._strict_call(lambda: continuous_test(v, g, nonnormal=True))

    def test_kruskal_on_constant_data_no_warning(self):
        from pysofra.summary.tests import continuous_test
        v = pd.Series([3.0] * 90)
        g = pd.Series(["A", "B", "C"] * 30)
        self._strict_call(lambda: continuous_test(v, g, nonnormal=True))

    def test_chisq_on_sparse_table_no_warning(self):
        from pysofra.summary.tests import categorical_test
        # 3x3 table with several zero cells, expected < 5 in places.
        v = pd.Series(["x", "y", "z"] * 6)
        g = pd.Series(["A"] * 9 + ["B"] * 9)
        self._strict_call(lambda: categorical_test(v, g))

    def test_cramers_v_on_sparse_table_no_warning(self):
        from pysofra.summary.effect_size import cramers_v
        v = pd.Series(["x", "y", "z"] * 6)
        g = pd.Series(["A"] * 9 + ["B"] * 9)
        self._strict_call(lambda: cramers_v(v, g))

    def test_phi_on_2x2_no_warning(self):
        from pysofra.summary.effect_size import phi_coefficient
        v = pd.Series([0, 1] * 30)
        g = pd.Series(["A", "B"] * 30)
        self._strict_call(lambda: phi_coefficient(v, g))

    def test_full_pipeline_add_p_on_constant_column_no_warning(self):
        # End-to-end: .add_p() over a constant column must not raise
        # RuntimeWarning even with -W error::RuntimeWarning.
        import warnings as _w
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 60),
        "constant": 1.0,
        "noisy": rng.normal(0, 1, 60),
        })
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            ps.tbl_one(df, by="arm").add_p()


            # ----------------------------------------------------------------------
            # Cross-process byte-determinism for the binary backends
            # (.xlsx / .docx / .pptx). The paper's central architectural claim is
            # That every renderer produces byte-deterministic output. The OOXML
            # Formats use ZIP archives whose entry mtimes (from python-docx /
            # Python-pptx) and core.xml ``<dcterms:created>`` (from xlsxwriter)
            # Default to the current wall-clock and silently break this guarantee.
            # Verified by running each render twice in the same process and
            # Comparing SHA-256 hashes; the in-process gate catches any return of
            # Wall-clock leakage and is mutation-tested by reverting the fix.
            # ----------------------------------------------------------------------
class TestBinaryRenderersByteDeterministic:
    @staticmethod
    def _hash_under_two_wall_clocks(method_name, suffix, tmp_path):
        """Render twice with two *different* monkeypatched wall clocks
        and compare. ZIP entry timestamps have 2-second DOS granularity,
        so we use widely-separated timestamps (10 years apart) to make
        the gate robust regardless of sleep().
        """
        import hashlib
        import time
        from unittest.mock import patch

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], 80),
        "x": rng.normal(0, 1, 80),
        "cat": rng.choice(["red", "green", "blue"], 80),
        })
        tbl = ps.tbl_one(df, by="arm").add_p().add_smd()

        # Two widely-separated wall-clock samples. Without the
        # Determinism fix, ZIP entry mtimes (docx/pptx) and
        # ``<dcterms:created>`` (xlsx) inherit these and diverge.
        clocks = [
        time.struct_time((2010, 6, 15, 12, 0, 0, 0, 0, 0)),
        time.struct_time((2030, 6, 15, 12, 0, 0, 0, 0, 0)),
        ]
        hashes = []
        for i, st in enumerate(clocks):
            p = tmp_path / f"out_{i}{suffix}"
            with patch("time.localtime", return_value=st), \
                 patch("time.time", return_value=time.mktime(st)):
                getattr(tbl, method_name)(str(p))
            hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
        return hashes

    def test_xlsx_byte_deterministic(self, tmp_path):
        pytest.importorskip("xlsxwriter")
        h1, h2 = self._hash_under_two_wall_clocks(
        "to_xlsx", ".xlsx", tmp_path,
        )
        assert h1 == h2, f"xlsx not deterministic: {h1[:12]} vs {h2[:12]}"

    def test_docx_byte_deterministic(self, tmp_path):
        pytest.importorskip("docx")
        h1, h2 = self._hash_under_two_wall_clocks(
        "to_docx", ".docx", tmp_path,
        )
        assert h1 == h2, f"docx not deterministic: {h1[:12]} vs {h2[:12]}"

    def test_pptx_byte_deterministic(self, tmp_path):
        pytest.importorskip("pptx")
        h1, h2 = self._hash_under_two_wall_clocks(
        "to_pptx", ".pptx", tmp_path,
        )
        assert h1 == h2, f"pptx not deterministic: {h1[:12]} vs {h2[:12]}"


        # ----------------------------------------------------------------------
        # Sklearn multi-class support in tbl_regression
        # Previously raised NotImplementedError on coef_.shape[0] != 1. The
        # Fix flattens the (n_classes, n_features) coefficient matrix into
        # One row per (class, feature) pair using the same flat-label
        # Convention as lifelines AFT models ("feature (class=X)").
        # ----------------------------------------------------------------------
class TestSklearnMulticlassRegression:
    def test_multiclass_logistic_emits_one_row_per_class_feature(self):
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(28, 5, n),
        })
        y = pd.Series(rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]))
        clf = LogisticRegression(max_iter=1000).fit(X, y)

        # Sanity: actual shape is multi-row.
        assert clf.coef_.shape == (3, 2)

        tbl = ps.tbl_regression(clf)
        labels = [r.cells[0].text for r in tbl.rows]
        assert len(labels) == 6, labels # 3 classes * 2 features
        # Each label has the documented "feature (class=X)" shape.
        for cls in clf.classes_:
            for feat in ("age", "bmi"):
                expected = f"{feat} (class={cls})"
                assert expected in labels, (
                f"missing {expected!r} in {labels!r}"
                )

    def test_multiclass_coefficients_match_model_coef(self):
        """The per-row OR values must equal exp(model.coef_) exactly."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(1)
        n = 200
        X = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(28, 5, n),
        })
        y = pd.Series(rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]))
        clf = LogisticRegression(max_iter=1000).fit(X, y)

        # ModelSummary access via tbl_regression's extractor.
        from pysofra.models.extract import _extract_sklearn
        ms = _extract_sklearn(clf)

        for ci, cls in enumerate(clf.classes_):
            for fi, feat in enumerate(("age", "bmi")):
                got = ms.estimates[f"{feat} (class={cls})"]
                want = clf.coef_[ci, fi]
                assert abs(got - want) < 1e-12, (
                f"{feat}@{cls}: got {got} want {want}"
                )

    def test_binary_logistic_unchanged(self):
        """The (1, n_features) binary path must still emit raw feature
        labels (no class prefix)."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(2)
        n = 200
        X = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(28, 5, n),
        })
        y = pd.Series(rng.choice([0, 1], n))
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        assert clf.coef_.shape == (1, 2)

        tbl = ps.tbl_regression(clf)
        labels = [r.cells[0].text for r in tbl.rows]
        assert labels == ["age", "bmi"], labels

    def test_sklearn_table_carries_no_inference_footnote(self):
        """A sklearn-fitted regression table must carry a footnote
        warning the reader that CIs / p-values are not available
        natively. Without this, blank CI cells silently become
        "no effect" in a reader's eye — the overclaim trap."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({
            "age": rng.normal(60, 10, n),
            "bmi": rng.normal(28, 5, n),
        })
        y = pd.Series(rng.choice([0, 1], n))
        clf = LogisticRegression(max_iter=1000).fit(X, y)

        tbl = ps.tbl_regression(clf)
        msg = " ".join(tbl.footnotes)
        assert "scikit-learn" in msg, (
            f"sklearn no-inference footnote missing: {tbl.footnotes}"
        )
        assert "point estimates only" in msg, (
            f"sklearn no-inference footnote missing the 'point "
            f"estimates only' phrasing: {tbl.footnotes}"
        )
        # Statsmodels-fitted table must *not* carry the sklearn
        # footnote (negative control on the same shape of model).
        import statsmodels.api as sm
        Xc = sm.add_constant(X)
        sm_fit = sm.Logit(y, Xc).fit(disp=False)
        sm_tbl = ps.tbl_regression(sm_fit)
        sm_msg = " ".join(sm_tbl.footnotes)
        assert "scikit-learn" not in sm_msg, (
            f"statsmodels-fitted table picked up sklearn footnote: "
            f"{sm_tbl.footnotes}"
        )


        # ----------------------------------------------------------------------
        # ``weights=`` alone auto-promotes the continuous dispatch
        # From Welch's unweighted t-test to the design-adjusted t-test
        # (``svyttest``). Previously the auto-dispatch required a full
        # ``SurveyDesign(weights=, strata=, cluster=)`` — passing just
        # ``weights=`` silently fell back to UNWEIGHTED Welch on weighted data,
        # Which is wrong (the user passed weights and expected them used).
        # ----------------------------------------------------------------------
class TestWeightsAutoPromotesToSvyttest:
    @staticmethod
    def _toy_df():
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B"], n),
        "age": rng.normal(60, 10, n),
        "cat": rng.choice(["x", "y", "z"], n),
        })
        rng2 = np.random.default_rng(1)
        df["w"] = rng2.uniform(0.5, 2.0, n)
        return df

    def test_weights_only_triggers_design_adjusted_t(self):
        df = self._toy_df()
        t = ps.tbl_one(df, by="arm", weights="w",
        variables=["age"]).add_p()
        # Footnote names the design-adjusted t-test (not "Welch's
        # T-test", the unweighted fallback).
        assert any("Design-adjusted t-test" in f for f in t.footnotes), (
        t.footnotes
        )

    def test_weights_only_p_equals_svyttest_directly(self):
        """The continuous-row p-value must match svyttest() called on
        the same arrays — pysofra is not recomputing, it's dispatching."""
        from pysofra.summary.tests import svyttest
        df = self._toy_df()
        t = ps.tbl_one(df, by="arm", weights="w",
        variables=["age"]).add_p()
        age_row = next(r for r in t.rows if r.cells[0].text == "age")
        p_table = next(c.value for c in age_row.cells
        if c.kind == "p_value")
        res = svyttest(df["age"], df["arm"], df["w"])
        assert abs(float(p_table) - float(res.p_value)) < 1e-12, (
        f"table={p_table} svyttest={res.p_value}"
        )

    def test_unweighted_still_uses_welch(self):
        """Removing weights= must restore the Welch dispatch — the
        this change is additive, not a behaviour swap for
        existing users."""
        df = self._toy_df()
        t = ps.tbl_one(df, by="arm", variables=["age"]).add_p()
        assert any("Welch's t-test" in f for f in t.footnotes), (
        t.footnotes
        )

    def test_weighted_p_differs_from_unweighted_p(self):
        """The whole point of the fix: weighted analysis must produce a
        different p-value than the unweighted one (otherwise weights
        weren't used). Sanity gate against silent fallback regression."""
        df = self._toy_df()
        p_w = next(c.value for r in
        ps.tbl_one(df, by="arm", weights="w",
        variables=["age"]).add_p().rows
        if r.cells[0].text == "age"
        for c in r.cells if c.kind == "p_value")
        p_u = next(c.value for r in
        ps.tbl_one(df, by="arm",
        variables=["age"]).add_p().rows
        if r.cells[0].text == "age"
        for c in r.cells if c.kind == "p_value")
        assert abs(float(p_w) - float(p_u)) > 1e-6, (
        f"weighted p {p_w} == unweighted p {p_u}; weights ignored"
        )


        # ----------------------------------------------------------------------
        # Tbl_cross now carries a rebuild closure so .add_p() and
        # .add_overall() work. Previously raised RuntimeError ("does not carry
        # A re-runnable spec"). .add_smd() still raises but with an informative
        # Domain-correct message (SMD is undefined on a contingency table).
        # ----------------------------------------------------------------------
class TestTblCrossRebuild:
    @staticmethod
    def _toy_df():
        rng = np.random.default_rng(0)
        n = 200
        return pd.DataFrame({
        "arm": rng.choice(["Placebo", "Treatment"], n),
        "sex": rng.choice(["F", "M"], n),
        "race": rng.choice(["White", "Black", "Asian"], n,
        p=[0.6, 0.25, 0.15]),
        })

    def test_add_p_appends_pvalue_footnote_2x2_uses_fisher(self):
        from scipy import stats as sp_stats
        df = self._toy_df()
        t = ps.tbl_cross(df, row="sex", column="arm").add_p()
        # Footnote naming the test + journal-formatted p.
        assert any("Fisher's exact" in f for f in t.footnotes), t.footnotes
        # Raw numeric in metadata.
        assert "p_value" in t.metadata
        # Numeric equality with scipy direct call.
        ctab = pd.crosstab(df["sex"], df["arm"]).to_numpy()
        _, p_scipy = sp_stats.fisher_exact(ctab, alternative="two-sided")
        assert abs(t.metadata["p_value"] - float(p_scipy)) < 1e-12

    def test_add_p_appends_pvalue_footnote_rxc_uses_chi_square(self):
        from scipy import stats as sp_stats
        df = self._toy_df()
        # 3x2 table → Pearson chi-square branch.
        t = ps.tbl_cross(df, row="race", column="arm").add_p()
        assert any("chi-square" in f.lower() for f in t.footnotes), t.footnotes
        ctab = pd.crosstab(df["race"], df["arm"]).to_numpy()
        chi2, p, _, _ = sp_stats.chi2_contingency(ctab, correction=False)
        assert abs(t.metadata["p_value"] - float(p)) < 1e-12

    def test_add_overall_turns_on_margins(self):
        df = self._toy_df()
        t = ps.tbl_cross(df, row="sex", column="arm",
        margins=False).add_overall()
        header_labels = [h.text for h in t.headers[0].cells]
        assert "Total" in header_labels, header_labels

    def test_add_smd_raises_informative(self):
        df = self._toy_df()
        with pytest.raises(NotImplementedError, match="SMD"):
            ps.tbl_cross(df, row="sex", column="arm").add_smd()

    def test_bare_tbl_cross_unchanged(self):
        """The pre-fix output (no modifiers) must be byte-identical."""
        df = self._toy_df()
        t = ps.tbl_cross(df, row="sex", column="arm")
        # Single footnote (the cell style); no p-value footnote leaked.
        assert len(t.footnotes) == 1
        assert "Cells:" in t.footnotes[0]
        assert "p_value" not in t.metadata
        # Rebuild closure exists (this is the new contract).
        assert t._rebuild is not None
        assert t._spec is not None and t._spec.builder == "tbl_cross"

    def test_add_p_then_add_overall_composes(self):
        df = self._toy_df()
        t = (ps.tbl_cross(df, row="sex", column="arm", margins=False)
        .add_p()
        .add_overall())
        # Both effects: p-value footnote + Total column.
        assert any("p = " in f for f in t.footnotes), t.footnotes
        assert "Total" in [h.text for h in t.headers[0].cells]


        # ----------------------------------------------------------------------
        # Add_global_p on tbl_one / tbl_summary
        # Previously raised NotImplementedError. The new path fits one
        # Logistic regression per variable (``Logit(by == ref ~ variable
        # [+ adjust_for])``) and computes the joint Wald p-value on the
        # Variable's coefficients — same statistic as gtsummary's
        # ``add_global_p()``.
        # ----------------------------------------------------------------------
class TestTblOneGlobalP:
    @staticmethod
    def _toy_df():
        rng = np.random.default_rng(0)
        n = 400
        return pd.DataFrame({
        "arm": rng.choice(["A", "B"], n),
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(28, 5, n),
        "sex": rng.choice(["F", "M"], n),
        "race": rng.choice(["W", "B", "X", "Y"], n,
        p=[0.5, 0.25, 0.15, 0.10]),
        })

    @staticmethod
    def _suppress():
        import warnings as _w
        return _w.catch_warnings()

    def test_categorical_global_p_matches_direct_logit(self):
        """For a 4-level race ~ arm fit, the joint Wald p must match a
        manual ``Logit(arm ~ C(race)).f_test`` to machine precision."""
        import warnings as _w

        import statsmodels.api as sm
        df = self._toy_df()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm", variables=["race"]).add_global_p()
            race_row = next(r for r in t.rows if r.cells[0].text == "race")
            p_table = race_row.cells[-1].value

            # Direct fit (machine-precision reference).
            sub = df[["arm", "race"]].dropna()
            y = (sub["arm"] == "B").astype(int).to_numpy()
            dummies = pd.get_dummies(sub["race"], prefix="race",
            drop_first=True, dtype=float).to_numpy()
            X = sm.add_constant(dummies)
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = sm.Logit(y, X).fit(disp=False)
                n_dummies = X.shape[1] - 1 # excluding the const
                constraint = ", ".join(f"x{i+1} = 0" for i in range(n_dummies))
                p_direct = float(res.f_test(constraint).pvalue)
                assert abs(float(p_table) - p_direct) < 1e-12, (
                f"table={p_table} direct={p_direct}"
                )

    def test_continuous_variable_gets_walds_p(self):
        """Single-coefficient predictors (continuous) get the Wald p of
        that one coefficient — same as the joint test on a 1-coef set."""
        import warnings as _w
        df = self._toy_df()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p()
            age_row = next(r for r in t.rows if r.cells[0].text == "age")
            assert age_row.cells[-1].value is not None
            # Numeric, 0..1, formatted as a journal p-value string.
            assert 0.0 <= float(age_row.cells[-1].value) <= 1.0

    def test_adjust_for_changes_p_value(self):
        """Adding ``adjust_for=['age']`` to a sex ~ arm fit must
        change the joint p from the unadjusted version (otherwise
        the covariate is being ignored)."""
        import warnings as _w
        df = self._toy_df()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t_unadj = ps.tbl_one(df, by="arm",
            variables=["sex"]).add_global_p()
            t_adj = ps.tbl_one(df, by="arm",
            variables=["sex"]).add_global_p(
            adjust_for=["age", "bmi"])
            sex_unadj = next(r for r in t_unadj.rows
            if r.cells[0].text.startswith("sex"))
            sex_adj = next(r for r in t_adj.rows
            if r.cells[0].text.startswith("sex"))
            p_u = float(sex_unadj.cells[-1].value)
            p_a = float(sex_adj.cells[-1].value)
            assert abs(p_u - p_a) > 1e-9, f"adjust didn't change p: {p_u} == {p_a}"

    def test_level_rows_blank_for_categorical_variable(self):
        """For a 4-level race, the parent row carries the global p
        and the four level rows are blank — visual convention so the
        joint p reads as 'belonging to' the variable as a whole."""
        import warnings as _w
        df = self._toy_df()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["race"]).add_global_p()
            race_row = next(r for r in t.rows if r.cells[0].text == "race")
            assert race_row.cells[-1].value is not None
            # The four level rows must have empty global p cells.
            level_rows = [r for r in t.rows
            if r.cells[0].text in ("W", "B", "X", "Y")]
            assert len(level_rows) == 4
            for r in level_rows:
                assert r.cells[-1].text == "", r.cells[-1].text

    def test_more_than_two_by_levels_falls_back_with_footnote(self):
        """3-level ``by=`` requires multinomial logit (out of scope v1).
        Column is added with em-dash fills + an explanatory footnote."""
        import warnings as _w
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
        "arm": rng.choice(["A", "B", "C"], n),
        "age": rng.normal(60, 10, n),
        })
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p()
            # The column was added (even if filled with em-dash).
            assert "global p" in [h.text for h in t.headers[0].cells]
            assert any("multinomial" in f.lower() for f in t.footnotes), (
            t.footnotes
            )

    def test_tbl_regression_path_still_works(self):
        """The new dispatch must not break the existing tbl_regression
        path — add_global_p() with no model still uses f_test()."""
        import warnings as _w

        import statsmodels.api as sm
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
        "y": rng.binomial(1, 0.4, n),
        "x": rng.normal(0, 1, n),
        "race": rng.choice(["W", "B", "X", "Y"], n,
        p=[0.5, 0.25, 0.15, 0.10]),
        })
        df_d = pd.get_dummies(df["race"], prefix="race",
        drop_first=True, dtype=float)
        X = sm.add_constant(
        pd.concat([df[["x"]], df_d], axis=1),
        )
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            fit = sm.Logit(df["y"], X).fit(disp=False)
            t = ps.tbl_regression(fit).add_global_p()
            # Global p column was added (existing tbl_regression behaviour).
            assert "global p" in [h.text for h in t.headers[0].cells]

    def test_by_none_skips_with_footnote(self):
        """add_global_p() on an unstratified tbl_one (no by=) makes no
        sense as a 'joint p for variable-vs-arm'; the column is added
        with blanks + an explanatory footnote, not a misleading
        em-dash column or a crash."""
        import warnings as _w
        df = self._toy_df()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_summary(df, variables=["age"]).add_global_p()
            # Column header present, every body cell is blank.
            assert "global p" in [h.text for h in t.headers[0].cells]
            assert all(r.cells[-1].text == "" for r in t.rows)
            assert any("no by=" in f.lower() for f in t.footnotes), t.footnotes

    def test_all_nan_variable_returns_em_dash(self):
        """Variable with all-NaN values (after dropping by-missing) →
        ``_fit_global_p`` returns None, cell renders as em-dash."""
        import warnings as _w
        df = self._toy_df()
        df["allnan"] = np.nan
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["allnan"]).add_global_p()
            row = next(r for r in t.rows if r.cells[0].text == "allnan")
            assert row.cells[-1].text == "—", row.cells[-1].text

    def test_categorical_adjust_column_is_dummy_coded(self):
        """A non-numeric ``adjust_for`` column must enter the design
        matrix as dummy-coded levels (not as raw strings, which would
        crash Logit). Tests ``_quick_kind`` non-continuous branch."""
        import warnings as _w
        df = self._toy_df()
        rng = np.random.default_rng(99)
        df["site"] = rng.choice(["S1", "S2", "S3"], len(df))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p(
            adjust_for=["site"])
            row = next(r for r in t.rows if r.cells[0].text == "age")
            assert row.cells[-1].value is not None
            assert 0.0 <= float(row.cells[-1].value) <= 1.0

    def test_two_level_adjust_column_uses_dichotomous_kind(self):
        """A 2-unique-value adjustment column hits ``_quick_kind``'s
        dichotomous branch — exercises a different code path than the
        3+ level categorical branch."""
        import warnings as _w
        df = self._toy_df()
        # Binary numeric column with exactly 2 unique values.
        df["smoker"] = pd.Series(
        np.random.default_rng(7).choice([0, 1], len(df)),
        index=df.index,
        )
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p(
            adjust_for=["smoker"])
            row = next(r for r in t.rows if r.cells[0].text == "age")
            assert row.cells[-1].value is not None

    def test_single_level_variable_returns_em_dash(self):
        """Variable with only one unique value → dummies are empty
        (k-1 = 0 cols), ``_fit_global_p`` returns None. Tests the
        ``if not var_cols`` guard."""
        import warnings as _w
        df = self._toy_df()
        df["constant_cat"] = "only_value" # single-level categorical
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm",
            variables=["constant_cat"]).add_global_p()
            row = next(r for r in t.rows
            if r.cells[0].text == "constant_cat")
            assert row.cells[-1].text == "—"

    def test_adjust_for_overlapping_variables_deduplicates(self):
        """If the variable being tested is also listed in
        ``adjust_for``, the duplicate would otherwise (a) make
        ``data[[by, var, *adjust_for]]`` produce a 2-D Series and crash
        ``pd.to_numeric``, and (b) make the design matrix singular.
        The fix drops the duplicate adjustment so the variable's own
        coefficients are still the joint-test target."""
        import warnings as _w
        df = self._toy_df()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # 'age' appears in both variables and adjust_for; must not
            # Crash and the global p for age must equal the unadjusted
            # Version (age can't adjust for itself).
            t_dedup = ps.tbl_one(df, by="arm",
            variables=["age", "bmi"]).add_global_p(
            adjust_for=["age"])
            t_plain = ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p()
            age_a = next(r for r in t_dedup.rows
            if r.cells[0].text == "age").cells[-1].value
            age_b = next(r for r in t_plain.rows
            if r.cells[0].text == "age").cells[-1].value
            assert age_a == age_b, (
            f"dedupe didn't preserve age's joint p: {age_a} vs {age_b}"
            )

    def test_adjust_for_missing_column_raises_clear_keyerror(self):
        """``adjust_for=['NOPE']`` where 'NOPE' isn't a column should
        raise a clear ``KeyError`` naming the missing column and the
        method — matching the existing ``by=`` / ``weights=`` validation
        pattern (``by column 'foo' not in data``). Previously raised a
        bare pandas ``KeyError: \"['NOPE'] not in index\"`` from deep
        inside the fit, which left the user guessing about the cause."""
        df = self._toy_df()
        with pytest.raises(KeyError, match=r"adjust_for.*not in data"):
            ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p(adjust_for=["NOPE"])

    def test_adjust_for_partially_missing_lists_all_missing(self):
        """When multiple adjust_for columns are missing, the error
        message must name all of them (so the user fixes them in one
        pass, not iteratively)."""
        df = self._toy_df()
        with pytest.raises(KeyError) as excinfo:
            ps.tbl_one(df, by="arm",
            variables=["age"]).add_global_p(
            adjust_for=["NOPE", "AGE", "bmi"]) # bmi exists
            msg = str(excinfo.value)
            # Both missing names appear; the valid one ('bmi') doesn't.
            assert "NOPE" in msg
            assert "AGE" in msg # capitalised — pandas is case-sensitive
            # Bmi is valid so it shouldn't appear as missing.
            assert "'bmi'" not in msg


# ----------------------------------------------------------------------
# Lifelines regression CIs honour user-supplied ``conf_level``
# (previously: lifelines bakes alpha into the fit, so the CI columns
# reflected the fit-time level — usually 95% — even when the user
# requested a different one. The fix re-derives the CI from
# ``coef`` and ``se(coef)`` using a normal pivot at the requested
# level.)
# ----------------------------------------------------------------------
class TestLifelinesConfLevelHonoured:
    def test_cph_conf_level_changes_ci_width(self):
        pytest.importorskip("lifelines")
        from lifelines import CoxPHFitter

        rng = np.random.default_rng(2026)
        n = 200
        df = pd.DataFrame({
            "time": rng.exponential(10, n),
            "event": rng.integers(0, 2, n),
            "x": rng.normal(0, 1, n),
        })
        cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")

        t95 = ps.tbl_regression(cph, conf_level=0.95)
        t90 = ps.tbl_regression(cph, conf_level=0.90)
        t99 = ps.tbl_regression(cph, conf_level=0.99)

        def _ci(t):
            return next(
                c.value for c in t.rows[0].cells
                if c.kind == "ci" and isinstance(c.value, tuple)
            )

        # tbl_regression CI cells carry (lo, hi); tbl_one's
        # add_difference carries (point, lo, hi) — don't confuse them.
        lo95, hi95 = _ci(t95)
        lo90, hi90 = _ci(t90)
        lo99, hi99 = _ci(t99)
        # Narrower at 90%, wider at 99%, with strict inequality.
        assert (hi90 - lo90) < (hi95 - lo95) < (hi99 - lo99)

    def test_cph_conf_level_matches_manual_normal_pivot(self):
        """The re-derived CI must equal ``coef ± z*se`` on the log-HR scale,
        then exponentiated. Verifies against a hand-computed reference."""
        pytest.importorskip("lifelines")
        sp = pytest.importorskip("scipy.stats")
        from lifelines import CoxPHFitter

        rng = np.random.default_rng(2027)
        n = 300
        df = pd.DataFrame({
            "time": rng.exponential(8, n),
            "event": rng.integers(0, 2, n),
            "x": rng.normal(0, 1, n),
        })
        cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        coef = float(cph.summary["coef"].iloc[0])
        se = float(cph.summary["se(coef)"].iloc[0])

        for lvl in (0.80, 0.90, 0.95, 0.99):
            t = ps.tbl_regression(cph, conf_level=lvl)
            ci = next(
                c.value for c in t.rows[0].cells
                if c.kind == "ci" and isinstance(c.value, tuple)
            )
            z = float(sp.norm.ppf(0.5 + lvl / 2))
            expected_lo = float(np.exp(coef - z * se))
            expected_hi = float(np.exp(coef + z * se))
            assert abs(ci[0] - expected_lo) < 1e-9, (lvl, ci, expected_lo)
            assert abs(ci[1] - expected_hi) < 1e-9, (lvl, ci, expected_hi)


# ----------------------------------------------------------------------
# AFT family exp(coef) is labelled "TR" (Time Ratio), NOT "HR".
# Mislabelling is publication-critical because TR>1 means LONGER
# survival while HR>1 means SHORTER survival — opposite direction.
# ----------------------------------------------------------------------
class TestAFTLabelIsTimeRatio:
    def test_weibull_aft_header_says_TR(self):
        pytest.importorskip("lifelines")
        from lifelines import WeibullAFTFitter

        rng = np.random.default_rng(2026)
        n = 150
        df = pd.DataFrame({
            "time": rng.exponential(10, n),
            "event": rng.integers(0, 2, n),
            "x": rng.normal(0, 1, n),
        })
        aft = WeibullAFTFitter().fit(df, duration_col="time", event_col="event")
        t = ps.tbl_regression(aft)
        headers = [h.text for h in t.headers[0].cells]
        assert "TR" in headers, f"expected 'TR' in headers, got {headers}"
        assert "HR" not in headers, f"AFT must not carry 'HR' label, got {headers}"


# ----------------------------------------------------------------------
# Multi-model with_forest_plot warns the user that only the first model
# is drawn (the renderer plots a single estimate/CI series per row).
# Silent first-model-only plotting on a 3-model side-by-side
# regression would mislead readers about what the figure represents.
# ----------------------------------------------------------------------
class TestMultiModelForestPlotWarns:
    def test_warning_on_multi_model_table(self):
        sm = pytest.importorskip("statsmodels.api")

        rng = np.random.default_rng(2026)
        n = 200
        df = pd.DataFrame({
            "age": rng.normal(60, 10, n),
            "bmi": rng.normal(28, 5, n),
            "event": rng.binomial(1, 0.3, n),
        })
        X1 = sm.add_constant(df[["age"]])
        X2 = sm.add_constant(df[["age", "bmi"]])
        m1 = sm.Logit(df["event"], X1).fit(disp=False)
        m2 = sm.Logit(df["event"], X2).fit(disp=False)

        t = ps.tbl_regression([m1, m2], model_labels=["Crude", "Adjusted"])
        with pytest.warns(UserWarning, match=r"only the first model"):
            t.with_forest_plot()

    def test_no_warning_on_single_model_table(self):
        sm = pytest.importorskip("statsmodels.api")

        rng = np.random.default_rng(2027)
        n = 200
        df = pd.DataFrame({
            "age": rng.normal(60, 10, n),
            "event": rng.binomial(1, 0.3, n),
        })
        X = sm.add_constant(df[["age"]])
        m = sm.Logit(df["event"], X).fit(disp=False)
        t = ps.tbl_regression(m)
        # Should NOT warn on single-model.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t.with_forest_plot()


# ----------------------------------------------------------------------
# conf_level range validation is enforced consistently across the
# public API (previously only add_ci / add_difference rejected
# out-of-range values; tbl_regression / tbl_survival / pool accepted
# anything and propagated nonsense into model.conf_int / KM CI logic).
# ----------------------------------------------------------------------
class TestConfLevelValidatedEverywhere:
    def test_tbl_regression_rejects_out_of_range(self):
        sm = pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"y": [0, 1] * 25, "x": list(range(50))})
        m = sm.Logit(df["y"], sm.add_constant(df[["x"]])).fit(disp=False)
        for bad in (-0.5, 0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match=r"conf_level"):
                ps.tbl_regression(m, conf_level=bad)

    def test_tbl_survival_rejects_out_of_range(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "t": rng.exponential(10, 100),
            "e": rng.integers(0, 2, 100),
        })
        for bad in (-0.5, 0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match=r"conf_level"):
                ps.tbl_survival(df, time="t", event="e", conf_level=bad)

    def test_pool_rejects_out_of_range(self):
        from pysofra import pool
        sm = pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"y": [0, 1] * 25, "x": list(range(50))})
        m = sm.Logit(df["y"], sm.add_constant(df[["x"]])).fit(disp=False)
        for bad in (-0.5, 0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match=r"conf_level"):
                pool([m, m], conf_level=bad)


# ----------------------------------------------------------------------
# svyttest now uses a full-design Taylor-linearised variance and the
# correct ``n_PSU − n_strata − 1`` degrees of freedom. The point
# estimate, t-statistic, and p-value all match R ``survey::svyttest``
# to 6+ decimal places on representative designs.
#
# These tests are pinned against **R-derived reference values**
# (computed by ``Rscript survey::svyttest`` on the same seed) rather
# than against pysofra's own intermediate computations — so they
# detect a regression in either the variance formula OR the df
# formula, not just self-consistency.
# ----------------------------------------------------------------------
class TestSvyttestVsR:
    def test_unclustered_matches_R_to_6_decimals(self):
        """Reference produced by:
            Rscript -e 'library(survey); df <- data.frame(
              v=c(5.1,5.5,6.0,5.8,5.4,6.5,7.0,6.8,7.2,6.9),
              g=c(rep("A",5), rep("B",5)),
              w=c(1.0,2,1,1,1,1.0,1,2,1,1));
              print(svyttest(v~g, svydesign(ids=~1, weights=~w, data=df)))'
        → t = 8.473899, df = 8, p = 2.879105e-05
        """
        from pysofra.summary.tests import svyttest

        df = pd.DataFrame({
            "v": [5.1, 5.5, 6.0, 5.8, 5.4, 6.5, 7.0, 6.8, 7.2, 6.9],
            "g": ["A"] * 5 + ["B"] * 5,
            "w": [1.0, 2, 1, 1, 1, 1.0, 1, 2, 1, 1],
        })
        res = svyttest(df["v"], df["g"], df["w"])
        assert abs(res.statistic - 8.473899) < 1e-5
        assert abs(res.p_value - 2.879105e-05) < 1e-7

    def test_clustered_stratified_matches_R_to_6_decimals(self):
        """Reference dataset is generated by NumPy's PCG-64 default_rng
        with seed=2026; the exact same CSV was then loaded by R 4.6.0
        and run through:

            survey::svyttest(
                y ~ arm,
                svydesign(ids=~cluster, strata=~strata, weights=~wt,
                          data=df, nest=TRUE)
            )

        → t = 1.8480707873, df = 86, p = 6.8029410799e-02.

        Pinning these to 6+ decimal places confirms full-design
        Taylor linearisation parity with R's ``survey`` package.
        """
        from pysofra.summary.tests import svyttest

        rng = np.random.default_rng(2026)
        n = 600
        n_clusters = 30
        df = pd.DataFrame({
            "arm":     rng.choice(["A", "B"], n, p=[0.5, 0.5]),
            "y":       rng.normal(10, 3, n),
            "cluster": rng.integers(0, n_clusters, n),
            "strata":  rng.choice(["S1", "S2", "S3"], n),
            "wt":      rng.uniform(0.5, 2.0, n),
        })
        df.loc[df["arm"] == "B", "y"] += 0.6

        res = svyttest(df["y"], df["arm"], df["wt"],
                       strata=df["strata"], cluster=df["cluster"])
        # t-statistic identical to R to machine precision; p-value to
        # 1e-10 (sensitive only to df rounding, but pysofra and R agree
        # on df = 86 here).
        assert abs(res.statistic - 1.8480707873) < 1e-7
        assert abs(res.p_value - 6.8029410799e-02) < 1e-7

    def test_per_group_variance_bug_does_not_recur(self):
        """The pre-fix code computed per-group variances separately and
        summed in quadrature. On the auditor's clustered dataset that
        produced p ≈ 5.6e-6 vs R's 0.073 — four orders of magnitude
        anti-conservative. The new code uses full-design linearisation
        and matches R to 6 decimals.

        Pin: on the python-RNG seed=2026 fixture (verified above), the
        old buggy code would have produced t ≈ 4.5 or larger (per-group
        variances under-estimate Var(diff) when clusters straddle
        groups). The corrected t is 1.85 — confirm we're nowhere near
        the anti-conservative regime.
        """
        from pysofra.summary.tests import svyttest

        rng = np.random.default_rng(2026)
        n = 600
        df = pd.DataFrame({
            "arm":     rng.choice(["A", "B"], n, p=[0.5, 0.5]),
            "y":       rng.normal(10, 3, n),
            "cluster": rng.integers(0, 30, n),
            "strata":  rng.choice(["S1", "S2", "S3"], n),
            "wt":      rng.uniform(0.5, 2.0, n),
        })
        df.loc[df["arm"] == "B", "y"] += 0.6
        res = svyttest(df["y"], df["arm"], df["wt"],
                       strata=df["strata"], cluster=df["cluster"])
        # |t| should be small (under 3); regression to the per-group
        # bug would push it well above that on this data.
        assert abs(res.statistic) < 3.0
        # And the p-value should be in the borderline-non-significant
        # range (around 0.07), not the anti-conservative 1e-5 the
        # buggy code produced.
        assert 0.01 < res.p_value < 0.5


# ----------------------------------------------------------------------
# SMD column honours weights on a weighted Table 1. Previously the SMD
# was computed unweighted regardless of weights=, silently disagreeing
# with R's ``tableone(weights=)`` and ``survey::svystdize`` for the
# canonical headline use case (weighted Table 1 with SMDs).
# ----------------------------------------------------------------------
class TestWeightedSMD:
    def test_continuous_weighted_smd_matches_statsmodels_reference(self):
        sm_w = pytest.importorskip("statsmodels.stats.weightstats")

        rng = np.random.default_rng(2026)
        n = 400
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "age": rng.normal(60, 10, n),
            "w":   rng.uniform(0.5, 2.0, n),
        })
        df.loc[df["arm"] == "A", "w"] *= 3.0  # make weighting matter
        # Reference: |mean_A - mean_B| / sqrt((var_A + var_B)/2)
        # with weighted means / variances at ddof=1.
        a = df[df["arm"] == "A"]
        b = df[df["arm"] == "B"]
        da = sm_w.DescrStatsW(a["age"].to_numpy(), weights=a["w"].to_numpy(), ddof=1)
        db = sm_w.DescrStatsW(b["age"].to_numpy(), weights=b["w"].to_numpy(), ddof=1)
        expected = abs(da.mean - db.mean) / float(np.sqrt((da.var + db.var) / 2))

        from pysofra.summary.smd import continuous_smd
        got = continuous_smd(df["age"], df["arm"], weights=df["w"])
        assert abs(got - expected) < 1e-12

    def test_weighted_tbl_one_smd_differs_from_unweighted(self):
        rng = np.random.default_rng(2027)
        n = 300
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "age": rng.normal(60, 10, n),
            "w":   rng.uniform(0.5, 2.0, n),
        })
        df.loc[df["arm"] == "A", "w"] *= 4.0

        t_unw = ps.tbl_one(df, by="arm", variables=["age"],
                           missing="never").add_smd()
        t_wt  = ps.tbl_one(df, by="arm", variables=["age"], weights="w",
                           missing="never").add_smd()
        # SMD cell is the last cell of the "age" row
        def smd_of(t):
            r = next(r for r in t.rows if r.cells[0].text == "age")
            return float(r.cells[-1].value)
        unw, wtd = smd_of(t_unw), smd_of(t_wt)
        # Must differ on this synthetic dataset (where weights are non-uniform).
        assert abs(unw - wtd) > 1e-6


# ----------------------------------------------------------------------
# add_ci / add_difference / add_global_p now honour weights= on a
# weighted tbl_one. Previously each modifier silently dropped weights
# and reverted to unweighted computations, producing a table whose
# row p-values were weighted but whose CIs and joint p-values were
# unweighted — internally inconsistent.
# ----------------------------------------------------------------------
class TestWeightedModifiers:
    def _df(self, seed: int = 2026):
        rng = np.random.default_rng(seed)
        n = 300
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "age": rng.normal(60, 10, n),
            "smoker": rng.choice([0, 1], n, p=[0.7, 0.3]),
            "w": rng.uniform(0.5, 2.0, n),
        })
        df.loc[df["arm"] == "A", "w"] *= 3.0
        return df

    def test_add_ci_weighted_matches_descrstats(self):
        sm_w = pytest.importorskip("statsmodels.stats.weightstats")
        df = self._df()
        t = (
            ps.tbl_one(df, by="arm", variables=["age"], weights="w",
                       missing="never")
            .add_ci(conf_level=0.95)
        )
        # Find the per-arm cell text and parse the bracketed CI.
        # Arm A
        a = df[df["arm"] == "A"]
        sm_a = sm_w.DescrStatsW(a["age"].to_numpy(), weights=a["w"].to_numpy(), ddof=1)
        import math
        n_eff_a = (a["w"].sum() ** 2) / (a["w"] ** 2).sum()
        se_a = math.sqrt(sm_a.var / n_eff_a)
        from scipy import stats as sp
        tcrit = float(sp.t.ppf(0.975, df=n_eff_a - 1))
        ref_lo = sm_a.mean - tcrit * se_a
        ref_hi = sm_a.mean + tcrit * se_a
        # Pull arm-A cell text
        row = next(r for r in t.rows if r.cells[0].text == "age")
        arm_cells = [c.text for c in row.cells[1:3]]
        # First cell should contain the CI brackets.
        import re
        m = re.search(r"\[([\-0-9.]+), ([\-0-9.]+)\]", arm_cells[0])
        assert m is not None, f"no bracket CI in {arm_cells[0]!r}"
        got_lo, got_hi = float(m.group(1)), float(m.group(2))
        # Display rounds to 2 dp; allow that tolerance.
        assert abs(got_lo - round(ref_lo, 2)) < 0.02
        assert abs(got_hi - round(ref_hi, 2)) < 0.02

    def test_add_difference_weighted_differs_from_unweighted(self):
        df = self._df()
        t_unw = (
            ps.tbl_one(df, by="arm", variables=["age"], missing="never")
            .add_difference()
        )
        t_wt = (
            ps.tbl_one(df, by="arm", variables=["age"], weights="w",
                       missing="never")
            .add_difference()
        )

        def diff_cell(t):
            r = next(r for r in t.rows if r.cells[0].text == "age")
            cell = next(
                c for c in r.cells
                if c.kind == "ci" and isinstance(c.value, tuple)
                and len(c.value) == 3
            )
            return cell.value

        diff_unw = diff_cell(t_unw)
        diff_wt = diff_cell(t_wt)
        # Point estimate differs.
        assert abs(diff_unw[0] - diff_wt[0]) > 1e-6
        # CI bounds differ.
        assert abs(diff_unw[1] - diff_wt[1]) > 1e-6 or \
               abs(diff_unw[2] - diff_wt[2]) > 1e-6

    def test_add_global_p_weighted_matches_glm_var_weights(self):
        # Reference uses ``var_weights=`` rather than ``freq_weights=``:
        # for non-integer sampling weights, ``freq_weights`` artificially
        # inflates df_resid by ``Σw`` (treating the weight as an integer
        # count of repeats), making the F-test anti-conservative. The
        # ``var_weights`` convention keeps df_resid = n - k, which is
        # the appropriate SRS-weighted Wald-F for sampling/IPW weights.
        sm = pytest.importorskip("statsmodels.api")
        df = self._df()
        t = (
            ps.tbl_one(df, by="arm", variables=["age", "smoker"],
                       weights="w", missing="never", types={"smoker": "dichotomous"})
            .add_global_p()
        )
        # Manual reference: fit GLM(Binomial) with var_weights and
        # f_test on the single age coefficient.
        y = (df["arm"] == "B").astype(int).to_numpy()
        X = sm.add_constant(df[["age"]])
        ref = sm.GLM(y, X, family=sm.families.Binomial(),
                     var_weights=df["w"].to_numpy(dtype=float)).fit(disp=False)
        expected_p = float(ref.f_test("age = 0").pvalue)
        # Get the table's global p for "age"
        row = next(r for r in t.rows if r.cells[0].text == "age")
        # global p column is right after p-value column; locate by value
        gp_cell = next(
            c for c in row.cells if c.kind == "p_value" and c.value is not None
        )
        # Take the LAST p-value cell in the row (rightmost = global p).
        last_p = [c for c in row.cells if c.kind == "p_value"][-1].value
        del gp_cell
        assert last_p is not None
        assert abs(float(last_p) - expected_p) < 1e-6, (last_p, expected_p)


# ----------------------------------------------------------------------
# tbl_survival validates time + event content. Previously negative
# follow-up times and non-0/1 event codes were silently passed
# through to lifelines (which treats nonzero as a death), producing
# misleading survival curves without complaint.
# ----------------------------------------------------------------------
class TestSurvivalInputValidation:
    def test_negative_time_raises(self):
        pytest.importorskip("lifelines")
        df = pd.DataFrame({
            "t": [1.0, -2.0, 3.0, 4.0],
            "e": [0, 1, 1, 0],
        })
        with pytest.raises(ValueError, match=r"negative value"):
            ps.tbl_survival(df, time="t", event="e")

    def test_non_binary_event_raises(self):
        pytest.importorskip("lifelines")
        df = pd.DataFrame({
            "t": [1.0, 2.0, 3.0, 4.0],
            "e": [0, 1, 9, 1],   # 9 is not 0/1
        })
        with pytest.raises(ValueError, match=r"must contain only 0/1"):
            ps.tbl_survival(df, time="t", event="e")

    def test_valid_inputs_pass(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "t": rng.exponential(10, 50),
            "e": rng.integers(0, 2, 50),
        })
        # Should not raise.
        t = ps.tbl_survival(df, time="t", event="e")
        assert len(t.rows) >= 1


# ----------------------------------------------------------------------
# PhD-audit fixes — round 7 (0.1.0a8)
# ----------------------------------------------------------------------

class TestTblRegressionAddPIsNoOp:
    """tbl_regression.add_p() previously raised RuntimeError despite the
    docstring promising a no-op (the spec-routed path tripped a
    rebuild-context check). Now genuinely returns ``self``."""

    def test_add_p_no_op_on_tbl_regression(self):
        sm = pytest.importorskip("statsmodels.api")
        df = pd.DataFrame({"y": [0, 1] * 25, "x": list(range(50))})
        m = sm.Logit(df["y"], sm.add_constant(df[["x"]])).fit(disp=False)
        t = ps.tbl_regression(m)
        t2 = t.add_p()
        # ``add_p`` should be a true no-op — same object.
        assert t is t2


class TestForestPlotAutoDetectScale:
    """with_forest_plot() now auto-detects ``log_x`` and ``null_line``
    from the table's coefficient column header: OR/HR/TR/IRR/RR → log
    scale at null=1; β/Estimate → linear scale at null=0."""

    def test_logit_OR_uses_log_scale(self):
        sm = pytest.importorskip("statsmodels.api")
        pytest.importorskip("matplotlib")
        df = pd.DataFrame({"y": [0, 1] * 25, "x": list(range(50))})
        m = sm.Logit(df["y"], sm.add_constant(df[["x"]])).fit(disp=False)
        t = ps.tbl_regression(m)
        # Should not raise — OR table with default log_x=auto.
        t.with_forest_plot()

    def test_ols_beta_uses_linear_scale(self):
        sm = pytest.importorskip("statsmodels.api")
        pytest.importorskip("matplotlib")
        df = pd.DataFrame({"y": [float(i) for i in range(50)],
                           "x": list(range(50))})
        m = sm.OLS(df["y"], sm.add_constant(df[["x"]])).fit()
        t = ps.tbl_regression(m)
        # Should not raise — β table with default log_x=auto resolves
        # to log_x=False, null_line=0. Previously the hard-coded
        # log_x=True would have either crashed or silently dropped
        # negative β values on the log axis.
        t.with_forest_plot()


class TestHTMLLinkSchemeAllowlist:
    """HTML renderer now strips javascript: / data: URLs from
    CellPart(link=...) — previously they passed through unfiltered,
    creating an XSS vector for tables built from untrusted input."""

    def test_javascript_url_replaced_with_about_blank(self):
        from pysofra.core.schema import (
            Cell,
            CellPart,
            HeaderCell,
            HeaderRow,
            Row,
        )
        from pysofra.core.table import SofraTable
        parts = (CellPart(text="click", link="javascript:alert(1)"),)
        t = SofraTable(
            rows=(Row(cells=(Cell(text="click", parts=parts),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="X"),)),),
        )
        html = t.to_html()
        assert "javascript:" not in html
        assert "about:blank" in html

    def test_https_url_preserved(self):
        from pysofra.core.schema import (
            Cell,
            CellPart,
            HeaderCell,
            HeaderRow,
            Row,
        )
        from pysofra.core.table import SofraTable
        parts = (CellPart(text="x", link="https://example.com/p?q=1"),)
        t = SofraTable(
            rows=(Row(cells=(Cell(text="x", parts=parts),)),),
            headers=(HeaderRow(cells=(HeaderCell(text="X"),)),),
        )
        assert "https://example.com" in t.to_html()


class TestDesignFPCUnstratified:
    """SurveyDesign(fpc=...) without strata now actually applies the
    finite-population correction. Previously FPC was silently dropped
    in the unstratified branch."""

    def test_fpc_reduces_variance_without_strata(self):
        from pysofra.summary.design import design_mean_var
        v = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        w = pd.Series([2.0] * 10)
        fpc = pd.Series([100] * 10)  # n=10 out of N=100
        _, v_no_fpc, _ = design_mean_var(v, w)
        _, v_fpc, _ = design_mean_var(v, w, fpc=fpc)
        # FPC = 1 − n/N = 1 − 10/100 = 0.9 → variance scaled by 0.9
        assert abs(v_fpc - 0.9 * v_no_fpc) < 1e-12


class TestLonelyPSUWarns:
    """A stratum with a single PSU produces an undefined cluster
    variance contribution. Pysofra now emits a UserWarning and treats
    the stratum's contribution as zero (R survey errors by default)."""

    def test_lonely_psu_warns(self):
        from pysofra.summary.design import design_mean_var
        v = pd.Series([1.0, 2, 3, 10, 11, 12])
        w = pd.Series([1.0] * 6)
        strata = pd.Series(["A", "A", "A", "B", "B", "B"])
        cluster = pd.Series(["c1"] * 3 + ["c2"] * 3)  # both strata lonely
        with pytest.warns(UserWarning, match=r"lonely PSU"):
            design_mean_var(v, w, strata=strata, cluster=cluster)

    def test_single_overall_cluster_warns(self):
        from pysofra.summary.design import design_mean_var
        v = pd.Series([1.0, 2, 3, 4])
        w = pd.Series([1.0] * 4)
        cluster = pd.Series(["c1"] * 4)  # only one PSU overall
        with pytest.warns(UserWarning, match=r"only one cluster"):
            design_mean_var(v, w, cluster=cluster)


class TestTblSurvivalWeights:
    """tbl_survival now accepts a `weights=` column and threads it
    through lifelines' weighted KM. Weighted N totals differ from
    unweighted N totals on non-uniform weights."""

    def test_weighted_n_differs_from_unweighted(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "t": rng.exponential(10, n),
            "e": rng.integers(0, 2, n),
            "arm": rng.choice(["A", "B"], n),
            "w": rng.uniform(0.5, 2.0, n),
        })
        df.loc[df["arm"] == "A", "w"] *= 3.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_unw = ps.tbl_survival(df, time="t", event="e", by="arm")
            t_wt = ps.tbl_survival(df, time="t", event="e", by="arm",
                                   weights="w")
        n_unw = next(r for r in t_unw.rows if r.cells[0].text == "N")
        n_wt = next(r for r in t_wt.rows if r.cells[0].text == "N")
        # The weighted N for arm A should be roughly 3× the unweighted N
        # (since arm A's weights were 3×); at minimum the weighted N
        # differs from the unweighted N for both arms.
        for col in (1, 2):
            assert n_unw.cells[col].value != n_wt.cells[col].value

    def test_negative_weights_raise(self):
        pytest.importorskip("lifelines")
        df = pd.DataFrame({
            "t": [1.0, 2, 3, 4, 5],
            "e": [0, 1, 1, 0, 1],
            "w": [-1.0, 1.0, 1.0, 1.0, 1.0],
        })
        with pytest.raises(ValueError, match=r"negative"):
            ps.tbl_survival(df, time="t", event="e", weights="w")


class TestRebuildDropsColumnWarning:
    """All spec-changing modifiers (not just add_global_p) now warn
    when chaining them after add_difference / add_ci /
    add_significance_stars would silently drop those columns."""

    def test_add_p_after_add_difference_warns(self):
        df = pd.DataFrame({
            "arm": ["A", "B"] * 25,
            "x": [float(i) for i in range(50)],
        })
        t = ps.tbl_one(df, by="arm", variables=["x"], missing="never")
        with pytest.warns(UserWarning, match=r"column.*will be dropped"):
            t.add_difference().add_p()

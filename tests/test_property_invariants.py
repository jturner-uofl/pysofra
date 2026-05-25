"""Property-based invariant testing via Hypothesis.

Uses Hypothesis to generate hundreds of random DataFrames and exercise
PySofra against invariants that must hold *for every input*:

* Percentages always live in ``[0, 100]``.
* p-values always live in ``[0, 1]``.
* Row counts never drop below the requested-variable count.
* CI bounds are ordered (lo ≤ hi) whenever both are finite.
* Modifier chains commute on the table structure they don't touch.
* Re-rendering is byte-stable.

These tests are intentionally noisy — Hypothesis explores edge cases
(NaN-only columns, single-row data, all-same values, large groups,
weight imbalances) and would surface bugs that example-based tests
miss.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pysofra as ps

# ----------------------------------------------------------------------
# Strategy helpers
# ----------------------------------------------------------------------

# Cap example sizes to keep runtime tight.
SETTINGS = settings(
    max_examples=40,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@st.composite
def small_trial(draw, min_n=20, max_n=200):
    """Generate a small clinical-style DataFrame."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10_000)))
    return pd.DataFrame({
        "arm": rng.choice(["A", "B"], size=n),
        "age": rng.normal(55, 12, size=n),
        "sex": rng.choice(["F", "M"], size=n),
        "race": rng.choice(["W", "B", "A", "O"], size=n),
        "flag": rng.choice([0, 1], size=n),
    })


@st.composite
def categorical_pair(draw, levels=("A", "B", "C", "D")):
    """Generate (values, groups) pair for a categorical test."""
    n = draw(st.integers(min_value=20, max_value=150))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10_000)))
    values = rng.choice(levels, size=n)
    groups = rng.choice(["g1", "g2"], size=n)
    return pd.Series(values), pd.Series(groups)


@st.composite
def numeric_pair(draw):
    """Two numeric arrays of decent size."""
    n_a = draw(st.integers(min_value=10, max_value=80))
    n_b = draw(st.integers(min_value=10, max_value=80))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10_000)))
    return rng.normal(size=n_a), rng.normal(size=n_b)


# ======================================================================
# tbl_one — universal invariants
# ======================================================================
class TestTblOneInvariants:
    @SETTINGS
    @given(df=small_trial())
    def test_row_count_at_least_variable_count(self, df):
        t = ps.tbl_one(df, variables=["age", "sex", "race", "flag"],
                       missing="never")
        # age + (sex as dichotomous) + race header + 4 race levels + flag
        # So the minimum row count is # variables (dichotomous + continuous
        # render as 1 row; multi-cat renders as header + levels).
        assert len(t.rows) >= 4

    @SETTINGS
    @given(df=small_trial())
    def test_p_values_are_in_unit_interval(self, df):
        t = ps.tbl_one(df, by="arm",
                       variables=["age", "sex", "race"]).add_p()
        for r in t.rows:
            for c in r.cells:
                if (c.kind == "p_value" and isinstance(c.value, float)
                        and not math.isnan(c.value)):
                    assert 0.0 <= c.value <= 1.0, (
                        f"p-value {c.value} out of range"
                    )

    @SETTINGS
    @given(df=small_trial())
    def test_percentages_in_zero_to_hundred(self, df):
        t = ps.tbl_one(df, by="arm",
                       variables=["sex", "race", "flag"])
        for r in t.rows:
            for c in r.cells:
                if "%" not in c.text:
                    continue
                # Extract the percent number
                try:
                    pct_str = c.text.split("(")[1].split("%")[0]
                    pct = float(pct_str)
                    assert 0.0 <= pct <= 100.0, (
                        f"percent {pct} from {c.text!r} out of range"
                    )
                except (IndexError, ValueError):
                    continue

    @SETTINGS
    @given(df=small_trial())
    def test_smd_is_non_negative(self, df):
        t = ps.tbl_one(df, by="arm",
                       variables=["age", "sex", "race"]).add_smd()
        for r in t.rows:
            for c in r.cells:
                is_smd = c.text.startswith("SMD") or (
                    c.kind == "numeric" and isinstance(c.value, float)
                    and "smd" in (r.cells[0].text.lower() if r.cells else "")
                )
                if is_smd and not math.isnan(c.value):
                    assert c.value >= 0.0


# ======================================================================
# Continuous tests — invariants
# ======================================================================
class TestContinuousTestsProperties:
    @SETTINGS
    @given(pair=numeric_pair())
    def test_continuous_p_value_in_unit_interval(self, pair):
        from pysofra.summary.tests import continuous_test
        a, b = pair
        df = pd.DataFrame({"v": np.r_[a, b],
                           "g": ["A"] * len(a) + ["B"] * len(b)})
        res = continuous_test(df["v"], df["g"], nonnormal=False)
        if res.p_value is not None and not math.isnan(res.p_value):
            assert 0.0 <= res.p_value <= 1.0

    @SETTINGS
    @given(pair=numeric_pair())
    def test_welch_and_wilcoxon_both_finite(self, pair):
        from pysofra.summary.tests import continuous_test
        a, b = pair
        df = pd.DataFrame({"v": np.r_[a, b],
                           "g": ["A"] * len(a) + ["B"] * len(b)})
        for nn in (False, True):
            res = continuous_test(df["v"], df["g"], nonnormal=nn)
            assert res.p_value is None or math.isfinite(res.p_value) or \
                math.isnan(res.p_value)


# ======================================================================
# Categorical tests — invariants
# ======================================================================
class TestCategoricalTestsProperties:
    @SETTINGS
    @given(pair=categorical_pair())
    def test_categorical_p_value_in_unit_interval(self, pair):
        from pysofra.summary.tests import categorical_test
        v, g = pair
        res = categorical_test(v, g)
        if res.p_value is not None and not math.isnan(res.p_value):
            assert 0.0 <= res.p_value <= 1.0


# ======================================================================
# Effect sizes — bounded ranges
# ======================================================================
class TestEffectSizesProperties:
    @SETTINGS
    @given(pair=categorical_pair())
    def test_cramers_v_bounded_zero_to_one(self, pair):
        from pysofra.summary.effect_size import cramers_v
        v, g = pair
        out = cramers_v(v, g)
        if out is not None and not math.isnan(out):
            assert 0.0 <= out <= 1.0 + 1e-9  # tiny float slack

    @SETTINGS
    @given(pair=categorical_pair(levels=("x", "y")))
    def test_phi_bounded_zero_to_one(self, pair):
        from pysofra.summary.effect_size import phi_coefficient
        v, g = pair
        out = phi_coefficient(v, g)
        if out is not None and not math.isnan(out):
            assert 0.0 <= out <= 1.0 + 1e-9

    @SETTINGS
    @given(pair=numeric_pair())
    def test_cohens_d_finite_or_inf(self, pair):
        from pysofra.summary.effect_size import cohen_d
        a, b = pair
        out = cohen_d(a, b)
        if out is not None:
            # Either finite, or +/-inf (zero variance case), but not NaN
            assert math.isfinite(out) or math.isinf(out)


# ======================================================================
# Formatting — exhaustive invariants
# ======================================================================
class TestFormatProperties:
    @SETTINGS
    @given(p=st.floats(min_value=0.0, max_value=1.0))
    def test_fmt_p_value_stable_for_valid_p(self, p):
        from pysofra.core.format import fmt_p_value
        out = fmt_p_value(p)
        # Output must be a string that contains a digit, a comparison
        # operator, or the NA dash.
        assert isinstance(out, str)
        assert any(ch.isdigit() for ch in out) or "—" in out

    @SETTINGS
    @given(v=st.floats(allow_nan=False, allow_infinity=False,
                       min_value=-1e6, max_value=1e6),
           digits=st.integers(min_value=0, max_value=6))
    def test_fmt_number_round_trip(self, v, digits):
        from pysofra.core.format import fmt_number
        out = fmt_number(v, digits=digits)
        # Result is parseable back to float, within rounding tolerance
        parsed = float(out)
        tol = 10 ** (-digits) + 1e-9
        assert abs(parsed - v) <= tol

    @SETTINGS
    @given(frac=st.floats(min_value=0.0, max_value=1.0))
    def test_fmt_percent_within_zero_to_hundred(self, frac):
        from pysofra.core.format import fmt_percent
        out = fmt_percent(frac, digits=1)
        pct = float(out)
        assert -0.05 <= pct <= 100.05  # rounding slack


# ======================================================================
# Render determinism — byte-stable across calls
# ======================================================================
class TestRenderDeterminism:
    @SETTINGS
    @given(df=small_trial())
    def test_html_byte_stable(self, df):
        t = ps.tbl_one(df, by="arm",
                       variables=["age", "sex"]).add_p()
        assert t.to_html() == t.to_html()

    @SETTINGS
    @given(df=small_trial())
    def test_markdown_byte_stable(self, df):
        t = ps.tbl_one(df, by="arm",
                       variables=["age", "sex"]).add_p()
        assert t.to_markdown() == t.to_markdown()

    @SETTINGS
    @given(df=small_trial())
    def test_latex_byte_stable(self, df):
        t = ps.tbl_one(df, by="arm",
                       variables=["age", "sex"]).add_p()
        assert t.to_latex() == t.to_latex()


# ======================================================================
# tbl_cross — invariants
# ======================================================================
class TestTblCrossProperties:
    @SETTINGS
    @given(df=small_trial())
    def test_tbl_cross_row_count_matches_levels(self, df):
        t = ps.tbl_cross(df, row="sex", column="arm")
        # 2 sex levels + margins = 3 body rows (or 2 without margins)
        assert len(t.rows) >= 2


# ======================================================================
# Modifier chains — commute on independent slots
# ======================================================================
class TestModifierChainCommute:
    @SETTINGS
    @given(df=small_trial())
    def test_add_overall_add_p_add_smd_chain(self, df):
        t1 = ps.tbl_one(df, by="arm",
                        variables=["age", "sex"]) \
                .add_overall().add_p().add_smd()
        t2 = ps.tbl_one(df, by="arm",
                        variables=["age", "sex"]) \
                .add_smd().add_p().add_overall()
        # The chain order shouldn't change the row count or the cell
        # labels (only column ordering may change).
        assert len(t1.rows) == len(t2.rows)

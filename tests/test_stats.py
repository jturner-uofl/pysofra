"""Unit tests for summary statistic computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysofra.summary.smd import (
    categorical_smd,
    categorical_smd_pair,
    continuous_smd,
    continuous_smd_pair,
)
from pysofra.summary.stats import categorical_stats, continuous_stats
from pysofra.summary.tests import categorical_test, continuous_test
from pysofra.summary.typing import infer_kind


class TestContinuousStats:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        st = continuous_stats(s)
        assert st.n == 5
        assert st.n_missing == 0
        assert st.mean == pytest.approx(3.0)
        # sample SD of 1..5 is sqrt(2.5)
        assert st.sd == pytest.approx(np.sqrt(2.5))
        assert st.median == 3.0
        assert st.min == 1.0
        assert st.max == 5.0

    def test_with_missing(self):
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        st = continuous_stats(s)
        assert st.n == 3
        assert st.n_missing == 2
        assert st.mean == pytest.approx(3.0)

    def test_all_missing(self):
        s = pd.Series([np.nan, np.nan])
        st = continuous_stats(s)
        assert st.n == 0
        assert st.n_missing == 2
        assert np.isnan(st.mean)


class TestCategoricalStats:
    def test_basic(self):
        s = pd.Series(["a", "b", "a", "c", "b", "a"])
        st = categorical_stats(s)
        assert st.n == 6
        assert st.counts == {"a": 3, "b": 2, "c": 1}
        assert st.levels == ("a", "b", "c")

    def test_explicit_levels_aligns(self):
        s = pd.Series(["a", "a", "b"])
        st = categorical_stats(s, levels=["a", "b", "c"])
        assert st.counts == {"a": 2, "b": 1, "c": 0}


class TestTypingInference:
    def test_bool(self):
        assert infer_kind(pd.Series([True, False, True])) == "dichotomous"

    def test_zero_one_ints(self):
        assert infer_kind(pd.Series([0, 1, 0, 1, 1])) == "dichotomous"

    def test_few_low_int_levels(self):
        # 3 distinct small ints → categorical
        assert infer_kind(pd.Series([0, 1, 2, 0, 1, 2])) == "categorical"

    def test_continuous_age(self):
        assert infer_kind(pd.Series([55, 62, 47, 68, 51, 75, 23, 90])) == "continuous"

    def test_string_two_levels(self):
        assert infer_kind(pd.Series(["M", "F", "M"])) == "dichotomous"

    def test_string_multi(self):
        assert infer_kind(pd.Series(["a", "b", "c"])) == "categorical"

    def test_ordered_categorical(self):
        s = pd.Series(pd.Categorical(["low", "high", "mid"],
                                     categories=["low", "mid", "high"], ordered=True))
        assert infer_kind(s) == "ordinal"


class TestContinuousTest:
    def test_two_group_ttest(self, small_trial):
        res = continuous_test(small_trial["age"], small_trial["arm"])
        assert res.p_value is not None
        assert "t-test" in res.test

    def test_two_group_nonnormal(self, small_trial):
        res = continuous_test(small_trial["age"], small_trial["arm"], nonnormal=True)
        assert res.p_value is not None
        assert "Wilcoxon" in res.test

    def test_three_group_anova(self, trial_with_three_arms):
        res = continuous_test(trial_with_three_arms["age"], trial_with_three_arms["arm"])
        assert res.p_value is not None
        assert "ANOVA" in res.test

    def test_three_group_kruskal(self, trial_with_three_arms):
        res = continuous_test(trial_with_three_arms["age"],
                              trial_with_three_arms["arm"], nonnormal=True)
        assert res.p_value is not None
        assert "Kruskal" in res.test


class TestCategoricalTest:
    def test_two_by_two_fisher(self):
        s = pd.Series(["F"] * 10 + ["M"] * 10)
        g = pd.Series(["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5)
        res = categorical_test(s, g)
        assert res.test == "Fisher's exact"
        assert res.p_value is not None

    def test_multi_by_multi_chisq(self, small_trial):
        res = categorical_test(small_trial["race"], small_trial["arm"])
        assert "chi-square" in res.test.lower() or "chi" in res.test.lower()


class TestSMD:
    def test_continuous_smd_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smd = continuous_smd_pair(a, a.copy())
        assert smd == pytest.approx(0.0)

    def test_continuous_smd_positive(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        smd = continuous_smd_pair(a, b)
        assert smd is not None and smd > 5.0  # very large effect

    def test_continuous_smd_function(self, small_trial):
        smd = continuous_smd(small_trial["age"], small_trial["arm"])
        assert smd is None or smd >= 0.0

    def test_categorical_smd_function(self, small_trial):
        smd = categorical_smd(small_trial["sex"], small_trial["arm"])
        assert smd is None or smd >= 0.0

    def test_categorical_smd_identical(self):
        # Genuinely identical category mass per group: both X and Y
        # see 5 "a" + 5 "b" (the old test paired v with g position-wise,
        # which produced complete separation — exactly the singular
        # case we now flag as infinite).
        v = pd.Series(["a"] * 5 + ["b"] * 5 + ["a"] * 5 + ["b"] * 5)
        g = pd.Series(["X"] * 10 + ["Y"] * 10)
        smd = categorical_smd(v, g)
        assert smd is not None and smd < 1e-6

    def test_categorical_smd_complete_separation_two_level(self):
        # Reviewer regression: group 1 is all "A", group 2 is all "B".
        # The average within-group covariance is the zero matrix, so the
        # Mahalanobis form is undefined. We must NOT silently return 0.
        v = pd.Series(["A", "A", "B", "B"])
        g = pd.Series([0, 0, 1, 1])
        smd = categorical_smd(v, g)
        assert smd is not None and not np.isfinite(smd)

    def test_categorical_smd_complete_separation_pair(self):
        # Same edge case at the lower-level pair API: K=2 categories,
        # complete separation.
        smd = categorical_smd_pair(np.array([1.0, 0.0]),
                                   np.array([0.0, 1.0]))
        assert smd is not None and not np.isfinite(smd)

    def test_categorical_smd_pair_truly_identical_returns_zero(self):
        # Sanity check the other branch of the new guard: identical
        # degenerate proportions must still report 0, not inf.
        smd = categorical_smd_pair(np.array([1.0, 0.0]),
                                   np.array([1.0, 0.0]))
        assert smd == pytest.approx(0.0)

    def test_categorical_smd_complete_separation_with_empty_level(self):
        # Sparse padding levels must not mask the bug either.
        v = pd.Series(["A", "A", "B", "B"])
        g = pd.Series([0, 0, 1, 1])
        smd = categorical_smd(v, g, levels=["A", "B", "C"])
        assert smd is not None and not np.isfinite(smd)

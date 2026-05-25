"""Tests for per-variable test overrides and the named-test registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps
from pysofra.summary.tests import available_tests, run_named_test


class TestNamedTestRegistry:
    def test_registry_keys(self):
        avail = available_tests()
        assert "wilcoxon" in avail["continuous"]
        assert "fisher" in avail["categorical"]
        assert "anova" in avail["continuous"]
        assert "kruskal" in avail["continuous"]

    def test_run_wilcoxon(self, small_trial):
        res = run_named_test("wilcoxon", small_trial["age"],
                             small_trial["arm"], kind="continuous")
        assert "Wilcoxon" in res.test
        assert res.p_value is not None

    def test_run_anova_three_groups(self, trial_with_three_arms):
        res = run_named_test("anova", trial_with_three_arms["age"],
                             trial_with_three_arms["arm"], kind="continuous")
        assert "ANOVA" in res.test

    def test_run_fisher_on_2x2(self):
        v = pd.Series(["F"] * 10 + ["M"] * 10)
        g = pd.Series(["A", "B"] * 10)
        res = run_named_test("fisher", v, g, kind="categorical")
        assert res.test == "Fisher's exact"

    def test_unknown_continuous_test_raises(self, small_trial):
        with pytest.raises(ValueError):
            run_named_test("not_a_test", small_trial["age"],
                           small_trial["arm"], kind="continuous")

    def test_unknown_categorical_test_raises(self, small_trial):
        with pytest.raises(ValueError):
            run_named_test("not_a_test", small_trial["sex"],
                           small_trial["arm"], kind="categorical")

    def test_unknown_kind_raises(self, small_trial):
        with pytest.raises(ValueError):
            run_named_test("welch", small_trial["age"],
                           small_trial["arm"], kind="ordinal")


class TestTblOneOverrides:
    def test_per_variable_override(self, small_trial):
        t = (
            ps.tbl_one(small_trial, by="arm",
                       tests={"age": "wilcoxon", "race": "fisher"})
              .add_p()
        )
        footnote = " ".join(t.footnotes)
        assert "Wilcoxon" in footnote
        # Default for bmi is still Welch
        assert "Welch" in footnote

    def test_overrides_dont_affect_other_vars(self, small_trial):
        # Overriding 'age' to wilcoxon shouldn't change bmi (which stays Welch)
        t = ps.tbl_one(small_trial, by="arm",
                       tests={"age": "wilcoxon"}).add_p()
        # Both tests should be present in metadata
        assert "Wilcoxon rank-sum" in t.metadata["tests"]
        assert "Welch's t-test" in t.metadata["tests"]


class TestQValues:
    def test_q_column_added(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),
            "x4": rng.normal(0, 1, n),
        })
        # Shift x1 between groups to guarantee a real signal
        df.loc[df.arm == "B", "x1"] += 1.5
        t = ps.tbl_one(df, by="arm").add_p().add_q()
        # One more column than without q
        ncols = len(t.headers[0].cells)
        assert ncols == 5  # Characteristic, A, B, p, q
        assert t.headers[0].cells[-1].text == "q-value"

    def test_q_values_are_monotone_w_p(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),
        })
        df.loc[df.arm == "B", "x1"] += 2.0  # very strong effect
        t = ps.tbl_one(df, by="arm").add_p().add_q()
        # Find all (p, q) pairs
        pq_pairs = []
        for r in t.rows:
            p_cell = next((c for c in r.cells if c.kind == "p_value"
                          and isinstance(c.value, (int, float))), None)
            q_cell = next((c for c in r.cells if c.kind == "q_value"
                          and isinstance(c.value, (int, float))), None)
            if p_cell and q_cell:
                pq_pairs.append((p_cell.value, q_cell.value))
        # BH-adjusted q-values are always >= p-values
        for p, q in pq_pairs:
            assert q >= p - 1e-9

    def test_add_q_footnote(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").add_p().add_q(method="fdr_bh")
        assert any("Benjamini" in f for f in t.footnotes)

    def test_bonferroni(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").add_p().add_q(method="bonferroni")
        assert any("Bonferroni" in f for f in t.footnotes)

    def test_add_q_without_explicit_add_p(self, small_trial):
        # add_q should implicitly enable p-values
        t = ps.tbl_one(small_trial, by="arm").add_q()
        # Both columns present
        labels = [c.text for c in t.headers[0].cells]
        assert "p-value" in labels
        assert "q-value" in labels

"""Tests for ``tbl_uvregression`` factor expansion.

Categorical predictors are dummy-encoded (first level = reference) and
rendered as a group-header row + one indented row per level — matching
``gtsummary::tbl_uvregression``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def mixed_predictors():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(55, 10, n),
        "race": rng.choice(["W", "B", "A", "O"], n),
        "sex": rng.choice(["F", "M"], n),
        "smoker": rng.choice([False, True], n),
    })
    df["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(
        0.04 * df["age"] - 0.5 * (df["sex"] == "M")
    )))).astype(int)
    return df


class TestFactorExpansion:
    def test_4_level_categorical_emits_header_plus_levels(self, mixed_predictors):
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True, predictors=["race"],
        )
        labels = [r.cells[0].text for r in t.rows]
        # 1 group header ('race') + 4 level rows (one of them the ref)
        assert labels[0] == "race"
        for lvl in ("A", "B", "O", "W"):
            assert lvl in labels, f"missing level row {lvl!r}"
        assert len(t.rows) == 5

    def test_reference_row_marked_with_em_dash(self, mixed_predictors):
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True, predictors=["race"],
        )
        # The reference row should have "— ref" in its estimate cell
        ref_row = next(r for r in t.rows
                       if "— ref" in r.cells[2].text)
        assert ref_row is not None

    def test_boolean_predictor_renders_yes_no_labels(self, mixed_predictors):
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True, predictors=["smoker"],
        )
        labels = [r.cells[0].text for r in t.rows]
        assert "smoker" in labels
        assert "No" in labels and "Yes" in labels

    def test_numeric_predictor_still_single_row(self, mixed_predictors):
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True, predictors=["age"],
        )
        # Numeric predictor: exactly one body row, no group header.
        assert len(t.rows) == 1
        assert t.rows[0].cells[0].text == "age"

    def test_level_n_counts_correct(self, mixed_predictors):
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True, predictors=["sex"],
        )
        # 'M' level should report count == number of M rows
        n_m = int((mixed_predictors["sex"] == "M").sum())
        m_row = next(r for r in t.rows if r.cells[0].text == "M")
        assert int(m_row.cells[1].text) == n_m

    def test_adjust_for_supports_categorical(self, mixed_predictors):
        pytest.importorskip("statsmodels.api")
        # Adjust for a categorical confounder while testing a numeric predictor.
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True, predictors=["age"],
            adjust_for=["race"],
        )
        # Output should still report just 'age' (one row), but the
        # estimate is now adjusted for race dummies.
        assert len(t.rows) == 1
        # Footnote names the adjustment
        assert any("adjusted for" in fn.lower() for fn in t.footnotes)

    def test_mixed_predictor_table_has_expected_row_count(self, mixed_predictors):
        # 1 numeric + 1 cat(4) + 1 cat(2) + 1 bool ⇒ 1 + 5 + 3 + 3 = 12 rows
        pytest.importorskip("statsmodels.api")
        t = ps.tbl_uvregression(
            mixed_predictors, outcome="y", method="Logit",
            exponentiate=True,
            predictors=["age", "race", "sex", "smoker"],
        )
        assert len(t.rows) == 12

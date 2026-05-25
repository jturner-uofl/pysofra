"""Tests for tbl_merge() and tbl_stack()."""

from __future__ import annotations

import pytest

import pysofra as ps


class TestMerge:
    def test_merge_two_tbl_one(self, small_trial):
        # Restrict to variables with stable row counts so the merge succeeds
        # regardless of which sex subset is non-empty.
        vars_ = ["age", "smoker"]
        t1 = ps.tbl_one(small_trial[small_trial["sex"] == "F"], by="arm",
                        variables=vars_, missing="never")
        t2 = ps.tbl_one(small_trial[small_trial["sex"] == "M"], by="arm",
                        variables=vars_, missing="never")
        merged = ps.tbl_merge([t1, t2], tab_spanners=["Female", "Male"])
        labels = [s.label for s in merged.spanning_headers]
        assert "Female" in labels and "Male" in labels

    def test_merge_drops_duplicate_label_column(self, small_trial):
        vars_ = ["age", "smoker"]
        t1 = ps.tbl_one(small_trial, by="arm", variables=vars_, missing="never")
        t2 = ps.tbl_one(small_trial, by="arm", variables=vars_, missing="never")
        merged = ps.tbl_merge([t1, t2])
        n_t1_cols = len(t1.headers[0].cells)
        n_t2_cols = len(t2.headers[0].cells)
        # label col dropped from t2
        assert len(merged.headers[0].cells) == n_t1_cols + n_t2_cols - 1

    def test_merge_requires_equal_rows(self, small_trial):
        # Slices keep both arms (small_trial = 30 "A" + 30 "B"). Without
        # the interleave we'd silently hit the single-level-by
        # warning gate.
        even = small_trial.iloc[::2]
        odd = small_trial.iloc[1::2]
        t1 = ps.tbl_one(even, by="arm", variables=["age", "sex"])
        t2 = ps.tbl_one(odd, by="arm", variables=["age", "sex", "bmi"])
        with pytest.raises(ValueError):
            ps.tbl_merge([t1, t2])


class TestStack:
    def test_stack_basic(self, small_trial):
        even = small_trial.iloc[::2]
        odd = small_trial.iloc[1::2]
        t1 = ps.tbl_one(even, by="arm", variables=["age", "sex"])
        t2 = ps.tbl_one(odd, by="arm", variables=["age", "sex"])
        stacked = ps.tbl_stack([t1, t2], group_labels=["Cohort 1", "Cohort 2"])
        labels = [r.cells[0].text for r in stacked.rows if r.is_group_header]
        assert "Cohort 1" in labels and "Cohort 2" in labels

    def test_stack_mismatched_cols_raises(self, small_trial):
        t1 = ps.tbl_one(small_trial, by="arm")
        t2 = ps.tbl_one(small_trial)  # no by → different col count
        with pytest.raises(ValueError):
            ps.tbl_stack([t1, t2])

"""Tests for tbl_one()."""

from __future__ import annotations

import pandas as pd
import pytest

import pysofra as ps


class TestTblOneBasics:
    def test_no_by(self, small_trial):
        t = ps.tbl_one(small_trial.drop(columns=["arm"]))
        assert len(t.rows) > 0
        # one label column + one "Overall" column
        assert len(t.headers[0].cells) == 2

    def test_with_by(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm")
        # label + 2 group columns
        assert len(t.headers[0].cells) == 3

    def test_add_p_adds_column(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").add_p()
        assert len(t.headers[0].cells) == 4
        assert "p" in t.headers[0].cells[-1].text.lower()

    def test_add_smd_adds_column(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").add_smd()
        assert len(t.headers[0].cells) == 4
        assert t.headers[0].cells[-1].text == "SMD"

    def test_add_overall_adds_column(self, small_trial):
        base = ps.tbl_one(small_trial, by="arm")
        t = base.add_overall()
        assert len(t.headers[0].cells) == len(base.headers[0].cells) + 1

    def test_chain_all(self, small_trial):
        t = (
            ps.tbl_one(small_trial, by="arm")
              .add_p()
              .add_smd()
              .add_overall()
              .theme("clinical")
        )
        n = len(small_trial["arm"].unique())
        # label + overall + n_groups + p + smd
        assert len(t.headers[0].cells) == 1 + 1 + n + 2

    def test_labels_used(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm", labels={"age": "Age (years)"})
        first_col_texts = [r.cells[0].text for r in t.rows]
        assert any("Age (years)" in s for s in first_col_texts)

    def test_types_override_forces_continuous(self):
        # smoker is 0/1 (dichotomous by default) — force continuous
        df = pd.DataFrame({"x": [0, 1, 0, 1, 1, 0], "g": ["A"] * 3 + ["B"] * 3})
        t = ps.tbl_one(df, by="g", types={"x": "continuous"})
        # Continuous rows are not "x = 1" prefix; should be just label
        labels = [r.cells[0].text for r in t.rows]
        assert "x = 1" not in labels
        assert "x" in labels

    def test_unknown_column_raises(self, small_trial):
        with pytest.raises(KeyError):
            ps.tbl_one(small_trial, by="not_a_column")

    def test_missing_row_appears_when_present(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm", missing="always")
        labels = [r.cells[0].text for r in t.rows]
        assert "Missing" in labels


class TestTblOneRendering:
    def test_html(self, small_trial):
        h = ps.tbl_one(small_trial, by="arm").add_p().to_html()
        assert "<table" in h and "</table>" in h
        assert "p-value" in h

    def test_markdown(self, small_trial):
        m = ps.tbl_one(small_trial, by="arm").to_markdown()
        assert m.startswith("|")
        assert "---" in m

    def test_repr_html(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm").add_p()
        h = t._repr_html_()
        assert "pysofra-wrap" in h

    def test_docx(self, small_trial, tmp_path):
        out = tmp_path / "t.docx"
        ps.tbl_one(small_trial, by="arm").add_p().to_docx(out)
        assert out.exists()
        assert out.stat().st_size > 1000


class TestTblOneNonnormal:
    def test_nonnormal_changes_summary(self, small_trial):
        t = ps.tbl_one(small_trial, by="arm", nonnormal=["age"])
        # Find the age row — should now contain median (Q1, Q3) format with commas
        for r in t.rows:
            if r.cells[0].text == "age":
                txt = r.cells[1].text
                # median (Q1, Q3) has two commas inside the parens
                assert txt.count(",") >= 1
                break
        else:
            pytest.fail("age row not found")

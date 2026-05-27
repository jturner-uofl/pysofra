"""Unit tests for the 0.1.0a10 additions:

* snapshot lock (.snapshot_hash / .lock_snapshot / .assert_snapshot)
* publication-safety checker (.check_safety / .with_safety_warnings)
* Quarto export (.to_quarto)
* Typst renderer (.to_typst / .to_typst_file)
* command-line interface (pysofra.cli)
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "arm": rng.choice(["A", "B"], 200),
        "age": rng.normal(50, 10, 200).round(1),
        "sex": rng.choice(["M", "F"], 200),
        "bmi": rng.normal(27, 4, 200).round(1),
    })


@pytest.fixture
def basic_table(small_df) -> ps.SofraTable:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ps.tbl_one(small_df, by="arm",
                          variables=["age", "sex", "bmi"], missing="never")


# ----------------------------------------------------------------------
# Snapshot lock
# ----------------------------------------------------------------------

class TestSnapshotLock:
    def test_snapshot_hash_is_deterministic(self, basic_table):
        h1 = basic_table.snapshot_hash()
        h2 = basic_table.snapshot_hash()
        assert h1 == h2
        # sha256 = 64 hex chars
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_snapshot_hash_ignores_theme(self, basic_table):
        # Theme changes are presentational; hash must not move.
        h1 = basic_table.snapshot_hash()
        themed = basic_table.theme("jama") if hasattr(basic_table, "theme") else basic_table
        assert themed.snapshot_hash() == h1

    def test_snapshot_hash_changes_with_data(self, small_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = ps.tbl_one(small_df, by="arm",
                            variables=["age"], missing="never")
            df_mut = small_df.copy()
            df_mut.loc[0, "age"] = 999
            t2 = ps.tbl_one(df_mut, by="arm",
                            variables=["age"], missing="never")
        assert t1.snapshot_hash() != t2.snapshot_hash()

    def test_lock_and_assert_roundtrip(self, basic_table, tmp_path):
        lock = tmp_path / "table.lock"
        basic_table.lock_snapshot(lock)
        assert lock.exists()
        payload = json.loads(lock.read_text())
        assert payload["schema_version"] == 1
        assert payload["sha256"] == basic_table.snapshot_hash()
        # Round-trip
        basic_table.assert_snapshot(lock)  # must not raise

    def test_assert_raises_on_drift(self, small_df, tmp_path):
        lock = tmp_path / "table.lock"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = ps.tbl_one(small_df, by="arm",
                            variables=["age"], missing="never")
        t1.lock_snapshot(lock)
        # Mutate data → different table → assert raises
        df_mut = small_df.copy()
        df_mut.loc[0:5, "age"] = 999
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t2 = ps.tbl_one(df_mut, by="arm",
                            variables=["age"], missing="never")
        with pytest.raises(AssertionError, match="Snapshot mismatch"):
            t2.assert_snapshot(lock)

    def test_assert_raises_filenotfound(self, basic_table, tmp_path):
        missing = tmp_path / "nope.lock"
        with pytest.raises(FileNotFoundError):
            basic_table.assert_snapshot(missing)

    def test_assert_diff_message_includes_change(self, small_df, tmp_path):
        lock = tmp_path / "table.lock"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps.tbl_one(small_df, by="arm",
                       variables=["age"], missing="never").lock_snapshot(lock)
            t2 = ps.tbl_one(small_df.assign(age=lambda d: d.age + 100),
                            by="arm", variables=["age"], missing="never")
        with pytest.raises(AssertionError) as exc_info:
            t2.assert_snapshot(lock)
        assert "Diff" in str(exc_info.value)


# ----------------------------------------------------------------------
# Publication-safety checker
# ----------------------------------------------------------------------

class TestSafetyChecker:
    def test_clean_table_no_warnings(self, basic_table):
        assert basic_table.check_safety() == []

    def test_extreme_proportion_flagged(self):
        rng = np.random.default_rng(0)
        # 100% YES in a column with n > 30
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], 100),
            "outcome": [1] * 100,
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm", variables=["outcome"],
                           missing="never")
        warns = t.check_safety()
        codes = {w.code for w in warns}
        assert "extreme_proportion" in codes

    def test_sd_exceeds_mean_flagged(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], 100),
            "wild": rng.normal(0.5, 50.0, 100),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm", variables=["wild"],
                           missing="never")
        warns = t.check_safety()
        assert any(w.code == "sd_exceeds_mean" for w in warns)

    def test_dominant_missing_flagged(self):
        df = pd.DataFrame({
            "arm": ["A"] * 60 + ["B"] * 60,
            "x": [None] * 100 + list(range(20)),  # 100/120 ≈ 83% missing
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm", variables=["x"])
        warns = t.check_safety()
        assert any(w.code == "dominant_missing" for w in warns)

    def test_with_safety_warnings_appends_footnotes(self, small_df):
        df = small_df.assign(outcome=lambda d: [1] * len(d))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ps.tbl_one(df, by="arm", variables=["outcome"],
                           missing="never").with_safety_warnings()
        joined = " ".join(t.footnotes)
        assert "SAFETY" in joined and "extreme_proportion" in joined

    def test_with_safety_warnings_noop_on_clean(self, basic_table):
        same = basic_table.with_safety_warnings()
        assert same.footnotes == basic_table.footnotes


# ----------------------------------------------------------------------
# Quarto export
# ----------------------------------------------------------------------

class TestQuartoExport:
    def test_bare_html_block(self, basic_table):
        out = basic_table.to_quarto(format="html")
        assert out.startswith("::: {=html}")
        assert out.rstrip().endswith(":::")
        assert "<table" in out  # actual HTML body present

    def test_bare_latex_block(self, basic_table):
        out = basic_table.to_quarto(format="latex")
        assert out.startswith("::: {=latex}")
        assert "\\begin{table}" in out or "\\begin{tabular}" in out

    def test_invalid_format_raises(self, basic_table):
        with pytest.raises(ValueError, match="format must be"):
            basic_table.to_quarto(format="pdf")

    def test_label_and_caption_wrap_block(self, basic_table):
        out = basic_table.to_quarto(format="html",
                                     label="tbl-baseline",
                                     caption="My table.")
        assert "::: {#tbl-baseline}" in out
        assert "My table." in out

    def test_non_tbl_label_warns(self, basic_table):
        with pytest.warns(UserWarning, match="should start with 'tbl-'"):
            basic_table.to_quarto(format="html", label="my-table")


# ----------------------------------------------------------------------
# Typst renderer
# ----------------------------------------------------------------------

class TestTypstRenderer:
    def test_render_produces_table_block(self, basic_table):
        out = basic_table.to_typst()
        assert "#table(" in out
        assert "table.header(" in out
        assert "columns:" in out

    def test_column_count_matches(self, basic_table):
        out = basic_table.to_typst()
        # basic_table has 4 columns: Characteristic, A, B (no p / smd here)
        assert "columns: 3" in out

    def test_special_chars_escaped(self):
        # A label containing $ and # should be escaped
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.DataFrame({"arm": ["A", "B"] * 10,
                               "rate": list(range(20))})
            t = ps.tbl_one(df, by="arm", variables=["rate"],
                           labels={"rate": "Rate ($/min)"},
                           missing="never")
        out = t.to_typst()
        # The $ in "Rate ($/min)" must be escaped
        assert "\\$" in out

    def test_write_to_file(self, basic_table, tmp_path):
        out_path = tmp_path / "table.typ"
        result = basic_table.to_typst_file(out_path)
        assert result == out_path
        assert out_path.exists()
        text = out_path.read_text()
        assert "#table(" in text

    def test_footnotes_preserved(self, basic_table):
        out = basic_table.to_typst()
        # tbl_one always adds at least the Mean (SD) footnote
        assert "Mean" in out or "_" in out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

class TestCLI:
    def _run(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pysofra.cli", *args],
            capture_output=True, text=True, check=False, **kwargs,
        )

    def test_version_command(self):
        r = self._run("version")
        assert r.returncode == 0
        assert ps.__version__ in r.stdout

    def test_help_command(self):
        r = self._run("--help")
        assert r.returncode == 0
        assert "table" in r.stdout
        assert "check" in r.stdout
        assert "version" in r.stdout

    def test_table_to_stdout_markdown(self, small_df, tmp_path):
        csv = tmp_path / "data.csv"
        small_df.to_csv(csv, index=False)
        r = self._run("table", str(csv), "--by", "arm",
                       "--vars", "age,sex,bmi", "--missing", "never")
        assert r.returncode == 0, r.stderr
        assert "| Characteristic |" in r.stdout
        assert "age" in r.stdout

    def test_table_writes_html(self, small_df, tmp_path):
        csv = tmp_path / "data.csv"
        out = tmp_path / "table.html"
        small_df.to_csv(csv, index=False)
        r = self._run("table", str(csv), "--by", "arm",
                       "--vars", "age", "--missing", "never",
                       "--out", str(out))
        assert r.returncode == 0, r.stderr
        assert out.exists()
        assert "<table" in out.read_text()

    def test_check_clean_returns_0(self, small_df, tmp_path):
        csv = tmp_path / "data.csv"
        small_df.to_csv(csv, index=False)
        r = self._run("check", str(csv), "--by", "arm",
                       "--vars", "age,sex,bmi", "--missing", "never")
        assert r.returncode == 0, r.stderr
        assert "OK" in r.stdout

    def test_check_dirty_returns_2(self, tmp_path):
        # 100% outcome → extreme_proportion → exit 2
        df = pd.DataFrame({
            "arm": ["A"] * 60 + ["B"] * 60,
            "outcome": [1] * 120,
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        r = self._run("check", str(csv), "--by", "arm",
                       "--vars", "outcome", "--missing", "never")
        assert r.returncode == 2
        assert "extreme_proportion" in r.stderr

    def test_unrecognised_extension_fails(self, small_df, tmp_path):
        bad = tmp_path / "data.xyz"
        bad.write_text("garbage")
        r = self._run("table", str(bad), "--by", "arm")
        assert r.returncode != 0

    def test_unrecognised_output_extension_fails(self, small_df, tmp_path):
        csv = tmp_path / "data.csv"
        out = tmp_path / "table.unknownfmt"
        small_df.to_csv(csv, index=False)
        r = self._run("table", str(csv), "--by", "arm",
                       "--vars", "age", "--missing", "never",
                       "--out", str(out))
        assert r.returncode != 0

"""Tests for tbl_survival()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def surv_df():
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(2026)
    n = 250
    df = pd.DataFrame({
        "arm":   rng.choice(["A", "B"], n),
        "time":  rng.exponential(scale=24, size=n),
        "event": rng.integers(0, 2, n),
    })
    return df


class TestTblSurvival:
    def test_basic_columns(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event", by="arm")
        # Statistic | A | B | p-value
        assert len(t.headers[0].cells) == 4
        assert t.headers[0].cells[0].text == "Statistic"
        assert t.headers[0].cells[-1].text == "p-value"

    def test_overall_only(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event")
        # No p-value column without `by`
        assert len(t.headers[0].cells) == 2
        assert t.headers[0].cells[1].text == "Overall"

    def test_includes_n_events_censored_median(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event", by="arm")
        labels = [r.cells[0].text for r in t.rows]
        assert "N" in labels
        assert "Events" in labels
        assert "Censored" in labels
        assert any(lab.startswith("Median survival") for lab in labels)

    def test_fixed_time_rows(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event", by="arm",
                            times=[6, 12], times_label="mo")
        labels = [r.cells[0].text for r in t.rows]
        assert any("S(6 mo)" in lab for lab in labels)
        assert any("S(12 mo)" in lab for lab in labels)
        # Each S(t) row's value should contain a % and an n=
        for r in t.rows:
            if r.cells[0].text.startswith("S("):
                # Skip the p-value column
                for c in r.cells[1:-1]:
                    assert "%" in c.text
                    assert "n=" in c.text

    def test_logrank_p_in_metadata(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event", by="arm")
        assert "logrank_p" in t.metadata
        assert isinstance(t.metadata["logrank_p"], float)

    def test_no_logrank_without_by(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event")
        assert t.metadata["logrank_p"] is None or "p-value" not in [
            c.text for c in t.headers[0].cells
        ]

    def test_html_renders(self, surv_df):
        t = ps.tbl_survival(surv_df, time="time", event="event", by="arm",
                            times=[12])
        html = t.to_html()
        assert "<table" in html
        assert "Median survival" in html

    def test_unknown_column_raises(self, surv_df):
        with pytest.raises(KeyError):
            ps.tbl_survival(surv_df, time="nope", event="event")

    def test_polars_input(self, surv_df):
        pl = pytest.importorskip("polars")
        pl_df = pl.from_pandas(surv_df)
        t = ps.tbl_survival(pl_df, time="time", event="event", by="arm")
        assert len(t.rows) >= 4

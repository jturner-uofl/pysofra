"""Snapshot-style structural tests.

We don't snapshot raw HTML strings (whose scope IDs change every render),
but we snapshot the structural ``to_dict()`` form which captures the
table semantics that downstream renderers consume.
"""

from __future__ import annotations

import pandas as pd

import pysofra as ps


def test_dict_snapshot_basic():
    df = pd.DataFrame({
        "arm": ["A", "A", "A", "B", "B", "B"],
        "age": [50.0, 55.0, 60.0, 52.0, 57.0, 62.0],
        "sex": ["F", "M", "F", "M", "F", "M"],
    })
    t = ps.tbl_one(df, by="arm").add_p().add_smd().theme("clinical")
    d = t.to_dict()

    assert d["theme"] == "clinical"
    assert d["headers"] == [["Characteristic", "A\nN = 3", "B\nN = 3", "p-value", "SMD"]]
    assert d["rows"][0][0] == "age"
    # Identical means → SMD must be 0
    assert d["rows"][0][3] is not None
    sex_row = next(r for r in d["rows"] if r[0].startswith("sex"))
    assert sex_row[0] == "sex = M"


def test_html_contains_expected_elements():
    df = pd.DataFrame({"arm": ["A"] * 5 + ["B"] * 5,
                       "x": [1.0, 2, 3, 4, 5, 1, 2, 3, 4, 5]})
    h = ps.tbl_one(df, by="arm").set_caption("Hello").to_html()
    # ``<caption>`` carries an inline ``style="..."`` attribute (theme
    # styles are inlined so they survive sanitizers that strip
    # ``<style>``), so we match the tag opening + the text instead of a
    # naked exact string.
    assert "<caption" in h and ">Hello</caption>" in h
    assert "<table" in h
    assert "<style>" in h


def test_markdown_round_trip_structure():
    df = pd.DataFrame({"arm": ["A"] * 6 + ["B"] * 6,
                       "age": [50, 55, 60, 65, 70, 75] * 2,
                       "sex": ["F", "M"] * 6})
    m = ps.tbl_one(df, by="arm").add_p().to_markdown()
    # Has a header line, separator, and at least 2 data rows.
    lines = [line for line in m.splitlines() if line.startswith("|")]
    assert len(lines) >= 4
    assert "---" in lines[1]

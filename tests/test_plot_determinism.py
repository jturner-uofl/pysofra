"""Determinism tests for plot-embedded tables.

The renderer-consistency suite (`test_renderer_consistency.py`)
verifies determinism for plain text-only tables. This file extends
that guarantee to tables that have an inline matplotlib plot
(`with_forest_plot`, `with_km_plot`) — historically the source of
non-deterministic SVG / PNG / PDF bytes via matplotlib's wall-clock
timestamps and process-random hash salt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


# ----------------------------------------------------------------------
# KM plot determinism
# ----------------------------------------------------------------------
class TestKMPlotDeterminism:
    def _build(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "arm": rng.choice(["A", "B"], 100),
            "time": rng.exponential(20, 100),
            "event": rng.integers(0, 2, 100),
        })
        return ps.tbl_survival(df, time="time", event="event",
                               by="arm").with_km_plot(
            risk_times=[5, 10, 20])

    def test_svg_byte_equal(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("lifelines")
        assert self._build().inline_plot.svg == \
               self._build().inline_plot.svg

    def test_png_byte_equal(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("lifelines")
        assert self._build().inline_plot.png_bytes == \
               self._build().inline_plot.png_bytes

    def test_pdf_byte_equal(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("lifelines")
        assert self._build().inline_plot.pdf_bytes == \
               self._build().inline_plot.pdf_bytes

    def test_html_byte_equal(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("lifelines")
        # End-to-end: the full HTML render including the embedded SVG
        # is byte-identical across calls.
        assert self._build().to_html() == self._build().to_html()


# ----------------------------------------------------------------------
# Forest plot determinism
# ----------------------------------------------------------------------
class TestForestPlotDeterminism:
    def _build(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({"x": rng.normal(size=n),
                           "z": rng.normal(size=n)})
        df["y"] = (rng.uniform(size=n) <
                   1 / (1 + np.exp(-(df["x"] + 0.3 * df["z"])))).astype(int)
        m = sm.Logit(df["y"], sm.add_constant(df[["x", "z"]])).fit(disp=False)
        return ps.tbl_regression(m, intercept=False,
                                 exponentiate=True).with_forest_plot()

    def test_svg_byte_equal(self):
        pytest.importorskip("matplotlib")
        assert self._build().inline_plot.svg == \
               self._build().inline_plot.svg

    def test_png_byte_equal(self):
        pytest.importorskip("matplotlib")
        assert self._build().inline_plot.png_bytes == \
               self._build().inline_plot.png_bytes

    def test_pdf_byte_equal(self):
        pytest.importorskip("matplotlib")
        assert self._build().inline_plot.pdf_bytes == \
               self._build().inline_plot.pdf_bytes


# ----------------------------------------------------------------------
# Direct probe of the matplotlib post-processors
# ----------------------------------------------------------------------
class TestStrippers:
    def test_strip_svg_removes_dc_date(self):
        from pysofra.plot.inline import _strip_svg_nondeterminism
        before = b"<dc:date>2026-05-21T07:39:19.123456</dc:date>"
        after = _strip_svg_nondeterminism(before)
        # Date is replaced with the fixed sentinel, not removed.
        assert b"2026-05-21T07:39:19" not in after
        assert b"<dc:date>" in after

    def test_strip_pdf_removes_creationdate(self):
        from pysofra.plot.inline import _strip_pdf_nondeterminism
        before = b"junk /CreationDate (D:20260521073919Z) more /ID [<DEAD> <BEEF>] tail"
        after = _strip_pdf_nondeterminism(before)
        assert b"D:20260521073919Z" not in after
        assert b"D:20260101000000Z" in after
        # /ID is also rewritten deterministically
        assert b"<DEAD>" not in after

    def test_strip_png_drops_time_chunk(self):
        from pysofra.plot.inline import _strip_png_nondeterminism
        # Minimal valid PNG signature + a tIME chunk + an IEND chunk
        sig = b"\x89PNG\r\n\x1a\n"
        time_chunk = (
            b"\x00\x00\x00\x07tIME"
            b"\x07\xea\x05\x15\x07\x27\x13"
            b"\xde\xad\xbe\xef"  # CRC stub
        )
        iend = b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"
        before = sig + time_chunk + iend
        after = _strip_png_nondeterminism(before)
        assert b"tIME" not in after
        assert after.endswith(iend)

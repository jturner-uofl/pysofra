"""Unit tests for formatting helpers."""

from __future__ import annotations

import math

from pysofra.core.format import (
    NA_STRING,
    fmt_ci,
    fmt_estimate_ci,
    fmt_mean_sd,
    fmt_median_iqr,
    fmt_n_pct,
    fmt_number,
    fmt_p_value,
    fmt_percent,
    fmt_smd,
)


class TestFmtNumber:
    def test_basic(self):
        assert fmt_number(3.14159, 2) == "3.14"

    def test_zero_digits(self):
        assert fmt_number(2.7, 0) == "3"

    def test_none(self):
        assert fmt_number(None) == NA_STRING

    def test_nan(self):
        assert fmt_number(float("nan")) == NA_STRING


class TestFmtPValue:
    def test_small(self):
        assert fmt_p_value(0.0001) == "<0.001"

    def test_large(self):
        assert fmt_p_value(0.991) == ">0.99"

    def test_typical(self):
        assert fmt_p_value(0.0234) == "0.023"

    def test_none(self):
        assert fmt_p_value(None) == NA_STRING

    def test_nan(self):
        assert fmt_p_value(float("nan")) == NA_STRING


class TestFmtNPct:
    def test_basic(self):
        assert fmt_n_pct(7, 28) == "7 (25.0%)"

    def test_zero_total(self):
        assert fmt_n_pct(0, 0).startswith("0 (")


class TestFmtMeanSdMedianIQR:
    def test_mean_sd(self):
        assert fmt_mean_sd(10.0, 2.0) == "10.00 (2.00)"

    def test_median_iqr(self):
        assert fmt_median_iqr(5.0, 3.0, 8.0, digits=1) == "5.0 (3.0, 8.0)"


class TestFmtCI:
    def test_ci(self):
        assert fmt_ci(1.2, 4.8) == "1.20, 4.80"

    def test_estimate_ci(self):
        assert fmt_estimate_ci(2.5, 1.2, 4.8, digits=1) == "2.5 (1.2, 4.8)"


class TestFmtSmd:
    def test_basic(self):
        assert fmt_smd(0.123) == "0.123"

    def test_none(self):
        assert fmt_smd(None) == NA_STRING


class TestFmtPercent:
    def test_basic(self):
        assert fmt_percent(0.234, digits=1) == "23.4"

    def test_nan(self):
        assert fmt_percent(math.nan) == NA_STRING

"""Formatting helpers for numbers, p-values, percents, and confidence intervals.

These are deterministic, locale-agnostic, and unit-testable. All rounding
uses banker's-rounding-free conventional half-up via Python's ``format`` mini
language so output matches what most statistical journals expect.
"""

from __future__ import annotations

import math
from typing import Final

NA_STRING: Final[str] = "—"


def fmt_number(value: float | int | None, digits: int = 2) -> str:
    """Format a numeric value to ``digits`` decimal places.

    ``None``, ``NaN``, infinite, and anything that can't be coerced to a
    ``float`` render as :data:`NA_STRING`. Both IEEE-754 negative zero
    AND small negative numbers that round to all-zero at ``digits``
    precision are normalised so cells never display as ``"-0.00"``
    (which is confusing and uninformative).
    """
    if value is None:
        return NA_STRING
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NA_STRING
    if math.isnan(v) or math.isinf(v):
        return NA_STRING
    out = f"{v:.{digits}f}"
    # If the formatted result is a "negative zero" representation —
    # leading minus on a string of zeros and a decimal point — drop
    # the sign. Covers both IEEE -0.0 (renders as "-0.00") and small
    # negative inputs that round to zero at this precision (e.g.
    # -0.001 at 2dp renders as "-0.00"). The information loss is
    # already in the round-to-2dp step; preserving the sign on what
    # the reader sees as zero would be misleading.
    if out.startswith("-") and set(out[1:]) <= {"0", "."}:
        out = out[1:]
    return out


def fmt_int(value: float | int | None) -> str:
    if value is None:
        return NA_STRING
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NA_STRING
    if math.isnan(v) or math.isinf(v):
        return NA_STRING
    return f"{int(round(v))}"


def fmt_percent(value: float | None, digits: int = 1) -> str:
    """Format a fraction (0–1) as a percent. Pass 0.234 → '23.4'.

    Negative-zero output ("-0.0") is normalised to "0.0" for the same
    reason :func:`fmt_number` does (it's confusing in publication tables).
    """
    if value is None:
        return NA_STRING
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return NA_STRING
    out = f"{100.0 * float(value):.{digits}f}"
    if out.startswith("-") and set(out[1:]) <= {"0", "."}:
        out = out[1:]
    return out


def fmt_n_pct(n: int, total: int, digits: int = 1) -> str:
    """Render ``n (xx.x%)``. If ``total`` is zero, returns ``n (—)``."""
    if total <= 0:
        return f"{int(n)} ({NA_STRING})"
    pct = 100.0 * n / total
    return f"{int(n)} ({pct:.{digits}f}%)"


def fmt_mean_sd(mean: float | None, sd: float | None, digits: int = 2) -> str:
    """Render ``mean (sd)`` in journal style."""
    return f"{fmt_number(mean, digits)} ({fmt_number(sd, digits)})"


def fmt_median_iqr(
    median: float | None,
    q1: float | None,
    q3: float | None,
    digits: int = 2,
) -> str:
    """Render ``median (Q1, Q3)``."""
    return f"{fmt_number(median, digits)} ({fmt_number(q1, digits)}, {fmt_number(q3, digits)})"


def fmt_range(lo: float | None, hi: float | None, digits: int = 2) -> str:
    return f"{fmt_number(lo, digits)}, {fmt_number(hi, digits)}"


def fmt_ci(
    lo: float | None,
    hi: float | None,
    digits: int = 2,
    *,
    sep: str = ", ",
) -> str:
    """Render a confidence interval as ``lo, hi``."""
    return f"{fmt_number(lo, digits)}{sep}{fmt_number(hi, digits)}"


def fmt_estimate_ci(
    estimate: float | None,
    lo: float | None,
    hi: float | None,
    digits: int = 2,
) -> str:
    """Render ``estimate (lo, hi)``."""
    return f"{fmt_number(estimate, digits)} ({fmt_ci(lo, hi, digits)})"


def fmt_p_value(p: float | None, digits: int = 3) -> str:
    """Journal-style p-value formatting.

    Rules:
      * ``None`` / ``NaN`` / infinite        → :data:`NA_STRING`
      * out-of-range (``p < 0`` or ``p > 1``) → :data:`NA_STRING`
        (silently coercing an invalid p-value would mask a real bug in
        the upstream computation)
      * ``p < 10^-digits``                    → ``"<0.001"`` (for ``digits=3``)
      * ``p > 0.99``                          → ``">0.99"``
      * otherwise                             → ``"0.xxx"``
    """
    if p is None:
        return NA_STRING
    if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
        return NA_STRING
    p = float(p)
    if p < 0.0 or p > 1.0:
        return NA_STRING
    threshold = 10 ** (-digits)
    if p < threshold:
        return f"<{threshold:.{digits}f}"
    if p > 0.99:
        return ">0.99"
    return f"{p:.{digits}f}"


def fmt_smd(smd: float | None, digits: int = 3) -> str:
    """Format a standardized mean difference. Always signed magnitude."""
    if smd is None:
        return NA_STRING
    if isinstance(smd, float) and (math.isnan(smd) or math.isinf(smd)):
        return NA_STRING
    return f"{float(smd):.{digits}f}"

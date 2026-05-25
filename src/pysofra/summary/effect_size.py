"""Effect-size helpers — Cohen's d, Hedges' g, Cramér's V, eta-squared.

Companion functions to the inferential tests in
:mod:`pysofra.summary.tests`. Effect sizes describe the *magnitude* of
a difference / association independently of sample size, and are
frequently requested alongside the p-values in clinical reports.

All functions accept aligned :class:`pandas.Series` for ``values`` /
``groups``; missing values are dropped pairwise. They return floats
(or ``None`` for degenerate input).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Continuous
# ----------------------------------------------------------------------

def cohen_d(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    """Cohen's d using the pooled standard deviation.

    Parameters
    ----------
    a, b
        Two independent samples (``pandas.Series`` or 1-D ``numpy``
        array). Non-numeric entries are coerced; ``NaN`` rows are
        dropped per array. Each sample must contain at least two
        finite values.

    Returns
    -------
    float or None
        ``d = (μ_a − μ_b) / s_pool``, where the pooled SD weights the
        two samples by their degrees of freedom:
        ``s_pool = sqrt(((n_a − 1)·s_a² + (n_b − 1)·s_b²) / (n_a + n_b − 2))``.
        Returns ``None`` if either sample has fewer than 2 finite
        observations. Returns ``0.0`` if the pooled SD is zero and
        the two means are identical; ``inf`` if the pooled SD is zero
        but the means differ (degenerate constant-sample case).

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
      Sciences* (2nd ed.). Lawrence Erlbaum.
    """
    a_arr = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(dtype=float)
    b_arr = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(dtype=float)
    n_a, n_b = a_arr.size, b_arr.size
    if n_a < 2 or n_b < 2:
        return None
    v_a = float(np.var(a_arr, ddof=1))
    v_b = float(np.var(b_arr, ddof=1))
    s_pool = math.sqrt(((n_a - 1) * v_a + (n_b - 1) * v_b) / (n_a + n_b - 2))
    if s_pool == 0:
        return 0.0 if a_arr.mean() == b_arr.mean() else float("inf")
    return (float(a_arr.mean()) - float(b_arr.mean())) / s_pool


def hedges_g(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    """Hedges' g — Cohen's d with the small-sample bias correction.

    ``g = d · J``, where ``J ≈ 1 − 3/(4(n_a+n_b) − 9)`` (Hedges 1981).
    """
    d = cohen_d(a, b)
    if d is None or math.isinf(d):
        return d
    n_a = int(pd.to_numeric(pd.Series(a), errors="coerce").dropna().size)
    n_b = int(pd.to_numeric(pd.Series(b), errors="coerce").dropna().size)
    denom = 4 * (n_a + n_b) - 9
    if denom <= 0:  # pragma: no cover — unreachable given cohen_d's n>=2 guard
        return d
    j = 1.0 - 3.0 / denom
    return d * j


def eta_squared(values: pd.Series, groups: pd.Series) -> float | None:
    """One-way ANOVA effect size: between-group / total sum-of-squares.

    Ranges ``[0, 1]``. Small ≈ 0.01, medium ≈ 0.06, large ≈ 0.14
    (Cohen 1988).
    """
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"),
                       "g": groups}).dropna()
    if df.empty:
        return None
    grand = float(df["v"].mean())
    ss_between = float((df.groupby("g")["v"]
                         .apply(lambda x: x.size * (x.mean() - grand) ** 2))
                        .sum())
    ss_total = float(((df["v"] - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    return ss_between / ss_total


def omega_squared(values: pd.Series, groups: pd.Series) -> float | None:
    """Less-biased counterpart to ``eta_squared`` (Hays 1973)."""
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"),
                       "g": groups}).dropna()
    if df.empty:
        return None
    k = int(df["g"].nunique())
    n = int(df.shape[0])
    if n - k <= 0 or k <= 1:
        return None
    grand = float(df["v"].mean())
    ss_between = float((df.groupby("g")["v"]
                         .apply(lambda x: x.size * (x.mean() - grand) ** 2))
                        .sum())
    ss_total = float(((df["v"] - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ms_within = (ss_total - ss_between) / (n - k)
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return float(max(0.0, omega))


# ----------------------------------------------------------------------
# Categorical
# ----------------------------------------------------------------------

def cramers_v(values: pd.Series, groups: pd.Series) -> float | None:
    """Cramér's V — chi-square effect size normalised to ``[0, 1]``.

    ``V = √(χ² / (N · (min(R, C) − 1)))``.
    """
    import warnings as _w

    from scipy import stats as sp_stats
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df.empty:
        return None
    ctab = pd.crosstab(df["v"], df["g"])
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        return None
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"), \
            _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        chi2, _, _, _ = sp_stats.chi2_contingency(ctab.to_numpy(), correction=False)
    n = float(ctab.values.sum())
    if n <= 0:  # pragma: no cover — guarded by the shape >= 2x2 check above
        return None
    min_dim = min(ctab.shape) - 1
    if min_dim <= 0:  # pragma: no cover — shape >= 2x2 guarantees min_dim >= 1
        return None
    return float(math.sqrt(chi2 / (n * min_dim)))


def phi_coefficient(values: pd.Series, groups: pd.Series) -> float | None:
    """Phi — Cramér's V special case for 2×2 tables; ``φ = √(χ²/N)``."""
    import warnings as _w

    from scipy import stats as sp_stats
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df.empty:
        return None
    ctab = pd.crosstab(df["v"], df["g"])
    if ctab.shape != (2, 2):
        return None
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"), \
            _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        chi2, _, _, _ = sp_stats.chi2_contingency(ctab.to_numpy(), correction=False)
    n = float(ctab.values.sum())
    if n <= 0:  # pragma: no cover — guarded by the (2,2) shape check above
        return None
    return float(math.sqrt(chi2 / n))


# ----------------------------------------------------------------------
# Auto dispatch (mirrors auto-test selection)
# ----------------------------------------------------------------------


def auto_effect_size(values: pd.Series, groups: pd.Series) -> tuple[str, float | None]:
    """Pick a sensible effect size for the variable kind / number of groups.

    Returns ``(name, value)`` so callers can both display the metric and
    label it in a footnote.
    """
    g_unique = pd.Series(groups).dropna().unique()
    n_groups = len(g_unique)

    # Continuous-looking?
    try:
        pd.to_numeric(values, errors="raise")
        continuous = True
    except (ValueError, TypeError):
        continuous = False

    if continuous and n_groups == 2:
        return "Cohen's d", cohen_d(
            values[groups == g_unique[0]],
            values[groups == g_unique[1]],
        )
    if continuous and n_groups >= 3:
        return "η²", eta_squared(values, groups)
    if not continuous and n_groups >= 2:
        ctab = pd.crosstab(values, groups)
        if ctab.shape == (2, 2):
            return "φ", phi_coefficient(values, groups)
        return "Cramér's V", cramers_v(values, groups)
    return "—", None

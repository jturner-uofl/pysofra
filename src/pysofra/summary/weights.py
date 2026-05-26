"""Weighted summary statistics for frequency-weighted Table 1.

These are *frequency* weights — each row carries a non-negative count.
For complex survey designs (cluster sampling, post-stratification),
users should pre-compute weights with a dedicated survey package and
pass them here as a single column.

Weighted statistics implemented:

* mean: ``Σ w_i x_i / Σ w_i``
* variance: unbiased frequency-weighted variance
  ``Σ w_i (x_i - μ)² / (Σ w_i - 1)``
* quantiles: linear-interpolation method on the weighted ECDF
* proportions: ``Σ w_i 1{x_i = level} / Σ w_i``

Weighted contingency tests use Rao–Scott-corrected chi-square, falling
back to a regular chi-square on the weighted observed table when no
design effect is available (which is the case for frequency weights —
the weights *are* the counts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _fsum(arr: np.ndarray) -> float:
    """Kahan-Babuška-Neumaier-equivalent compensated sum.

    Wraps :func:`math.fsum` which provides full-precision (i.e.
    exactly-rounded) summation for ``float`` iterables. The cost is
    O(n) and the benefit is independence from accumulation order —
    important for weighted sums where ``Σ w_i x_i`` and
    ``Σ w_i (x_i − μ)²`` can lose 5–6 digits of precision under
    naïve ``np.sum`` on long arrays with mixed magnitudes (e.g.
    NHANES-scale survey weights that span 4–5 orders of magnitude).
    Empty input returns ``0.0`` (matching ``math.fsum([])``).
    """
    if arr.size == 0:
        return 0.0
    # ``math.fsum`` consumes any iterable of floats; converting to
    # a Python list once is cheaper than tolist() inside a loop.
    return math.fsum(arr.tolist())


@dataclass(frozen=True)
class WeightedContinuousStats:
    n_eff: float       # effective sample size (sum of weights)
    n_missing: float   # weighted count of missing values
    mean: float
    sd: float
    median: float
    q1: float
    q3: float
    min: float
    max: float


def weighted_continuous_stats(
    values: pd.Series,
    weights: pd.Series,
) -> WeightedContinuousStats:
    """Frequency-weighted summary of a continuous variable."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")

    valid = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    v_v = v[valid]
    w_v = w[valid]

    n_missing = _fsum(w[np.isnan(v) & ~np.isnan(w)])
    n_eff = _fsum(w_v)

    if n_eff <= 0 or v_v.size == 0:
        nan = float("nan")
        return WeightedContinuousStats(0.0, n_missing, nan, nan, nan, nan, nan, nan, nan)

    # Compensated weighted mean and variance.  ``math.fsum`` is
    # exactly-rounded, so ``Σ w_i x_i`` and ``Σ w_i (x_i − μ)²`` no
    # longer leak precision under heterogeneous sampling weights.
    mean = _fsum(w_v * v_v) / n_eff
    # Frequency-weighted unbiased variance is undefined when the effective
    # sample size collapses to one (or fewer). NaN propagates through
    # ``fmt_mean_sd`` so the cell shows ``—`` rather than ``(0.00)``.
    var = (
        _fsum(w_v * (v_v - mean) ** 2) / (n_eff - 1)
        if n_eff > 1
        else float("nan")
    )
    sd = float(np.sqrt(max(var, 0.0))) if not np.isnan(var) else float("nan")

    median, q1, q3 = (_weighted_quantile(v_v, w_v, q) for q in (0.5, 0.25, 0.75))

    return WeightedContinuousStats(
        n_eff=n_eff,
        n_missing=n_missing,
        mean=mean,
        sd=sd,
        median=median,
        q1=q1,
        q3=q3,
        min=float(np.min(v_v)),
        max=float(np.max(v_v)),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Linear-interpolation weighted quantile.

    ``q`` is the desired probability level in ``[0, 1]``. The CDF is
    computed at midpoint positions so that the method matches the
    behaviour of NumPy's ``np.quantile(method='linear')`` in the
    equal-weights limit.
    """
    if values.size == 0 or weights.size == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumw = np.cumsum(w)
    total = cumw[-1]
    if total <= 0:
        return float("nan")
    # Position of the q-th quantile in the weighted ECDF.
    target = q * (total - w[0]) + 0.5 * w[0]  # midpoint adjustment
    # Cumulative midpoints.
    midpoints = cumw - 0.5 * w
    return float(np.interp(target, midpoints, v))


@dataclass(frozen=True)
class WeightedCategoricalStats:
    n_eff: float
    n_missing: float
    counts: dict[object, float]
    levels: tuple[object, ...]


def weighted_categorical_stats(
    values: pd.Series,
    weights: pd.Series,
    levels: list[object] | tuple[object, ...] | None = None,
) -> WeightedCategoricalStats:
    """Frequency-weighted counts per level."""
    df = pd.DataFrame({"v": values, "w": pd.to_numeric(weights, errors="coerce")})
    n_missing = float(df.loc[df["v"].isna() & df["w"].notna(), "w"].sum())
    df = df.dropna()
    df = df[df["w"] > 0]

    if levels is None:
        if isinstance(values.dtype, pd.CategoricalDtype):
            level_list = list(values.cat.categories)
        else:
            level_list = sorted(df["v"].unique(), key=_safe_sort_key)
    else:
        level_list = list(levels)

    counts: dict[object, float] = {lvl: 0.0 for lvl in level_list}
    for lvl, sub in df.groupby("v", observed=True):
        counts[lvl] = float(sub["w"].sum())

    n_eff = float(sum(counts.values()))
    return WeightedCategoricalStats(
        n_eff=n_eff,
        n_missing=n_missing,
        counts=counts,
        levels=tuple(level_list),
    )


def _safe_sort_key(x: object) -> tuple[int, float | str]:
    if isinstance(x, bool):
        return (0, float(int(x)))
    if isinstance(x, (int, float)):
        return (0, float(x))
    if isinstance(x, str):
        return (1, x)
    return (2, repr(x))

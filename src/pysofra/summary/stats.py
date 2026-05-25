"""Summary statistic computations for continuous and categorical variables.

Pure functions that take a pandas Series (or groupby slice) and return a
small dataclass of statistics. Format-free — formatting belongs to
:mod:`pysofra.core.format`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContinuousStats:
    n: int
    n_missing: int
    mean: float
    sd: float
    median: float
    q1: float
    q3: float
    min: float
    max: float


def continuous_stats(series: pd.Series) -> ContinuousStats:
    """Compute continuous summary statistics. Tolerates all-NaN slices."""
    s = pd.to_numeric(series, errors="coerce")
    n_total = len(s)
    valid = s.dropna()
    n = int(valid.size)
    n_missing = int(n_total - n)

    if n == 0:
        nan = float("nan")
        return ContinuousStats(0, n_missing, nan, nan, nan, nan, nan, nan, nan)

    arr = valid.to_numpy(dtype=float)
    # When the array contains ``inf``/``-inf``, numpy's ``mean`` / ``std``
    # / ``quantile`` emit ``RuntimeWarning: invalid value encountered in
    # subtract`` (and similar). Under ``filterwarnings = error`` — which
    # this project's own pyproject.toml sets and which is a common
    # user-side ``-W error`` posture — those warnings escalate to
    # exceptions and crash ``tbl_one`` on perfectly legal data. An
    # earlier fix in ``infer_kind`` handled the ``int(np.inf) →
    # OverflowError`` path but did not reach this downstream stats
    # site. Wrap arithmetic in ``np.errstate`` + ``catch_warnings`` so
    # the stats compute cleanly to ``nan`` / ``inf`` (which the
    # formatters then render as em-dash).
    with np.errstate(invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = float(np.mean(arr))
        # Sample SD (ddof=1) is undefined for n=1; we report NaN so renderers
        # can show ``—`` rather than a misleading "0.00".
        sd = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        median = float(np.median(arr))
        q1, q3 = (float(x) for x in np.quantile(arr, [0.25, 0.75]))
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
    return ContinuousStats(
        n=n,
        n_missing=n_missing,
        mean=mean,
        sd=sd,
        median=median,
        q1=q1,
        q3=q3,
        min=arr_min,
        max=arr_max,
    )


@dataclass(frozen=True)
class CategoricalStats:
    n: int
    n_missing: int
    counts: dict[object, int]  # ordered by level
    levels: tuple[object, ...]


def categorical_stats(
    series: pd.Series,
    levels: list[object] | tuple[object, ...] | None = None,
) -> CategoricalStats:
    """Compute counts per level.

    If ``levels`` is provided, levels missing from the series are included
    with count 0 (so that grouped tables align across strata).
    """
    s = series
    n_missing = int(s.isna().sum())
    valid = s.dropna()

    if levels is None:
        if isinstance(s.dtype, pd.CategoricalDtype):
            level_list: list[object] = list(s.cat.categories)
        else:
            level_list = sorted(valid.unique(), key=_safe_sort_key)
    else:
        level_list = list(levels)

    counts: dict[object, int] = {lvl: 0 for lvl in level_list}
    vc = valid.value_counts(dropna=False)
    for lvl, c in vc.items():
        if lvl in counts:
            counts[lvl] = int(c)
        else:
            # Out-of-spec level (only when caller passed explicit levels).
            counts[lvl] = int(c)
            level_list.append(lvl)

    return CategoricalStats(
        n=int(valid.size),
        n_missing=n_missing,
        counts=counts,
        levels=tuple(level_list),
    )


def _safe_sort_key(x: object) -> tuple[int, object]:
    """Sort key that puts numerics first, then strings, then everything else.

    Avoids ``TypeError`` from mixed-type uniques (e.g. ``[1, "a"]``).
    """
    if isinstance(x, bool):
        return (0, int(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        return (0, float(x))
    if isinstance(x, str):
        return (1, x)
    return (2, repr(x))

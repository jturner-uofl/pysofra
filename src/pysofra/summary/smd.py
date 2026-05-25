"""Standardized mean differences (SMDs) across groups.

For two groups, the SMD is computed as |mean_1 - mean_2| / sd_pool for
continuous variables and as the multivariate categorical SMD for factors.

For three or more groups we report the *maximum pairwise* SMD, which is
the convention used by ``tableone``. Users who want a different summary
(mean pairwise, single-reference) can post-process the metadata.

References
----------
Yang, D., & Dalton, J. E. (2012). A unified approach to measuring the
effect size between two groups using SAS. SAS Global Forum.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def continuous_smd_pair(a: np.ndarray, b: np.ndarray) -> float | None:
    """SMD between two continuous samples using pooled SD."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return None
    # Same ``inf``-safety wrap as ``continuous_stats``: numpy's mean / var
    # emit ``RuntimeWarning`` on inf-bearing inputs which escalates to
    # an exception under the project's ``filterwarnings = error`` gate.
    # The resulting ``mean = inf`` / ``var = nan`` are handled by the
    # explicit ``sd_pool == 0.0`` and ``ma == mb`` checks below.
    import warnings as _w
    with np.errstate(invalid="ignore", over="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ma, mb = float(np.mean(a)), float(np.mean(b))
        va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        sd_pool = float(np.sqrt((va + vb) / 2.0))
    if sd_pool == 0.0:
        return 0.0 if ma == mb else float("inf")
    return abs(ma - mb) / sd_pool


def categorical_smd_pair(p1: np.ndarray, p2: np.ndarray) -> float | None:
    """Multivariate categorical SMD between two proportion vectors.

    ``p1`` and ``p2`` are proportion vectors over the same K levels.
    Uses the Yang & Dalton (2012) formulation with K-1 dimensions.

    Edge cases. When the (K-1) average covariance ``S`` is the zero
    matrix the multivariate quadratic form is undefined; this happens
    when both groups have all mass on a single (possibly different)
    category, including the *complete-separation* case
    (e.g. group 1 = "A", group 2 = "B" only). Returning the
    Mahalanobis distance via the pseudo-inverse silently yields zero
    in that case, which would report perfect balance — the opposite
    of the truth. We therefore return ``inf`` when the contrast is
    nonzero and the covariance is degenerate, and ``0`` when both
    are zero (groups truly identical).
    """
    if p1.size != p2.size or p1.size < 2:
        return None
    # Use K-1 categories to avoid singular covariance.
    p1 = p1[:-1]
    p2 = p2[:-1]
    diff = p1 - p2
    # Mean covariance matrix S = (S1 + S2) / 2
    s1 = np.diag(p1) - np.outer(p1, p1)
    s2 = np.diag(p2) - np.outer(p2, p2)
    s = (s1 + s2) / 2.0
    # Degenerate covariance: either no variability within either group
    # (each group concentrates on one category) or the K-1 contrast
    # space is empty. ``pinv`` returns a near-zero matrix here, which
    # would falsely report a zero SMD even under complete separation.
    if np.allclose(s, 0.0):
        return 0.0 if np.allclose(diff, 0.0) else float("inf")
    try:
        s_inv = np.linalg.pinv(s)
    except np.linalg.LinAlgError:
        return None
    val = float(diff @ s_inv @ diff)
    if val < 0:  # numerical
        val = 0.0
    return float(np.sqrt(val))


def continuous_smd(values: pd.Series, groups: pd.Series) -> float | None:
    """Maximum pairwise continuous SMD across groups."""
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": groups}).dropna()
    if df.empty:
        return None
    by_group = {g: x["v"].to_numpy() for g, x in df.groupby("g", observed=True)}
    keys = list(by_group)
    if len(keys) < 2:
        return None
    if len(keys) == 2:
        return continuous_smd_pair(by_group[keys[0]], by_group[keys[1]])
    pair_results = [
        continuous_smd_pair(by_group[a], by_group[b])
        for a, b in combinations(keys, 2)
    ]
    pairs_cont: list[float] = [p for p in pair_results if p is not None]
    return max(pairs_cont) if pairs_cont else None


def categorical_smd(
    values: pd.Series,
    groups: pd.Series,
    levels: list[object] | tuple[object, ...] | None = None,
) -> float | None:
    """Maximum pairwise categorical SMD across groups."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df.empty:
        return None
    ctab = pd.crosstab(df["v"], df["g"])
    if levels is not None:
        ctab = ctab.reindex(index=list(levels), fill_value=0)
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        return None
    col_totals = ctab.sum(axis=0).replace(0, np.nan)
    props = (ctab / col_totals).fillna(0.0)
    keys = list(props.columns)
    if len(keys) == 2:
        return categorical_smd_pair(props[keys[0]].to_numpy(), props[keys[1]].to_numpy())
    cat_pair_results = [
        categorical_smd_pair(props[a].to_numpy(), props[b].to_numpy())
        for a, b in combinations(keys, 2)
    ]
    pairs_cat: list[float] = [p for p in cat_pair_results if p is not None]
    return max(pairs_cat) if pairs_cat else None

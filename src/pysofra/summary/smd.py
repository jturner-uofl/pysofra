"""Standardized mean differences (SMDs) across groups.

For two groups, the SMD is computed as |mean_1 - mean_2| / sd_pool for
continuous variables and as the multivariate categorical SMD for factors.

For three or more groups we report the *maximum pairwise* SMD, which is
the convention used by ``tableone``. Users who want a different summary
(mean pairwise, single-reference) can post-process the metadata.

Weighted SMDs. When ``weights=`` is supplied, the per-group means /
variances / proportions are replaced by their weighted counterparts using
the frequency-weight convention (divisor ``Σw − 1`` for variance, matching
``statsmodels.stats.weightstats.DescrStatsW``). On a survey-weighted
``tbl_one`` this is the only way the SMD column can agree with a
weighted analysis done in R's ``survey`` / ``tableone(weights=)``.

References
----------
Yang, D., & Dalton, J. E. (2012). A unified approach to measuring the
effect size between two groups using SAS. SAS Global Forum.
"""

from __future__ import annotations

import warnings as _w
from itertools import combinations

import numpy as np
import pandas as pd


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean + variance with divisor ``Σw − 1``.

    Matches ``statsmodels.stats.weightstats.DescrStatsW`` (with the
    default ddof=1) and R's ``Hmisc::wtd.var(..., type="unbiased")``.

    Notes
    -----
    This divisor is the textbook "frequency-weight" / "unbiased"
    convention. It is also a sensible choice for "reliability" /
    "importance" weights that average near 1.

    For *sampling* weights where ``Σw`` represents an inflated
    population total (e.g. NHANES weights that sum to millions), this
    convention undercounts the residual df by ``Σw − n`` and produces
    an overconfident variance. For such designs, use
    :class:`~pysofra.SurveyDesign` end-to-end (which routes through
    :func:`~pysofra.summary.design.design_mean_var` with proper Taylor
    linearisation, FPC, and cluster handling) rather than passing a
    raw ``weights=`` column to ``tbl_one``.
    """
    sw = float(w.sum())
    if sw <= 1.0:
        return float("nan"), float("nan")
    with np.errstate(invalid="ignore", over="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        mean = float((w * x).sum() / sw)
        var = float((w * (x - mean) ** 2).sum() / (sw - 1.0))
    return mean, var


def continuous_smd_pair(
    a: np.ndarray, b: np.ndarray,
    *,
    wa: np.ndarray | None = None,
    wb: np.ndarray | None = None,
) -> float | None:
    """SMD between two continuous samples using pooled SD.

    When ``wa`` / ``wb`` are supplied, the per-group mean and variance are
    computed using the frequency-weight convention (see
    :func:`_weighted_mean_var`) and the pooled SD is the unweighted
    average of the two within-group variances.
    """
    # Strip NaN from x and the matching weight position.
    mask_a = ~np.isnan(a)
    mask_b = ~np.isnan(b)
    a = a[mask_a]
    b = b[mask_b]
    if wa is not None:
        wa = wa[mask_a]
    if wb is not None:
        wb = wb[mask_b]
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return None

    # Same ``inf``-safety wrap as ``continuous_stats``: numpy's mean / var
    # emit ``RuntimeWarning`` on inf-bearing inputs which escalates to
    # an exception under the project's ``filterwarnings = error`` gate.
    # The resulting ``mean = inf`` / ``var = nan`` are handled by the
    # explicit ``sd_pool == 0.0`` and ``ma == mb`` checks below.
    with np.errstate(invalid="ignore", over="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        if wa is None and wb is None:
            ma, mb = float(np.mean(a)), float(np.mean(b))
            va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        else:
            wa_use = wa if wa is not None else np.ones_like(a, dtype=float)
            wb_use = wb if wb is not None else np.ones_like(b, dtype=float)
            ma, va = _weighted_mean_var(a, wa_use)
            mb, vb = _weighted_mean_var(b, wb_use)
        sd_pool = float(np.sqrt((va + vb) / 2.0))
    if sd_pool == 0.0:
        return 0.0 if ma == mb else float("inf")
    if not np.isfinite(sd_pool) or not (np.isfinite(ma) and np.isfinite(mb)):
        return None
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


def continuous_smd(
    values: pd.Series,
    groups: pd.Series,
    *,
    weights: pd.Series | None = None,
) -> float | None:
    """Maximum pairwise continuous SMD across groups.

    When ``weights`` is supplied it is treated as a frequency weight per
    observation: each per-group mean / variance is weighted accordingly
    (see :func:`continuous_smd_pair`).
    """
    df = pd.DataFrame({
        "v": pd.to_numeric(values, errors="coerce"),
        "g": groups,
    })
    if weights is not None:
        df["w"] = pd.to_numeric(weights, errors="coerce")
        df = df.dropna(subset=["v", "g", "w"])
        df = df[df["w"] > 0]
    else:
        df = df.dropna(subset=["v", "g"])
    if df.empty:
        return None

    by_group_v: dict[object, np.ndarray] = {
        g: x["v"].to_numpy() for g, x in df.groupby("g", observed=True)
    }
    by_group_w: dict[object, np.ndarray] | None
    if weights is not None:
        by_group_w = {
            g: x["w"].to_numpy(dtype=float)
            for g, x in df.groupby("g", observed=True)
        }
    else:
        by_group_w = None
    keys = list(by_group_v)
    if len(keys) < 2:
        return None

    def _pair(a_key: object, b_key: object) -> float | None:
        wa = by_group_w[a_key] if by_group_w is not None else None
        wb = by_group_w[b_key] if by_group_w is not None else None
        return continuous_smd_pair(
            by_group_v[a_key], by_group_v[b_key], wa=wa, wb=wb,
        )

    if len(keys) == 2:
        return _pair(keys[0], keys[1])
    pair_results = [_pair(a, b) for a, b in combinations(keys, 2)]
    pairs_cont: list[float] = [p for p in pair_results if p is not None]
    return max(pairs_cont) if pairs_cont else None


def categorical_smd(
    values: pd.Series,
    groups: pd.Series,
    levels: list[object] | tuple[object, ...] | None = None,
    *,
    weights: pd.Series | None = None,
) -> float | None:
    """Maximum pairwise categorical SMD across groups.

    When ``weights`` is supplied, the per-group, per-level proportions
    used by the Yang–Dalton formula are weighted proportions
    (``Σw·I[level=l] / Σw`` within each group) instead of unweighted
    counts / totals.
    """
    if weights is None:
        df = pd.DataFrame({"v": values, "g": groups}).dropna()
        if df.empty:
            return None
        ctab = pd.crosstab(df["v"], df["g"])
    else:
        df = pd.DataFrame({
            "v": values,
            "g": groups,
            "w": pd.to_numeric(weights, errors="coerce"),
        }).dropna(subset=["v", "g", "w"])
        df = df[df["w"] > 0]
        if df.empty:
            return None
        # Weighted contingency: Σw per (level, group). ``unstack`` accepts
        # only ``int | str | dict | None`` for ``fill_value`` under strict
        # typing; we pass an int 0 and cast afterwards so any zero-Σw
        # cells stay numeric.
        ctab = (
            df.groupby(["v", "g"], observed=True)["w"]
            .sum()
            .unstack(fill_value=0)
            .astype(float)
        )
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

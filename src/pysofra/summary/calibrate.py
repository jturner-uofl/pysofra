"""Survey-weight calibration — post-stratification and raking.

Both algorithms scale the design weights so that the weighted marginal
totals over selected variables match supplied population targets.

* :func:`post_stratify` solves the case where the calibration variables
  partition the sample into a single cross-classification (e.g. age × sex
  cells). The new weight for each row is the design weight multiplied
  by the cell-level ratio of population total to weighted sample total.

* :func:`rake` (a.k.a. iterative proportional fitting) handles the more
  common case where targets are given for several variables marginally
  but not for their joint cross-classification. Weights are scaled one
  variable at a time, repeatedly, until convergence.

Both functions return a new pandas Series of calibrated weights with the
same index as the input. Use the result as the ``weights`` column of a
:class:`SurveyDesign`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def post_stratify(
    data: pd.DataFrame,
    base_weights: pd.Series | str,
    *,
    strata_cols: list[str] | tuple[str, ...],
    targets: Mapping[tuple[object, ...], float] | pd.Series,
) -> pd.Series:
    """Post-stratification calibration over a complete cross-classification.

    Parameters
    ----------
    data
        Source dataframe.
    base_weights
        Either the column name of design weights in ``data`` or a Series
        aligned to ``data.index``.
    strata_cols
        One or more columns whose Cartesian product defines the
        post-strata.
    targets
        Population totals for each stratum. Accepts either:

        * a ``dict``-like keyed by tuples whose length equals
          ``len(strata_cols)`` (e.g. ``{('M', '<50'): 1200, ...}``), or
        * a ``pandas.Series`` indexed by those tuples
          (a ``MultiIndex.Series``).

    Returns
    -------
    pandas.Series
        Calibrated weights, aligned to ``data.index``.

    Raises
    ------
    KeyError
        When a stratum present in the data is missing from ``targets``.
    """
    if isinstance(base_weights, str):
        bw = pd.to_numeric(data[base_weights], errors="coerce").astype(float)
    else:
        bw = pd.to_numeric(base_weights, errors="coerce").astype(float)

    strata_cols = list(strata_cols)
    if not strata_cols:
        raise ValueError("post_stratify requires at least one strata column.")
    key = data[strata_cols].apply(
        lambda row: tuple(row.tolist()) if len(strata_cols) > 1 else row.iloc[0],
        axis=1,
    )
    weighted_totals = bw.groupby(key).sum()

    targets_dict = (
        targets.to_dict()
        if isinstance(targets, pd.Series)
        else dict(targets)
    )
    # Allow scalar keys when there's only one strata column.
    if len(strata_cols) == 1:
        targets_dict = {(k if isinstance(k, tuple) else (k,)): v
                        for k, v in targets_dict.items()}
        key = key.map(lambda x: x if isinstance(x, tuple) else (x,))
        weighted_totals.index = [
            (i if isinstance(i, tuple) else (i,))
            for i in weighted_totals.index
        ]

    missing = [
        k for k in weighted_totals.index
        if (k if isinstance(k, tuple) else (k,)) not in targets_dict
    ]
    if missing:
        raise KeyError(f"post_stratify: targets missing strata {missing[:5]}...")

    scale_map = {
        stratum: float(targets_dict[stratum if isinstance(stratum, tuple) else (stratum,)])
        / float(total)
        if total > 0 else 0.0
        for stratum, total in weighted_totals.items()
    }
    return bw * key.map(scale_map).astype(float)


def rake(
    data: pd.DataFrame,
    base_weights: pd.Series | str,
    *,
    margins: Mapping[str, Mapping[object, float]],
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.Series:
    """Raking (iterative proportional fitting) over marginal targets.

    Parameters
    ----------
    data
        Source dataframe.
    base_weights
        Either the column name in ``data`` or an aligned Series.
    margins
        Mapping of *variable → {level: target_total}*. Each variable's
        targets are summed during one iteration; the algorithm cycles
        through the variables until the weights stabilise.
    max_iter
        Maximum number of full sweeps over ``margins``.
    tol
        Convergence threshold on the largest relative change in any
        weight between iterations.

    Returns
    -------
    pandas.Series
        Calibrated weights aligned to ``data.index``.
    """
    if isinstance(base_weights, str):
        w = pd.to_numeric(data[base_weights], errors="coerce").astype(float)
    else:
        w = pd.to_numeric(base_weights, errors="coerce").astype(float)
    w = w.copy()

    if not margins:
        return w

    # Validate columns / targets up front.
    for col, target_levels in margins.items():
        if col not in data.columns:
            raise KeyError(f"rake: column {col!r} not in data")
        present = set(data[col].dropna().unique())
        missing_targets = present - set(target_levels)
        if missing_targets:
            raise KeyError(
                f"rake: column {col!r} has levels with no target: {sorted(missing_targets)[:5]}"
            )

    for _ in range(max_iter):
        max_rel = 0.0
        for col, target_levels in margins.items():
            for lvl, target in target_levels.items():
                mask = data[col] == lvl
                total = float(w[mask].sum())
                if total <= 0:
                    continue
                factor = float(target) / total
                old = w[mask].copy()
                w.loc[mask] = old * factor
                # Track the worst relative change for convergence.
                if old.sum() > 0:
                    rel = abs((w[mask].sum() - old.sum()) / old.sum())
                    max_rel = max(max_rel, rel)
        if max_rel < tol:
            break

    return w


def design_effect(weights: pd.Series) -> float:
    """Kish's design-effect estimate: ``DEFF ≈ n · Σw² / (Σw)²``.

    A quick QC check after calibration — large DEFF (≫ 1) means the
    weights are highly variable and effective sample size is low.

    Negative weights are not meaningful in a design context (they would
    flip the contribution of a row), so they are excluded from the
    computation. If any are present, a ``UserWarning`` flags how many
    rows were dropped — matching the same behaviour as ``tbl_one(...,
    weights=...)``. Returns ``nan`` when no positive weights remain.
    """
    w_raw = pd.to_numeric(weights, errors="coerce").dropna()
    n_negative = int((w_raw < 0).sum())
    if n_negative:
        import warnings
        warnings.warn(
            f"design_effect: weights column contains {n_negative} negative "
            "value(s); rows with negative weight are excluded from the "
            "design-effect estimate.",
            UserWarning,
            stacklevel=2,
        )
    w = w_raw[w_raw > 0]
    if w.empty:
        return float("nan")
    n = len(w)
    return float(n * (w ** 2).sum() / (w.sum() ** 2))


# silence unused-import lint
_ = np

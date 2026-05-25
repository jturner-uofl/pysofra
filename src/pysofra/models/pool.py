"""Multiple-imputation pooling — Rubin's rules.

Combines a list of fitted models (one per imputed dataset) into a
single :class:`~pysofra.models.extract.ModelSummary` ready for
:func:`pysofra.tbl_regression`.

Implementation
--------------

* Pooled point estimate: arithmetic mean of imputation-specific
  estimates.
* Total variance ``T = Ubar + (1 + 1/m) * B`` (Rubin 1987), with
  within-imputation variance ``Ubar`` recovered from the
  per-imputation CIs and between-imputation variance ``B`` taken as
  the sample variance of the estimates.
* Degrees of freedom: **Rubin (1987)** ``ν = (m-1)·(1 + 1/r)²`` where
  ``r = ((1+1/m)·B) / Ubar``. The Barnard–Rubin (1999) refinement
  ``ν* = (ν · ν_obs) / (ν + ν_obs)`` further trims ``ν`` to respect
  the complete-data degrees of freedom but requires per-imputation
  ``df_resid``, which PySofra does not currently extract for every
  supported model family. For small per-imputation residual df this
  means the CIs / p-values are very slightly narrower than R's
  ``mice::pool`` would produce; the practical difference is
  negligible for the typical clinical-trial sample size (n ≳ 60).

References
----------
* Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in
  Surveys*. Wiley.
* Barnard, J. & Rubin, D.B. (1999). Small-sample degrees of freedom
  with multiple imputation. *Biometrika* 86 (4), 948–955.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .extract import ModelSummary, extract


def pool(models: list[Any], *, conf_level: float = 0.95) -> ModelSummary:
    """Pool a list of fitted models via Rubin's rules.

    Returns a :class:`ModelSummary` whose estimates / CIs / p-values
    reflect the across-imputation combination. Pass the result directly
    into :func:`pysofra.tbl_regression`.

    Each input must be a fitted model recognised by
    :func:`pysofra.models.extract.extract` — statsmodels, lifelines,
    sklearn (sklearn has no SEs so the pool degenerates to a simple
    mean-of-coefficients).
    """
    if len(models) < 2:
        raise ValueError(
            "pool requires at least two imputed-dataset fits "
            f"(got {len(models)})."
        )
    summaries = [extract(m, conf_level=conf_level) for m in models]
    coef_names = list(summaries[0].estimates.index)

    # Each summary must share the same coefficients to pool them coherently.
    for s in summaries[1:]:
        if list(s.estimates.index) != coef_names:
            raise ValueError(
                "All imputed fits must share the same coefficient names; "
                "got different sets."
            )

    m = len(summaries)
    Qbar = pd.Series(
        np.mean([s.estimates.to_numpy() for s in summaries], axis=0),
        index=coef_names,
    )

    # Within-imputation variance Ubar (mean of squared SE estimates) —
    # derived from CI half-widths so it works for any model with CIs.
    ses = np.zeros((m, len(coef_names)), dtype=float)
    z_crit = float(sp_stats.norm.ppf(0.5 + conf_level / 2))
    for i, s in enumerate(summaries):
        half = (s.ci_hi.to_numpy() - s.ci_lo.to_numpy()) / 2.0
        ses[i, :] = half / z_crit
    Ubar = np.nanmean(ses ** 2, axis=0)

    # Between-imputation variance B.
    Q = np.array([s.estimates.to_numpy() for s in summaries])
    B = np.var(Q, axis=0, ddof=1)

    # Total variance T = Ubar + (1 + 1/m) * B.
    T = Ubar + (1.0 + 1.0 / m) * B
    se_pool = np.sqrt(np.maximum(T, 0.0))

    # Rubin (1987) degrees of freedom.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ((1.0 + 1.0 / m) * B) / np.where(Ubar > 0, Ubar, np.nan)
        df_old = (m - 1) * (1.0 + 1.0 / np.where(r > 0, r, np.nan)) ** 2
    df_old = np.where(np.isfinite(df_old), df_old, 10_000.0)

    # Compute CI bounds and p-values from the pooled t-statistic.
    t_crit = sp_stats.t.ppf(0.5 + conf_level / 2, df=df_old)
    ci_lo = Qbar.to_numpy() - t_crit * se_pool
    ci_hi = Qbar.to_numpy() + t_crit * se_pool

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = Qbar.to_numpy() / np.where(se_pool > 0, se_pool, np.nan)
    p_vals = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=df_old)
    p_vals = np.where(np.isfinite(p_vals), p_vals, float("nan"))

    return ModelSummary(
        estimates=Qbar.astype(float),
        ci_lo=pd.Series(ci_lo, index=coef_names, dtype=float),
        ci_hi=pd.Series(ci_hi, index=coef_names, dtype=float),
        pvalues=pd.Series(p_vals, index=coef_names, dtype=float),
        family=f"Pooled MI ({m} imputations) — Rubin's rules",
        natural_exponentiate=summaries[0].natural_exponentiate,
    )

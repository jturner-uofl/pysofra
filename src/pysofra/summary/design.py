"""Survey design object for variance estimation under complex sampling.

The :class:`SurveyDesign` dataclass mirrors the headline fields of R's
``survey::svydesign``:

* ``weights`` — column carrying the sampling weight for each row.
* ``strata`` — optional stratification variable. Within each stratum,
  PSUs are assumed independent; variance is summed across strata.
* ``cluster`` — optional primary-sampling-unit (PSU) variable. When
  given, the variance of any estimator is computed across cluster
  totals rather than individual observations (Taylor linearization
  for the mean).
* ``fpc`` — optional finite-population-correction column (population
  size in each stratum, or per-cluster if no strata). Used to scale
  the variance by ``(1 - n/N)`` per stratum.

This is a *first-order* implementation: it covers what the vast
majority of survey-weighted clinical / epidemiology pipelines need
(stratified single-stage and clustered single-stage designs with FPC).
Multi-stage designs and post-stratification calibration remain on the
roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurveyDesign:
    """Column-name bundle describing a survey-design structure.

    ``cluster`` accepts either a single column name (single-stage
    cluster sampling) or a tuple of names (multi-stage). For multi-stage
    designs PySofra currently uses the outermost PSU for variance
    estimation and a footnote will name the second-stage column as
    "nested within" — full multi-stage Taylor linearisation is planned.

    ``replicate_weights`` and ``replicate_scale`` enable the jackknife
    family of variance estimators: every replicate column carries
    weights with one PSU dropped, and the variance is computed as
    ``replicate_scale * Σ (θ̂_r − θ̂)²``.  The ``"jk1"`` default sets
    ``replicate_scale`` to ``(n − 1)/n`` automatically.
    """

    weights: str
    strata: str | None = None
    cluster: str | tuple[str, ...] | None = None
    fpc: str | None = None
    replicate_weights: tuple[str, ...] | None = None
    replicate_type: str = "jk1"   # 'jk1' | 'jkn' | 'bootstrap'

    @property
    def primary_cluster(self) -> str | None:
        if self.cluster is None:
            return None
        if isinstance(self.cluster, tuple):
            return self.cluster[0] if self.cluster else None
        return self.cluster

    def validate(self, data: pd.DataFrame) -> None:
        for name, col in (("weights", self.weights),
                          ("strata", self.strata),
                          ("fpc", self.fpc)):
            if col is not None and col not in data.columns:
                raise KeyError(f"{name} column {col!r} not in data")
        if self.cluster is not None:
            cluster_cols = (
                self.cluster if isinstance(self.cluster, tuple)
                else (self.cluster,)
            )
            for c in cluster_cols:
                if c not in data.columns:
                    raise KeyError(f"cluster column {c!r} not in data")
        if self.replicate_weights is not None:
            missing = [c for c in self.replicate_weights if c not in data.columns]
            if missing:
                raise KeyError(f"replicate_weights columns not in data: {missing}")
            if self.replicate_type not in ("jk1", "jkn", "bootstrap"):
                raise ValueError(
                    f"replicate_type must be 'jk1', 'jkn', or 'bootstrap'; "
                    f"got {self.replicate_type!r}."
                )


# ----------------------------------------------------------------------
# Variance estimators
# ----------------------------------------------------------------------


def design_mean_var(
    values: pd.Series,
    weights: pd.Series,
    *,
    strata: pd.Series | None = None,
    cluster: pd.Series | None = None,
    fpc: pd.Series | None = None,
) -> tuple[float, float, float]:
    """Estimate the survey-weighted mean and its design-based variance.

    Returns ``(mean, variance, n_eff)``.

    For a simple stratified design, this implements the Taylor-series
    linearisation:

        Var(ŷ) = Σ_h (1 - f_h) · (n_h / (n_h - 1)) · Σ_{i in h} (w_i (y_i - ŷ))²
                                                       /  (Σ w_i)²

    For a clustered design (no strata, single-stage), the variance is
    computed across PSU totals.

    When both strata and clusters are given, the formula nests cluster
    variance within strata.
    """
    v = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").astype(float)
    mask = ~(v.isna() | w.isna()) & (w > 0)
    v = v[mask]
    w = w[mask]
    if strata is not None:
        strata = strata[mask].reset_index(drop=True)
    if cluster is not None:
        cluster = cluster[mask].reset_index(drop=True)
    if fpc is not None:
        fpc = pd.to_numeric(fpc, errors="coerce")[mask].reset_index(drop=True)
    v = v.reset_index(drop=True)
    w = w.reset_index(drop=True)

    total_w = float(w.sum())
    if total_w <= 0 or v.size == 0:
        return float("nan"), float("nan"), 0.0

    mean = float((w * v).sum() / total_w)
    n = int(v.size)

    # Residuals for the mean estimator.
    e = w.to_numpy() * (v.to_numpy() - mean)

    if strata is None and cluster is None:
        var_num = float(np.sum(e ** 2)) * (n / max(n - 1, 1))
    elif strata is None:
        assert cluster is not None
        # Single-stage clusters; sum residuals within each cluster, take
        # variance across cluster totals.
        s_per_cluster = pd.Series(e).groupby(cluster).sum().to_numpy()
        n_clust = int(s_per_cluster.size)
        if n_clust > 1:
            mean_cluster = float(s_per_cluster.mean())
            var_num = float(np.sum((s_per_cluster - mean_cluster) ** 2)) \
                * (n_clust / (n_clust - 1))
        else:
            var_num = 0.0
    else:
        # Stratified, possibly with clusters within strata.
        var_num = 0.0
        s = pd.Series(e)
        strata_series = pd.Series(strata)
        for _stratum, idx in strata_series.groupby(strata_series).indices.items():
            idx_arr = np.asarray(idx)
            e_h = s.iloc[idx_arr].to_numpy()
            if cluster is not None:
                c_h = cluster.iloc[idx_arr].to_numpy()
                psu_totals = pd.Series(e_h).groupby(c_h).sum().to_numpy()
                n_h = int(psu_totals.size)
                if n_h > 1:
                    mean_h = float(psu_totals.mean())
                    contrib = float(np.sum((psu_totals - mean_h) ** 2)) \
                        * (n_h / (n_h - 1))
                else:
                    contrib = 0.0
            else:
                n_h = int(e_h.size)
                contrib = (
                    float(np.sum(e_h ** 2)) * (n_h / (n_h - 1))
                    if n_h > 1
                    else 0.0
                )

            if fpc is not None and idx_arr.size > 0:
                fpc_h = float(fpc.iloc[idx_arr].iloc[0])
                # FPC = 1 - n/N
                f_h = min(1.0, idx_arr.size / max(fpc_h, 1.0))
                contrib *= 1.0 - f_h

            var_num += contrib

    variance = var_num / (total_w ** 2)
    return mean, variance, total_w


# ----------------------------------------------------------------------
# Replicate-weight variance
# ----------------------------------------------------------------------


def replicate_mean_var(
    values: pd.Series,
    base_weights: pd.Series,
    replicate_weights: list[pd.Series] | tuple[pd.Series, ...],
    *,
    replicate_type: str = "jk1",
) -> tuple[float, float, float]:
    """Variance of a weighted mean from replicate weights.

    The full-sample estimator uses ``base_weights``; each replicate gives
    a perturbed estimate, and the variance is

        Var(θ̂) = c · Σ_r (θ̂_r − θ̂)²

    where ``c`` is the replicate-type scaling: ``(R-1)/R`` for ``jk1``,
    ``1/R`` for ``bootstrap``. ``jkn`` (BRR/stratified jackknife) uses
    ``(R-1)/R`` too; users who need a different scale should pass the
    appropriate replicate weights and use ``bootstrap`` for the
    unscaled form.
    """
    v = pd.to_numeric(values, errors="coerce").astype(float)
    bw = pd.to_numeric(base_weights, errors="coerce").astype(float)
    mask = ~(v.isna() | bw.isna()) & (bw > 0)
    v = v[mask].reset_index(drop=True)
    bw = bw[mask].reset_index(drop=True)

    total_w = float(bw.sum())
    if total_w <= 0 or v.empty:
        return float("nan"), float("nan"), 0.0
    theta_hat = float((v * bw).sum() / total_w)

    R = len(replicate_weights)
    if R == 0:
        return theta_hat, 0.0, total_w

    sq_dev = 0.0
    for rw in replicate_weights:
        rw_arr = pd.to_numeric(rw, errors="coerce").astype(float)
        rw_arr = rw_arr[mask].reset_index(drop=True)
        w_pos = rw_arr.where(rw_arr > 0, 0.0)
        denom = float(w_pos.sum())
        if denom <= 0:
            continue
        theta_r = float((v * w_pos).sum() / denom)
        sq_dev += (theta_r - theta_hat) ** 2

    scale = 1.0 / R if replicate_type == "bootstrap" else (R - 1.0) / R
    return theta_hat, scale * sq_dev, total_w

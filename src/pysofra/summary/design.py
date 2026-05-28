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

    # Emit a UserWarning when ANY stratum or cluster has exactly one PSU
    # (the "lonely PSU" problem). The standard cluster-variance estimator
    # needs at least 2 PSUs to compute a within-stratum sum-of-squares;
    # with 1 PSU it silently contributes 0, *under-estimating* the
    # variance. R ``survey::svyrecvar`` errors by default
    # (``options(survey.lonely.psu='fail')``); we warn and contribute 0
    # so the rest of the table still renders, but the user is on notice.
    import warnings as _w
    if strata is not None and cluster is not None:
        n_psu_per_stratum = (
            pd.Series(cluster).groupby(pd.Series(strata)).nunique()
        )
        lonely = n_psu_per_stratum[n_psu_per_stratum < 2]
        if not lonely.empty:
            _w.warn(
                f"{len(lonely)} stratum/strata contain only one cluster "
                "('lonely PSU'); their variance contribution is taken as "
                "zero, which UNDER-ESTIMATES the design-based variance. "
                "R's survey package errors by default on this. Combine "
                "or drop the affected strata, or use a replicate-weight "
                "variance estimator.",
                UserWarning,
                stacklevel=2,
            )
    elif cluster is not None:
        if int(pd.Series(cluster).nunique()) < 2:
            _w.warn(
                "design has only one cluster overall; the cluster-robust "
                "variance is undefined and reported as zero. Add more "
                "clusters or drop the cluster argument.",
                UserWarning,
                stacklevel=2,
            )

    if strata is None and cluster is None:
        var_num = float(np.sum(e ** 2)) * (n / max(n - 1, 1))
        # Apply FPC in the unstratified-unclustered branch too. The
        # standard formula is ``Var(ȳ_w) ∝ (1 − n/N) · Σ residuals²``;
        # earlier alphas only applied FPC inside the stratified branch,
        # silently dropping it for designs configured with only
        # ``weights=`` and ``fpc=``.
        if fpc is not None and fpc.size > 0:
            fpc_val = float(fpc.iloc[0])
            f = min(1.0, n / max(fpc_val, 1.0))
            var_num *= 1.0 - f
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
        # Apply FPC in the clustered-unstratified branch as well.
        if fpc is not None and fpc.size > 0:
            fpc_val = float(fpc.iloc[0])
            f = min(1.0, n_clust / max(fpc_val, 1.0))
            var_num *= 1.0 - f
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
                    # Centre on the stratum-specific PSU-total mean.
                    # The textbook Taylor estimator within a stratum is
                    # ``(n_h/(n_h−1))·Σ(s_hk − s̄_h)²``; the previous
                    # ``Σ s_hk²`` form (which centres on the global zero)
                    # over-states the variance whenever the per-stratum
                    # influence-function mean is non-zero, which it
                    # generally is in stratified data.
                    mean_h = float(psu_totals.mean())
                    contrib = float(np.sum((psu_totals - mean_h) ** 2)) \
                        * (n_h / (n_h - 1))
                else:
                    contrib = 0.0
            else:
                n_h = int(e_h.size)
                if n_h > 1:
                    # Same correction in the stratified-unclustered
                    # case: centre on the stratum mean, not on zero.
                    mean_h = float(e_h.mean())
                    contrib = float(np.sum((e_h - mean_h) ** 2)) \
                        * (n_h / (n_h - 1))
                else:
                    contrib = 0.0

            if fpc is not None and idx_arr.size > 0:
                fpc_h = float(fpc.iloc[idx_arr].iloc[0])
                # FPC = 1 - n/N
                f_h = min(1.0, idx_arr.size / max(fpc_h, 1.0))
                contrib *= 1.0 - f_h

            var_num += contrib

    variance = var_num / (total_w ** 2)
    return mean, variance, total_w


# ----------------------------------------------------------------------
# Design-based (Taylor-linearised) GLM covariance — svyglm sandwich
# ----------------------------------------------------------------------

# Map a statsmodels family class name → the GLM variance function
# V(μ) = dμ/dη for the *canonical* link. For canonical links the
# estimating-function score simplifies to  u_i = w_i (y_i − μ_i) x_i
# and the bread is  A = Xᵀ diag(w_i · V(μ_i)) X. These three families
# cover every design= path PySofra supports.
_CANONICAL_VARFUNC = {
    "Binomial": lambda mu: mu * (1.0 - mu),   # logit link
    "Poisson":  lambda mu: mu,                # log link
    "Gaussian": lambda mu: np.ones_like(mu),  # identity link
}


def survey_glm_vcov(
    fitted_glm: object,
    weights: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    cluster: np.ndarray | None = None,
    fpc: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Design-based (Taylor-linearised) covariance for a survey GLM.

    Reproduces R ``survey::svyglm`` to numerical precision for the
    canonical-link Binomial / Poisson / Gaussian families. The
    estimator is the standard linearisation sandwich

        V = A⁻¹ · B · A⁻¹

    where the *bread* ``A = Xᵀ diag(wᵢ V(μᵢ)) X`` is the weighted GLM
    information matrix and the *meat* ``B`` is the design-based
    variance of the per-observation score totals
    ``uᵢ = wᵢ (yᵢ − μᵢ) xᵢ``, aggregated to PSU level and summed across
    strata exactly as :func:`design_mean_var` does for a scalar mean.

    Parameters
    ----------
    fitted_glm
        A fitted ``statsmodels`` GLM results object (must expose
        ``.model.exog``, ``.model.endog``, ``.fittedvalues``,
        ``.params`` and ``.model.family``).
    weights
        Sampling-weight vector, aligned to the model rows.
    strata, cluster, fpc
        Design columns aligned to the model rows. ``cluster`` is the
        PSU id; ``strata`` the stratum id; ``fpc`` the per-stratum
        finite-population size.

    Returns
    -------
    (vcov, df_design)
        ``vcov`` is the k×k design-based covariance matrix (coefficient
        order matches ``fitted_glm.params``). ``df_design`` is the
        survey design degrees of freedom ``n_PSU − n_strata`` used for
        t-based CIs (R ``svyglm`` convention); ``float('inf')`` when
        no cluster structure is present.

    Raises
    ------
    NotImplementedError
        If the GLM family is not one of the supported canonical-link
        families (Binomial, Poisson, Gaussian).
    """
    family_name = type(fitted_glm.model.family).__name__
    varfunc = _CANONICAL_VARFUNC.get(family_name)
    if varfunc is None:
        raise NotImplementedError(
            f"survey_glm_vcov supports canonical-link Binomial / Poisson "
            f"/ Gaussian only; got family {family_name!r}. For other "
            f"families compute the design SE in R survey::svyglm."
        )

    X = np.asarray(fitted_glm.model.exog, dtype=float)
    y = np.asarray(fitted_glm.model.endog, dtype=float)
    mu = np.asarray(fitted_glm.fittedvalues, dtype=float)
    w = np.asarray(weights, dtype=float)
    n, k = X.shape

    # Bread: A = Xᵀ diag(w · V(μ)) X
    d = w * varfunc(mu)
    A = X.T @ (X * d[:, None])
    A_inv = np.linalg.pinv(A)

    # Per-observation score: uᵢ = wᵢ (yᵢ − μᵢ) xᵢ
    resid = w * (y - mu)
    U = X * resid[:, None]          # n × k

    # Meat: design-based variance of the score totals, nested PSU-within-
    # stratum exactly like design_mean_var. With no strata/cluster this
    # reduces to the with-replacement HC0-style Σ uᵢ uᵢᵀ scaled n/(n−1).
    B = np.zeros((k, k))
    n_psu_total = 0
    n_strata_total = 0

    def _stratum_meat(U_h: np.ndarray, clust_h: np.ndarray | None,
                       fpc_h: float | None) -> tuple[np.ndarray, int]:
        if clust_h is not None:
            # Sum scores to PSU totals, then between-PSU SSCP
            uniq = pd.unique(clust_h)
            psu_tot = np.vstack([
                U_h[clust_h == c].sum(axis=0) for c in uniq
            ])
            m = psu_tot.shape[0]
        else:
            psu_tot = U_h
            m = psu_tot.shape[0]
        if m < 2:
            return np.zeros((k, k)), m
        dev = psu_tot - psu_tot.mean(axis=0)
        contrib = (m / (m - 1.0)) * (dev.T @ dev)
        if fpc_h is not None and fpc_h > 0:
            f_h = min(1.0, m / fpc_h)
            contrib *= 1.0 - f_h
        return contrib, m

    if strata is None and cluster is None:
        dev = U - U.mean(axis=0)
        B = (n / (n - 1.0)) * (dev.T @ dev)
        n_psu_total, n_strata_total = n, 0
    elif strata is None:
        contrib, m = _stratum_meat(U, np.asarray(cluster), None)
        B += contrib
        n_psu_total, n_strata_total = m, 1
    else:
        s_arr = np.asarray(strata)
        for s in pd.unique(s_arr):
            sel = s_arr == s
            U_h = U[sel]
            clust_h = (np.asarray(cluster)[sel]
                       if cluster is not None else None)
            fpc_h = (float(np.asarray(fpc)[sel][0])
                     if fpc is not None else None)
            contrib, m = _stratum_meat(U_h, clust_h, fpc_h)
            B += contrib
            n_psu_total += m
            n_strata_total += 1

    vcov = A_inv @ B @ A_inv
    df_design = (float(n_psu_total - n_strata_total)
                 if cluster is not None else float("inf"))
    return vcov, df_design


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

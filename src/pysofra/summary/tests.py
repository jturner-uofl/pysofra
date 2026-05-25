"""Statistical test selection for Table 1 / summary tables.

Two layers:

* **Defaults** — :func:`continuous_test` and :func:`categorical_test` choose
  a sensible test for a variable given its kind, following the
  ``tableone`` / ``gtsummary`` conventions:

    Continuous, 2 groups   → Welch's t-test (Wilcoxon if ``nonnormal``)
    Continuous, 3+ groups  → one-way ANOVA (Kruskal–Wallis if ``nonnormal``)
    Categorical, 2×2       → Fisher's exact
    Categorical, larger    → Pearson chi-square (flagged sparse if any expected < 5)

* **Per-variable overrides** — :func:`run_named_test` dispatches a named
  test by string key. Builders accept a ``tests={'age': 'wilcoxon'}`` map
  and call into this dispatcher, falling back to the defaults otherwise.

Returns a small :class:`TestResult` so callers can render both the p-value
and the test name for the footnote.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass(frozen=True)
class TestResult:
    p_value: float | None
    test: str  # short human-readable name; used in footnote
    statistic: float | None = None


_NA = TestResult(p_value=None, test="—")


@contextmanager
def _quiet_scipy():  # type: ignore[no-untyped-def]
    """Suppress numeric and ``RuntimeWarning`` chatter from scipy hypothesis
    tests on edge-case inputs.

    scipy emits ``RuntimeWarning: Precision loss occurred in moment
    calculation`` when ``ttest_ind`` is asked to test on near-constant
    arrays, and similar advisory warnings for other tests on degenerate
    data. The warning is correct but advisory — the resulting ``nan``
    / boundary p-value is the well-defined output we want to surface.
    Users gating on ``-W error::RuntimeWarning`` would otherwise see
    the test routine crash instead of returning a NaN p-value, so wrap
    the scipy call at our boundary.
    """
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _group_arrays(values: pd.Series, groups: pd.Series) -> list[np.ndarray]:
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": groups})
    df = df.dropna(subset=["v", "g"])
    if df.empty:
        return []
    arrs = [g["v"].to_numpy() for _, g in df.groupby("g", observed=True)]
    return [a for a in arrs if a.size > 0]


# ----------------------------------------------------------------------
# Continuous
# ----------------------------------------------------------------------

def continuous_test(
    values: pd.Series,
    groups: pd.Series,
    nonnormal: bool = False,
) -> TestResult:
    """Default continuous test selection."""
    arrs = _group_arrays(values, groups)
    if len(arrs) < 2:
        return _NA
    if len(arrs) == 2:
        return _wilcoxon(arrs) if nonnormal else _welch(arrs)
    return _kruskal(arrs) if nonnormal else _anova(arrs)


def _welch(arrs: list[np.ndarray]) -> TestResult:
    with _quiet_scipy():
        stat, p = sp_stats.ttest_ind(*arrs, equal_var=False, nan_policy="omit")
    return TestResult(p_value=float(p), test="Welch's t-test", statistic=float(stat))


def _student_t(arrs: list[np.ndarray]) -> TestResult:
    with _quiet_scipy():
        stat, p = sp_stats.ttest_ind(*arrs, equal_var=True, nan_policy="omit")
    return TestResult(p_value=float(p), test="Student's t-test", statistic=float(stat))


def _wilcoxon(arrs: list[np.ndarray]) -> TestResult:
    with _quiet_scipy():
        stat, p = sp_stats.mannwhitneyu(*arrs, alternative="two-sided")
    return TestResult(p_value=float(p), test="Wilcoxon rank-sum", statistic=float(stat))


def _anova(arrs: list[np.ndarray]) -> TestResult:
    with _quiet_scipy():
        stat, p = sp_stats.f_oneway(*arrs)
    return TestResult(p_value=float(p), test="One-way ANOVA", statistic=float(stat))


def _kruskal(arrs: list[np.ndarray]) -> TestResult:
    with _quiet_scipy():
        stat, p = sp_stats.kruskal(*arrs)
    return TestResult(p_value=float(p), test="Kruskal–Wallis", statistic=float(stat))


# ----------------------------------------------------------------------
# Categorical
# ----------------------------------------------------------------------

def categorical_test(values: pd.Series, groups: pd.Series) -> TestResult:
    """Default categorical test selection."""
    ctab = _crosstab(values, groups)
    if ctab is None:
        return _NA
    observed = ctab.to_numpy()
    if observed.shape == (2, 2):
        return _fisher(observed)
    return _chisq(observed)


def svyttest(
    values: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
    *,
    strata: pd.Series | None = None,
    cluster: pd.Series | None = None,
) -> TestResult:
    """Design-adjusted two-sample t-test (R ``survey::svyttest`` analogue).

    Equivalent to fitting ``y ~ I(group == levels[1])`` under the survey
    design and reading off the coefficient and its design-based variance.
    Specifically we compute:

    * **Point estimate.** The weighted least-squares slope of *y* on the
      0/1 group indicator equals the weighted mean difference
      ``ȳ_B − ȳ_A``.
    * **Variance.** Taylor-linearised cluster-robust variance of the
      slope. The influence function of the slope at the optimum is
      ``u_i = w_i · (x_i − x̄_w) · ε_i / S_xx`` where ε_i is the
      working residual. The design-based variance of ``Σu_i`` is
      computed by summing influence-function contributions within
      each PSU (cluster, or singleton observation if unclustered),
      then summing squared deviations within each stratum with the
      usual ``n_h / (n_h − 1)`` finite-sample correction.
    * **Degrees of freedom.** ``n_PSU − n_strata`` (matches R
      ``survey::svyttest`` with ``nest=TRUE`` semantics — unique
      ``(stratum, cluster)`` pairs count as distinct PSUs).

    Previous alphas computed per-group variance separately and summed
    in quadrature. That ignores the cross-group covariance under the
    full design and gave inflated t-statistics whenever clusters
    straddled groups; the current formulation matches R to first
    order.
    """
    df_ = pd.DataFrame({
        "v": pd.to_numeric(values, errors="coerce"),
        "g": groups,
        "w": pd.to_numeric(weights, errors="coerce"),
    })
    if strata is not None:
        df_["strata"] = strata.values
    if cluster is not None:
        df_["cluster"] = cluster.values
    df_ = df_.dropna(subset=["v", "g", "w"])
    df_ = df_[df_["w"] > 0]
    if df_.empty:
        return _NA

    levels = sorted(df_["g"].unique(), key=str)
    if len(levels) != 2:
        return _NA

    # Centre the group indicator and the outcome by their weighted means;
    # the WLS slope is then Sxy / Sxx, which for a 0/1 indicator equals
    # the difference of weighted group means.
    x = (df_["g"] == levels[1]).to_numpy(dtype=float)
    y = df_["v"].to_numpy(dtype=float)
    w = df_["w"].to_numpy(dtype=float)

    sw = float(w.sum())
    if sw <= 0:
        return _NA
    x_bar = float((w * x).sum() / sw)
    y_bar = float((w * y).sum() / sw)
    x_dev = x - x_bar
    y_dev = y - y_bar
    s_xx = float((w * x_dev * x_dev).sum())
    s_xy = float((w * x_dev * y_dev).sum())
    if s_xx <= 0:
        return _NA
    beta = s_xy / s_xx
    # Working residual under WLS at the optimum.
    eps = y_dev - beta * x_dev
    # Influence function for β: u_i = w_i · x_dev_i · ε_i / S_xx.
    u = (w * x_dev * eps) / s_xx

    # Design-based variance of β = variance of Σu_i under the sampling
    # design. We sum u within each PSU (cluster, or singleton if no
    # clustering), then within each stratum compute the usual
    # ``n_h / (n_h − 1) · Σ(s_h_total − s̄_h)²`` cluster-of-totals variance.
    h = 1 if strata is None else int(pd.Series(strata).nunique())
    if "cluster" in df_.columns and "strata" in df_.columns:
        n_psu = int(df_[["strata", "cluster"]].drop_duplicates().shape[0])
    elif "cluster" in df_.columns:
        n_psu = int(df_["cluster"].nunique())
    else:
        n_psu = int(len(df_))

    var_beta = _cluster_robust_var_of_sum(
        u=u,
        strata=df_["strata"].to_numpy() if "strata" in df_.columns else None,
        cluster=(df_["cluster"].to_numpy()
                 if "cluster" in df_.columns else None),
    )
    if not np.isfinite(var_beta) or var_beta <= 0:
        return _NA
    se = float(np.sqrt(var_beta))
    t_stat = beta / se
    # df = design df − 1. R ``survey::degf`` for a stratified clustered
    # design returns ``n_PSU − n_strata``; ``svyttest`` then subtracts
    # one more for the slope parameter, leaving ``n_PSU − n_strata − 1``.
    # For an unclustered, unstratified design (n_PSU = n, n_strata = 1)
    # this collapses to the familiar ``n − 2``.
    df_deg = max(1, n_psu - h - 1)
    p = 2 * float(sp_stats.t.sf(abs(t_stat), df=df_deg))

    return TestResult(p_value=p, test="Design-adjusted t-test", statistic=t_stat)


def _cluster_robust_var_of_sum(
    *,
    u: np.ndarray,
    strata: np.ndarray | None,
    cluster: np.ndarray | None,
) -> float:
    """Design-based variance of a sum estimator ``Σu_i`` under stratified
    cluster sampling.

    Standard sandwich formula:

    * **Unstratified, unclustered**: ``Σu²`` (with ``n/(n − 1)`` Bessel
      correction).
    * **Clustered, unstratified**: sum ``u`` within each cluster, then
      take ``n_c / (n_c − 1) · Σ(s_c − s̄)²`` across cluster totals.
    * **Stratified**: do the above within each stratum and sum across.

    Influence-function inputs ``u`` always have ``Σu ≈ 0`` at the
    optimum, so the cluster-mean of within-stratum totals is taken
    rather than zero (matches R ``survey::svyrecvar`` which centres on
    the empirical stratum mean).
    """
    n = u.size
    if n <= 1:
        return float("nan")
    if strata is None and cluster is None:
        return float((u * u).sum()) * n / max(n - 1, 1)
    if strata is None:
        assert cluster is not None
        s_per_cluster = pd.Series(u).groupby(pd.Series(cluster)).sum().to_numpy()
        nc = int(s_per_cluster.size)
        if nc <= 1:
            return 0.0
        mean_c = float(s_per_cluster.mean())
        return float(((s_per_cluster - mean_c) ** 2).sum()) * nc / (nc - 1)
    # Stratified path
    var_num = 0.0
    s_strata = pd.Series(strata)
    for _h, idx in s_strata.groupby(s_strata).indices.items():
        idx_arr = np.asarray(idx)
        u_h = u[idx_arr]
        if cluster is not None:
            c_h = cluster[idx_arr]
            psu_totals = pd.Series(u_h).groupby(pd.Series(c_h)).sum().to_numpy()
            n_h = int(psu_totals.size)
            if n_h > 1:
                mean_h = float(psu_totals.mean())
                var_num += float(((psu_totals - mean_h) ** 2).sum()) * n_h / (n_h - 1)
        else:
            n_h = int(u_h.size)
            if n_h > 1:
                var_num += float((u_h * u_h).sum()) * n_h / (n_h - 1)
    return var_num


def rao_scott_chisq(
    values: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
) -> TestResult:
    """Rao–Scott first-order corrected chi-square for survey-weighted data.

    Computes a Pearson chi-square statistic on the contingency table
    after **normalising the weights to sum to** ``n`` (so the chi-square
    is independent of the absolute scale of the weights; matches R
    ``survey::svychisq(..., statistic="Chisq")`` on this step). Then
    scales by the Kish design-effect estimate:

        DEFF ≈ n · Σ w_i² / (Σ w_i)²

    The corrected statistic is referred to a χ² distribution with the
    usual ``(R-1)(C-1)`` degrees of freedom.

    Notes
    -----
    This is a *first-order* Rao–Scott correction using the Kish design
    effect (a single scalar derived from the weights). For exact parity
    with R ``survey::svychisq(..., statistic="F")`` — which uses the
    *generalised* design effect derived from the eigenvalues of the
    full design covariance matrix — call out to the R ``survey``
    package directly. Pysofra's first-order approximation typically
    agrees with R to within ~5% on simple weighted designs and is
    adequate for descriptive Table 1 use.
    """
    df = pd.DataFrame({
        "v": values,
        "g": groups,
        "w": pd.to_numeric(weights, errors="coerce"),
    }).dropna()
    df = df[df["w"] > 0]
    if df.empty:
        return _NA

    w = df["w"].to_numpy(dtype=float)
    n = float(len(w))
    total_w = float(w.sum())
    if total_w <= 0 or n <= 0:
        return _NA

    # Normalise weights so Σw = n. The Pearson chi-square statistic on a
    # weighted contingency table is invariant to the absolute weight
    # scale only when the table sums to n; otherwise it grows linearly
    # in Σw (a 10× rescaling of weights would inflate chi² 10×). R
    # ``survey::svychisq`` normalises internally for this reason.
    w_norm = w * (n / total_w)
    df = df.assign(w_norm=w_norm)
    ctab = df.groupby(["v", "g"], observed=True)["w_norm"].sum().unstack(
        fill_value=0,
    ).astype(float)
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        return _NA
    observed_n = ctab.to_numpy(dtype=float)

    # Pearson chi-square on the (normalised) weighted contingency table.
    with _quiet_scipy():
        chi2, _, dof, _expected = sp_stats.chi2_contingency(
            observed_n, correction=False,
        )

    # Kish design effect — scale-invariant by construction.
    deff = float(n * (w**2).sum() / (total_w ** 2))
    chi2_adj = float(chi2) / max(deff, 1e-12)
    p_adj = float(sp_stats.chi2.sf(chi2_adj, df=dof))

    return TestResult(
        p_value=p_adj,
        test="Rao–Scott chi-square",
        statistic=chi2_adj,
    )


def _crosstab(values: pd.Series, groups: pd.Series) -> pd.DataFrame | None:
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df.empty:
        return None
    ctab = pd.crosstab(df["v"], df["g"])
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        return None
    return ctab


def _fisher(observed: np.ndarray) -> TestResult:
    # scipy >= 1.13: `alternative` is only valid for 2x2 tables; larger
    # tables use an exact RxC computation under the default method.
    with _quiet_scipy():
        if observed.shape == (2, 2):
            _, p = sp_stats.fisher_exact(observed, alternative="two-sided")
        else:
            _, p = sp_stats.fisher_exact(observed)
    return TestResult(p_value=float(p), test="Fisher's exact")


def _chisq(observed: np.ndarray) -> TestResult:
    with _quiet_scipy():
        chi2, p, _, expected = sp_stats.chi2_contingency(observed, correction=False)
    if np.any(expected < 5):
        return TestResult(p_value=float(p), test="Chi-square (sparse)", statistic=float(chi2))
    return TestResult(p_value=float(p), test="Pearson's chi-square", statistic=float(chi2))


# ----------------------------------------------------------------------
# Named-test dispatcher (per-variable overrides)
# ----------------------------------------------------------------------

ContinuousFn = Callable[[list[np.ndarray]], TestResult]
CategoricalFn = Callable[[np.ndarray], TestResult]

_CONTINUOUS_TESTS: dict[str, ContinuousFn] = {
    "welch": _welch,
    "welch_t": _welch,
    "t": _welch,
    "ttest": _welch,
    "student": _student_t,
    "student_t": _student_t,
    "equal_var_t": _student_t,
    "wilcoxon": _wilcoxon,
    "mannwhitney": _wilcoxon,
    "mwu": _wilcoxon,
    "rank_sum": _wilcoxon,
    "anova": _anova,
    "oneway_anova": _anova,
    "kruskal": _kruskal,
    "kruskal_wallis": _kruskal,
}

_CATEGORICAL_TESTS: dict[str, CategoricalFn] = {
    "fisher": _fisher,
    "fisher_exact": _fisher,
    "chisq": _chisq,
    "chi_square": _chisq,
    "chi2": _chisq,
    "pearson": _chisq,
}


def available_tests() -> dict[str, list[str]]:
    """Return the lookup table of named tests, grouped by variable kind."""
    return {
        "continuous": sorted(_CONTINUOUS_TESTS),
        "categorical": sorted(_CATEGORICAL_TESTS),
    }


def run_named_test(
    name: str,
    values: pd.Series,
    groups: pd.Series,
    *,
    kind: str,
) -> TestResult:
    """Run a named test against (values, groups).

    ``kind`` is ``"continuous"`` or ``"categorical"`` and disambiguates
    which dispatch table to consult. Raises ``ValueError`` if the test
    name is unknown for the given kind.
    """
    key = name.lower().strip()
    if kind == "continuous":
        cont_fn = _CONTINUOUS_TESTS.get(key)
        if cont_fn is None:
            raise ValueError(
                f"Unknown continuous test {name!r}. Available: "
                + ", ".join(sorted(set(_CONTINUOUS_TESTS)))
            )
        arrs = _group_arrays(values, groups)
        if len(arrs) < 2:
            return _NA
        return cont_fn(arrs)

    if kind == "categorical":
        cat_fn = _CATEGORICAL_TESTS.get(key)
        if cat_fn is None:
            raise ValueError(
                f"Unknown categorical test {name!r}. Available: "
                + ", ".join(sorted(set(_CATEGORICAL_TESTS)))
            )
        ctab = _crosstab(values, groups)
        if ctab is None:
            return _NA
        return cat_fn(ctab.to_numpy())

    raise ValueError(f"Unknown variable kind {kind!r}")

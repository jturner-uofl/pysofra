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
    """Design-adjusted two-sample t-test (svyttest analogue).

    For two groups, the test statistic is

        t = (ȳ₂_w − ȳ₁_w) / SE(diff)

    where ȳᵢ_w is the weighted group mean and the SE is computed via
    Taylor linearisation (using :func:`pysofra.summary.design.design_mean_var`
    once per group, summed across groups for the variance of the
    difference). Compared against a ``t`` distribution with ``Σ n_h − H``
    degrees of freedom (where ``H`` is the number of strata if given, or
    one otherwise).
    """
    from .design import design_mean_var

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

    means: list[float] = []
    vars_: list[float] = []
    n_per: list[int] = []
    for lvl in levels:
        sub = df_[df_["g"] == lvl]
        m, v, _ = design_mean_var(
            sub["v"], sub["w"],
            strata=sub.get("strata"),
            cluster=sub.get("cluster"),
        )
        means.append(m)
        vars_.append(v)
        n_per.append(int(len(sub)))

    diff = means[1] - means[0]
    se = float(np.sqrt(vars_[0] + vars_[1])) if not (
        np.isnan(vars_[0]) or np.isnan(vars_[1])
    ) else float("nan")
    if not np.isfinite(se) or se == 0:
        return _NA
    t_stat = diff / se
    # Degrees of freedom: total n minus number of strata (1 if unstratified).
    h = 1 if strata is None else int(pd.Series(strata).nunique())
    df_deg = max(1, sum(n_per) - h)
    p = 2 * float(sp_stats.t.sf(abs(t_stat), df=df_deg))

    return TestResult(p_value=p, test="Design-adjusted t-test", statistic=t_stat)


def rao_scott_chisq(
    values: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
) -> TestResult:
    """Rao–Scott first-order corrected chi-square for survey-weighted data.

    Computes a Pearson chi-square statistic on the *weighted* contingency
    table, then scales it by an estimated design effect (DEFF) derived
    from the weights:

        DEFF ≈ n * Σ w_i² / (Σ w_i)²

    The corrected statistic is referred to a χ² distribution with the
    usual ``(R-1)(C-1)`` degrees of freedom. This is the first-order
    Rao–Scott correction; for full second-order accuracy a generalised
    design matrix is required and is left to dedicated survey packages.
    """
    df = pd.DataFrame({
        "v": values,
        "g": groups,
        "w": pd.to_numeric(weights, errors="coerce"),
    }).dropna()
    df = df[df["w"] > 0]
    if df.empty:
        return _NA

    # Weighted contingency table.
    ctab = df.groupby(["v", "g"], observed=True)["w"].sum().unstack(
        fill_value=0,
    ).astype(float)
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        return _NA
    observed_w = ctab.to_numpy(dtype=float)

    # Pearson chi-square on the weighted observed table.
    with _quiet_scipy():
        chi2, _, dof, _expected = sp_stats.chi2_contingency(
            observed_w, correction=False,
        )

    w = df["w"].to_numpy(dtype=float)
    n = float(len(w))
    deff = float(n * (w**2).sum() / (w.sum() ** 2)) if w.sum() > 0 else 1.0
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

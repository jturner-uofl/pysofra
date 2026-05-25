"""Coefficient extraction from fitted models.

Different libraries expose fitted-model summaries in different ways. This
module abstracts the extraction into a single :class:`ModelSummary` and
detects the source by duck-typing — we never hard-import optional
dependencies (lifelines, sklearn) at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSummary:
    """Per-coefficient summary used by :func:`tbl_regression`.

    ``estimates`` / ``ci_lo`` / ``ci_hi`` / ``pvalues`` are aligned Series
    indexed by coefficient name. ``family`` is a short human-readable
    string (``"Logit"``, ``"OLS"``, ``"CoxPHFitter"``, ``"LogisticRegression"``)
    used to pick a sensible estimate column label.
    """

    estimates: pd.Series
    ci_lo: pd.Series
    ci_hi: pd.Series
    pvalues: pd.Series
    family: str
    natural_exponentiate: bool  # whether exp() is the natural reporting metric


def extract(model: Any, conf_level: float = 0.95) -> ModelSummary:
    """Extract a :class:`ModelSummary` from any supported fitted model."""
    qualname = f"{type(model).__module__}.{type(model).__name__}"

    # PooledSummary (multiple-imputation Rubin'd results) — already extracted.
    if isinstance(model, ModelSummary):
        return model

    # lifelines first — its fitters don't all expose the statsmodels
    # ``.params`` interface (CoxPHFitter exposes ``.params_`` and ``.summary``).
    if qualname.startswith("lifelines."):
        return _extract_lifelines(model, conf_level)

    # statsmodels — Results wrapper, recognised by the .params/.bse interface.
    # MixedLM and GEE both honour this interface; the family-label helper
    # picks them out.
    if hasattr(model, "params") and hasattr(model, "pvalues") and hasattr(model, "conf_int"):
        return _extract_statsmodels(model, conf_level)

    # sklearn linear models — recognised by .coef_ + a `predict`/`fit` method.
    # We extract point estimates only; CIs / p-values are not natively
    # available and are filled with NaN.
    if hasattr(model, "coef_") and (hasattr(model, "predict") or hasattr(model, "fit")):
        return _extract_sklearn(model)

    raise TypeError(
        f"Unsupported model type {qualname!r}. "
        "tbl_regression supports statsmodels Results, lifelines fitters, "
        "and sklearn linear models."
    )


# ----------------------------------------------------------------------
# statsmodels
# ----------------------------------------------------------------------

def _extract_statsmodels(model: Any, conf_level: float) -> ModelSummary:
    params = pd.Series(model.params)
    pvalues = pd.Series(getattr(model, "pvalues", pd.Series(dtype=float)))
    try:
        ci = model.conf_int(alpha=1.0 - conf_level)
    except TypeError:  # pragma: no cover — older statsmodels signature
        ci = model.conf_int()
    ci = pd.DataFrame(ci)
    ci.columns = ["lo", "hi"]
    ci = ci.reindex(params.index)

    family_label = _statsmodels_family_label(model)
    natural_exp = _is_log_link(family_label)

    return ModelSummary(
        estimates=params.astype(float),
        ci_lo=ci["lo"].astype(float),
        ci_hi=ci["hi"].astype(float),
        pvalues=pvalues.reindex(params.index).astype(float),
        family=family_label,
        natural_exponentiate=natural_exp,
    )


def _statsmodels_family_label(model: Any) -> str:
    cls = type(model).__name__
    fam = getattr(model, "family", None)
    if fam is not None:
        return f"{cls} ({type(fam).__name__})"
    inner = getattr(model, "model", None)
    if inner is not None:
        inner_name = type(inner).__name__
        # MixedLM / GEE add their own family/link info that's worth surfacing.
        if "MixedLM" in inner_name:
            return f"{cls} (MixedLM)"
        if "GEE" in inner_name or "GeneralizedEstimatingEquations" in inner_name:
            cov = getattr(inner, "cov_struct", None)
            if cov is not None:
                return f"{cls} (GEE, {type(cov).__name__})"
            return f"{cls} (GEE)"
        return f"{cls} ({inner_name})"
    return cls


def _is_log_link(family_label: str) -> bool:
    fl = family_label.lower()
    return any(k in fl for k in ("logit", "binomial", "probit", "poisson",
                                 "negativebinomial"))


# ----------------------------------------------------------------------
# lifelines (Cox PH, AFT, etc.)
# ----------------------------------------------------------------------

def _extract_lifelines(model: Any, conf_level: float) -> ModelSummary:
    """Extract coefficients from a fitted lifelines regression model.

    ``model.summary`` is a DataFrame with the standard columns
    ``coef``, ``coef lower X%``, ``coef upper X%``, ``p``. The exact column
    names vary by lifelines version and confidence level — we resolve them
    dynamically.
    """
    if not hasattr(model, "summary"):
        raise TypeError(
            "lifelines model has no .summary attribute; "
            "make sure you called .fit() before tbl_regression()."
        )
    summary = model.summary
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("lifelines .summary is not a DataFrame.")

    # Find the CI columns. Lifelines reports ``coef lower 95%`` /
    # ``coef upper 95%`` by default; we accept any matching lower/upper
    # pair.
    lo_col = _find_col(summary, ["coef lower"])
    hi_col = _find_col(summary, ["coef upper"])
    if lo_col is None or hi_col is None:
        raise ValueError(
            f"Could not locate CI columns in lifelines summary "
            f"(columns: {list(summary.columns)})."
        )

    estimates = summary["coef"].astype(float)
    pvalues = summary["p"].astype(float) if "p" in summary.columns else pd.Series(
        [float("nan")] * len(summary), index=summary.index
    )

    # Lifelines bakes the CI level into the fit (alpha=0.05 by default),
    # so the ``coef lower/upper X%`` columns reflect the fit-time alpha,
    # not the user's requested ``conf_level``. To honour ``conf_level``
    # without re-fitting the model, re-derive the CI directly from
    # ``coef`` and ``se(coef)`` using a normal pivot. Falls back to the
    # lifelines-provided columns only when no SE column is present.
    se_col = _find_col(summary, ["se(coef)"])
    if se_col is not None:
        import numpy as _np
        from scipy import stats as _sp_stats
        z = float(_sp_stats.norm.ppf(0.5 + conf_level / 2))
        se = summary[se_col].astype(float)
        ci_lo = estimates - z * se
        ci_hi = estimates + z * se
        # Hide ``_np`` reference so linters don't flag it as unused.
        del _np
    else:
        ci_lo = summary[lo_col].astype(float)
        ci_hi = summary[hi_col].astype(float)
    # AFT models (Weibull / log-logistic / log-normal) carry a MultiIndex
    # ``(param, covariate)`` index — e.g. ``('lambda_', 'age')``. Renderers
    # expect string row labels; flatten with ``covariate (param)`` so the
    # table reads naturally ("age (lambda_)") rather than emitting a tuple
    # that crashes the markdown / HTML escapers.
    if isinstance(estimates.index, pd.MultiIndex):
        flat = [f"{cov} ({param})" for param, cov in estimates.index]
        estimates.index = pd.Index(flat)
        ci_lo.index = pd.Index(flat)
        ci_hi.index = pd.Index(flat)
        pvalues.index = pd.Index(flat)

    family = type(model).__name__
    # Cox returns exp(coef) as a Hazard Ratio; the AFT family (Weibull,
    # LogNormal, LogLogistic) returns exp(coef) as a Time Ratio. Both are
    # the natural "exponentiate me" output of the fitter, so we set
    # natural_exp=True; the column header label is chosen downstream by
    # ``_default_estimate_label`` in regression.py which selects "HR"
    # for Cox and "TR" for AFT.
    natural_exp = True
    return ModelSummary(
        estimates=estimates,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        pvalues=pvalues,
        family=family,
        natural_exponentiate=natural_exp,
    )


def _find_col(df: pd.DataFrame, prefixes: list[str]) -> str | None:
    # ``df.columns`` items are ``Hashable`` (e.g. tuples for MultiIndex,
    # ints for default-named frames), so coerce to ``str`` for both the
    # match and the return — keeps the declared ``str | None`` return
    # type honest under strict typing.
    for col in df.columns:
        s = str(col).lower()
        if any(s.startswith(p) for p in prefixes):
            return str(col)
    return None


# ----------------------------------------------------------------------
# sklearn (point estimates only; no native CIs)
# ----------------------------------------------------------------------

def _extract_sklearn(model: Any) -> ModelSummary:
    coef = np.atleast_2d(model.coef_)
    n_outputs, n_features = coef.shape

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = np.array([f"x{i}" for i in range(n_features)])
    feature_names = list(feature_names)

    family = type(model).__name__
    natural_exp = "logistic" in family.lower() or "poisson" in family.lower()

    if n_outputs == 1:
        # Binary / single-output: one coefficient vector. Index the
        # ModelSummary by raw feature name.
        labels = list(feature_names)
        values = coef[0, :]
    else:
        # Multi-class (e.g. LogisticRegression(multi_class='multinomial')
        # with 3+ classes, or one-vs-rest). ``coef_`` is
        # (n_classes, n_features); pull the per-class labels from
        # ``model.classes_`` when available. Flatten to one row per
        # (class, feature) pair using the same ``"feature (class=X)"``
        # convention as lifelines AFT models so renderers see clean
        # string labels (the index must be hashable strings — see the
        # AFT path).
        classes = getattr(model, "classes_", None)
        if classes is None:  # pragma: no cover — sklearn fits always set classes_
            classes = np.array([f"class_{k}" for k in range(n_outputs)])
        class_labels = [str(c) for c in classes]
        labels = [
            f"{feat} (class={cls})"
            for cls in class_labels for feat in feature_names
        ]
        values = coef.reshape(-1)

    estimates = pd.Series(values, index=labels, dtype=float)
    nan = pd.Series([float("nan")] * len(labels),
                    index=labels, dtype=float)

    return ModelSummary(
        estimates=estimates,
        ci_lo=nan.copy(),
        ci_hi=nan.copy(),
        pvalues=nan.copy(),
        family=family,
        natural_exponentiate=natural_exp,
    )

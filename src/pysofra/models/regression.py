"""Regression tables — equivalent to ``gtsummary::tbl_regression``.

Supports any of:

* **statsmodels** — ``OLS``, ``GLM``, ``Logit``, ``Probit``, ``Poisson``,
  ``NegativeBinomial``, etc. (anything that exposes
  ``.params`` / ``.pvalues`` / ``.conf_int()``).
* **lifelines** — ``CoxPHFitter``, ``WeibullAFTFitter``,
  ``LogNormalAFTFitter``, and similar regression fitters with ``.summary``.
* **sklearn** — ``LinearRegression``, ``LogisticRegression`` (binary),
  ``Lasso``, ``Ridge`` etc. Point estimates only; sklearn does not expose
  confidence intervals.

Pass a single model for a one-model table, or a list for a side-by-side
multi-model comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.format import fmt_number, fmt_p_value
from ..core.schema import HeaderCell, HeaderRow, Row, make_cell
from ..core.table import SofraTable, TableSpec
from .extract import ModelSummary, extract


def tbl_regression(
    model: Any | list[Any],
    *,
    exponentiate: bool | None = None,
    conf_level: float = 0.95,
    digits: int = 2,
    labels: dict[str, str] | None = None,
    intercept: bool = False,
    estimate_label: str | None = None,
    model_labels: list[str] | None = None,
    design: Any = None,
    data: Any = None,
) -> SofraTable:
    """Build a regression results table.

    Parameters
    ----------
    model
        A fitted model, or a list of fitted models for a multi-model
        side-by-side table.
    exponentiate
        If ``True``, exponentiate point estimates and CI bounds (ORs / HRs
        / IRRs). ``None`` (default) auto-selects: ``True`` for log-link
        models (Logit / Poisson / Cox / Weibull AFT), ``False`` otherwise.
    conf_level
        Confidence level for the CI column (default 95%).
    digits
        Decimal places for estimates and CI bounds.
    labels
        Mapping from coefficient name → display label. Shared across all
        models in a multi-model table.
    intercept
        Whether to include the intercept row.
    estimate_label
        Custom header label for the estimate column. Defaults to ``OR`` /
        ``HR`` / ``IRR`` / ``β`` / ``Estimate`` based on the detected
        model family.
    model_labels
        For multi-model tables, the spanning-header label for each model
        (defaults to ``Model 1``, ``Model 2``, ...).
    design
        Optional :class:`~pysofra.SurveyDesign`. When provided, the fit
        is re-summarised with cluster-robust standard errors (Taylor
        linearisation matching ``survey::svyglm`` to first order). The
        ``data`` argument is required for statsmodels models when a
        design with cluster columns is given.
    data
        Source dataframe — needed only when ``design=`` references
        columns that the fitted model didn't already see.
    """
    if not (0.0 < conf_level < 1.0):
        raise ValueError(
            f"conf_level must lie in the open interval (0, 1); "
            f"got {conf_level!r}."
        )
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    if not models:
        raise ValueError("tbl_regression requires at least one model.")

    if design is not None:
        # ``data`` may be a single DataFrame (shared by every model) or a
        # list of one DataFrame per model when each fit was on a different
        # slice.
        if isinstance(data, (list, tuple)):
            if len(data) != len(models):
                raise ValueError(
                    "When data= is a list it must have one DataFrame per model "
                    f"(got {len(data)} for {len(models)} models)."
                )
            datas = list(data)
        else:
            datas = [data] * len(models)
        models = [
            _refit_with_design(m, design, d)
            for m, d in zip(models, datas, strict=True)
        ]

    summaries = [extract(m, conf_level=conf_level) for m in models]
    labels = dict(labels or {})

    if len(summaries) == 1:
        tbl = _build_single(
            summaries[0],
            exponentiate=exponentiate,
            conf_level=conf_level,
            digits=digits,
            labels=labels,
            intercept=intercept,
            estimate_label=estimate_label,
        )
        # Attach the fitted model so add_global_p() can run Wald F-tests.
        from dataclasses import replace as _replace
        new_md = dict(tbl.metadata)
        new_md["model"] = models[0]
        new_md["design"] = design
        return _replace(tbl, metadata=new_md)

    return _build_multi(
        summaries,
        exponentiate=exponentiate,
        conf_level=conf_level,
        digits=digits,
        labels=labels,
        intercept=intercept,
        estimate_label=estimate_label,
        model_labels=model_labels,
    )


# ----------------------------------------------------------------------
# Single-model
# ----------------------------------------------------------------------

def _build_single(
    summary: ModelSummary,
    *,
    exponentiate: bool | None,
    conf_level: float,
    digits: int,
    labels: dict[str, str],
    intercept: bool,
    estimate_label: str | None,
) -> SofraTable:
    exp = summary.natural_exponentiate if exponentiate is None else bool(exponentiate)
    label = estimate_label or _default_estimate_label(summary.family, exp)

    keep = [n for n in summary.estimates.index if intercept or not _is_intercept(n)]

    header_cells = (
        HeaderCell(text="Variable", align="left"),
        HeaderCell(text=label),
        HeaderCell(text=f"{int(round(conf_level * 100))}% CI"),
        HeaderCell(text="p-value"),
    )
    headers = (HeaderRow(cells=header_cells),)

    rows: list[Row] = []
    for name in keep:
        rows.append(_render_coef_row(
            summary, name, exp=exp, digits=digits, labels=labels,
        ))

    footnotes = _footnotes(summary.family, exp, conf_level, label, has_ci=True)
    spec = TableSpec(
        builder="tbl_regression",
        options={
            "exponentiate": exp,
            "conf_level": conf_level,
            "digits": digits,
            "intercept": intercept,
        },
    )
    return SofraTable(
        rows=tuple(rows),
        headers=headers,
        footnotes=tuple(footnotes),
        metadata={"builder": "tbl_regression", "family": summary.family},
        _spec=spec,
        _rebuild=None,
    )


# ----------------------------------------------------------------------
# Multi-model
# ----------------------------------------------------------------------

def _build_multi(
    summaries: list[ModelSummary],
    *,
    exponentiate: bool | None,
    conf_level: float,
    digits: int,
    labels: dict[str, str],
    intercept: bool,
    estimate_label: str | None,
    model_labels: list[str] | None,
) -> SofraTable:
    if model_labels is not None and len(model_labels) != len(summaries):
        raise ValueError(
            f"model_labels has {len(model_labels)} entries but {len(summaries)} models."
        )
    model_labels = model_labels or [f"Model {i + 1}" for i in range(len(summaries))]

    # Union of coefficient names, ordered by first appearance across models.
    coef_order: list[str] = []
    seen: set[str] = set()
    for s in summaries:
        for n in s.estimates.index:
            if not intercept and _is_intercept(n):
                continue
            if n not in seen:
                seen.add(n)
                coef_order.append(n)

    # Per-model exponentiate decision (each model may have a different link).
    exp_per = [
        s.natural_exponentiate if exponentiate is None else bool(exponentiate)
        for s in summaries
    ]
    labels_per = [
        estimate_label or _default_estimate_label(s.family, e)
        for s, e in zip(summaries, exp_per, strict=True)
    ]

    # Header: Variable, then for each model: {label}, CI, p
    header_cells = [HeaderCell(text="Variable", align="left")]
    spanning = []
    col = 1
    for label, ml in zip(labels_per, model_labels, strict=True):
        spanning.append((ml, col, col + 2))
        header_cells.append(HeaderCell(text=label))
        header_cells.append(HeaderCell(text=f"{int(round(conf_level * 100))}% CI"))
        header_cells.append(HeaderCell(text="p"))
        col += 3
    headers = (HeaderRow(cells=tuple(header_cells)),)

    from ..core.schema import SpanningHeader
    spanning_headers = tuple(
        SpanningHeader(label=ml, start=s, end=e) for ml, s, e in spanning
    )

    rows: list[Row] = []
    for name in coef_order:
        cells = [make_cell(labels.get(name, name), align="left")]
        for s, e in zip(summaries, exp_per, strict=True):
            if name in s.estimates.index:
                est = float(s.estimates[name])
                lo = float(s.ci_lo[name]) if name in s.ci_lo.index else float("nan")
                hi = float(s.ci_hi[name]) if name in s.ci_hi.index else float("nan")
                p = float(s.pvalues[name]) if name in s.pvalues.index else float("nan")
                if e:
                    with np.errstate(over="ignore"):
                        est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
                cells.append(make_cell(fmt_number(est, digits), value=est,
                                       kind="numeric", align="right"))
                cells.append(make_cell(
                    f"{fmt_number(lo, digits)}, {fmt_number(hi, digits)}",
                    value=(lo, hi), kind="ci", align="right",
                ))
                cells.append(make_cell(fmt_p_value(p), value=p, kind="p_value",
                                       align="right"))
            else:
                cells.append(make_cell("—", value=None, align="right"))
                cells.append(make_cell("—", value=None, align="right"))
                cells.append(make_cell("—", value=None, align="right"))
        rows.append(Row(cells=tuple(cells)))

    footnotes: list[str] = []
    for s, e, ml in zip(summaries, exp_per, model_labels, strict=True):
        footnotes.append(f"{ml}: {s.family}{' (exponentiated)' if e else ''}.")
    footnotes.append(f"CI = {int(round(conf_level * 100))}% confidence interval.")

    return SofraTable(
        rows=tuple(rows),
        headers=headers,
        spanning_headers=spanning_headers,
        footnotes=tuple(footnotes),
        metadata={"builder": "tbl_regression", "n_models": len(summaries)},
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _render_coef_row(
    summary: ModelSummary,
    name: str,
    *,
    exp: bool,
    digits: int,
    labels: dict[str, str],
) -> Row:
    est = float(summary.estimates[name])
    lo = float(summary.ci_lo[name]) if name in summary.ci_lo.index else float("nan")
    hi = float(summary.ci_hi[name]) if name in summary.ci_hi.index else float("nan")
    p = float(summary.pvalues[name]) if name in summary.pvalues.index else float("nan")

    if exp:
        # Suppress the standard "overflow encountered in exp" warning from
        # pathological estimates (e.g. perfect-separation logits). The
        # formatter already renders inf as ``—``.
        with np.errstate(over="ignore"):
            est_disp, lo_disp, hi_disp = np.exp(est), np.exp(lo), np.exp(hi)
    else:
        est_disp, lo_disp, hi_disp = est, lo, hi

    label = labels.get(name, name)
    return Row(cells=(
        make_cell(label, align="left"),
        make_cell(fmt_number(est_disp, digits), value=est_disp,
                  kind="numeric", align="right"),
        make_cell(f"{fmt_number(lo_disp, digits)}, {fmt_number(hi_disp, digits)}",
                  value=(lo_disp, hi_disp), kind="ci", align="right"),
        make_cell(fmt_p_value(p), value=p, kind="p_value", align="right"),
    ))


def _footnotes(family: str, exp: bool, conf_level: float, label: str,
               has_ci: bool) -> list[str]:
    out: list[str] = []
    if exp:
        out.append(
            f"{label} = exponentiated coefficient; "
            f"CI = {int(round(conf_level * 100))}% confidence interval."
        )
    else:
        out.append(f"CI = {int(round(conf_level * 100))}% confidence interval.")
    if family:
        out.append(f"Model: {family}.")
    if not has_ci:  # pragma: no cover — every caller currently passes has_ci=True
        out.append("Note: CIs not available for this model type.")
    return out


def _is_intercept(name: str) -> bool:
    return str(name).lower() in {"intercept", "const", "(intercept)"}


def _default_estimate_label(family_label: str, exponentiate: bool) -> str:
    fl = family_label.lower()
    if exponentiate:
        if "cox" in fl or "phreg" in fl:
            return "HR"
        if "weibull" in fl or "lognormal" in fl or "loglogistic" in fl:
            # AFT family: exp(coef) is a TIME RATIO (also called Acceleration
            # Factor), not a hazard ratio. TR > 1 means LONGER survival;
            # HR > 1 means SHORTER survival — the two parameters point in
            # opposite directions. Mislabelling AFT as "HR" is publication-
            # critical because a reader will draw the wrong clinical
            # conclusion.
            return "TR"
        if "logit" in fl or "binomial" in fl or "probit" in fl or "logistic" in fl:
            return "OR"
        if "poisson" in fl or "negativebinomial" in fl:
            return "IRR"
        return "exp(β)"
    if "ols" in fl or "linear" in fl or "gls" in fl or "wls" in fl:
        return "β"
    return "Estimate"


# ----------------------------------------------------------------------
# Design-aware refit (svyglm parity, first-order)
# ----------------------------------------------------------------------

def _refit_with_design(model: Any, design: Any, data: Any) -> Any:
    """Re-fit a statsmodels model using design-aware point estimates + SEs.

    Reproduces R's ``survey::svyglm`` to first order:

    * If ``design.weights`` is set, the model is re-fit with the
      sampling weights folded in — ``WLS(weights=)`` for OLS,
      ``GLM(freq_weights=)`` for binomial / Poisson families. Point
      estimates therefore match a weighted analysis (not the original
      unweighted fit).
    * If ``design.cluster`` is set, the variance estimator switches to
      cluster-robust (``cov_type='cluster'``) keyed by the first-stage
      PSU. Otherwise HC1 is used.

    ``design.strata`` and ``design.fpc`` are not yet exhibited by the
    refit — full strata-aware Taylor linearisation for GLM-family
    regression is outside the scope of statsmodels' built-in variance
    estimators. We log this limitation but do not refuse to run when
    only strata is set; the cluster / HC1 SE remains a valid (if
    conservative) approximation. To get the exact stratified estimator
    use R's ``survey::svyglm`` directly.
    """
    import numpy as np
    import pandas as pd

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "fit"):
        raise ValueError(
            "design= currently supports statsmodels-style results only."
        )

    # Use the original DataFrames where available so coefficient names
    # survive the refit (statsmodels uses the column index as the
    # params index).
    endog = getattr(inner.data, "orig_endog", None)
    if endog is None:  # pragma: no cover — modern statsmodels always attaches orig_endog
        endog = inner.endog
    exog = getattr(inner.data, "orig_exog", None)
    if exog is None:  # pragma: no cover — modern statsmodels always attaches orig_exog
        exog = inner.exog

    # ------------------------------------------------------------------
    # Build the weight + cluster vectors from `data`. They must be in
    # the same row order as endog/exog — statsmodels keeps them in the
    # frame's natural order after `dropna`-style cleaning, so we trust
    # the user to pass the same `data` that the model was fit on.
    # ------------------------------------------------------------------
    weights_arr = None
    if design.weights is not None:
        if data is None:
            raise ValueError(
                "Pass data= to tbl_regression when design has weights."
            )
        w = pd.to_numeric(data[design.weights], errors="coerce").to_numpy(dtype=float)
        if len(w) != len(endog):
            raise ValueError(
                f"design.weights length {len(w)} does not match the model's "
                f"endog length {len(endog)}; pass the same DataFrame the "
                f"model was fit on."
            )
        # Validate weights with the same policy as ``tbl_one``: a refit
        # silently zeroing bad rows would make the SE of a design-aware
        # model disagree silently with what ``tbl_one(design=…)`` produces
        # on the same column. Loud failure here.
        bad_mask = ~(np.isfinite(w) & (w > 0))
        if bool(bad_mask.any()):
            n_bad = int(bad_mask.sum())
            raise ValueError(
                f"design.weights contains {n_bad} non-positive / non-finite "
                "value(s). Drop or correct those rows before calling "
                "tbl_regression(design=…)."
            )
        weights_arr = w

    cluster_arr = None
    if design.cluster is not None:
        # SurveyDesign requires `weights` (no default), so the earlier
        # `if design.weights is not None` branch will already have
        # captured data=None and the length mismatch. Both raises here
        # are kept as defence-in-depth in case SurveyDesign ever gains
        # an optional-weights mode.
        if data is None:  # pragma: no cover — guarded by required design.weights
            raise ValueError(
                "Pass data= to tbl_regression when design has cluster columns."
            )
        clust = data[design.primary_cluster].to_numpy()
        if len(clust) != len(endog):  # pragma: no cover — guarded by upstream length check
            raise ValueError(
                f"design.cluster length {len(clust)} does not match the model's "
                f"endog length {len(endog)}."
            )
        cluster_arr = clust

    # ------------------------------------------------------------------
    # Re-fit point estimates with weights when applicable. We dispatch
    # by the original model class so the user's choice of OLS / Logit /
    # Poisson / GLM is preserved.
    # ------------------------------------------------------------------
    cov_kwds: dict[str, Any] = {}
    if cluster_arr is not None:
        cov_type = "cluster"
        cov_kwds["groups"] = cluster_arr
    else:
        cov_type = "HC1"

    if weights_arr is None:
        # Weight-free path: keep the original model class, just swap
        # in the design-based variance estimator.
        return inner.fit(cov_type=cov_type, cov_kwds=cov_kwds)

    # Weighted path. Pick the correct refit recipe by model family.
    inner_name = type(inner).__name__
    try:
        import statsmodels.api as sm
    except ImportError as e:  # pragma: no cover — guarded by upstream import
        raise ImportError("design= requires statsmodels.") from e

    # statsmodels emits SpecificationWarning when combining var_weights
    # with cov_type='cluster' in GLM. The combination is what every
    # design-based regression library uses (R's survey::svyglm,
    # Stata's svyset/regress), so we suppress the warning locally.
    #
    # We use ``var_weights`` (not ``freq_weights``) for sampling /
    # IPW weights: ``freq_weights`` scales ``df_resid`` by ``Σw``
    # (treating w as an integer count of repeats), which inflates the
    # effective sample size when weights are non-integer and produces
    # anti-conservative SEs. ``var_weights`` keeps ``df_resid = n − k``,
    # which is the SRS-weighted convention matching R ``svyglm`` to
    # first order. (For frequency-weighted analysis where each weight
    # IS an integer count, the user can pre-replicate the data.)
    import warnings as _w
    try:
        from statsmodels.tools.sm_exceptions import SpecificationWarning
    except ImportError:  # pragma: no cover
        SpecificationWarning = Warning

    def _fit(refit: Any) -> Any:
        with _w.catch_warnings():
            _w.simplefilter("ignore", SpecificationWarning)
            return refit.fit(cov_type=cov_type, cov_kwds=cov_kwds)

    if inner_name == "OLS":
        return _fit(sm.WLS(endog, exog, weights=weights_arr))

    if inner_name == "GLM":
        return _fit(sm.GLM(endog, exog, family=inner.family,
                           var_weights=weights_arr))

    if inner_name == "Logit":
        return _fit(sm.GLM(endog, exog, family=sm.families.Binomial(),
                           var_weights=weights_arr))

    if inner_name == "Poisson":
        return _fit(sm.GLM(endog, exog, family=sm.families.Poisson(),
                           var_weights=weights_arr))

    raise NotImplementedError(  # pragma: no cover — exotic statsmodels model
        f"design= with weights does not yet support {inner_name!r}. "
        f"Supported model classes are OLS, GLM, Logit, Poisson. "
        f"For other models, either drop the weights from the design or "
        f"open an issue describing your model."
    )

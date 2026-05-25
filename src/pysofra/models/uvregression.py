"""Univariable regression — one model per predictor, stacked side-by-side.

Equivalent to R ``gtsummary::tbl_uvregression``. For each predictor,
a separate regression of ``outcome ~ predictor`` (optionally
``+ adjust_for``) is fit; results are stacked vertically into a single
table.

Categorical predictors are dummy-encoded (first level = reference). A
group-header row is emitted for each multi-level predictor; each
non-reference level becomes its own indented body row, matching the
gtsummary layout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..core.frames import to_pandas
from ..core.schema import Cell, HeaderCell, HeaderRow, Row, make_cell
from ..core.table import SofraTable, TableSpec
from .extract import extract
from .regression import _default_estimate_label

# Separator used when dummy-encoding a categorical predictor's columns.
# Triple underscore makes accidental collisions with real column names
# vanishingly rare.
_DUMMY_SEP = "___"


def _is_continuous(col: pd.Series) -> bool:
    """A predictor is treated as continuous iff its dtype is numeric
    and *not* boolean (booleans are dichotomous)."""
    return (
        pd.api.types.is_numeric_dtype(col)
        and not pd.api.types.is_bool_dtype(col)
    )


def _expand_predictor(
    sub: pd.DataFrame, pred: str,
) -> tuple[pd.DataFrame, list[tuple[str | None, str, bool]]]:
    """Return ``(design_frame, level_specs)`` for one predictor.

    For a numeric predictor this is the identity: one design column,
    one output row, no reference level.

    For a categorical predictor we drop the first level (the reference)
    and dummy-encode the rest. The returned ``level_specs`` is a list
    of ``(design_column_name_or_None, display_label, is_reference)``
    tuples — one tuple per *displayed* row, ordered top-to-bottom.
    """
    col = sub[pred]
    if _is_continuous(col):
        return pd.DataFrame({pred: col}), [(pred, pred, False)]

    # Categorical / boolean — enumerate levels.
    if isinstance(col.dtype, pd.CategoricalDtype):
        levels: list[Any] = list(col.cat.categories)
    elif pd.api.types.is_bool_dtype(col):
        levels = [False, True]
    else:
        levels = sorted(col.dropna().unique(), key=str)

    if len(levels) < 2:
        # Single-level → nothing to fit.
        empty = pd.DataFrame(index=sub.index)
        return empty, []

    ref = levels[0]
    dummies = pd.get_dummies(col, prefix=pred, prefix_sep=_DUMMY_SEP, dtype=float)
    ref_col = f"{pred}{_DUMMY_SEP}{ref}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    # Drop unused levels — pd.Categorical creates a dummy column for
    # every declared category, even if no observation belongs to it.
    # An all-zero column is collinear with the intercept and breaks the
    # fit; remove it so the reference set excludes phantom levels.
    zero_var = [c for c in dummies.columns if dummies[c].sum() == 0]
    if zero_var:
        dummies = dummies.drop(columns=zero_var)

    # Order rows: reference first (label only, no fit), then each
    # non-reference level. Boolean columns get nicer display labels.
    def _fmt_level(x: Any) -> str:
        if isinstance(x, bool):
            return "Yes" if x else "No"
        return str(x)

    level_specs: list[tuple[str | None, str, bool]] = [
        (None, _fmt_level(ref), True)
    ]
    for lvl in levels[1:]:
        cname = f"{pred}{_DUMMY_SEP}{lvl}"
        if cname not in dummies.columns:
            # Level present in `levels` but every observation in `sub`
            # was a different level (categorical with unused categories).
            # Skip it.
            continue
        level_specs.append((cname, _fmt_level(lvl), False))

    return dummies, level_specs


def tbl_uvregression(
    data: Any,
    *,
    outcome: str,
    predictors: list[str] | None = None,
    method: Callable[..., Any] | str = "OLS",
    method_kwargs: dict[str, Any] | None = None,
    adjust_for: list[str] | None = None,
    exponentiate: bool | None = None,
    conf_level: float = 0.95,
    digits: int = 2,
    labels: dict[str, str] | None = None,
) -> SofraTable:
    """Univariable regression — one model per predictor.

    Parameters
    ----------
    data
        Source dataframe (pandas or polars).
    outcome
        Column name of the response variable.
    predictors
        Predictor columns. Defaults to every column except ``outcome``
        and any ``adjust_for`` covariates (numeric *and* categorical).
    method
        Either a callable that takes ``(y, X)`` and returns a fitted
        statsmodels-style results object, or one of the string aliases
        ``"OLS"``, ``"Logit"``, ``"Poisson"``, ``"GLM"``.
    method_kwargs
        Extra keyword arguments forwarded to the model class.
    adjust_for
        Optional list of covariates included in every univariable fit
        (matching ``gtsummary``'s ``include`` argument). Adjustment
        covariates are themselves dummy-encoded if categorical.
    exponentiate
        If ``True``, exponentiate point estimates and CI bounds.
        ``None`` (default) auto-selects based on the model family.
    conf_level
        Confidence level for the CI column.
    digits
        Decimal places for estimates and CI bounds.
    labels
        Mapping from predictor name → display label. Applied to the
        group-header row for categorical predictors.

    Notes
    -----
    For a categorical predictor with K levels the result has
    ``K`` rows: a header naming the variable, plus ``K`` indented
    rows (the reference level rendered as ``— ref``, and one row
    per non-reference level with its estimate / CI / p-value).
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "tbl_uvregression requires statsmodels. "
            "Install with `pip install statsmodels`."
        ) from e

    df = to_pandas(data)
    if outcome not in df.columns:
        raise KeyError(f"outcome column {outcome!r} not in data")
    df = df.dropna(subset=[outcome])

    adjust_for = list(adjust_for or [])
    if predictors is None:
        excluded = {outcome, *adjust_for}
        predictors = [c for c in df.columns if c not in excluded]

    # Predictor / adjust_for overlap doesn't make sense ("regress y on x
    # adjusted for x") and would also break design-matrix assembly
    # because pandas returns a DataFrame, not a Series, when a column
    # name is duplicated in a slice.
    overlap = sorted(set(predictors) & set(adjust_for))
    if overlap:
        raise ValueError(
            f"Predictor(s) {overlap} also appear in adjust_for; remove "
            f"from one of the two."
        )
    if outcome in predictors:
        raise ValueError(
            f"outcome {outcome!r} must not appear in predictors."
        )
    if outcome in adjust_for:
        raise ValueError(
            f"outcome {outcome!r} must not appear in adjust_for."
        )

    labels = dict(labels or {})
    method_kwargs = dict(method_kwargs or {})

    model_factory: Callable[..., Any]
    if callable(method):
        model_factory = method
    elif method == "OLS":
        model_factory = sm.OLS
    elif method == "Logit":
        model_factory = sm.Logit
    elif method == "Poisson":
        model_factory = sm.Poisson
    elif method == "GLM":
        model_factory = sm.GLM
    else:
        raise ValueError(
            f"Unknown method {method!r}; pass a callable or one of "
            "'OLS', 'Logit', 'Poisson', 'GLM'."
        )

    # Pre-expand the adjust_for block once — it's shared across rows.
    if adjust_for:
        sub_adjust = df[adjust_for]
        adjust_design_blocks: list[pd.DataFrame] = []
        for a in adjust_for:
            adj_design, _ = _expand_predictor(sub_adjust, a)
            adjust_design_blocks.append(adj_design)
        adjust_block_template = (
            pd.concat(adjust_design_blocks, axis=1)
            if adjust_design_blocks else pd.DataFrame(index=df.index)
        )
    else:
        adjust_block_template = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Fit one regression per predictor.
    # ------------------------------------------------------------------
    rows: list[Row] = []
    families: set[str] = set()
    exp_per: list[bool] = []
    failed: list[str] = []

    for pred in predictors:
        # Build the working sub-frame: outcome + adjust_for + this predictor.
        cols_needed = [outcome, pred, *adjust_for]
        sub = df[cols_needed].dropna()
        if sub.empty:
            failed.append(pred)
            continue

        pred_design, level_specs = _expand_predictor(sub, pred)
        if pred_design.empty or not level_specs:
            failed.append(pred)
            continue

        adjust_block = (
            adjust_block_template.loc[sub.index]
            if not adjust_block_template.empty else None
        )
        # Stitch predictor + adjust into a single design matrix.
        if adjust_block is not None and not adjust_block.empty:
            design_X = pd.concat([pred_design, adjust_block], axis=1)
        else:
            design_X = pred_design.copy()
        X = sm.add_constant(design_X)

        try:
            fit = model_factory(sub[outcome], X, **method_kwargs).fit(disp=False)
        except Exception:
            failed.append(pred)
            continue

        summary = extract(fit, conf_level=conf_level)
        families.add(summary.family)
        exp = summary.natural_exponentiate if exponentiate is None else bool(exponentiate)
        exp_per.append(exp)

        n_sub = int(len(sub))
        display_label = labels.get(pred, pred)
        n_levels = len(level_specs)
        is_categorical_predictor = n_levels > 1 or (
            n_levels == 1 and level_specs[0][2]  # reference-only edge case
        )

        if not is_categorical_predictor:
            # Numeric / continuous → single row, no header.
            design_col = level_specs[0][0]
            assert design_col is not None
            if design_col not in summary.estimates.index:
                failed.append(pred)
                continue
            rows.append(_one_predictor_row(
                design_col, summary, exp=exp, digits=digits,
                label=display_label, n=n_sub, indent=0,
                source_name=pred,
            ))
            continue

        # Multi-level categorical → group header + one row per level.
        rows.append(_group_header_row(
            display_label, source_name=pred, n=n_sub, n_cols=5,
        ))
        for design_col, lvl_label, is_ref in level_specs:
            if is_ref:
                rows.append(_reference_row(lvl_label, source_name=pred))
                continue
            if design_col is None or design_col not in summary.estimates.index:  # pragma: no cover — would require a singular fit dropping a non-ref column
                failed.append(f"{pred}={lvl_label}")
                continue
            # Count the level by summing the dummy column (avoids
            # string-vs-bool comparison pitfalls when reversing the
            # mangle).
            level_n = int(pred_design[design_col].sum())
            rows.append(_one_predictor_row(
                design_col, summary, exp=exp, digits=digits,
                label=lvl_label, n=level_n,
                indent=1, source_name=pred,
            ))

    if not rows and not failed:
        raise ValueError("No predictors produced a coefficient.")

    # Estimate label uses the first family / first exponentiate setting.
    family_label = next(iter(families)) if families else "?"
    est_label = _default_estimate_label(family_label, exp_per[0] if exp_per else False)

    headers = (HeaderRow(cells=(
        HeaderCell(text="Predictor", align="left"),
        HeaderCell(text="N"),
        HeaderCell(text=est_label),
        HeaderCell(text=f"{int(round(conf_level * 100))}% CI"),
        HeaderCell(text="p-value"),
    )),)

    footnotes: list[str] = []
    if adjust_for:
        footnotes.append("Each variable's coefficient is adjusted for: "
                         f"{', '.join(adjust_for)}.")
    else:
        footnotes.append(
            "Each row is a univariable regression of the outcome on the "
            "named predictor."
        )
    if any(exp_per):
        footnotes.append(
            f"{est_label} = exponentiated coefficient; "
            f"CI = {int(round(conf_level * 100))}% confidence interval."
        )
    else:
        footnotes.append(f"CI = {int(round(conf_level * 100))}% confidence interval.")
    if families:
        footnotes.append(f"Model: {next(iter(families))}.")
    if failed:
        footnotes.append(
            f"{len(failed)} predictor(s) / level(s) failed to converge or "
            f"had no data: {', '.join(failed)}."
        )

    spec = TableSpec(
        builder="tbl_uvregression",
        options={
            "outcome": outcome,
            "predictors": tuple(predictors),
            "method": method if isinstance(method, str) else method.__name__,
            "exponentiate": exponentiate,
            "conf_level": conf_level,
            "digits": digits,
        },
    )
    return SofraTable(
        rows=tuple(rows),
        headers=headers,
        footnotes=tuple(footnotes),
        metadata={
            "builder": "tbl_uvregression",
            "family": next(iter(families), None),
            "failed": failed,
        },
        _spec=spec,
    )


def _group_header_row(label: str, *, source_name: str, n: int, n_cols: int) -> Row:
    """Bold predictor-name row introducing a categorical predictor's levels."""
    cells = [make_cell(label, align="left", bold=True)]
    cells.append(make_cell(str(n), value=n, kind="numeric", align="right"))
    for _ in range(n_cols - 2):
        cells.append(Cell(text="", value=None))
    return Row(cells=tuple(cells), is_group_header=True,
               metadata={"variable": source_name})


def _reference_row(level_label: str, *, source_name: str) -> Row:
    """The reference level — no estimate, marked '— ref'."""
    return Row(cells=(
        make_cell(level_label, align="left", indent=1),
        Cell(text="", value=None),
        make_cell("— ref", value=None, kind="numeric", align="right"),
        Cell(text="", value=None, kind="ci"),
        Cell(text="", value=None, kind="p_value"),
    ), metadata={"variable": source_name})


def _one_predictor_row(
    design_col: str,
    summary: Any,  # ModelSummary
    *,
    exp: bool,
    digits: int,
    label: str,
    n: int,
    indent: int = 0,
    source_name: str | None = None,
) -> Row:
    from math import exp as _exp

    from ..core.format import fmt_number, fmt_p_value

    est = float(summary.estimates[design_col])
    lo = float(summary.ci_lo[design_col]) if design_col in summary.ci_lo.index else float("nan")
    hi = float(summary.ci_hi[design_col]) if design_col in summary.ci_hi.index else float("nan")
    p = float(summary.pvalues[design_col]) if design_col in summary.pvalues.index else float("nan")

    def _safe_exp(x: float) -> float:
        try:
            return _exp(x)
        except OverflowError:
            return float("inf") if x > 0 else 0.0

    if exp:
        est_d, lo_d, hi_d = _safe_exp(est), _safe_exp(lo), _safe_exp(hi)
    else:
        est_d, lo_d, hi_d = est, lo, hi

    return Row(cells=(
        make_cell(label, align="left", indent=indent),
        make_cell(str(n), value=n, kind="numeric", align="right"),
        make_cell(fmt_number(est_d, digits), value=est_d,
                  kind="numeric", align="right"),
        make_cell(f"{fmt_number(lo_d, digits)}, {fmt_number(hi_d, digits)}",
                  value=(lo_d, hi_d), kind="ci", align="right"),
        make_cell(fmt_p_value(p), value=p, kind="p_value", align="right"),
    ), metadata={"variable": source_name} if source_name else {})

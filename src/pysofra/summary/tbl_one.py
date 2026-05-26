"""Table 1 builder — baseline characteristic tables.

Mirrors the workflow of R's ``tableone`` while staying Pythonic:

.. code-block:: python

    import pysofra as ps

    (
        ps.tbl_one(df, by="treatment")
          .add_p()
          .add_smd()
          .add_overall()
    )

The function returns a :class:`~pysofra.core.SofraTable` that renders
beautifully in notebooks and exports to HTML/Markdown/DOCX.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.format import (
    fmt_mean_sd,
    fmt_median_iqr,
    fmt_p_value,
    fmt_smd,
)
from ..core.frames import to_pandas
from ..core.schema import Cell, HeaderCell, HeaderRow, Row, make_cell
from ..core.table import SofraTable, TableSpec
from .design import SurveyDesign, design_mean_var, replicate_mean_var
from .smd import categorical_smd, continuous_smd
from .stats import continuous_stats
from .tests import (
    categorical_test,
    continuous_test,
    rao_scott_chisq,
    run_named_test,
    svyttest,
)
from .typing import VarKind, apply_overrides, infer_kind
from .weights import weighted_continuous_stats


def _is_cat_dtype(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


DEFAULT_OVERALL_LABEL = "Overall"
MISSING_LABEL = "Missing"


def tbl_one(
    data: pd.DataFrame,
    *,
    by: str | None = None,
    variables: list[str] | None = None,
    labels: dict[str, str] | None = None,
    types: dict[str, VarKind] | None = None,
    nonnormal: list[str] | None = None,
    tests: dict[str, str] | None = None,
    weights: str | None = None,
    design: SurveyDesign | None = None,
    digits: int = 2,
    pct_digits: int = 1,
    missing: str = "ifany",
    include_missing: bool | None = None,
) -> SofraTable:
    """Build a Table 1.

    Parameters
    ----------
    data
        Source dataframe.
    by
        Optional column name to stratify on. If omitted, a single
        ``Overall`` column is produced.
    variables
        Explicit list of variables to include. Defaults to all columns
        other than ``by``.
    labels
        Mapping of column name → display label.
    types
        Override automatic variable typing on a per-column basis.
    nonnormal
        Continuous variables that should be summarised as
        ``median (Q1, Q3)`` and tested with rank-based tests.
    tests
        Per-variable statistical test overrides, e.g.
        ``{'age': 'wilcoxon', 'race': 'fisher'}``. See
        :func:`pysofra.summary.tests.available_tests` for the registry.
    weights
        Column name carrying non-negative frequency weights. When
        supplied, continuous summaries become weighted means / variances
        and categorical summaries become weighted proportions. The
        weights column is excluded from the variable list automatically.
    design
        A :class:`SurveyDesign` describing a complex sampling structure
        (weights + optional strata, clusters, and FPC). When provided,
        variance estimates use Taylor linearisation instead of the
        simple frequency-weighted formula. If both ``weights`` and
        ``design`` are passed, ``design`` wins.
    digits
        Decimal places for continuous summaries.
    pct_digits
        Decimal places for percentages.
    missing
        ``"ifany"`` (default) — include a *Missing* row only when there is
        missing data; ``"always"`` — always include; ``"never"``.
    include_missing
        Deprecated alias for ``missing``. ``True`` maps to ``"ifany"``,
        ``False`` to ``"never"``.
    """
    if include_missing is not None:
        missing = "ifany" if include_missing else "never"
    if missing not in ("ifany", "always", "never"):
        raise ValueError("missing must be one of 'ifany', 'always', 'never'")

    data = to_pandas(data)
    # Duplicate-column-name check — without this, downstream pandas calls
    # raise a confusing ``AttributeError`` on the duplicated Series.
    duplicate_cols = [c for c in data.columns if list(data.columns).count(c) > 1]
    if duplicate_cols:
        raise ValueError(
            f"tbl_one cannot accept a DataFrame with duplicate column names; "
            f"got duplicates: {sorted(set(duplicate_cols))}."
        )
    if by is not None and by not in data.columns:
        raise KeyError(f"by column {by!r} not in data")
    if design is not None and weights is not None and weights != design.weights:
        import warnings
        warnings.warn(
            f"Both weights={weights!r} and design= were provided; "
            f"using design.weights={design.weights!r}. Pass only one.",
            UserWarning,
            stacklevel=2,
        )
    if design is not None:
        design.validate(data)
        weights = design.weights
    if weights is not None and weights not in data.columns:
        raise KeyError(f"weights column {weights!r} not in data")
    if weights is not None:
        # Reject negative or all-zero weights. Earlier alphas merely warned
        # and "dropped" the offending rows, but that left the displayed
        # output in inconsistent states (e.g. ``N = -1`` in a group column
        # when ``int(Σw)`` was computed naively, or ``N = 0`` when every
        # weight was zero). For a publication-ready library it's safer to
        # raise loudly: the user must fix their weights column.
        w_col = pd.to_numeric(data[weights], errors="coerce")
        n_neg = int((w_col < 0).sum())
        if n_neg > 0:
            raise ValueError(
                f"weights column {weights!r} contains {n_neg} negative value(s). "
                "Negative weights are not supported; drop or correct them "
                "before calling tbl_one()."
            )
        total = float(w_col.fillna(0.0).sum())
        if total <= 0:
            raise ValueError(
                f"weights column {weights!r} has total weight {total!r}; "
                "at least one positive weight is required."
            )
    excluded: set[str] = {c for c in (by, weights) if c is not None}
    if design is not None:
        if design.strata is not None:
            excluded.add(design.strata)
        if design.cluster is not None:
            if isinstance(design.cluster, tuple):
                excluded.update(design.cluster)
            else:
                excluded.add(design.cluster)
        if design.fpc is not None:
            excluded.add(design.fpc)
        if design.replicate_weights is not None:
            excluded.update(design.replicate_weights)
    if variables is None:
        variables = [c for c in data.columns if c not in excluded]
    else:
        missing_cols = [v for v in variables if v not in data.columns]
        if missing_cols:
            raise KeyError(f"variables not in data: {missing_cols}")
        # Reject duplicate entries early — silently de-duplicating would
        # produce a table whose row count doesn't match the user's list,
        # while keeping duplicates would emit identical rows.
        seen: dict[str, int] = {}
        for v in variables:
            seen[v] = seen.get(v, 0) + 1
        dupes = [v for v, n in seen.items() if n > 1]
        if dupes:
            raise ValueError(
                f"variables contains duplicate names: {dupes}. "
                "Each variable may appear at most once."
            )
        # Warn when the user-supplied variables list overlaps the design /
        # stratification columns; silently dropping them is surprising.
        overlap = [v for v in variables if v in excluded]
        if overlap:
            import warnings
            warnings.warn(
                f"variables={overlap} overlap with the by= / weights / design "
                "columns and were excluded.",
                UserWarning,
                stacklevel=2,
            )
        variables = [v for v in variables if v not in excluded]

    labels = dict(labels or {})
    nonnormal_set = set(nonnormal or [])
    inferred = {v: infer_kind(data[v]) for v in variables}
    kinds = apply_overrides(inferred, types)
    tests_map = dict(tests or {})

    # Warn the user if labels / nonnormal / tests reference columns that
    # are NOT in the final variable list — those entries are silently
    # ignored otherwise, leading to wrong tests (a ``nonnormal=["hbac1"]``
    # typo → Welch instead of Wilcoxon, no warning), wrong row labels
    # (``labels={"hbac1": "HbA1c"}`` typo → raw column name in the table),
    # and wrong tests (``tests={"hbac1": "wilcoxon"}`` typo → default).
    _var_set = set(variables)
    _bad_labels = sorted(set(labels) - _var_set)
    _bad_nonnormal = sorted(nonnormal_set - _var_set)
    _bad_tests = sorted(set(tests_map) - _var_set)
    if _bad_labels or _bad_nonnormal or _bad_tests:
        import warnings
        msgs = []
        if _bad_labels:
            msgs.append(f"labels={_bad_labels}")
        if _bad_nonnormal:
            msgs.append(f"nonnormal={_bad_nonnormal}")
        if _bad_tests:
            msgs.append(f"tests={_bad_tests}")
        warnings.warn(
            f"Variables referenced but not in the table: {'; '.join(msgs)}. "
            f"Check for typos against {sorted(_var_set)!r}. The entries were "
            "ignored.",
            UserWarning,
            stacklevel=2,
        )

    spec = TableSpec(
        builder="tbl_one",
        options={
            "by": by,
            "variables": tuple(variables),
            "labels": labels,
            "kinds": kinds,
            "nonnormal": frozenset(nonnormal_set),
            "tests": tests_map,
            "weights": weights,
            "design": design,
            "digits": int(digits),
            "pct_digits": int(pct_digits),
            "missing": missing,
            "p_value": False,
            "smd": False,
            "overall": False,
            "overall_label": DEFAULT_OVERALL_LABEL,
            "q_value": False,
            "q_method": "fdr_bh",
        },
    )
    # We close over the *original data* so spec changes don't lose it.
    return _build(data, spec)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _build(data: pd.DataFrame, spec: TableSpec) -> SofraTable:
    """Construct a SofraTable from a (data, spec) pair."""
    opts = spec.options
    by: str | None = opts["by"]
    variables: tuple[str, ...] = opts["variables"]
    labels: dict[str, str] = opts["labels"]
    kinds: dict[str, VarKind] = opts["kinds"]
    nonnormal: frozenset[str] = opts["nonnormal"]
    tests_map: dict[str, str] = opts.get("tests", {}) or {}
    weights_col: str | None = opts.get("weights")
    design: SurveyDesign | None = opts.get("design")
    digits: int = opts["digits"]
    pct_digits: int = opts["pct_digits"]
    missing_mode: str = opts["missing"]
    show_p: bool = bool(opts["p_value"])
    show_smd: bool = bool(opts["smd"])
    show_overall: bool = bool(opts["overall"])
    overall_label: str = opts["overall_label"]
    show_q: bool = bool(opts.get("q_value"))
    q_method: str = opts.get("q_method", "fdr_bh")
    bold_p_mode: bool = False
    bold_p_threshold: float = 0.05

    if by is None:
        group_keys: list[Any] = [overall_label]
        group_masks = {overall_label: pd.Series(True, index=data.index)}
        show_overall = False  # already overall-only
    else:
        # Drop rows missing the stratification variable; record N dropped.
        by_series = data[by]
        # Preserve categorical / sorted order
        if _is_cat_dtype(by_series):
            group_keys = [k for k in by_series.cat.categories if (by_series == k).any()]
        else:
            group_keys = sorted(by_series.dropna().unique(), key=_sort_key)
        # cast to plain list[Any]
        group_keys = list(group_keys)
        group_masks = {k: (by_series == k) for k in group_keys}
        # Guard against degenerate ``by=`` columns
        # silently produced an unstratified or empty table. Match the R9
        # policy (clear UserWarning when the user's intent doesn't match
        # the input) so the user knows to drop ``by=`` or fix the column.
        if len(group_keys) == 0:
            import warnings
            warnings.warn(
                f"by={by!r} has no non-missing values; the resulting table "
                "has no stratification columns. Pass by=None for an "
                "explicitly unstratified summary, or fix the column.",
                UserWarning,
                stacklevel=2,
            )
        elif len(group_keys) == 1:
            import warnings
            warnings.warn(
                f"by={by!r} has only one non-missing level "
                f"({group_keys[0]!r}); the resulting table has a single "
                "stratum column and no between-group statistics. Pass "
                "by=None for an explicitly unstratified summary.",
                UserWarning,
                stacklevel=2,
            )

    if weights_col is not None:
        w_series = pd.to_numeric(data[weights_col], errors="coerce").fillna(0.0)
        n_per_group = {k: float(w_series[group_masks[k]].sum()) for k in group_keys}
        n_overall = (
            float(w_series.sum())
            if by is None
            else float(w_series[data[by].notna()].sum())
        )
    else:
        w_series = None
        n_per_group = {k: int(group_masks[k].sum()) for k in group_keys}
        n_overall = int(len(data)) if by is None else int(data[by].notna().sum())

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------
    header_cells: list[HeaderCell] = [HeaderCell(text="Characteristic", align="left")]

    def _fmt_n(val: float | int) -> str:
        if isinstance(val, float):
            return f"{val:,.1f}" if val != int(val) else f"{int(val):,}"
        return f"{val:,}"

    if show_overall:
        header_cells.append(
            HeaderCell(text=f"{overall_label}\nN = {_fmt_n(n_overall)}")
        )
    for k in group_keys:
        header_cells.append(
            HeaderCell(text=f"{_fmt_level(k)}\nN = {_fmt_n(n_per_group[k])}")
        )
    if show_p:
        header_cells.append(HeaderCell(text="p-value"))
    if show_q:
        header_cells.append(HeaderCell(text="q-value"))
    if show_smd:
        header_cells.append(HeaderCell(text="SMD"))

    headers: tuple[HeaderRow, ...] = (HeaderRow(cells=tuple(header_cells)),)

    # ------------------------------------------------------------------
    # Body rows
    # ------------------------------------------------------------------
    rows: list[Row] = []
    test_names: set[str] = set()

    for var in variables:
        kind = kinds[var]
        label = labels.get(var, var)
        is_nonnormal = var in nonnormal

        test_override = tests_map.get(var)

        if kind == "continuous":
            row_blocks, test_used = _continuous_rows(
                data, var, label, by, group_keys, group_masks,
                digits=digits,
                pct_digits=pct_digits,
                show_overall=show_overall,
                show_p=show_p,
                show_q=show_q,
                show_smd=show_smd,
                nonnormal=is_nonnormal,
                missing_mode=missing_mode,
                bold_p_mode=bold_p_mode,
                bold_p_threshold=bold_p_threshold,
                test_override=test_override,
                weights=w_series,
                design=design,
            )
            rows.extend(row_blocks)
            if test_used:
                test_names.add(test_used)
        else:
            row_blocks, test_used = _categorical_rows(
                data, var, label, by, group_keys, group_masks,
                kind=kind,
                pct_digits=pct_digits,
                show_overall=show_overall,
                show_p=show_p,
                show_q=show_q,
                show_smd=show_smd,
                missing_mode=missing_mode,
                bold_p_mode=bold_p_mode,
                bold_p_threshold=bold_p_threshold,
                test_override=test_override,
                weights=w_series,
                design=design,
            )
            rows.extend(row_blocks)
            if test_used:
                test_names.add(test_used)

    # ------------------------------------------------------------------
    # Footnotes
    # ------------------------------------------------------------------
    footnotes: list[str] = []
    # Continuous summary footnote
    cont_vars = [v for v in variables if kinds[v] == "continuous"]
    nn_vars = [v for v in cont_vars if v in nonnormal]
    nm_vars = [v for v in cont_vars if v not in nonnormal]
    design_with_variance = (
        design is not None and weights_col is not None
        and (design.strata is not None or design.cluster is not None)
    )
    if nm_vars and design_with_variance:
        footnotes.append(
            "Mean (SE) for continuous variables (design-based "
            "Taylor-linearised variance)."
        )
    elif nm_vars:
        footnotes.append("Mean (SD) for continuous variables.")
    if nn_vars:
        labelled = ", ".join(labels.get(v, v) for v in nn_vars)
        footnotes.append(f"Median (Q1, Q3) for: {labelled}.")
    cat_vars = [v for v in variables if kinds[v] != "continuous"]
    if cat_vars:
        footnotes.append("n (%) for categorical variables.")
    if show_p and test_names:
        footnotes.append("Tests: " + "; ".join(sorted(test_names)) + ".")
    if show_q:
        footnotes.append(f"q-value = {_q_method_label(q_method)} adjusted p-value.")
    if show_smd:
        footnotes.append("SMD = standardized mean difference (max pairwise).")

    if show_q:
        rows = _patch_q_values(rows, method=q_method)

    # ------------------------------------------------------------------
    # add_global_p() — joint Wald p-value per variable, fitted as
    # ``Logit(by == reference_level ~ variable + adjust_for)`` for
    # each variable. Adds a "global p" column to the right of any
    # existing p-value column. Requires a 2-level ``by=``; >2-level
    # ``by=`` is out of scope for v1 (would require multinomial
    # logit).
    # ------------------------------------------------------------------
    if opts.get("global_p"):
        rows, headers, footnotes_extra = _attach_global_p(
            data=data, by=by,
            variables=variables, kinds=kinds, labels=labels,
            rows=rows, headers=headers,
            adjust_for=tuple(opts.get("global_p_adjust_for", ()) or ()),
            weights_col=weights_col,
        )
        footnotes = list(footnotes) + list(footnotes_extra)

    def _rebuild_fn(new_spec: TableSpec) -> SofraTable:
        return _build(data, new_spec)

    return SofraTable(
        rows=tuple(rows),
        headers=headers,
        footnotes=tuple(footnotes),
        metadata={"builder": "tbl_one", "tests": sorted(test_names)},
        _spec=spec,
        _rebuild=_rebuild_fn,
    )


# ----------------------------------------------------------------------
# q-value post-processing
# ----------------------------------------------------------------------

_Q_METHOD_LABELS = {
    "fdr_bh": "Benjamini–Hochberg",
    "fdr_by": "Benjamini–Yekutieli",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "hommel": "Hommel",
    "sidak": "Šidák",
}


def _q_method_label(method: str) -> str:
    return _Q_METHOD_LABELS.get(method, method)


def _patch_q_values(rows: list[Row], *, method: str) -> list[Row]:
    """Walk rows, collect p-values, compute q-values, patch q-value cells."""
    # Find rows that have *both* a numeric p-value cell and a q-value placeholder.
    # NaN p-values are silently skipped — feeding them to ``multipletests``
    # contaminates the entire adjustment (statsmodels returns NaN for every
    # output), which would wrongly null out the q-values of valid rows.
    import math
    from dataclasses import replace as dc_replace

    from statsmodels.stats.multitest import multipletests

    pairs: list[tuple[int, int, int, float]] = []  # (row_idx, p_col, q_col, p_val)
    for i, r in enumerate(rows):
        p_col = q_col = None
        for j, c in enumerate(r.cells):
            if c.kind == "p_value" and isinstance(c.value, (int, float)):
                if math.isnan(float(c.value)):
                    p_col = None  # poison; skip this row entirely
                    break
                p_col = j
            elif c.kind == "q_value":
                q_col = j
        if p_col is not None and q_col is not None:
            pairs.append((i, p_col, q_col, float(rows[i].cells[p_col].value)))

    if not pairs:
        return rows

    pvals = [p for _, _, _, p in pairs]
    _, qvals, _, _ = multipletests(pvals, method=method)

    new_rows = list(rows)
    for (i, _p_col, q_col, _p), q in zip(pairs, qvals, strict=True):
        old_row = new_rows[i]
        new_cells = list(old_row.cells)
        new_cells[q_col] = dc_replace(
            new_cells[q_col],
            text=fmt_p_value(float(q), digits=3),
            value=float(q),
        )
        new_rows[i] = dc_replace(old_row, cells=tuple(new_cells))
    return new_rows


def _fmt_level(k: Any) -> str:
    if isinstance(k, bool):
        return "Yes" if k else "No"
    return str(k)


def _fmt_weighted_n_pct(n: float, total: float, pct_digits: int) -> str:
    """Render ``n (xx.x%)`` with weighted (possibly non-integer) counts."""
    if total <= 0:
        n_str = f"{n:,.1f}" if n != int(n) else f"{int(n):,}"
        return f"{n_str} (—)"
    pct = 100.0 * n / total
    n_str = f"{n:,.1f}" if n != int(n) else f"{int(n):,}"
    return f"{n_str} ({pct:.{pct_digits}f}%)"


def _sort_key(x: Any) -> tuple[int, Any]:
    if isinstance(x, bool):
        return (0, int(x))
    if isinstance(x, (int, float)):
        return (0, float(x))
    if isinstance(x, str):
        return (1, x)
    return (2, repr(x))


# ----------------------------------------------------------------------
# Continuous rows
# ----------------------------------------------------------------------

def _continuous_rows(
    data: pd.DataFrame,
    var: str,
    label: str,
    by: str | None,
    group_keys: list[Any],
    group_masks: dict[Any, pd.Series],
    *,
    digits: int,
    pct_digits: int,
    show_overall: bool,
    show_p: bool,
    show_q: bool,
    show_smd: bool,
    nonnormal: bool,
    missing_mode: str,
    bold_p_mode: bool,
    bold_p_threshold: float,
    test_override: str | None = None,
    weights: pd.Series | None = None,
    design: SurveyDesign | None = None,
) -> tuple[list[Row], str | None]:
    """Produce 1 (+ optional missing) rows for one continuous variable."""

    def _summary_for(mask: pd.Series) -> str:
        # Design-based: report mean (SE) when the user has opted into a
        # complex design (strata or cluster). For weight-only designs we
        # fall through to the frequency-weighted mean (SD) path below.
        if design is not None and weights is not None and (
            design.strata is not None
            or design.cluster is not None
            or design.replicate_weights is not None
        ):
            if design.replicate_weights is not None:
                rep_series = [data.loc[mask, c] for c in design.replicate_weights]
                mean, var_, n_eff = replicate_mean_var(
                    data.loc[mask, var],
                    weights.loc[mask],
                    rep_series,
                    replicate_type=design.replicate_type,
                )
            else:
                mean, var_, n_eff = design_mean_var(
                    data.loc[mask, var],
                    weights.loc[mask],
                    strata=(data.loc[mask, design.strata]
                            if design.strata else None),
                    cluster=(data.loc[mask, design.primary_cluster]
                             if design.cluster else None),
                    fpc=(data.loc[mask, design.fpc]
                         if design.fpc else None),
                )
            if n_eff <= 0:
                return "—"
            import math
            se = math.sqrt(max(var_, 0.0)) if not math.isnan(var_) else float("nan")
            return fmt_mean_sd(mean, se, digits=digits)
        if weights is not None:
            st = weighted_continuous_stats(data.loc[mask, var], weights.loc[mask])
            if st.n_eff <= 0:
                return "—"
            if nonnormal:
                return fmt_median_iqr(st.median, st.q1, st.q3, digits=digits)
            return fmt_mean_sd(st.mean, st.sd, digits=digits)
        cs = continuous_stats(data.loc[mask, var])
        if cs.n == 0:
            return "—"
        if nonnormal:
            return fmt_median_iqr(cs.median, cs.q1, cs.q3, digits=digits)
        return fmt_mean_sd(cs.mean, cs.sd, digits=digits)

    p_value: float | None = None
    test_used: str | None = None
    if show_p and by is not None:
        if test_override is not None:
            res = run_named_test(test_override, data[var], data[by], kind="continuous")
        elif weights is not None:
            # Any weighted call gets the design-adjusted two-sample
            # t-test (Taylor-linearised; ``svyttest`` analogue) on the
            # 2-group case. Strata/cluster from ``design`` are honoured
            # when present; bare ``weights=`` falls through with
            # ``strata=None``/``cluster=None``, which still gives a
            # weighted SE rather than the unweighted Welch fallback the
            # earlier behaviour silently produced.
            two_grp = len(set(data[by].dropna().unique())) == 2
            if two_grp:
                strata_col = (data[design.strata]
                              if design is not None and design.strata else None)
                cluster_col = (data[design.primary_cluster]
                               if design is not None and design.primary_cluster
                               else None)
                res = svyttest(
                    data[var], data[by], weights,
                    strata=strata_col,
                    cluster=cluster_col,
                )
            else:
                # >2 groups under weights: design-adjusted F-test
                # (svyglm-based) is not yet implemented. We emit a
                # one-time ``UserWarning`` rather than silently
                # returning an unweighted p-value — the auditor's red
                # flag for prior alphas was that this fallback was
                # invisible to the user on a weighted Table 1.
                import warnings as _w
                _w.warn(
                    f"continuous variable {var!r} has >2 groups under "
                    "weighted Table 1: design-adjusted F-test is not "
                    "implemented; falling back to the *unweighted* "
                    "ANOVA / Kruskal-Wallis p-value. For a weighted "
                    "test on multi-group data, run the comparison "
                    "manually via R survey::svyglm or a pairwise "
                    "weighted t-test (svyttest) with multiplicity "
                    "adjustment.",
                    UserWarning,
                    stacklevel=2,
                )
                res = continuous_test(data[var], data[by], nonnormal=nonnormal)
        else:
            res = continuous_test(data[var], data[by], nonnormal=nonnormal)
        p_value = res.p_value
        test_used = res.test if res.p_value is not None else None

    smd_val: float | None = None
    if show_smd and by is not None:
        smd_val = continuous_smd(data[var], data[by], weights=weights)

    bold_row = (
        bold_p_mode
        and p_value is not None
        and p_value < bold_p_threshold
    )

    cells: list[Cell] = [make_cell(label, align="left", bold=bold_row)]
    if show_overall:
        cells.append(
            make_cell(_summary_for(pd.Series(True, index=data.index)),
                      kind="numeric", align="right")
        )
    for k in group_keys:
        cells.append(
            make_cell(_summary_for(group_masks[k]), kind="numeric", align="right")
        )
    if show_p:
        cells.append(make_cell(fmt_p_value(p_value), value=p_value, kind="p_value",
                               align="right", bold=bold_row))
    if show_q:
        # Placeholder; patched by _patch_q_values after build.
        cells.append(make_cell("", value=None, kind="q_value", align="right"))
    if show_smd:
        cells.append(make_cell(fmt_smd(smd_val), value=smd_val, kind="numeric", align="right"))

    rows: list[Row] = [Row(cells=tuple(cells))]

    _maybe_append_missing(rows, data, var, group_keys, group_masks,
                          show_overall, show_p, show_q, show_smd,
                          pct_digits=pct_digits, missing_mode=missing_mode,
                          weights=weights)
    return rows, test_used


# ----------------------------------------------------------------------
# Categorical rows
# ----------------------------------------------------------------------

def _categorical_rows(
    data: pd.DataFrame,
    var: str,
    label: str,
    by: str | None,
    group_keys: list[Any],
    group_masks: dict[Any, pd.Series],
    *,
    kind: VarKind,
    pct_digits: int,
    show_overall: bool,
    show_p: bool,
    show_q: bool,
    show_smd: bool,
    missing_mode: str,
    bold_p_mode: bool,
    bold_p_threshold: float,
    test_override: str | None = None,
    weights: pd.Series | None = None,
    design: SurveyDesign | None = None,
) -> tuple[list[Row], str | None]:
    """Produce a header row + one row per level (+ optional missing)."""
    s_all = data[var]
    # Determine levels from full data so all groups share them.
    if _is_cat_dtype(s_all):
        levels: list[Any] = list(s_all.cat.categories)
    else:
        levels = sorted(s_all.dropna().unique(), key=_sort_key)

    # All-NaN variable: emit a single "no data" row and any missing-row
    # follow-up. Without this short-circuit the multi-level path produces
    # a confusing group header with empty group cells.
    if len(levels) == 0:
        empty_cells: list[Cell] = [make_cell(label, align="left", bold=True)]
        if show_overall:
            empty_cells.append(make_cell("—", value=None, kind="numeric", align="right"))
        for _ in group_keys:
            empty_cells.append(make_cell("—", value=None, kind="numeric", align="right"))
        if show_p:
            empty_cells.append(make_cell("—", value=None, kind="p_value", align="right"))
        if show_q:
            empty_cells.append(make_cell("—", value=None, kind="q_value", align="right"))
        if show_smd:
            empty_cells.append(make_cell("—", value=None, kind="numeric", align="right"))
        empty_rows: list[Row] = [Row(cells=tuple(empty_cells))]
        _maybe_append_missing(empty_rows, data, var, group_keys, group_masks,
                              show_overall, show_p, show_q, show_smd,
                              pct_digits=pct_digits, missing_mode=missing_mode,
                              weights=weights)
        return empty_rows, None

    p_value: float | None = None
    test_used: str | None = None
    if show_p and by is not None:
        if test_override is not None:
            res = run_named_test(test_override, s_all, data[by], kind="categorical")
        elif weights is not None:
            # Survey-weighted data → Rao–Scott corrected chi-square.
            # The implementation here uses the *first-order* Kish-DEFF
            # approximation, which only knows about the weight values
            # (not strata or clusters). Under a SurveyDesign with strata
            # or clusters the correct correction uses the full design
            # covariance (Rao & Scott 1981, eqn 3.4), which we do not
            # implement. Warn the user so they don't take the p-value as
            # publication-grade and point them at R ``survey::svychisq``.
            if design is not None and (
                design.strata is not None or design.cluster is not None
            ):
                import warnings
                warnings.warn(
                    f"Rao–Scott chi-square for {var!r}: pysofra uses the "
                    "first-order Kish-DEFF approximation which does not "
                    "account for stratification or clustering in the "
                    "provided SurveyDesign. The reported p-value may "
                    "disagree with R ``survey::svychisq`` by 10–15% or "
                    "more. For design-grade chi-square inference on "
                    "complex surveys, call ``survey::svychisq`` in R.",
                    UserWarning,
                    stacklevel=2,
                )
            res = rao_scott_chisq(s_all, data[by], weights)
        else:
            res = categorical_test(s_all, data[by])
        p_value = res.p_value
        test_used = res.test if res.p_value is not None else None

    smd_val: float | None = None
    if show_smd and by is not None:
        smd_val = categorical_smd(
            s_all, data[by], levels=levels, weights=weights,
        )

    bold_row = bold_p_mode and p_value is not None and p_value < bold_p_threshold

    def _weighted_n_tot(mask: pd.Series, target_value: Any) -> tuple[float, float]:
        sub = data.loc[mask]
        valid = sub[var].notna()
        if weights is not None:
            w_sub = weights.loc[sub.index]
            tot = float(w_sub[valid].sum())
            n_match = float(w_sub[valid & (sub[var] == target_value)].sum())
        else:
            tot = float(valid.sum())
            n_match = float((sub[var] == target_value).sum())
        return n_match, tot

    # Dichotomous: render as a single row "var, level = level1" with n (%)
    # for the second (success) level. This matches gtsummary defaults.
    if kind == "dichotomous" and len(levels) == 2:
        success = levels[1]
        success_label = _fmt_level(success)
        row_label = f"{label} = {success_label}"
        cells: list[Cell] = [make_cell(row_label, align="left", bold=bold_row)]
        if show_overall:
            n, tot = _weighted_n_tot(pd.Series(True, index=data.index), success)
            cells.append(make_cell(_fmt_weighted_n_pct(n, tot, pct_digits),
                                   kind="numeric", align="right"))
        for k in group_keys:
            n, tot = _weighted_n_tot(group_masks[k], success)
            cells.append(make_cell(_fmt_weighted_n_pct(n, tot, pct_digits),
                                   kind="numeric", align="right"))
        if show_p:
            cells.append(make_cell(fmt_p_value(p_value), value=p_value,
                                   kind="p_value", align="right", bold=bold_row))
        if show_q:
            cells.append(make_cell("", value=None, kind="q_value", align="right"))
        if show_smd:
            cells.append(make_cell(fmt_smd(smd_val), value=smd_val,
                                   kind="numeric", align="right"))
        rows: list[Row] = [Row(cells=tuple(cells))]
        _maybe_append_missing(rows, data, var, group_keys, group_masks,
                              show_overall, show_p, show_q, show_smd,
                              pct_digits=pct_digits, missing_mode=missing_mode,
                              weights=weights)
        return rows, test_used

    # Multi-level categorical: header row with overall N + p-value + SMD,
    # then one indented row per level.
    rows = []
    hdr: list[Cell] = [make_cell(label, align="left", bold=True)]
    if show_overall:
        hdr.append(make_cell("", value=None))
    for _ in group_keys:
        hdr.append(make_cell("", value=None))
    if show_p:
        hdr.append(make_cell(fmt_p_value(p_value), value=p_value,
                             kind="p_value", align="right",
                             bold=bold_row))
    if show_q:
        hdr.append(make_cell("", value=None, kind="q_value", align="right"))
    if show_smd:
        hdr.append(make_cell(fmt_smd(smd_val), value=smd_val,
                             kind="numeric", align="right"))
    rows.append(Row(cells=tuple(hdr), is_group_header=True))

    for lvl in levels:
        cells = [make_cell(f"{_fmt_level(lvl)}", align="left", indent=1)]
        if show_overall:
            n, tot = _weighted_n_tot(pd.Series(True, index=data.index), lvl)
            cells.append(make_cell(_fmt_weighted_n_pct(n, tot, pct_digits),
                                   kind="numeric", align="right"))
        for k in group_keys:
            n, tot = _weighted_n_tot(group_masks[k], lvl)
            cells.append(make_cell(_fmt_weighted_n_pct(n, tot, pct_digits),
                                   kind="numeric", align="right"))
        if show_p:
            cells.append(make_cell("", value=None))
        if show_q:
            cells.append(make_cell("", value=None))
        if show_smd:
            cells.append(make_cell("", value=None))
        rows.append(Row(cells=tuple(cells)))

    _maybe_append_missing(rows, data, var, group_keys, group_masks,
                          show_overall, show_p, show_q, show_smd,
                          pct_digits=pct_digits, missing_mode=missing_mode,
                          weights=weights)
    return rows, test_used


def _maybe_append_missing(
    rows: list[Row],
    data: pd.DataFrame,
    var: str,
    group_keys: list[Any],
    group_masks: dict[Any, pd.Series],
    show_overall: bool,
    show_p: bool,
    show_q: bool,
    show_smd: bool,
    *,
    pct_digits: int,
    missing_mode: str,
    weights: pd.Series | None = None,
) -> None:
    if missing_mode == "never":
        return
    if weights is not None:
        n_miss_overall = float(weights[data[var].isna()].sum())
    else:
        n_miss_overall = float(data[var].isna().sum())
    if missing_mode == "ifany" and n_miss_overall == 0:
        return

    cells: list[Cell] = [make_cell(MISSING_LABEL, align="left", indent=1)]
    if show_overall:
        tot = float(weights.sum()) if weights is not None else float(len(data))
        cells.append(make_cell(_fmt_weighted_n_pct(n_miss_overall, tot, pct_digits),
                               kind="numeric", align="right"))
    for k in group_keys:
        mask = group_masks[k]
        if weights is not None:
            n_miss = float(weights.loc[mask][data.loc[mask, var].isna()].sum())
            tot = float(weights.loc[mask].sum())
        else:
            n_miss = float(data.loc[mask, var].isna().sum())
            tot = float(mask.sum())
        cells.append(make_cell(_fmt_weighted_n_pct(n_miss, tot, pct_digits),
                               kind="numeric", align="right"))
    if show_p:
        cells.append(make_cell("", value=None))
    if show_q:
        cells.append(make_cell("", value=None))
    if show_smd:
        cells.append(make_cell("", value=None))
    rows.append(Row(cells=tuple(cells)))


# ----------------------------------------------------------------------
# add_global_p() — joint Wald p-value per variable for tbl_one /
# tbl_summary. Re-fits a logistic regression per variable on the
# source data (`Logit(by == ref ~ variable [+ adjust_for])`) and
# computes the joint p-value over the variable's coefficients. Adds
# a "global p" column to each row.
#
# Single-coefficient predictors (continuous, dichotomous) get the
# Wald p of that one coefficient. Multi-level categorical predictors
# (k levels → k-1 dummies) get the joint Wald F-test across all
# dummies — same statistic as gtsummary's ``add_global_p()``.
#
# v1 scope: 2-level ``by=``. With ≥3-level ``by=`` the joint test
# requires multinomial logit, which is left out of scope; the column
# is filled with em-dash and a footnote explains why.
# ----------------------------------------------------------------------


def _attach_global_p(
    *,
    data: pd.DataFrame,
    by: str | None,
    variables: tuple[str, ...],
    kinds: dict[str, VarKind],
    labels: dict[str, str],
    rows: list[Row],
    headers: tuple[HeaderRow, ...],
    adjust_for: tuple[str, ...],
    weights_col: str | None = None,
) -> tuple[list[Row], tuple[HeaderRow, ...], list[str]]:
    """Attach a joint-p column to a tbl_one table.

    Walks the existing rows, identifies which rows belong to which
    variable (by label-matching the first cell), and inserts a new
    "global p" column carrying the joint Wald p-value for each
    variable. Single-coefficient predictors get the Wald p directly;
    multi-coefficient (categorical) predictors get the F-test joint p.

    Parameters
    ----------
    data
        The source DataFrame closed over by the ``tbl_one`` rebuild.
    by
        The stratifying column name; ``None`` is unsupported and
        causes the function to return rows unchanged with a footnote
        explaining the column was skipped.
    variables, kinds, labels
        Variable list + kind / label maps from the spec.
    rows
        The already-built body rows (will be re-emitted with one
        extra cell appended).
    headers
        The already-built header rows (will be re-emitted with one
        extra header cell appended).
    adjust_for
        Tuple of column names to include as covariates. Each is
        treated as continuous if numeric and categorical (dummy
        coded) otherwise.

    Returns
    -------
    new_rows, new_headers, extra_footnotes
        Rows / headers with the new column inserted, plus any
        explanatory footnotes (e.g. for variables that couldn't be
        fit).
    """
    extra_footnotes: list[str] = []

    # Validate adjust_for columns up-front — fail fast with a clear
    # ``KeyError`` rather than letting pandas raise a generic
    # ``KeyError: ['NOPE'] not in index`` from deep inside the fit.
    # Matches the validation pattern for ``by=`` and ``weights=``.
    missing_adj = [c for c in adjust_for if c not in data.columns]
    if missing_adj:
        raise KeyError(
            f"add_global_p: adjust_for column(s) {missing_adj!r} not in data"
        )

    if by is None:
        extra_footnotes.append(
            "add_global_p: skipped (no by= column).",
        )
        # Append blank cells so the column shape stays consistent.
        return (
            _append_blank_column(rows),
            _append_header_column(headers, "global p"),
            extra_footnotes,
        )

    by_series = data[by]
    levels = sorted(by_series.dropna().unique(), key=_sort_key)
    if len(levels) != 2:
        # Multinomial logit is out of scope for v1.
        extra_footnotes.append(
            f"add_global_p: by={by!r} has {len(levels)} levels; "
            "v1 supports only 2-level stratification (multinomial "
            "logit not implemented).",
        )
        return (
            _append_blank_column(rows, fill="—"),
            _append_header_column(headers, "global p"),
            extra_footnotes,
        )

    # Compute one joint p-value per variable.
    p_per_var: dict[str, float | None] = {}
    for var in variables:
        p_per_var[var] = _fit_global_p(
            data=data, by=by, by_levels=levels,
            var=var, kind=kinds[var], adjust_for=adjust_for,
            weights_col=weights_col,
        )

    # Walk rows, map each to its variable, and append a cell. The map
    # uses the existing row labels:
    #   - Continuous / categorical parent row: label = labels.get(var, var)
    #   - Dichotomous: label = "varlabel = displayed_level"
    #   - Categorical level rows: indented level text (parent row above)
    #   - Missing sub-row: label = MISSING_LABEL
    # We rely on the build order: variables are processed sequentially
    # and the parent row of each variable always appears before its
    # level / missing sub-rows.
    var_label_to_var = {labels.get(v, v): v for v in variables}
    # For dichotomous "var = Level" rows, also map the prefix.
    dichot_label_to_var = {
        f"{labels.get(v, v)} = ": v for v in variables if kinds[v] == "dichotomous"
    }

    new_rows: list[Row] = []
    for r in rows:
        first = r.cells[0].text
        # Identify variable for this row.
        matched_var: str | None = None
        if first in var_label_to_var:
            matched_var = var_label_to_var[first]
        else:
            for prefix, v in dichot_label_to_var.items():
                if first.startswith(prefix):
                    matched_var = v
                    break

        if matched_var is not None:
            p = p_per_var.get(matched_var)
            cell = make_cell(
                fmt_p_value(p) if p is not None else "—",
                value=p, kind="p_value", align="right",
            )
        else:
            # Sub-row (categorical level, missing): blank so the
            # joint-p is visually anchored to the variable's parent row.
            cell = make_cell("", value=None)

        new_rows.append(
            Row(cells=tuple(list(r.cells) + [cell]),
                is_group_header=r.is_group_header),
        )

    new_headers = _append_header_column(headers, "global p")
    if adjust_for:
        extra_footnotes.append(
            "global p: joint Wald test on the variable's coefficients "
            f"from Logit({by} ~ variable + "
            f"{' + '.join(adjust_for)}).",
        )
    else:
        extra_footnotes.append(
            "global p: joint Wald test on the variable's coefficients "
            f"from Logit({by} ~ variable).",
        )
    return new_rows, new_headers, extra_footnotes


def _fit_global_p(
    *,
    data: pd.DataFrame,
    by: str,
    by_levels: list[Any],
    var: str,
    kind: VarKind,
    adjust_for: tuple[str, ...],
    weights_col: str | None = None,
) -> float | None:
    """Fit one logistic regression and return the joint Wald p-value
    for ``var``'s coefficients.

    Implementation choices:
      * Outcome encoded as ``by == by_levels[1]`` (alphabetically
        second level) so the reference is well-defined.
      * Variable encoded based on inferred kind: continuous columns
        used as-is; dichotomous / categorical columns one-hot encoded
        via ``pd.get_dummies(drop_first=True)``.
      * Adjustment columns each encoded the same way (numeric →
        as-is; non-numeric → dummies).
      * Joint test built as a constraint string
        ``"c1 = 0, c2 = 0, ..."`` over the variable's columns and
        passed to ``model.f_test`` (matches what ``add_global_p()``
        does for ``tbl_regression``).
      * Singular design / convergence failure → ``None`` (renders as
        em-dash; never a misleading numeric).
    """
    # Build the working frame: drop rows with NaN in any required col.
    # Deduplicate column references — when the variable being tested is
    # also listed in ``adjust_for``, the duplicate would (a) make
    # ``data[cols]`` produce a 2-D selection that crashes
    # ``pd.to_numeric``, and (b) make the design matrix singular.
    # Variable always wins; the matching adjustment column is dropped.
    seen: set[str] = set()
    cols: list[str] = []
    for c in (by, var, *adjust_for):
        if c not in seen:
            seen.add(c)
            cols.append(c)
    # If the table was built with frequency weights, include the weights
    # column so the joint Wald-F test uses the same weighted model the
    # rest of the table is based on. Without this, the row-level
    # p-values and global p would silently disagree.
    if weights_col is not None and weights_col not in seen:
        seen.add(weights_col)
        cols.append(weights_col)
    sub = data[cols].dropna()
    if sub.empty or sub[by].nunique() < 2:
        return None

    y = (sub[by] == by_levels[1]).astype(int).to_numpy()

    var_cols = _design_columns(sub, var, kind)
    if not var_cols:
        return None
    adj_cols: list[tuple[str, Any]] = []
    for a in adjust_for:
        if a == var:
            continue  # already in var_cols
        akind = _quick_kind(sub[a])
        for name, col in _design_columns(sub, a, akind):
            adj_cols.append((name, col))

    # Stack into a single design matrix.
    import numpy as np
    import statsmodels.api as sm
    X_parts = [c for _, c in var_cols] + [c for _, c in adj_cols]
    X = np.column_stack(X_parts)
    # Add a constant column (intercept).
    X = sm.add_constant(X, has_constant="add")

    # Column-name registry: index 0 is the const, then var_cols, then
    # adj_cols.
    col_names = ["const"] + [n for n, _ in var_cols] + [n for n, _ in adj_cols]
    if X.shape[1] != len(col_names):  # pragma: no cover — defensive
        return None

    import warnings as _w
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # statsmodels convergence chatter
            # Honour weights by routing through GLM(Binomial). We use
            # ``var_weights`` rather than ``freq_weights``: ``freq_weights``
            # treats the weight as an integer *count of repeats* and so
            # scales ``df_resid`` by ``Σw`` — which dramatically inflates
            # the effective sample size for non-integer sampling weights
            # (a survey weight calibrated to a 200k population would push
            # df_resid to 200k instead of n). ``var_weights`` keeps
            # ``df_resid = n − k``, which is the appropriate convention
            # for sampling / IPW weights where the weight does not
            # represent a count. For full design-based inference (with
            # strata or clusters) use ``ps.SurveyDesign`` end-to-end;
            # the joint p test here is an SRS-weighted Wald-F.
            if weights_col is not None:
                w_arr = sub[weights_col].to_numpy(dtype=float)
                fam = sm.families.Binomial()
                res = sm.GLM(y, X, family=fam, var_weights=w_arr).fit(disp=False)
            else:
                res = sm.Logit(y, X).fit(disp=False, method="newton",
                                          maxiter=100)
    except Exception:  # pragma: no cover — defensive: singular design / no convergence
        return None

    if not hasattr(res, "f_test"):  # pragma: no cover
        return None

    # Build the joint hypothesis: variable's dummies = 0.
    var_names = [n for n, _ in var_cols]
    constraint = ", ".join(
        f"x{col_names.index(n)} = 0" for n in var_names
    )
    try:
        ftest = res.f_test(constraint)
        return float(ftest.pvalue)
    except Exception:  # pragma: no cover
        return None


def _design_columns(
    sub: pd.DataFrame, var: str, kind: VarKind,
) -> list[tuple[str, Any]]:
    """Return a list of (column_name, numpy_array) pairs for a single
    variable, dummy-coded if categorical."""
    s = sub[var]
    if kind == "continuous":
        return [(var, pd.to_numeric(s, errors="coerce").to_numpy())]
    # Dichotomous and categorical both go through one-hot encoding;
    # ``drop_first=True`` keeps the design full-rank.
    dummies = pd.get_dummies(s, prefix=var, drop_first=True, dtype=float)
    return [(c, dummies[c].to_numpy()) for c in dummies.columns]


def _quick_kind(s: pd.Series) -> VarKind:
    """Best-effort kind inference for adjustment columns."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
        return "continuous"
    if s.nunique() <= 2:
        return "dichotomous"
    return "categorical"


def _append_header_column(
    headers: tuple[HeaderRow, ...], label: str,
) -> tuple[HeaderRow, ...]:
    out = []
    for hr in headers:
        new_cells = list(hr.cells) + [HeaderCell(text=label, align="center", bold=True)]
        out.append(HeaderRow(cells=tuple(new_cells)))
    return tuple(out)


def _append_blank_column(rows: list[Row], fill: str = "") -> list[Row]:
    return [
        Row(cells=tuple(list(r.cells) + [make_cell(fill, value=None)]),
            is_group_header=r.is_group_header)
        for r in rows
    ]
